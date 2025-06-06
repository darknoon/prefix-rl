# %% [markdown]
# From [the RLRF paper](https://arxiv.org/pdf/2505.20793)
# > We preprocess this data by rounding decimals to two
# significant figures, removing XML headers, and filtering out samples with excessively long URLs or
# embedded base64 images, which could lead the model to memorize irrelevant content

# %%
from datasets import (
    load_dataset,
    Dataset,
    Features,
    Value,
    concatenate_datasets,
    DatasetDict,
    Image as ImageFeature,
)

from transformers import AutoTokenizer
from PIL import Image

import subprocess
import tempfile
import os
from pathlib import Path
import io
import re

from svg_rlrf_reward import rasterize_svg


rasterize_features = Features(
    {
        "image": ImageFeature(),
        "width": Value("float32"),
        "height": Value("float32"),
    }
)


def rasterize(example):
    try:
        img_bytes, tree, (width, height) = rasterize_svg(
            example["Svg"], min_target=512, max_target=1536
        )
        # todo: use tree to enrich metadata
        return {
            "image": Image.open(io.BytesIO(img_bytes)),
            "width": width,
            "height": height,
        }
    except Exception:
        return {"image": None, "width": None, "height": None}


def strip_kvg_attributes(svg_text: str) -> dict:
    # Remove all kvg: attributes (e.g. kvg:element="foo", kvg:position='bar'), even before closing tags
    # Handles cases like: <g id="kvg:07a50" kvg:element="穐">
    # Also handles attributes before self-closing tags: <path ... kvg:type="㇒"/>
    svg_text = re.sub(
        r'\s+kvg:[\w-]+=(["\']).*?\1(?=\s|/?>)', ' data-kvg-stripped="true"', svg_text
    )
    return {"Svg": svg_text}


# A number of SVGs are cut off in the dataset. We should filter them.
def has_balanced_svg_tags(svg_text: str):
    open_tags = len(re.findall(r"<svg\b[^>]*>", svg_text, flags=re.IGNORECASE))
    close_tags = len(re.findall(r"</svg\s*>", svg_text, flags=re.IGNORECASE))
    if open_tags == 0 or close_tags == 0:
        return False
    if open_tags != close_tags:
        return False
    return True


def svgo_single(
    input_path, output_path, svgo_args, svgo_version: str = "latest"
) -> tuple[str, None] | tuple[None, str]:
    """Returns either the optimized SVG or the error message from svgo."""
    single_cmd = [
        "npx",
        "-y",
        f"svgo@{svgo_version}",
        "-i",
        input_path,
        "-o",
        output_path,
        *svgo_args,
    ]
    proc = subprocess.run(single_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not os.path.exists(output_path):
        return None, proc.stderr.decode("utf-8")
    else:
        with open(output_path, "r", encoding="utf-8") as f:
            return f.read(), None


# Latest version of svgo fails on some files, so we use an older version
def svgo_batch(
    svg_texts,
    names: list[str] | None = None,
    precision: int = 2,
    pretty: bool = False,
    svgo_version: str = "3.1.0",
    retry_with_latest: bool = True,
    write_failures_to: Path | None = None,
) -> tuple[list[str], list[str]]:
    """
    Optimize a batch of SVGs by writing them to a temp directory and running svgo on the directory.
    Returns a list of optimized SVG strings in the same order.
    If the batch fails, runs on individual files so that only the bad files are excluded.
    Uses the same input/output directories for individual files.
    """

    def filename(i: int) -> str:
        if names is None:
            return f"{i:06d}.svg"
        else:
            return names[i]

    if write_failures_to is not None:
        write_failures_to = Path(write_failures_to)
        write_failures_to.mkdir(parents=True, exist_ok=True)

    svgo_args = ["--precision", str(precision)]
    if pretty:
        svgo_args.append("--pretty")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        # Write each SVG to a file in the input directory
        for i, svg in enumerate(svg_texts):
            fname = filename(i)
            with open(os.path.join(input_dir, fname), "w", encoding="utf-8") as f:
                f.write(svg)
        # Run svgo on the directory

        cmd = [
            "npx",
            "-y",
            f"svgo@{svgo_version}",
            "-f",
            input_dir,
            "-o",
            output_dir,
            *svgo_args,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        optimized_svgs = []
        errors = []
        batch_failed = proc.returncode != 0
        if batch_failed:
            print(
                f"SVGO on {[filename(i) for i in range(len(svg_texts))]} failed, retrying with individual files."
            )
            # Try to optimize each SVG individually, reusing the same input/output dirs
            for i, svg in enumerate(svg_texts):
                fname = filename(i)
                input_path = os.path.join(input_dir, fname)
                output_path = os.path.join(output_dir, fname)

                optimized_svg, error = svgo_single(
                    input_path, output_path, svgo_args, svgo_version
                )

                if optimized_svg is None:
                    if write_failures_to is not None:
                        with open(
                            write_failures_to / fname, "w", encoding="utf-8"
                        ) as f:
                            f.write(svg)
                        with open(
                            write_failures_to / f"{fname}_error.txt",
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(error)
                    else:
                        print(f"SVGO failed on {input_path}")
                        print(error)

                    if retry_with_latest:
                        # Try again with latest version of svgo
                        optimized_svg, error = svgo_single(
                            input_path, output_path, svgo_args, svgo_version="latest"
                        )
                        if optimized_svg is not None:
                            print(f"SVGO fixed {input_path} on latest version.")

                # If svgo failed to write output, exclude the file from the dataset (write None to the list)
                optimized_svgs.append(optimized_svg)
                errors.append(error)
        else:
            # Read optimized SVGs in the same order using out_fnames
            for i in range(len(svg_texts)):
                out_path = os.path.join(output_dir, filename(i))
                if not os.path.exists(out_path):
                    print(f"SVGO failed on {out_path}")
                    # If svgo failed to write output, fallback to original
                    optimized_svgs.append(None)
                    errors.append("File not found")
                else:
                    with open(out_path, "r", encoding="utf-8") as f:
                        optimized_svgs.append(f.read())
                        errors.append(None)
        return optimized_svgs, errors


# feature for hf.Features
svgo_metadata = {
    "original_size": Value("int32"),
    "optimized_size": Value("int32"),
    "bytes_saved": Value("int32"),
    "saved_ratio": Value("float32"),
    "svgo_error": Value("string"),
}

failures_dir = Path("svgo_failures")


def svg_optim_batched(batch, write_failures_to=failures_dir, **kwargs):
    optimized_svgs, errors = svgo_batch(
        batch["Svg"], batch["Filename"], write_failures_to=write_failures_to, **kwargs
    )
    metas = []
    for original_svg, optimized_svg, error in zip(batch["Svg"], optimized_svgs, errors):
        if optimized_svg is None:
            metas.append(
                {
                    "original_size": len(original_svg),
                    "optimized_size": -1,
                    "bytes_saved": -1,
                    "saved_ratio": -1,
                    "svgo_error": error,
                }
            )
        else:
            metas.append(
                {
                    "original_size": len(original_svg),
                    "optimized_size": len(optimized_svg),
                    "bytes_saved": len(original_svg) - len(optimized_svg),
                    "saved_ratio": (
                        (len(original_svg) - len(optimized_svg)) / len(original_svg)
                    ),
                    "svgo_error": None,
                }
            )
    return {"Svg": optimized_svgs, "svgo": metas}


def no_base64_image(svg_text):
    # Look for <image ... xlink:href="data:image/..." or href="data:image/..."
    return not re.search(r'(xlink:href|href)\s*=\s*["\']data:image\/', svg_text)


# filter out any images that are a solid color
def not_solid_color(image: Image.Image):
    assert image.mode == "RGB", "Image must be RGB"
    extrema = image.getextrema()
    return any(lo != hi for lo, hi in extrema)


def add_token_counts(tokenizer: AutoTokenizer, ex: dict[str, list[any]]):
    svgs_tokenized = tokenizer(ex["Svg"], padding=False, truncation=False)["input_ids"]
    return {"svg_token_count": [len(t) for t in svgs_tokenized]}


def no_text(svg_text: str):
    return not re.search(r"<text", svg_text)


def run_all_processing(
    subset: Dataset,
    min_tokens: int = 100,
    tokenizer_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    num_proc: int | None = None,
):
    """
    Run all processing steps in sequence.
    """
    num_proc = num_proc or os.cpu_count() or 4
    features_with_svgo: Features = subset.features.copy()
    features_with_svgo.update(rasterize_features)
    features_with_svgo["svgo"] = svgo_metadata

    # Tokenize and filter by token count
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, fast=True)

    def add_token_counts(ex: dict[str, list[any]]):
        svgs_tokenized = tokenizer(ex["Svg"], padding=False, truncation=False)[
            "input_ids"
        ]
        return {"svg_token_count": [len(t) for t in svgs_tokenized]}

    # 1. Remove solid-color images
    filtered = (
        subset.filter(lambda ex: len(ex["Svg"]) < 16_000, desc="len() < 16k chars")
        .filter(
            has_balanced_svg_tags,
            input_columns="Svg",
            num_proc=num_proc,
            desc="Balanced <svg> vs </svg>",
        )
        .filter(
            no_base64_image,
            input_columns="Svg",
            num_proc=num_proc,
            desc="Filtering base64 images",
        )
        .filter(
            no_text,
            input_columns="Svg",
            num_proc=num_proc,
            desc="Filtering <text> elements",
        )
        .map(rasterize, num_proc=num_proc, desc="Rasterizing")
        .filter(
            lambda im: im is not None,
            input_columns="image",
            desc="Filtering Empty Images",
            batch_size=512,
            num_proc=num_proc,
        )
        .filter(
            not_solid_color,
            input_columns="image",
            num_proc=num_proc,
            desc="Not Solid Color",
        )
        .filter(
            strip_kvg_attributes,
            input_columns="Svg",
            num_proc=num_proc,
            desc="Filtering kvg:*",
        )
        .map(
            svg_optim_batched,
            batched=True,
            batch_size=32,
            features=features_with_svgo,
            num_proc=num_proc,
            desc="Optimizing",
        )
        .filter(
            lambda svg: svg is not None,
            input_columns="Svg",
            num_proc=num_proc,
            desc="Filtering Unoptimizable SVGs",
        )
        .map(
            add_token_counts,
            batched=True,
            batch_size=256,
            num_proc=num_proc,
            desc="Tokenizing",
        )
        .filter(
            lambda count: count >= min_tokens,
            input_columns="svg_token_count",
            num_proc=num_proc,
            desc="Token Count",
        )
    )
    return filtered


# Local version
def main(
    input_dataset_name: str = "starvector/svg-stack",
    tokenizer_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    chunk_size: int = 10_000,
    test_split_ratio: float = 0.05,
):
    # Load dataset
    input_train = load_dataset(input_dataset_name, split="train")
    input_val = load_dataset(input_dataset_name, split="val")

    # Split train into train and test
    split = input_train.train_test_split(test_size=test_split_ratio, seed=42)
    train_dataset = split["train"]
    test_dataset = split["test"]

    def process_in_chunks(dataset, name="dataset"):
        num_rows = len(dataset)
        chunks = []
        for i in range(0, num_rows, chunk_size):
            print(f"Processing {name} chunk {i}–{min(i + chunk_size, num_rows)}")
            chunk = dataset.select(range(i, min(i + chunk_size, num_rows)))
            processed = run_all_processing(chunk, tokenizer_name=tokenizer_name)
            chunks.append(processed)
        return concatenate_datasets(chunks) if len(chunks) > 1 else chunks[0]

    print("Building train dataset")
    train = process_in_chunks(train_dataset, name="train")
    print("Building test dataset")
    test = process_in_chunks(test_dataset, name="test")
    print("Building val dataset")
    val = process_in_chunks(input_val.select(range(len(input_val))), name="val")
    print("Done")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    dataset_final = DatasetDict({"train": train, "val": val, "test": test})
    dataset_final.push_to_hub("darknoon/svg-stack-filtered")


if __name__ == "__main__":
    main()
