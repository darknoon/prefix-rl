from env.svg.svg_rlrf_reward import (
    svg_env,
    get_image_comparator,
    rasterize_svg,
    extract_svg_text,
    compute_rewards,
    PreprocessedResponse,
    ImageComparisonResult,
    SVGRewards,
    MergedResult,
    write_debug_images_dict,
)
from cairosvg.parser import ParseError
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image, ImageChops
from openai import OpenAI
import os
from base64 import b64encode
from io import BytesIO
from typing import Callable, TypedDict, Optional, Tuple, Any
import argparse
from functools import partial
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import json
import logging
import shutil

load_dotenv()


class DatasetConfig(TypedDict):
    name: str
    split: str
    image_key: str
    completion_key: str
    prompt_key: Optional[str]


class Example(TypedDict):
    prompt: str
    image: Image.Image
    completion: str


datasets: dict[str, DatasetConfig] = {
    "simple-shapes": {
        "name": "darknoon/simple-shapes-svg",
        "split": "train",
        "image_key": "image",
        "completion_key": "svg",
        "prompt_key": None,
    },
    "svg-stack": {
        "name": "darknoon/svg-stack-filtered",
        "split": "test",
        "image_key": "image",
        "completion_key": "completion",
        "prompt_key": "prompt",
    },
}


def image_to_data_url(image: Image.Image) -> str:
    with BytesIO() as buffered:
        image.save(buffered, format="PNG")
        u = b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{u}"


def prompt_to_messages(prompt: str, image: Image.Image) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"detail": "auto", "url": image_to_data_url(image)},
                },
            ],
        }
    ]


def openai_client(
    prompt: str, image: Image.Image, model_name: str = "gpt-4.1-mini"
) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name, messages=prompt_to_messages(prompt, image)
    )
    return response.choices[0].message.content


def vllm_client(prompt: str, image: Image.Image, server_url: str) -> str:
    client = OpenAI(base_url=server_url)
    response = client.chat.completions.create(
        model="", messages=prompt_to_messages(prompt, image)
    )
    return response.choices[0].message.content


def debug_compare_md(
    result: Optional[PreprocessedResponse] = None,
    image_scores: Optional[ImageComparisonResult] = None,
    id: str = "0",
    output_dir: Path = Path("."),
    error: Optional[str] = None,
) -> Tuple[str, dict]:
    output = []
    rel_paths = {}
    if error is not None:
        output.append("# Debug Compare (Error)\n\n")
        output.append(f"## Error for example {id}:\n\n")
        output.append("```")
        output.append(f"{error}\n")
        output.append("```")
        return "".join(output), rel_paths
    if result is None or image_scores is None:
        output.append("# Debug Compare\n\n")
        output.append(f"No result or image scores available for example {id}.\n")
        return "".join(output), rel_paths

    rel_paths = write_debug_images_dict(result, image_scores, output_dir, f"{id}_")

    output.append("# Debug Compare\n\n")
    output.append("## Ground Truth vs Generated Image\n\n")
    # Show images side-by-side using HTML
    output.append(
        f'<div style="display: flex; gap: 20px;">'
        f'<div style="text-align: center;"><img src="{rel_paths["svg_gt"]}" style="max-width: 300px;"><br><b>Ground Truth</b></div>'
        f'<div style="text-align: center;"><img src="{rel_paths["svg"]}" style="max-width: 300px;"><br><b>Generated</b></div>'
        f"</div>\n\n"
    )
    # Add a details element with the canny images
    output.append(
        f"<details>\n"
        f"  <summary>Show Canny Edge Images</summary>\n"
        f'  <div style="display: flex; gap: 20px;">'
        f'<div style="text-align: center;"><img src="{rel_paths["canny_gt"]}" style="max-width: 300px;"><br><b>Canny GT</b></div>'
        f'<div style="text-align: center;"><img src="{rel_paths["canny"]}" style="max-width: 300px;"><br><b>Canny Generated</b></div>'
        f"</div>\n"
        f"</details>\n\n"
    )
    # Add a details element with the diff image and canny diff image side by side
    output.append(
        f"<details>\n"
        f"  <summary>Show Diff</summary>\n"
        f'  <div style="display: flex; gap: 20px;">'
        f'<div style="text-align: center;"><img src="{rel_paths["diff"]}" style="max-width: 300px;"><br><b>Diff</b></div>'
        f'<div style="text-align: center;"><img src="{rel_paths["diff_canny"]}" style="max-width: 300px;"><br><b>Canny Diff</b></div>'
        f"</div>\n"
        f"</details>\n\n"
    )
    output.append("## Image Comparison Scores\n\n")
    output.append(f"- **l2**: {image_scores.l2}\n")
    output.append(f"- **l2_canny**: {image_scores.l2_canny}\n")
    output.append(f"- **dreamsim**: {image_scores.dreamsim}\n")
    output.append(f"- **dreamsim_canny**: {image_scores.dreamsim_canny}\n")
    return "".join(output), rel_paths


class LLMClientException(Exception):
    pass


def eval_example(
    example: Example, client: Callable[[str, Image.Image], str]
) -> tuple[
    SVGRewards | None,
    ImageComparisonResult | None,
    PreprocessedResponse | None,
    Exception | None,
]:
    prompt = example["prompt"]
    gt_svg = extract_svg_text(example["completion"])
    if gt_svg is None:
        return None, None, None, ValueError("Ground truth error: No SVG found")
    # 2. Image loading
    try:
        gt_image = example["image"]
        if gt_image is None:
            svg_im_bytes, _, _ = rasterize_svg(gt_svg)
            gt_image = Image.open(BytesIO(svg_im_bytes))
        if not isinstance(gt_image, Image.Image):
            raise ValueError(f"Expected Image.Image, got {type(gt_image)}: {gt_image}")
    except Exception as e:
        return None, None, None, e
    # 3. Run completions
    try:
        completion = client(prompt, gt_image)
    except Exception as e:
        return None, None, None, e
    # 4. Parse and rasterize SVG
    try:
        rasterized = svg_env(completion, gt_svg)
    except Exception as e:
        return None, None, None, e
    # 5. Image comparison
    try:
        image_scores = get_image_comparator().compare_images(
            rasterized.svg_im, rasterized.svg_im_gt
        )
    except Exception as e:
        return None, None, rasterized, e
    # 6. Reward computation
    try:
        rewards = compute_rewards(rasterized, image_scores)
    except Exception as e:
        return None, image_scores, rasterized, e
    # OK!
    return rewards, image_scores, rasterized, None


def default_prompt(image: Image.Image) -> str:
    return f"Please recreate the image as accurately as possible as an SVG of width {image.width} and height {image.height}."


class Stats(TypedDict):
    mean: float
    std: float
    min: float
    max: float


def calculate_stats(results: list[SVGRewards]) -> dict[str, Stats]:
    if not results:
        return {}
    # Collect all reward keys
    reward_keys = ["overall", "l2", "l2_canny", "dreamsim", "dreamsim_canny"]
    stats: dict[str, Stats] = {}
    for key in reward_keys:
        values = [getattr(r, key) for r in results]
        stats[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return stats


def run_eval(
    config: DatasetConfig,
    client: Callable[[str, Image.Image], str],
    output_dir: Path,
    num_eval_examples: int = 100,
    debug_dump: bool = False,
    num_workers: int = 1,
    model_name: str | None = None,
):
    if model_name is None:
        raise ValueError("model_name must be provided")

    dataset_name = config["name"]
    split = config["split"]
    # streaming=True allows us to not load the entire dataset (which is very large)
    dataset = load_dataset(dataset_name, split=f"{split}", streaming=True)
    # Take only the first num_eval_examples from the streaming dataset
    dataset = dataset.take(num_eval_examples)
    image_key = config["image_key"]
    completion_key = config["completion_key"]
    prompt_key = config["prompt_key"]

    def process_example(x: dict) -> Example:
        """Convert the dataset example to the correct format"""
        image = x[image_key]
        prompt = x[prompt_key] if prompt_key is not None else default_prompt(image)
        completion = x[completion_key]
        return {
            "prompt": prompt,
            "image": image,
            "completion": completion,
        }

    dataset = dataset.map(process_example, remove_columns=dataset.column_names)
    example_list = list(dataset)

    # Prepare debug output directory and markdown path if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "eval_debug.md"

    # Setup logging to file
    log_path = output_dir / "log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    jsonl_path = output_dir / "llm_responses.jsonl"

    def worker(i: int, example: Example) -> tuple[SVGRewards, str, dict]:
        id = f"{i:05d}"
        image_paths = {}
        rewards, image_scores, result, exc = eval_example(example, client)
        status = "OK" if exc is None else "FAIL"
        error_message = None
        if exc is not None:
            if isinstance(exc, ParseError):
                error_message = "SVG_PARSE"
            else:
                error_message = str(exc)
        debug_md = None
        # Write markdown and debug images if requested
        if debug_dump:
            debug_md, image_paths = debug_compare_md(
                result, image_scores, id=id, output_dir=output_dir, error=error_message
            )
        # Try to get LLM response if possible
        prompt = example["prompt"]
        svg_gt = extract_svg_text(example["completion"])
        # Build record for jsonl
        record: MergedResult = {
            "id": id,
            "prompt": prompt,
            "response": result.response_str,
            "svg": result.svg,
            "svg_gt": svg_gt,
            # dump images if we have them - map logical names to field names
            "svg_image": image_paths.get("svg", ""),
            "svg_gt_image": image_paths.get("svg_gt", ""),
            "canny": image_paths.get("canny", ""),
            "canny_gt": image_paths.get("canny_gt", ""),
            "diff": image_paths.get("diff", ""),
            "diff_canny": image_paths.get("diff_canny", ""),
            **{
                (f"reward_{k}"): getattr(rewards, k, None)
                for k in ["overall", "l2", "l2_canny", "dreamsim", "dreamsim_canny"]
            },
            "status": status,
            "error": error_message,
        }
        return (rewards, debug_md, record)

    results = []
    debug_mds = []
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, i, ex) for i, ex in enumerate(example_list)]
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Evaluating examples",
        ):
            rewards, md, record = fut.result()
            results.append(rewards)
            if md is not None:
                debug_mds.append(md)
            records.append(record)

    stats = calculate_stats(results)
    if debug_dump:
        with md_path.open("w") as f:
            for md in debug_mds:
                f.write(md)
            f.write("# Total Rewards\n\n")
            for key, stat in stats.items():
                mean, std, min_, max_ = (
                    stat["mean"],
                    stat["std"],
                    stat["min"],
                    stat["max"],
                )
                f.write(
                    f"**{key}**: {mean:.2f} ± {std:.2f} ({min_:.2f}-{max_:.2f})\n\n"
                )

    # Write summarized stats as CSV
    csv_path = output_dir / "stats_summary.csv"
    with csv_path.open("w") as f:
        f.write("model_name,reward_key,mean,std,min,max\n")
        for key, stat in stats.items():
            mean, std, min_, max_ = (
                stat["mean"],
                stat["std"],
                stat["min"],
                stat["max"],
            )
            f.write(f"{model_name},{key},{mean:.4f},{std:.4f},{min_:.4f},{max_:.4f}\n")

    # Write all LLM responses and metadata to jsonl
    with jsonl_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return results


def anthropic_client(
    prompt: str, image: Image.Image, model_name: str = "claude-3-opus-20240229"
) -> str:
    # Stub for Anthropic client
    raise NotImplementedError("Anthropic client not implemented yet.")


def google_client(
    prompt: str, image: Image.Image, model_name: str = "gemini-pro-vision"
) -> str:
    # Stub for Google client
    raise NotImplementedError("Google client not implemented yet.")


parser = argparse.ArgumentParser(description="Evaluate SVG generation models.")
parser.add_argument(
    "--dataset",
    type=str,
    default="simple-shapes",
    choices=list(datasets.keys()),
    help="Dataset config to use.",
)
parser.add_argument(
    "-n",
    "--num_eval_examples",
    type=int,
    default=10,
    help="Number of examples to evaluate.",
)
parser.add_argument(
    "--client",
    type=str,
    default="openai",
    choices=["openai", "anthropic", "google", "vllm"],
    help="Which client/model to use.",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="gpt-4.1-mini",
    help="Model name for OpenAI/Anthropic/Google clients. For vllm, just determines what directory to write to, you must load the correct model separately.",
)
parser.add_argument(
    "--vllm_endpoint",
    type=str,
    default=None,
    help="Base URL for vllm endpoint (required if client is vllm).",
)
parser.add_argument(
    "--debug_dump", action="store_true", help="Dump debug markdown with images."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to write debug markdown and images (default: ./eval/{dataset_name}/)",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=1,
    help="Number of parallel workers to use for evaluation.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    dataset_config = datasets[args.dataset]

    if args.client == "openai":
        client_fn = partial(openai_client, model_name=args.model_name)
    elif args.client == "anthropic":
        client_fn = partial(anthropic_client, model_name=args.model_name)
    elif args.client == "google":
        client_fn = partial(google_client, model_name=args.model_name)
    elif args.client == "vllm":
        if not args.vllm_endpoint:
            raise ValueError(
                "--vllm_endpoint must be specified when using vllm client."
            )
        client_fn = partial(vllm_client, server_url=args.vllm_endpoint)
    else:
        raise ValueError(f"Unknown client: {args.client}")

    # Set output_dir to default if not provided
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        dataset_safe = dataset_config["name"].replace("/", "_")
        model_safe = args.model_name.replace("/", "_")
        dir_name = f"{dataset_safe}_{model_safe}"
        output_dir = Path("eval") / dir_name

    rewards = run_eval(
        dataset_config,
        client_fn,
        output_dir,
        num_eval_examples=args.num_eval_examples,
        debug_dump=args.debug_dump,
        num_workers=args.num_workers,
        model_name=args.model_name,
    )

    print(
        f"Output written to: {output_dir.resolve().relative_to(Path.cwd()) if not output_dir.is_absolute() else output_dir.resolve()}"
    )
