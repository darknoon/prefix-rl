from env.svg.svg_rlrf_reward import (
    svg_env,
    get_image_comparator,
    rasterize_svg,
    extract_svg_text,
    compute_rewards,
    PreprocessedResponse,
    ImageComparisonResult,
    SVGRewards,
)
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image, ImageChops
from openai import OpenAI
import os
from base64 import b64encode
from io import BytesIO
from typing import Callable, TypedDict, Optional
import argparse
from functools import partial
import numpy as np
from pathlib import Path
from tqdm import tqdm

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
    result: PreprocessedResponse,
    image_scores: ImageComparisonResult,
    id: str = "0",
    output_dir: str = ".",
):
    diff = ImageChops.difference(result.svg_im, result.svg_im_gt)
    diff_canny = ImageChops.difference(image_scores.canny_im, image_scores.canny_im_ref)

    # Save ground truth and generated images to files with id prefix in output_dir
    output_dir = Path(output_dir)
    gt_filename = output_dir / f"{id}_gt.png"
    gen_filename = output_dir / f"{id}_gen.png"
    canny_im_filename = output_dir / f"{id}_canny_im.png"
    canny_im_ref_filename = output_dir / f"{id}_canny_im_ref.png"
    diff_filename = output_dir / f"{id}_diff.png"
    diff_canny_filename = output_dir / f"{id}_diff_canny.png"

    result.svg_im_gt.save(gt_filename)
    result.svg_im.save(gen_filename)
    image_scores.canny_im.save(canny_im_filename)
    image_scores.canny_im_ref.save(canny_im_ref_filename)
    diff.save(diff_filename)
    diff_canny.save(diff_canny_filename)

    output = []
    output.append("# Debug Compare\n\n")
    output.append("## Ground Truth vs Generated Image\n\n")
    # Show images side-by-side using HTML
    output.append(
        f'<div style="display: flex; gap: 20px;">'
        f'<div style="text-align: center;"><img src="{gt_filename.name}" style="max-width: 300px;"><br><b>Ground Truth</b></div>'
        f'<div style="text-align: center;"><img src="{gen_filename.name}" style="max-width: 300px;"><br><b>Generated</b></div>'
        f"</div>\n\n"
    )
    # Add a details element with the canny images
    output.append(
        f"<details>\n"
        f"  <summary>Show Canny Edge Images</summary>\n"
        f'  <div style="display: flex; gap: 20px;">'
        f'<div style="text-align: center;"><img src="{canny_im_ref_filename.name}" style="max-width: 300px;"><br><b>Canny GT</b></div>'
        f'<div style="text-align: center;"><img src="{canny_im_filename.name}" style="max-width: 300px;"><br><b>Canny Generated</b></div>'
        f"</div>\n"
        f"</details>\n\n"
    )
    # Add a details element with the diff image and canny diff image side by side
    diff_filename = f"{id}_diff.png"
    diff_canny_filename = f"{id}_diff_canny.png"
    output.append(
        f"<details>\n"
        f"  <summary>Show Diff</summary>\n"
        f'  <div style="display: flex; gap: 20px;">'
        f'<div style="text-align: center;"><img src="{diff_filename}" style="max-width: 300px;"><br><b>Diff</b></div>'
        f'<div style="text-align: center;"><img src="{diff_canny_filename}" style="max-width: 300px;"><br><b>Canny Diff</b></div>'
        f"</div>\n"
        f"</details>\n\n"
    )
    output.append("## Image Comparison Scores\n\n")
    output.append(f"- **l2**: {image_scores.l2}\n")
    output.append(f"- **l2_canny**: {image_scores.l2_canny}\n")
    output.append(f"- **dreamsim**: {image_scores.dreamsim}\n")
    output.append(f"- **dreamsim_canny**: {image_scores.dreamsim_canny}\n")
    return "".join(output)


def eval_example(example: Example, client: Callable[[str, Image.Image], str]):
    prompt = example["prompt"]
    gt_svg = extract_svg_text(example["completion"])
    gt_image = example["image"]
    if gt_image is None:
        svg_im_bytes, _, _ = rasterize_svg(gt_svg)
        gt_image = Image.open(BytesIO(svg_im_bytes))
    elif isinstance(gt_image, list):
        # multiple images in a single example?
        if len(gt_image) == 1:
            gt_image = gt_image[0]
        else:
            raise ValueError(f"Expected 1 image, got {len(gt_image)}")

    if not isinstance(gt_image, Image.Image):
        raise ValueError(f"Expected Image.Image, got {type(gt_image)}")

    response = client(prompt, gt_image)
    result = svg_env(response, gt_svg)
    if result is not None:
        image_scores = get_image_comparator().compare_images(
            result.svg_im, result.svg_im_gt
        )
        rewards = compute_rewards(result, image_scores)
    return rewards, image_scores, result


def default_prompt(image: Image.Image) -> str:
    return f"Please recreate the image as accurately as possible as an SVG of width {image.width} and height {image.height}."


def calculate_stats(results: list[SVGRewards]) -> dict:
    if not results:
        return {}
    # Collect all reward keys
    reward_keys = ["overall", "l2", "l2_canny", "dreamsim", "dreamsim_canny"]
    stats = {}
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
    num_eval_examples: int = 100,
    debug_dump: bool = False,
    output_dir: Path | None = None,
):
    dataset_name = config["name"]
    split = config["split"]
    # streaming=True allows us to not load the entire dataset (which is very large)
    dataset = load_dataset(dataset_name, split=f"{split}", streaming=True)

    # INSERT_YOUR_CODE
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

    results = []
    if debug_dump:
        name_safe = dataset_name.replace("/", "_")
        model_safe = (
            client.keywords["model_name"]
            if hasattr(client, "keywords") and "model_name" in client.keywords
            else "model"
        ).replace("/", "_")
        dir_name = f"{name_safe}_{model_safe}"
        if output_dir is None:
            output_dir = Path("eval") / dir_name
        if output_dir.exists():
            import shutil

            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / f"eval_debug_{dir_name}.md"
        with md_path.open("w") as f:
            for i, example in enumerate(tqdm(dataset, desc="Evaluating examples")):
                id = f"{i:05d}"
                rewards, image_scores, result = eval_example(example, client)
                f.write(
                    debug_compare_md(result, image_scores, id=id, output_dir=output_dir)
                )
                results.append(rewards)
                f.flush()
            f.write("# Total Rewards\n\n")
            f.write(f"{calculate_stats(results)}\n")
    else:
        for example in tqdm(dataset, desc="Evaluating examples"):
            rewards, image_scores, result = eval_example(example, client)
            results.append(rewards)
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
    default=5,
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
    help="Model name for OpenAI/Anthropic/Google clients. Ignored for vllm.",
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

    rewards = run_eval(
        dataset_config,
        client_fn,
        num_eval_examples=args.num_eval_examples,
        debug_dump=args.debug_dump,
        output_dir=args.output_dir,
    )
