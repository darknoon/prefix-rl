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
    MIN_REWARD,
    UsageData,
)
from xml.etree.ElementTree import ParseError
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image, ImageChops
from openai import AsyncOpenAI
import anthropic
from google import genai
from google.genai import types
import os
import httpx
from base64 import b64encode
from io import BytesIO
from typing import Callable, TypedDict, Optional, Tuple, Any, Awaitable
import argparse
from functools import partial
import numpy as np
from pathlib import Path
from tqdm import tqdm
import asyncio
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


def image_to_base64(image: Image.Image) -> str:
    """Helper to convert PIL Image to base64 string"""
    with BytesIO() as buffered:
        image.save(buffered, format="PNG")
        return b64encode(buffered.getvalue()).decode("utf-8")


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


async def openai_client(
    prompt: str,
    image: Image.Image,
    model_name: str = "gpt-4.1-mini",
    temperature: float = 1.0,
) -> tuple[str, str | None, UsageData]:
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model_name,
        messages=prompt_to_messages(prompt, image),
        temperature=temperature,
    )

    usage = UsageData(
        prompt_tokens=response.usage.prompt_tokens if response.usage else None,
        completion_tokens=response.usage.completion_tokens if response.usage else None,
        reasoning_tokens=None,
        total_tokens=response.usage.total_tokens if response.usage else None,
    )

    return response.choices[0].message.content, None, usage


async def openai_reasoning_client(
    prompt: str,
    image: Image.Image,
    model_name: str = "o1-mini",
    temperature: float = 0.1,
) -> tuple[str, str | None, UsageData]:
    """OpenAI reasoning client using Responses API to capture thinking tokens"""
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model_name,
        messages=prompt_to_messages(prompt, image),
        temperature=temperature,
    )

    content = response.choices[0].message.content or ""

    # For reasoning models, check for reasoning tokens in usage
    reasoning_tokens = None
    if response.usage and hasattr(response.usage, "completion_tokens_details"):
        details = response.usage.completion_tokens_details
        if details and hasattr(details, "reasoning_tokens"):
            reasoning_tokens = details.reasoning_tokens

    usage = UsageData(
        prompt_tokens=response.usage.prompt_tokens if response.usage else None,
        completion_tokens=response.usage.completion_tokens if response.usage else None,
        reasoning_tokens=reasoning_tokens,
        total_tokens=response.usage.total_tokens if response.usage else None,
    )

    return content, None, usage


def health_check_url(server_url: str) -> str:
    base_url = server_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]  # Remove /v1
    return f"{base_url}/health"


def normalize_openai_base_url(server_url: str) -> str:
    """Return a base URL that always ends with /v1 for OpenAI-compatible clients.

    Accepts inputs like:
    - https://host
    - https://host/
    - https://host/v1
    - https://host/v1/
    and normalizes them to https://host/v1
    """
    base_url = server_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


async def check_vllm_server_health(server_url: str) -> bool:
    """Check if the vLLM server is online and healthy. Give a few minutes for the server to start."""

    health_url = health_check_url(server_url)
    max_attempts = 12  # Try for up to 2 minutes (12 * 10s)
    timeout = 10.0
    delay = 1

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(health_url)
                if response.status_code == 200:
                    print(f"✅ vLLM server health check passed: {health_url}")
                    return True
                else:
                    print(
                        f"❌ Health check failed with status {response.status_code}: {response.text}"
                    )
        except Exception as e:
            print(f"Health check attempt {attempt} failed with error: {e}")

        if attempt < max_attempts:
            print(
                f"Waiting {delay} seconds before retrying health check (attempt {attempt + 1}/{max_attempts})..."
            )
            await asyncio.sleep(delay)
        else:
            print("❌ vLLM server did not become healthy after several attempts.")

    return False


async def vllm_client(
    prompt: str,
    image: Image.Image,
    server_url: str,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    temperature: float = 1.0,
) -> tuple[str, str | None, UsageData]:
    client = AsyncOpenAI(
        base_url=normalize_openai_base_url(server_url),
        api_key=os.getenv("VLLM_API_KEY", ""),
    )
    response = await client.chat.completions.create(
        model=model_name,
        messages=prompt_to_messages(prompt, image),
        temperature=temperature,
    )

    usage = UsageData(
        prompt_tokens=response.usage.prompt_tokens if response.usage else None,
        completion_tokens=response.usage.completion_tokens if response.usage else None,
        reasoning_tokens=None,
        total_tokens=response.usage.total_tokens if response.usage else None,
    )

    return response.choices[0].message.content, None, usage


async def openrouter_client(
    prompt: str,
    image: Image.Image,
    model_name: str = "openrouter/horizon-beta",
    temperature: float = 1.0,
) -> tuple[str, str | None, UsageData]:
    """OpenRouter client using OpenAI-compatible API."""
    client = AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    response = await client.chat.completions.create(
        model=model_name,
        messages=prompt_to_messages(prompt, image),
        temperature=temperature,
    )
    usage = UsageData(
        prompt_tokens=response.usage.prompt_tokens if response.usage else None,
        completion_tokens=response.usage.completion_tokens if response.usage else None,
        reasoning_tokens=None,
        total_tokens=response.usage.total_tokens if response.usage else None,
    )
    return response.choices[0].message.content, None, usage


def debug_compare_md(
    result: Optional[PreprocessedResponse] = None,
    image_scores: Optional[ImageComparisonResult] = None,
    rewards: Optional[SVGRewards] = None,
    id: str = "0",
    output_dir: Path = Path("."),
    error: Optional[str] = None,
) -> Tuple[str, dict]:
    output = []
    rel_paths = {}
    if error is not None:
        output.append("# Debug Compare (Error)\n\n")
        output.append(f"## Error for example {id}:\n\n")
        output.append("```\n")
        output.append(f"{error}\n")
        output.append("```\n")
        return "".join(output), rel_paths
    if result is None or image_scores is None:
        output.append("# Debug Compare\n\n")
        output.append(f"No result or image scores available for example {id}.\n")
        return "".join(output), rel_paths

    rel_paths = write_debug_images_dict(result, image_scores, output_dir, f"{id}_")

    output.append(f"## Example {id}\n\n")
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
    output.append("### Scores\n\n")

    def fmt_score(x: float | None):
        return f"{x:.4f}" if x is not None else "N/A"

    output.append("| Metric | Raw Score | Reward |\n")
    output.append("|--------|-----------|--------|\n")

    metrics = [
        ("L2", "l2"),
        ("L2 Canny", "l2_canny"),
        ("DreamSim", "dreamsim"),
        ("DreamSim Canny", "dreamsim_canny"),
    ]

    for name, key in metrics:
        raw = fmt_score(getattr(image_scores, key, None)) if image_scores else "N/A"
        reward = fmt_score(getattr(rewards, key, None)) if rewards else "N/A"
        output.append(f"| {name} | {raw} | {reward} |\n")

    if rewards:
        # Length shows actual response length as raw score
        response_length = (
            len(result.response_str) if result and result.response_str else 0
        )
        output.append(f"| Format | N/A | {fmt_score(rewards.format)} |\n")
        output.append(f"| Length | {response_length} | {fmt_score(rewards.length)} |\n")
        output.append(f"| **Overall** | N/A | **{fmt_score(rewards.overall)}** |\n")

    output.append("\n")
    return "".join(output), rel_paths


class LLMClientException(Exception):
    pass


async def eval_example(
    example: Example,
    client: Callable[[str, Image.Image], Awaitable[tuple[str, str | None, UsageData]]],
) -> tuple[
    SVGRewards | None,
    ImageComparisonResult | None,
    PreprocessedResponse | None,
    Exception | None,
    str | None,
    UsageData | None,
]:
    # Initialize variables for early returns
    thinking: str | None = None
    usage: UsageData | None = None

    prompt = example["prompt"]
    gt_svg = extract_svg_text(example["completion"])
    if gt_svg is None:
        return (
            None,
            None,
            None,
            ValueError("Ground truth error: No SVG found"),
            thinking,
            usage,
        )
    # 2. Image loading
    try:
        gt_image = example["image"]
        if gt_image is None:
            svg_im_bytes, _, _ = rasterize_svg(gt_svg)
            gt_image = Image.open(BytesIO(svg_im_bytes))
        if not isinstance(gt_image, Image.Image):
            raise ValueError(f"Expected Image.Image, got {type(gt_image)}: {gt_image}")
    except Exception as e:
        return None, None, None, e, thinking, usage
    # 3. Run completions
    try:
        completion, thinking, usage = await client(prompt, gt_image)
    except Exception as e:
        return None, None, None, e, thinking, usage
    # 4. Parse and rasterize SVG
    try:
        rasterized = svg_env(completion, gt_svg, thinking, usage)
    except Exception as e:
        return None, None, None, e, thinking, usage
    # 5. Image comparison
    try:
        image_scores = get_image_comparator().compare_images(
            rasterized.svg_im, rasterized.svg_im_gt
        )
    except Exception as e:
        return None, None, rasterized, e, thinking, usage
    # 6. Reward computation
    try:
        rewards = compute_rewards(rasterized, image_scores)
    except Exception as e:
        return None, image_scores, rasterized, e, thinking, usage
    # OK!
    return rewards, image_scores, rasterized, None, thinking, usage


def default_prompt(image: Image.Image) -> str:
    return f"Please recreate the image as accurately as possible as an SVG of width {image.width} and height {image.height}."


class Stats(TypedDict):
    mean: float
    std: float
    min: float
    max: float


def calculate_stats(
    results: list[dict[str, float]],
    keys: list[str] = ["overall", "l2", "l2_canny", "dreamsim", "dreamsim_canny"],
) -> dict[str, Stats]:
    if not results:
        return {}
    stats: dict[str, Stats] = {}
    for key in keys:
        # gather only present, non-None values
        valid = [r.get(key) for r in results if r.get(key) is not None]
        if not valid:
            # no data for this key, don't cause an error
            stats[key] = {"mean": None, "std": None, "min": None, "max": None}
            continue
        arr = np.array(valid, dtype=float)
        stats[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return stats


async def run_eval(
    config: DatasetConfig,
    client: Callable[[str, Image.Image], Awaitable[tuple[str, str | None]]],
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
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "eval_debug.md"

    # Setup logging to file
    log_path = output_dir / "log.txt"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # Reduce noise from HTTP requests and other verbose loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger.info(
        f"Starting evaluation: {num_eval_examples} examples from {dataset_name} using {model_name} with {num_workers} workers"
    )

    jsonl_path = output_dir / "llm_responses.jsonl"

    async def worker(
        i: int, example: Example
    ) -> tuple[SVGRewards, str, dict, str | None, UsageData | None]:
        id = f"{i:05d}"
        logger.debug(f"Starting evaluation of example {id}")

        image_paths = {}
        rewards, image_scores, result, exc, thinking, usage = await eval_example(
            example, client
        )
        status = "OK" if exc is None else "FAIL"
        error_message = None

        if exc is not None:
            if isinstance(exc, ParseError):
                error_message = "SVG_PARSE"
            else:
                error_message = str(exc)

            # Log failure with completion details
            completion_text = (
                result.response_str if result is not None else "No completion available"
            )
            logger.error(f"Example {id} FAILED: {error_message}")
            logger.error(
                f"Example {id} completion that failed: {completion_text[:500]}{'...' if len(completion_text) > 500 else ''}"
            )
        else:
            logger.debug(f"Example {id} completed successfully")

        # Create zero rewards for failures
        if rewards is None:
            rewards = SVGRewards(
                format=0.0,
                length=0.0,
                l2=MIN_REWARD,
                l2_canny=MIN_REWARD,
                dreamsim=MIN_REWARD,
                dreamsim_canny=MIN_REWARD,
                overall=MIN_REWARD,
            )

        # Write markdown and debug images if requested
        if debug_dump:
            debug_md, image_paths = debug_compare_md(
                result,
                image_scores,
                rewards,
                id=id,
                output_dir=output_dir,
                error=error_message,
            )
        else:
            debug_md = None
        # Try to get LLM response if possible
        prompt = example["prompt"]
        svg_gt = extract_svg_text(example["completion"])
        # Build record for jsonl
        record: MergedResult = {
            "id": id,
            "prompt": prompt,
            "response": result.response_str if result is not None else None,
            "thinking": thinking,
            "svg": result.svg if result is not None else None,
            "svg_gt": svg_gt,
            # usage data
            "prompt_tokens": usage["prompt_tokens"] if usage else None,
            "completion_tokens": usage["completion_tokens"] if usage else None,
            "reasoning_tokens": usage["reasoning_tokens"] if usage else None,
            "total_tokens": usage["total_tokens"] if usage else None,
            # dump images if we have them - map logical names to field names
            "svg_image": image_paths.get("svg", ""),
            "svg_gt_image": image_paths.get("svg_gt", ""),
            "canny": image_paths.get("canny", ""),
            "canny_gt": image_paths.get("canny_gt", ""),
            "diff": image_paths.get("diff", ""),
            "diff_canny": image_paths.get("diff_canny", ""),
            # raw image scores
            **{
                k: getattr(image_scores, k, None) if image_scores is not None else None
                for k in ["l2", "l2_canny", "dreamsim", "dreamsim_canny"]
            },
            # rewards
            **{
                (f"reward_{k}"): getattr(rewards, k, None)
                for k in [
                    "format",
                    "length",
                    "overall",
                    "l2",
                    "l2_canny",
                    "dreamsim",
                    "dreamsim_canny",
                ]
            },
            "status": status,
            "error": error_message,
        }
        return (rewards, debug_md, record, thinking, usage)

    # Use parallel arrays with same length, indexed by original order
    results = [None] * len(example_list)
    debug_mds = [None] * len(example_list)
    records = [None] * len(example_list)
    thinkings = [None] * len(example_list)
    usages = [None] * len(example_list)

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(num_workers)

    async def bounded_worker(i: int, example: Example):
        async with semaphore:
            return i, await worker(i, example)

    # Create tasks for all examples
    tasks = [bounded_worker(i, ex) for i, ex in enumerate(example_list)]

    # Execute tasks with progress bar
    for task in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Evaluating examples",
    ):
        i, (rewards, md, record, thinking, usage) = await task
        results[i] = rewards
        debug_mds[i] = md
        records[i] = record
        thinkings[i] = thinking
        usages[i] = usage

    # Calculate success rate
    total_examples = len(results)
    failed_examples = sum(1 for r in records if r["status"] == "FAIL")
    success_rate = (
        (total_examples - failed_examples) / total_examples
        if total_examples > 0
        else 0.0
    )

    logger.info(
        f"Evaluation completed: {total_examples} examples processed, {failed_examples} failed, {success_rate:.1%} success rate"
    )

    # Build list of metrics including raw values and rewards
    metric_keys = [
        "l2",
        "l2_canny",
        "dreamsim",
        "dreamsim_canny",
        "reward_format",
        "reward_length",
        "reward_l2",
        "reward_l2_canny",
        "reward_dreamsim",
        "reward_dreamsim_canny",
        "reward_overall",
    ]
    stats = calculate_stats(records, keys=metric_keys)
    stats["success_rate"] = {
        "mean": success_rate,
        "std": None,
        "min": None,
        "max": None,
    }

    if debug_dump:
        with md_path.open("w") as f:
            # Write header with model and dataset info
            f.write("# SVG Evaluation Results\n\n")
            f.write(f"**Model:** {model_name}  \n")
            f.write(f"**Dataset:** {dataset_name}  \n")
            f.write(f"**Examples:** {num_eval_examples}  \n")

            # Write individual example results in order
            for md in debug_mds:
                f.write(md)

            f.write("# Summary Statistics\n\n")

            def fmt_stat(x: float | None):
                return f"{x:.2f}" if x is not None else "N/A"

            f.write("| Metric | Mean | Std | Min | Max |\n")
            f.write("|--------|------|-----|-----|-----|\n")

            for key, stat in stats.items():
                mean, std, min_, max_ = (
                    stat["mean"],
                    stat["std"],
                    stat["min"],
                    stat["max"],
                )
                f.write(
                    f"| {key} | {fmt_stat(mean)} | {fmt_stat(std)} | {fmt_stat(min_)} | {fmt_stat(max_)} |\n"
                )

    # Write summarized stats as CSV
    csv_path = output_dir / "stats_summary.csv"

    def fmt(x: float | None):
        return f"{x:.4f}" if x is not None else ""

    with csv_path.open("w") as f:
        f.write("model_name,metric,mean,std,min,max\n")
        for key, stat in stats.items():
            mean, std, min_, max_ = (
                stat["mean"],
                stat["std"],
                stat["min"],
                stat["max"],
            )
            f.write(
                f"{model_name},{key},{fmt(mean)},{fmt(std)},{fmt(min_)},{fmt(max_)}\n"
            )

    # Write all LLM responses and metadata to jsonl
    with jsonl_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return results


async def anthropic_client(
    prompt: str,
    image: Image.Image,
    model_name: str = "claude-3-5-sonnet-20241022",
    thinking_budget: int = 4000,
    temperature: float = 1.0,
) -> tuple[str, str | None, UsageData]:
    """Anthropic Claude client that returns (response, thinking) tuple"""
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Enable thinking for Claude 4 models
    thinking_config = None
    if "claude-sonnet-4" in model_name or "claude-4" in model_name:
        thinking_config = {"type": "enabled", "budget_tokens": thinking_budget}

    message_params = {
        "model": model_name,
        "max_tokens": 4000,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_to_base64(image),
                        },
                    },
                ],
            }
        ],
    }

    if thinking_config:
        message_params["thinking"] = thinking_config

    message = await client.messages.create(**message_params)

    # Extract response text and thinking
    response_text = ""
    thinking_text = ""

    for content_block in message.content:
        if content_block.type == "text":
            response_text = content_block.text
        elif content_block.type == "thinking":
            thinking_text = content_block.text

    usage = UsageData(
        prompt_tokens=message.usage.input_tokens if message.usage else None,
        completion_tokens=message.usage.output_tokens if message.usage else None,
        reasoning_tokens=None,  # Anthropic doesn't separate reasoning tokens yet
        total_tokens=(message.usage.input_tokens + message.usage.output_tokens)
        if message.usage
        else None,
    )

    return response_text, thinking_text if thinking_text else None, usage


async def google_client(
    prompt: str,
    image: Image.Image,
    model_name: str = "gemini-2.5-flash",
    thinking_budget: int = 4000,
    temperature: float = 1.0,
) -> tuple[str, str | None, UsageData]:
    """Google Gemini client that returns (response, thinking) tuple"""
    client = genai.Client(api_key=os.getenv("GOOGLE_GENERATIVE_AI_API_KEY"))

    with BytesIO() as buffered:
        image.save(buffered, format="PNG")
        image_data = buffered.getvalue()

    # Prepare content with text and image
    contents = [
        types.Content(
            parts=[
                types.Part(text=prompt),
                types.Part(
                    inline_data=types.Blob(mime_type="image/png", data=image_data)
                ),
            ]
        )
    ]

    # Configure thinking for 2.5 models
    config = None
    if "2.5" in model_name:
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True, thinking_budget=thinking_budget
            ),
            temperature=temperature,
        )
    else:
        config = types.GenerateContentConfig(temperature=temperature)

    response = await client.aio.models.generate_content(
        model=model_name, contents=contents, config=config
    )

    # Extract response text and thinking
    response_text = response.text
    thinking_text = None

    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.thought:
                    if thinking_text is None:
                        thinking_text = ""
                    thinking_text += part.text

    if response.usage_metadata:
        usage = UsageData(
            prompt_tokens=response.usage_metadata.prompt_token_count,
            completion_tokens=response.usage_metadata.candidates_token_count,
            reasoning_tokens=response.usage_metadata.thoughts_token_count,
            total_tokens=response.usage_metadata.total_token_count,
        )
    else:
        usage = UsageData(
            prompt_tokens=None,
            completion_tokens=None,
            reasoning_tokens=None,
            total_tokens=None,
        )

    return response_text, thinking_text, usage


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
    choices=["openai", "openai-responses", "anthropic", "google", "vllm", "openrouter"],
    help="Which client/model to use.",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="gpt-4.1-mini",
    help="Model name for OpenAI/OpenAI-responses/Anthropic/Google/vLLM clients. For vllm, you must also start the vLLM server separately with the correct model.",
)
parser.add_argument(
    "--vllm_endpoint",
    type=str,
    default=None,
    help="Base URL for vLLM endpoint (with or without /v1 is fine; required if client is vllm).",
)
parser.add_argument(
    "--debug_dump",
    action="store_true",
    default=True,
    help="Dump debug markdown with images.",
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
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="Temperature for generation (default: 1.0). Higher values make output more random, lower values make it more deterministic.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    dataset_config = datasets[args.dataset]

    if args.client == "openai":
        client_fn = partial(
            openai_client, model_name=args.model_name, temperature=args.temperature
        )
    elif args.client == "openai-responses":
        client_fn = partial(
            openai_reasoning_client,
            model_name=args.model_name,
            temperature=args.temperature,
        )
    elif args.client == "anthropic":
        client_fn = partial(
            anthropic_client,
            model_name=args.model_name,
            thinking_budget=4000,
            temperature=args.temperature,
        )
    elif args.client == "google":
        client_fn = partial(
            google_client,
            model_name=args.model_name,
            thinking_budget=4000,
            temperature=args.temperature,
        )
    elif args.client == "openrouter":
        client_fn = partial(
            openrouter_client,
            model_name=args.model_name,
            temperature=args.temperature,
        )
    elif args.client == "vllm":
        if not args.vllm_endpoint:
            raise ValueError(
                "--vllm_endpoint must be specified when using vllm client."
            )

        # Check vLLM server health before starting evaluation
        print("🔍 Checking vLLM server health...")
        if not asyncio.run(check_vllm_server_health(args.vllm_endpoint)):
            raise LLMClientException(
                f"vLLM server at {args.vllm_endpoint} is not healthy. Please ensure the server is running."
            )
        print("✅ vLLM server health check passed!")

        client_fn = partial(
            vllm_client,
            server_url=args.vllm_endpoint,
            model_name=args.model_name,
            temperature=args.temperature,
        )
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

    rewards = asyncio.run(
        run_eval(
            dataset_config,
            client_fn,
            output_dir,
            num_eval_examples=args.num_eval_examples,
            debug_dump=args.debug_dump,
            num_workers=args.num_workers,
            model_name=args.model_name,
        )
    )

    print(
        f"Output written to: {output_dir.resolve().relative_to(Path.cwd()) if not output_dir.is_absolute() else output_dir.resolve()}"
    )
