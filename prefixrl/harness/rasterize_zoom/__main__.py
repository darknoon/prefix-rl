"""
CLI entry point for the rasterize-zoom agentic evaluation harness.
"""

import argparse
import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import TypedDict, Optional

from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
import numpy as np

from .evaluator import AgenticEvaluator, AgenticEvalConfig, EvalResult

load_dotenv()

logger = logging.getLogger(__name__)


class DatasetConfig(TypedDict):
    name: str
    split: str
    image_key: str
    completion_key: str
    prompt_key: Optional[str]


DATASETS: dict[str, DatasetConfig] = {
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


def default_prompt(image: Image.Image) -> str:
    return f"Please recreate this {image.width}x{image.height} image as an SVG."


class Stats(TypedDict):
    mean: float | None
    std: float | None
    min: float | None
    max: float | None


def calculate_stats(
    values: list[float | None],
) -> Stats:
    """Calculate statistics for a list of values."""
    valid = [v for v in values if v is not None]
    if not valid:
        return {"mean": None, "std": None, "min": None, "max": None}

    arr = np.array(valid, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


async def run_eval(
    config: DatasetConfig,
    output_dir: Path,
    eval_config: AgenticEvalConfig,
    num_eval_examples: int = 10,
    num_workers: int = 1,
) -> list[EvalResult]:
    """Run agentic evaluation on a dataset."""
    dataset_name = config["name"]
    split = config["split"]

    # Load dataset (streaming to avoid loading everything)
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    dataset = dataset.take(num_eval_examples)

    image_key = config["image_key"]
    prompt_key = config["prompt_key"]

    # Convert to list of examples
    examples = []
    for i, row in enumerate(dataset):
        image = row[image_key]
        prompt = row[prompt_key] if prompt_key else default_prompt(image)
        examples.append(
            {
                "id": f"{i:05d}",
                "image": image,
                "prompt": prompt,
            }
        )

    # Setup output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update eval config debug dir
    eval_config.debug_dir = output_dir

    # Setup logging
    log_path = output_dir / "log.txt"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    # Reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger.info(
        f"Starting agentic evaluation: {num_eval_examples} examples from {dataset_name} "
        f"using {eval_config.model_name} with max_turns={eval_config.max_turns}"
    )

    evaluator = AgenticEvaluator(eval_config)

    # Run evaluations
    results: list[EvalResult] = []
    semaphore = asyncio.Semaphore(num_workers)

    async def evaluate_example(example: dict) -> EvalResult:
        async with semaphore:
            return await evaluator.evaluate(
                target_image=example["image"],
                example_id=example["id"],
                prompt=example["prompt"],
            )

    tasks = [evaluate_example(ex) for ex in examples]

    for task in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Evaluating examples",
    ):
        result = await task
        results.append(result)

    # Sort by example_id
    results.sort(key=lambda r: r.example_id)

    # Calculate statistics
    l2_values = [
        r.final_metrics.get("l2_distance") if r.final_metrics else None for r in results
    ]
    dreamsim_values = [
        r.final_metrics.get("dreamsim_distance") if r.final_metrics else None
        for r in results
    ]
    turn_counts = [r.total_turns for r in results]

    stats = {
        "l2_distance": calculate_stats(l2_values),
        "dreamsim_distance": calculate_stats(dreamsim_values),
        "total_turns": calculate_stats([float(t) for t in turn_counts]),
    }

    # Count statuses
    status_counts = {}
    for r in results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    success_rate = status_counts.get("completed", 0) / len(results) if results else 0.0

    logger.info(f"Evaluation completed. Success rate: {success_rate:.1%}")
    logger.info(f"Status counts: {status_counts}")
    
    l2_mean = stats['l2_distance']['mean']
    dreamsim_mean = stats['dreamsim_distance']['mean']
    turns_mean = stats['total_turns']['mean']
    
    if l2_mean is not None:
        logger.info(f"L2 distance: mean={l2_mean:.4f}")
    else:
        logger.info("L2 distance: N/A (no successful rasterizations)")
    
    if dreamsim_mean is not None:
        logger.info(f"DreamSim distance: mean={dreamsim_mean:.4f}")
    else:
        logger.info("DreamSim distance: N/A (no successful rasterizations)")
    
    if turns_mean is not None:
        logger.info(f"Total turns: mean={turns_mean:.1f}")

    # Write summary CSV
    csv_path = output_dir / "stats_summary.csv"
    
    def fmt_stat(val: float | None) -> str:
        return f"{val:.6f}" if val is not None else ""
    
    with csv_path.open("w") as f:
        f.write("model_name,metric,mean,std,min,max\n")
        for metric, stat in stats.items():
            f.write(
                f"{eval_config.model_name},{metric},"
                f"{fmt_stat(stat['mean'])},"
                f"{fmt_stat(stat['std'])},"
                f"{fmt_stat(stat['min'])},"
                f"{fmt_stat(stat['max'])}\n"
            )

    # Write detailed results JSONL
    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("w") as f:
        for r in results:
            record = {
                "id": r.example_id,
                "total_turns": r.total_turns,
                "final_metrics": r.final_metrics,
                "status": r.status,
                "error": r.error,
            }
            f.write(json.dumps(record) + "\n")

    # Write overall summary
    summary_path = output_dir / "eval_summary.json"
    summary = {
        "dataset": dataset_name,
        "model": eval_config.model_name,
        "max_turns": eval_config.max_turns,
        "num_examples": len(results),
        "status_counts": status_counts,
        "success_rate": success_rate,
        "stats": stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    # Write debug markdown
    md_path = output_dir / "eval_debug.md"
    with md_path.open("w") as f:
        f.write("# Agentic SVG Evaluation Results\n\n")
        f.write(f"**Model:** {eval_config.model_name}  \n")
        f.write(f"**Dataset:** {dataset_name}  \n")
        f.write(f"**Examples:** {len(results)}  \n")
        f.write(f"**Max Turns:** {eval_config.max_turns}  \n\n")

        f.write("## Summary Statistics\n\n")
        f.write("| Metric | Mean | Std | Min | Max |\n")
        f.write("|--------|------|-----|-----|-----|\n")
        for metric, stat in stats.items():
            mean = f"{stat['mean']:.4f}" if stat["mean"] is not None else "N/A"
            std = f"{stat['std']:.4f}" if stat["std"] is not None else "N/A"
            min_ = f"{stat['min']:.4f}" if stat["min"] is not None else "N/A"
            max_ = f"{stat['max']:.4f}" if stat["max"] is not None else "N/A"
            f.write(f"| {metric} | {mean} | {std} | {min_} | {max_} |\n")

        f.write("\n## Individual Results\n\n")
        for r in results:
            f.write(f"## Example {r.example_id}\n\n")

            # Show target vs final result side by side
            target_path = f"{r.example_id}_svg_gt.png"
            result_img_path = f"{r.example_id}_svg.png"

            f.write(
                f'<div style="display: flex; gap: 20px;">'
                f'<div style="text-align: center;"><img src="{target_path}" style="max-width: 300px;"><br><b>Ground Truth</b></div>'
                f'<div style="text-align: center;"><img src="{result_img_path}" style="max-width: 300px;"><br><b>Generated</b></div>'
                f"</div>\n\n"
            )

            # Summary line
            l2 = r.final_metrics.get("l2_distance", "N/A") if r.final_metrics else "N/A"
            ds = r.final_metrics.get("dreamsim_distance", "N/A") if r.final_metrics else "N/A"
            f.write(f"**Turns:** {r.total_turns} | **L2:** {l2} | **DreamSim:** {ds} | **Status:** {r.status}\n\n")

            if r.error:
                f.write(f"**Error:** {r.error}\n\n")

            # Collapsible turn details
            f.write("<details>\n")
            f.write("<summary>Show Turn-by-Turn Progress</summary>\n\n")

            # Build metrics progress table
            f.write("| Turn | Actions | L2 | DreamSim |\n")
            f.write("|------|---------|----|-----------|\n")
            for turn in r.turn_history:
                actions = ", ".join(tc["name"] for tc in turn.tool_calls) if turn.tool_calls else "(finished)"
                if turn.metrics:
                    l2_val = f"{turn.metrics.get('l2_distance', 0):.4f}"
                    ds_val = f"{turn.metrics.get('dreamsim_distance', 0):.4f}"
                else:
                    l2_val = "-"
                    ds_val = "-"
                f.write(f"| {turn.turn_number} | {actions} | {l2_val} | {ds_val} |\n")

            # Show intermediate rasterizations if any
            raster_files = []
            for i in range(1, 100):
                raster_path = f"{r.example_id}_rasterize_{i:03d}.png"
                if (output_dir / raster_path).exists():
                    raster_files.append(raster_path)
                else:
                    break

            if raster_files:
                f.write("\n**Rasterization Progress:**\n\n")
                f.write('<div style="display: flex; gap: 10px; flex-wrap: wrap;">\n')
                for i, rp in enumerate(raster_files, 1):
                    f.write(
                        f'<div style="text-align: center;"><img src="{rp}" style="max-width: 150px;"><br><small>#{i}</small></div>\n'
                    )
                f.write("</div>\n\n")

            # Show turn details
            f.write("\n**Turn Details:**\n\n")
            for turn in r.turn_history:
                f.write(f"**Turn {turn.turn_number}:**\n")
                if turn.tool_calls:
                    for tc in turn.tool_calls:
                        tool_name = tc["name"]
                        tool_input = tc.get("input", {})
                        if tool_name == "write":
                            content = tool_input.get("content", "")
                            preview = content[:200] + "..." if len(content) > 200 else content
                            f.write(f"- `write`: {len(content)} chars\n")
                        elif tool_name == "search_replace":
                            old = tool_input.get("old_string", "")[:50]
                            new = tool_input.get("new_string", "")[:50]
                            f.write(f"- `search_replace`: \"{old}\" → \"{new}\"\n")
                        elif tool_name == "rasterize_svg":
                            f.write(f"- `rasterize_svg`\n")
                        else:
                            f.write(f"- `{tool_name}`\n")
                if turn.response_text:
                    text_preview = turn.response_text[:200]
                    if len(turn.response_text) > 200:
                        text_preview += "..."
                    f.write(f"  > {text_preview}\n")
                f.write("\n")

            f.write("</details>\n\n")
            f.write("---\n\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run agentic SVG evaluation with rasterize-zoom harness."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="simple-shapes",
        choices=list(DATASETS.keys()),
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "-n",
        "--num_examples",
        type=int,
        default=10,
        help="Number of examples to evaluate.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use.",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=10,
        help="Maximum turns per example.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for generation.",
    )
    parser.add_argument(
        "--no_debug",
        action="store_true",
        help="Disable debug output.",
    )

    args = parser.parse_args()

    dataset_config = DATASETS[args.dataset]

    # Generate output directory name
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        dataset_safe = dataset_config["name"].replace("/", "_")
        model_safe = args.model_name.replace("/", "_")
        dir_name = f"{dataset_safe}_{model_safe}_rasterize_zoom"
        output_dir = Path("eval") / dir_name

    eval_config = AgenticEvalConfig(
        max_turns=args.max_turns,
        debug_output=not args.no_debug,
        debug_dir=output_dir,
        model_name=args.model_name,
        temperature=args.temperature,
    )

    results = asyncio.run(
        run_eval(
            config=dataset_config,
            output_dir=output_dir,
            eval_config=eval_config,
            num_eval_examples=args.num_examples,
            num_workers=args.num_workers,
        )
    )

    print(f"\nOutput written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
