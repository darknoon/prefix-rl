"""
Agentic evaluation loop for SVG generation.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from base64 import b64encode
from io import BytesIO

import anthropic
from PIL import Image

from .virtual_fs import VirtualFS
from .tools import ToolExecutor, TOOL_DEFINITIONS, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    """Record of a single turn in the evaluation."""

    turn_number: int
    response_text: str | None
    tool_calls: list[dict]
    tool_results: list[dict]
    metrics: dict | None = None
    svg_content: str | None = None


@dataclass
class EvalResult:
    """Result of an agentic evaluation."""

    example_id: str
    total_turns: int
    final_metrics: dict | None
    turn_history: list[TurnRecord]
    status: str  # "completed", "turn_limit_reached", "error"
    error: str | None = None
    final_svg: str | None = None
    saved_files: dict[str, Path] = field(default_factory=dict)


@dataclass
class AgenticEvalConfig:
    """Configuration for agentic evaluation."""

    max_turns: int = 10
    raster_min_size: int = 512
    raster_max_size: int = 1536
    debug_output: bool = True
    debug_dir: Path = field(default_factory=lambda: Path("debug"))
    model_name: str = "claude-sonnet-4-20250514"
    temperature: float = 1.0


SYSTEM_PROMPT = """You are an AI assistant that recreates images as SVG files.

## Task
Recreate the target image as accurately as possible as an SVG.

## Available Files
- `target.png`: The image you must recreate

## Available Tools
- `write`: Write new SVG content to result.svg
- `search_replace`: Make targeted edits to result.svg
- `rasterize_svg`: Render your SVG and get distance metrics compared to target (lower is better)
- `zoom_image`: Crop a region to inspect details
- `list_files`: See all available files
- `read_file`: Read file content

## Strategy
1. Analyze the target image to understand its structure (shapes, colors, layout)
2. Write an initial SVG attempt
3. Use `rasterize_svg` to see your metrics - l2_distance and dreamsim_distance (lower is better, 0 = perfect)
4. Compare your rasterized image to the target
5. Iterate: use `search_replace` for small fixes or `write` for major rewrites
6. Continue until metrics are satisfactory or you can't improve further

When you're done and satisfied with your result, respond with just a text message (no tool calls) indicating you're finished."""


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return b64encode(buffer.getvalue()).decode("utf-8")


class AgenticEvaluator:
    """Runs agentic evaluation with tool-use loop."""

    def __init__(self, config: AgenticEvalConfig):
        self.config = config
        self.client = anthropic.Anthropic()

    async def evaluate(
        self,
        target_image: Image.Image,
        example_id: str,
        prompt: str | None = None,
    ) -> EvalResult:
        """
        Run agentic evaluation for a single example.

        Args:
            target_image: The target image to recreate
            example_id: Unique identifier for this example
            prompt: Optional custom prompt (uses default if None)

        Returns:
            EvalResult with turn history and final metrics
        """
        # Setup virtual filesystem
        vfs = VirtualFS()
        vfs.write_image("target.png", target_image)
        vfs.write("result.svg", "")  # Initialize empty SVG file

        # Setup tool executor
        executor = ToolExecutor(
            vfs,
            target_filename="target.png",
            svg_filename="result.svg",
            raster_min_size=self.config.raster_min_size,
            raster_max_size=self.config.raster_max_size,
        )

        # Build initial message with target image
        user_prompt = prompt or f"Please recreate this {target_image.width}x{target_image.height} image as an SVG."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_to_base64(target_image),
                        },
                    },
                ],
            }
        ]

        turn_history: list[TurnRecord] = []
        final_metrics: dict | None = None
        status = "completed"
        error_msg: str | None = None

        # Main evaluation loop
        for turn_num in range(1, self.config.max_turns + 1):
            logger.debug(f"Example {example_id}: Starting turn {turn_num}")

            try:
                response = self.client.messages.create(
                    model=self.config.model_name,
                    max_tokens=8192,
                    temperature=self.config.temperature,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                )
            except Exception as e:
                logger.error(f"Example {example_id}: API error on turn {turn_num}: {e}")
                status = "error"
                error_msg = str(e)
                break

            # Extract response text and tool calls
            response_text = None
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    response_text = block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            # Execute tool calls
            tool_results = []
            turn_metrics = None

            if tool_calls:
                # Add assistant message with tool calls
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool
                tool_result_content = []
                for tc in tool_calls:
                    result = executor.execute(tc["name"], tc["input"])
                    tool_results.append(
                        {
                            "tool_id": tc["id"],
                            "tool_name": tc["name"],
                            "result": result.to_dict(),
                        }
                    )

                    # Track metrics if rasterize was called
                    if tc["name"] == "rasterize_svg" and result.success:
                        turn_metrics = result.data.get("metrics")
                        final_metrics = turn_metrics

                    # Build tool result for Claude
                    # Handle image results specially
                    if result.success and result.data.get("type") == "image":
                        # Return the image inline
                        filename = result.data.get("file")
                        img = vfs.read_image(filename)
                        if img:
                            tool_result_content.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tc["id"],
                                    "content": [
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/png",
                                                "data": image_to_base64(img),
                                            },
                                        },
                                        {"type": "text", "text": json.dumps(result.to_dict())},
                                    ],
                                }
                            )
                        else:
                            tool_result_content.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tc["id"],
                                    "content": result.to_json(),
                                }
                            )
                    elif tc["name"] == "rasterize_svg" and result.success:
                        # Return the rasterized image along with metrics
                        filename = result.data.get("image_file")
                        img = vfs.read_image(filename)
                        if img:
                            tool_result_content.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tc["id"],
                                    "content": [
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/png",
                                                "data": image_to_base64(img),
                                            },
                                        },
                                        {"type": "text", "text": json.dumps(result.to_dict())},
                                    ],
                                }
                            )
                        else:
                            tool_result_content.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tc["id"],
                                    "content": result.to_json(),
                                }
                            )
                    elif tc["name"] == "zoom_image" and result.success:
                        # Return the zoomed image
                        filename = result.data.get("image_file")
                        img = vfs.read_image(filename)
                        if img:
                            tool_result_content.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tc["id"],
                                    "content": [
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/png",
                                                "data": image_to_base64(img),
                                            },
                                        },
                                        {"type": "text", "text": json.dumps(result.to_dict())},
                                    ],
                                }
                            )
                        else:
                            tool_result_content.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tc["id"],
                                    "content": result.to_json(),
                                }
                            )
                    else:
                        tool_result_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tc["id"],
                                "content": result.to_json(),
                            }
                        )

                # Add tool results as user message
                messages.append({"role": "user", "content": tool_result_content})

            # Record this turn
            svg_content = vfs.read_text("result.svg")
            turn_record = TurnRecord(
                turn_number=turn_num,
                response_text=response_text,
                tool_calls=tool_calls,
                tool_results=tool_results,
                metrics=turn_metrics,
                svg_content=svg_content if svg_content else None,
            )
            turn_history.append(turn_record)

            # Check if model is done (no tool calls)
            if not tool_calls:
                logger.info(f"Example {example_id}: Model finished after {turn_num} turns")
                break

            # Check stop reason
            if response.stop_reason == "end_turn" and not tool_calls:
                logger.info(f"Example {example_id}: Model indicated completion")
                break

        else:
            # Loop completed without break = turn limit reached
            status = "turn_limit_reached"
            logger.warning(f"Example {example_id}: Turn limit reached ({self.config.max_turns})")

        # Get final SVG
        final_svg = vfs.read_text("result.svg")

        # Write debug output if enabled
        saved_files: dict[str, Path] = {}
        if self.config.debug_output:
            saved_files = self._write_debug_output(
                example_id, vfs, target_image, turn_history, final_metrics, status, error_msg
            )

        return EvalResult(
            example_id=example_id,
            total_turns=len(turn_history),
            final_metrics=final_metrics,
            turn_history=turn_history,
            status=status,
            error=error_msg,
            final_svg=final_svg,
            saved_files=saved_files,
        )

    def _write_debug_output(
        self,
        example_id: str,
        vfs: VirtualFS,
        target_image: Image.Image,
        turn_history: list[TurnRecord],
        final_metrics: dict | None,
        status: str,
        error: str | None,
    ) -> dict[str, Path]:
        """Write debug output for an evaluation. Returns paths to saved files."""
        output_dir = self.config.debug_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files: dict[str, Path] = {}

        # Save target image (ground truth) - only once per example
        target_path = output_dir / f"{example_id}_svg_gt.png"
        target_image.save(target_path)
        saved_files["target"] = target_path

        # Save rasterized attempt images with example_id prefix
        for filename in vfs.list_files():
            if filename.startswith("rasterize_") and filename.endswith(".png"):
                img = vfs.read_image(filename)
                if img:
                    # rasterize_001.png -> {example_id}_rasterize_001.png
                    out_path = output_dir / f"{example_id}_{filename}"
                    img.save(out_path)
                    saved_files[filename] = out_path

        # Save final SVG
        final_svg = vfs.read_text("result.svg")
        if final_svg:
            svg_path = output_dir / f"{example_id}_svg.svg"
            svg_path.write_text(final_svg)
            saved_files["svg"] = svg_path

            # Also rasterize final SVG for display
            try:
                from prefixrl.reward.svg.cairosvg import rasterize_svg
                final_img, _, _ = rasterize_svg(
                    final_svg,
                    min_target=self.config.raster_min_size,
                    max_target=self.config.raster_max_size,
                )
                final_img_path = output_dir / f"{example_id}_svg.png"
                final_img.save(final_img_path)
                saved_files["svg_rendered"] = final_img_path
            except Exception:
                pass  # Skip if rasterization fails

        return saved_files
