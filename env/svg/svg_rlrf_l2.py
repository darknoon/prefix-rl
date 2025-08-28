import re
import os
import sys

# Set up DYLD_LIBRARY_PATH for macOS Cairo support (Homebrew installation)
if sys.platform == "darwin":  # macOS
    homebrew_lib = "/opt/homebrew/lib"
    if os.path.exists(homebrew_lib):
        current_path = os.environ.get("DYLD_LIBRARY_PATH", "")
        if homebrew_lib not in current_path:
            os.environ["DYLD_LIBRARY_PATH"] = f"{homebrew_lib}:{current_path}"

import cairosvg.parser
import cairosvg.surface

from dataclasses import dataclass
import torch
from PIL import Image as PILImage
import io
import numpy as np
import torchvision.transforms as T
from typing import Any
import logging
import time
from concurrent.futures import ThreadPoolExecutor

# Minimum reward value for failed evaluations
MIN_REWARD = -1.0

REWARD_WEIGHTS = {
    "format": 1.0,
    "length": 1.0,
    "l2": 1.0,
}
REWARD_WEIGHT_SUM = sum(REWARD_WEIGHTS.values())


def weighted_reward(format: float, length: float, l2: float) -> float:
    return (
        format * REWARD_WEIGHTS["format"]
        + length * REWARD_WEIGHTS["length"]
        + l2 * REWARD_WEIGHTS["l2"]
    ) / REWARD_WEIGHT_SUM


logger = logging.getLogger(__name__)


def l2(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute L2 distance between two tensors."""
    h, w, c = x.shape
    den = (h * w * c) ** 0.5
    return (torch.norm(x - y) / den).item()


@torch.no_grad()
def compute_image_l2(im: PILImage.Image, im_ref: PILImage.Image) -> float | None:
    """
    Compute L2 distance between two images, or return None on error.
    """

    # does [0,255] -> [0,1] automatically
    to_tensor = T.ToTensor()

    try:
        # Convert PIL images to RGB
        im = im.convert("RGB")
        # Resize image to the reference image size for fair comparison
        im = im.resize(im_ref.size, PILImage.Resampling.LANCZOS)
        im_ref = im_ref.convert("RGB")
        return l2(to_tensor(im), to_tensor(im_ref))

    except Exception as e:
        logger.error(f"Error during image comparison: {e}")
        return None


def get_svg_size(tree: cairosvg.parser.Tree) -> tuple[float, float]:
    width = tree.get("width")
    height = tree.get("height")

    # Handle percentage values - default to 512 for percentages
    if width and width.endswith("%"):
        width = "512"
    if height and height.endswith("%"):
        height = "512"

    if width is None or height is None:
        # Get viewBox if size not specified
        viewbox = tree.get("viewBox")
        if viewbox:
            # viewBox format is "min-x min-y width height"
            parts = viewbox.split()
            if len(parts) == 4:
                width = parts[2]
                height = parts[3]

    # Default to 512x512 if no size info found
    width = width or "512"
    height = height or "512"

    # Strip any remaining non-numeric characters (like 'px')
    width = "".join(c for c in str(width) if c.isdigit() or c == ".")
    height = "".join(c for c in str(height) if c.isdigit() or c == ".")

    # Final fallback if we still don't have valid numbers
    try:
        width = float(width) if width else 512.0
        height = float(height) if height else 512.0
    except ValueError:
        width = 512.0
        height = 512.0

    return width, height


def compute_svg_raster_scale(
    width: float, height: float, min_target: int = 512, max_target: int = 1536
) -> tuple[float, int, int]:
    """
    Compute the scale and output size for rasterizing an SVG such that:
    - The smallest side is at least min_target px
    - The largest side is at most max_target px
    - Aspect ratio is preserved
    Returns: (scale, output_width, output_height)
    """
    # Compute scale factors for both constraints
    scale_min = min_target / min(width, height)
    scale_max = max_target / max(width, height)

    # The scale must be at least scale_min, but not so large that the largest side exceeds max_target
    scale = max(scale_min, 1.0)
    if max(width, height) * scale > max_target:
        scale = scale_max

    output_width = int(round(width * scale))
    output_height = int(round(height * scale))
    return scale, output_width, output_height


def rasterize_svg(
    svg_content: str, min_target: int = 512, max_target: int = 1536
) -> tuple[bytes, Any, tuple[float, float]]:
    """Rasterize SVG content to PNG image bytes using CairoSVG.

    The output image will have its smallest side at least min_target,
    and its longest side at most max_target, preserving aspect ratio.
    """
    tree = cairosvg.parser.Tree(bytestring=svg_content.encode("utf-8"))
    # Get the intrinsic width and height from the SVG
    width, height = get_svg_size(tree)

    # Compute scale and output size using the factored-out function
    scale, output_width, output_height = compute_svg_raster_scale(
        width, height, min_target=min_target, max_target=max_target
    )

    output = io.BytesIO()
    dpi = 96
    parent_width = width
    parent_height = height
    background_color = "white"
    instance = cairosvg.surface.PNGSurface(
        tree,
        output,
        dpi,
        None,
        parent_width,
        parent_height,
        scale,
        output_width,
        output_height,
        background_color,
    )
    instance.finish()
    image_bytes = output.getvalue()
    return image_bytes, tree, (width, height)


def format_reward(predict: str) -> float:
    """Check if response follows the expected format with <think> and <answer> tags."""
    pattern = re.compile(
        r"\s{0,3}<think>.*?</think>\s*<answer>.*?</answer>\s{0,3}", re.DOTALL
    )
    return 1.0 if re.fullmatch(pattern, predict) else 0.0


def extract_svg_text(full_response: str) -> str | None:
    """Extract the last SVG element from the response."""
    # First try to extract from <answer> tags
    answer_match = re.search(r"<answer>(.*?)</answer>", full_response, re.DOTALL)
    content = answer_match.group(1) if answer_match else full_response

    # Find all SVG elements and return the last one
    svg_matches = list(re.finditer(r"<svg.*?</svg>", content, re.DOTALL))
    return svg_matches[-1].group(0).strip() if svg_matches else None


def length_reward(L_pred: float, L_gt: float) -> float:
    """
    Compute the SVG Length Deviation reward:
        R_len = 1 - ( (1/L_gt) * max(0, L_pred - L_gt/2) )^2

    Args:
        L_pred: predicted SVG token length
        L_gt:   ground-truth SVG token length

    Returns:
        R_len: reward in [−∞, 1], penalizing over-length predictions
    """
    # how much we exceed half the ground-truth length
    excess = max(0.0, L_pred - 0.5 * L_gt)
    # normalized penalty squared
    penalty = (excess / L_gt) ** 2
    return 1.0 - penalty


def l2_reward(l2_distance: float) -> float:
    """Convert L2 distance to reward."""
    return max(min(1.0 - l2_distance, 1.0), -1.0)  # clip to [-1, 1]


@dataclass
class SVGRewards:
    format: float
    length: float
    l2: float
    score: float

    def __init__(self, format: float, length: float, l2: float):
        self.format = format
        self.length = length
        self.l2 = l2
        self.score = weighted_reward(format, length, l2)


def safe_rasterize_svg(svg_content: str) -> PILImage.Image | None:
    """Rasterize SVG content to PNG image bytes using CairoSVG.
    Returns None if the SVG is invalid.
    """
    try:
        svg_im_bytes, _, _ = rasterize_svg(svg_content)
        return PILImage.open(io.BytesIO(svg_im_bytes))
    except Exception as e:
        logger.warning(f"Failed to rasterize SVG: {e}")
        return None


def compute_rewards(response_str: str, svg_gt: str) -> SVGRewards:
    """Compute all rewards for an SVG generation response."""

    # Always compute format reward
    format_r = format_reward(response_str)
    invalid_svg = SVGRewards(
        format=format_r,
        length=MIN_REWARD,
        l2=MIN_REWARD,
    )

    svg_text = extract_svg_text(response_str)
    if svg_text is None:
        # Failed to parse/render - return minimum rewards except format
        return invalid_svg

    svg_im = safe_rasterize_svg(svg_text)
    if svg_im is None:
        return invalid_svg

    svg_gt_im = safe_rasterize_svg(svg_gt)
    if svg_gt_im is None:
        return invalid_svg

    # Compute length reward (expected format: <think>...</think><answer>SVG</answer>)
    gt_len = len(f"<think></think><answer>{svg_text}</answer>")
    length_r = length_reward(len(response_str), gt_len)

    # Compute L2 distance
    l2_dist = compute_image_l2(svg_im, svg_gt_im)
    if l2_dist is None:
        return invalid_svg
    l2_r = l2_reward(l2_dist)

    return SVGRewards(format=format_r, length=length_r, l2=l2_r)


def compute_rewards_dict(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[Any] = None,
    **kwargs,
) -> list[dict[str, float]]:
    """Compute rewards for a batch of SVG generation inputs (VERL interface)."""
    batch_size = len(solution_strs)
    logger.info(f"Processing batch of {batch_size} rewards (simplified L2 version)")

    start_time = time.time()

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(compute_rewards, sol_str, gt)
            for sol_str, gt in zip(solution_strs, ground_truths)
        ]
        rewards: list[SVGRewards] = [future.result() for future in futures]

    elapsed = time.time() - start_time
    logger.info(
        f"Batch SVG L2 reward computation took {elapsed * 1000:.0f} ms "
        f"({(elapsed * 1000) / batch_size:.1f} ms/item) on CPU"
    )

    # Convert SVGRewards to plain dicts for verl
    return [vars(r) for r in rewards]


if __name__ == "__main__":
    # Simple test
    def circle_svg(r: float) -> str:
        return f"<svg width='512' height='512'><circle cx='256' cy='256' r='{r:.0f}' stroke='black' stroke-width='3' fill='red' /></svg>"

    test_cases = [
        (
            "Identical circles",
            f"<think>test</think><answer>{circle_svg(200)}</answer>",
            circle_svg(200),
        ),
        (
            "Different circles",
            f"<think>test</think><answer>{circle_svg(180)}</answer>",
            circle_svg(200),
        ),
        ("Missing tags", circle_svg(200), circle_svg(200)),
        (
            "Invalid SVG",
            "<think>test</think><answer>Not an SVG</answer>",
            circle_svg(200),
        ),
    ]

    print("Testing simplified L2 reward function:")
    for desc, response, ground_truth in test_cases:
        rewards = compute_rewards(response, ground_truth)
        print(f"\n{desc}:")
        print(
            f"  format={rewards.format:.2f}, length={rewards.length:.2f}, "
            f"l2={rewards.l2:.2f}, score={rewards.score:.2f}"
        )
