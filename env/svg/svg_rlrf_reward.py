# Copyright 2025 Andrew Pouliot
# Based on https://arxiv.org/pdf/2505.20793

import re
import cairosvg.parser
import cairosvg.surface

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
import io
import numpy as np
import torchvision.transforms as T
import os
import tempfile
import wandb
from pathlib import Path
from typing import TypedDict, Optional, Literal
import logging
import traceback

from dreamsim import dreamsim

# Minimum reward value for failed evaluations
MIN_REWARD = -1.0

REWARD_WEIGHTS = {
    "format": 1.0,
    "length": 1.0,
    "l2": 1.0,
    "l2_canny": 1.0,
    "dreamsim": 1.0,
    "dreamsim_canny": 1.0,
}


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    """Normalize image tensor to zero mean and unit variance."""
    # img_tensor should be [C, H, W] or [B, C, H, W]
    if x.ndim == 3:
        # Add batch dimension if not present
        x = x.unsqueeze(0)

    # Calculate mean and std across spatial dimensions
    # Keep dims to ensure broadcasting works correctly
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = (
        x.std(dim=(-2, -1), keepdim=True) + 1e-6
    )  # Add epsilon to avoid division by zero

    # Normalize
    return (x - mean) / std


def canny(
    image_tensor: torch.Tensor,
    device: torch.device | str,
    low_thresh_factor: float = 0.1,
    high_thresh_factor: float = 0.2,
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.0,
    dilate_kernel_size: int = 3,
    dilate_iterations: int = 1,
    final_blur_size: int = 13,
    final_blur_sigma: float = 1.0,
) -> torch.Tensor:
    """
    Applies Canny-like edge detection using PyTorch, followed by dilation and Gaussian blur.
    Input should be a tensor [B, C, H, W] or [C, H, W] in range [0, 1].
    Output is a tensor of same spatial dimensions with edge intensities.
    """
    if isinstance(device, str):
        device = torch.device(device)

    # Ensure 4D input [B, C, H, W]
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Convert to grayscale if needed
    if image_tensor.shape[1] == 3:
        # weights for RGB to grayscale conversion
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(
            1, 3, 1, 1
        )
        img_gray = (image_tensor * rgb_weights).sum(dim=1, keepdim=True)
    else:
        img_gray = image_tensor

    # 1. Initial Gaussian Blur
    blur_transform = T.GaussianBlur(
        kernel_size=(gaussian_kernel_size, gaussian_kernel_size),
        sigma=(gaussian_sigma, gaussian_sigma),
    )
    img_blurred = blur_transform(img_gray)

    # 2. Sobel Filters for Gradients
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=img_blurred.dtype
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=img_blurred.dtype
    ).view(1, 1, 3, 3)

    grad_x = F.conv2d(img_blurred, sobel_x, padding=1)
    grad_y = F.conv2d(img_blurred, sobel_y, padding=1)

    # 3. Gradient Magnitude
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

    # Normalize magnitude to 0-1
    grad_magnitude = grad_magnitude / (grad_magnitude.max() + 1e-6)

    # 4. Thresholding (simplified Canny)
    edges = (grad_magnitude > high_thresh_factor).float()

    # 5. Dilation
    if dilate_iterations > 0:
        dilation_kernel = torch.ones(
            1,
            1,
            dilate_kernel_size,
            dilate_kernel_size,
            device=device,
            dtype=edges.dtype,
        )
        for _ in range(dilate_iterations):
            edges = F.conv2d(
                edges,
                dilation_kernel,
                padding=dilate_kernel_size // 2,
                groups=edges.shape[1],  # Apply to each channel independently
            )
            edges = (edges > 0).float()  # Threshold to keep binary

    # 6. Final Gaussian Blur
    final_blur = T.GaussianBlur(
        kernel_size=(final_blur_size, final_blur_size),
        sigma=(final_blur_sigma, final_blur_sigma),
    )
    edges_blurred = final_blur(edges)

    assert edges_blurred.ndim == 4, (
        f"Canny should return 4D tensor [B,C,H,W], got {edges_blurred.ndim}D: {edges_blurred.shape}"
    )
    return edges_blurred


def edge_to_pil(edge_map_tensor: torch.Tensor) -> PILImage.Image:
    """Converts a edge map tensor [B, 1, H, W] or [1, H, W] to a 1-channel PIL Image."""
    if edge_map_tensor.ndim == 4:
        edge_map_tensor = edge_map_tensor.squeeze(0)  # Remove batch dim
    if edge_map_tensor.ndim != 3:
        raise ValueError(
            f"Edge map tensor must be 3D after removing batch dim, got {edge_map_tensor.ndim} dimensions"
        )

    # Convert to PIL and then to RGB
    edge_map_pil = T.ToPILImage()(edge_map_tensor.cpu())
    return edge_map_pil


@dataclass
class ImageComparisonResult:
    l2: float  # L2 distance between normalized images
    l2_canny: float  # L2 distance between Canny edge maps
    dreamsim: float  # DreamSim distance (lower is more similar)
    dreamsim_canny: float  # DreamSim distance on edge maps
    canny_im: PILImage.Image  # Canny edge map of the image
    canny_im_ref: PILImage.Image  # Canny edge map of the reference image


def l2(x: torch.Tensor, y: torch.Tensor) -> float:
    return torch.norm(x - y) / np.sqrt(x.numel())


logger = logging.getLogger(__name__)


class ImageComparator:
    def __init__(
        self,
        device: str = "cpu",
        dreamsim_cache_dir: str = os.environ.get("DREAMSIM_CACHE_DIR", "./models"),
    ):
        self.device = device
        logger.info("Loading DreamSim model (from dreamsim library)...")
        try:
            self.dreamsim_model, self.dreamsim_preprocess = dreamsim(
                pretrained=True, device=self.device, cache_dir=dreamsim_cache_dir
            )
            self.dreamsim_model.eval()
            logger.info("DreamSim model and preprocessor loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load DreamSim model from library: {e}")
            logger.error(
                "Please ensure the 'dreamsim' library is installed and torch.hub caching is working."
            )
            raise

    @torch.no_grad()
    def compare_images(
        self, im: PILImage.Image, im_ref: PILImage.Image
    ) -> ImageComparisonResult | None:
        """
        Compare two images using multiple metrics:
        1. L2 distance between normalized images
        2. L2 distance between Canny edge maps (with dilation and blur)
        3. DreamSim distance on original images
        4. DreamSim distance on edge maps
        """

        to_tensor = T.ToTensor()

        try:
            # Convert PIL images to RGB
            im = im.convert("RGB")
            im_ref = im_ref.convert("RGB")

            # Convert PIL to tensors [0,1]
            tensor = to_tensor(im).to(self.device)
            tensor_ref = to_tensor(im_ref).to(self.device)

            # 1. L2 on normalized images
            l2_distance = l2(normalize_image(tensor), normalize_image(tensor_ref))

            # 2. L2 on Canny edge maps
            edge_map = canny(tensor, self.device)
            edge_map_ref = canny(tensor_ref, self.device)
            assert edge_map.shape == edge_map_ref.shape, (
                f"Edge maps shape mismatch: {edge_map.shape} vs {edge_map_ref.shape}"
            )
            l2_canny = l2(edge_map, edge_map_ref)

            # 3. DreamSim
            try:
                processed_im = self.dreamsim_preprocess(im)
                processed_im_ref = self.dreamsim_preprocess(im_ref)
                dreamsim_distance = float(
                    self.dreamsim_model(processed_im, processed_im_ref)
                )
            except Exception as e:
                logger.error(f"DreamSim failed on main images: {e}")
                logger.error(f"Image sizes: {im.size}, {im_ref.size}")
                logger.error(f"Image modes: {im.mode}, {im_ref.mode}")
                raise

            # 4. DreamSim Canny
            pil_canny = edge_to_pil(edge_map)
            pil_canny_ref = edge_to_pil(edge_map_ref)
            try:
                processed_canny = self.dreamsim_preprocess(pil_canny.convert("RGB"))
                processed_canny_ref = self.dreamsim_preprocess(
                    pil_canny_ref.convert("RGB")
                )
                logger.info(
                    f"DreamSim Canny input shapes: {processed_canny.shape}, {processed_canny_ref.shape}"
                )
                dreamsim_canny_distance = float(
                    self.dreamsim_model(processed_canny, processed_canny_ref)
                )
            except Exception as e:
                logger.error(f"DreamSim failed on Canny images: {e}")
                logger.error(
                    f"Canny image: {pil_canny.size} ({pil_canny.mode}), ref: {pil_canny_ref.size} ({pil_canny_ref.mode})"
                )
                raise

            return ImageComparisonResult(
                l2=float(l2_distance),
                l2_canny=float(l2_canny),
                dreamsim=dreamsim_distance,
                dreamsim_canny=dreamsim_canny_distance,
                canny_im=pil_canny,
                canny_im_ref=pil_canny_ref,
            )

        except Exception as e:
            import traceback

            logger.error(f"Error during image comparison: {e}")
            logger.error("Backtrace:")
            logger.error(traceback.format_exc())
            return None


def get_svg_size(tree: cairosvg.parser.Tree) -> tuple[int, int]:
    width = tree.get("width")
    height = tree.get("height")

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

    return float(width), float(height)


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
) -> bytes:
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
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>\s{0,3}", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def extract_svg_text(full_response: str) -> str | None:
    pattern = re.compile(r"<svg.*?</svg>", re.DOTALL)
    match = re.search(pattern, full_response)
    return match.group(0).strip() if match else None


image_comparator: ImageComparator | None = None


# TODO: support unloading the weights so we can use GPU / not take up vram
def get_image_comparator() -> ImageComparator:
    global image_comparator
    if image_comparator is None:
        image_comparator = ImageComparator()
    return image_comparator


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


@dataclass
class PreprocessedResponse:
    response_str: str
    svg: str
    svg_gt: str
    svg_im: PILImage.Image
    svg_im_gt: PILImage.Image


def expected_response(
    thinking: str = "", svg: str = "", add_line_breaks: bool = False
) -> str:
    if svg == "":
        raise ValueError("SVG cannot be empty")
    # we don't really care about a bit of whitespace, so test that we don't penalize it harshly
    sp = "\n" if add_line_breaks else ""
    return f"{sp}<think>{thinking}</think>{sp}<answer>{svg}</answer>{sp}"


def clip(x: float, min_val: float, max_val: float) -> float:
    return max(min(x, max_val), min_val)


def l2_reward(l2_distance: float) -> float:
    return clip(1.0 - l2_distance, -1.0, 1.0)


def dreamsim_reward(dreamsim: float) -> float:
    return 1.0 - 2.0 * dreamsim


def svg_env(response_str: str, svg_gt: str) -> PreprocessedResponse | None:
    """
    Turns a response and ground truth svg (just the content within <answer> tags) into rasterized images for comparison.
    Also computes the format and length rewards.
    """
    svg = extract_svg_text(response_str)
    if svg is None:
        return None
    svg_im_bytes, _, _ = rasterize_svg(svg)
    svg_im = PILImage.open(io.BytesIO(svg_im_bytes))
    svg_gt_bytes, _, _ = rasterize_svg(svg_gt)
    svg_gt_im = PILImage.open(io.BytesIO(svg_gt_bytes))
    return PreprocessedResponse(
        response_str=response_str,
        svg=svg,
        svg_gt=svg_gt,
        svg_im=svg_im,
        svg_im_gt=svg_gt_im,
    )


def reward_from_distance(distance: float) -> float:
    return 1.0 - distance


@dataclass
class SVGRewards:
    format: float
    length: float
    l2: float
    l2_canny: float
    dreamsim: float
    dreamsim_canny: float
    overall: float


class MergedResult(TypedDict):
    id: str
    prompt: str
    response: str
    svg: str
    svg_gt: str
    status: Literal["OK", "FAIL"]
    error: str | None
    # dump images if we have them - using logical names from write_debug_images_dict
    svg_image: str  # filename for generated SVG image
    svg_gt_image: str  # filename for ground truth SVG image
    canny: str  # filename for generated canny image
    canny_gt: str  # filename for ground truth canny image
    diff: str  # filename for diff image
    diff_canny: str  # filename for canny diff image
    # raw image scores
    score_l2: float | None
    score_l2_canny: float | None
    score_dreamsim: float | None
    score_dreamsim_canny: float | None
    # rewards
    reward_format: float
    reward_length: float
    reward_l2: float
    reward_l2_canny: float
    reward_dreamsim: float
    reward_dreamsim_canny: float
    reward_overall: float


def compute_rewards(
    p: PreprocessedResponse, image_scores: ImageComparisonResult | None = None
) -> SVGRewards:
    try:
        format_r = format_reward(p.response_str)
        length_r = length_reward(len(p.response_str), len(expected_response(svg=p.svg)))
    except Exception as e:
        print(f"Error in image comparison: {e}")
        image_scores = None

    weight_sum = sum(REWARD_WEIGHTS.values())

    BAD = MIN_REWARD
    # Build scores dict with only available metrics
    rewards = {
        "format": format_r,
        "length": length_r,
        # penalize hard if we can't render the image
        "l2": BAD,
        "l2_canny": BAD,
        "dreamsim": BAD,
        "dreamsim_canny": BAD,
    }

    if image_scores is not None:
        rewards.update(
            {
                "l2": l2_reward(image_scores.l2),
                "l2_canny": l2_reward(image_scores.l2_canny),
                "dreamsim": dreamsim_reward(image_scores.dreamsim),
                "dreamsim_canny": dreamsim_reward(image_scores.dreamsim_canny),
            }
        )

    rewards["overall"] = sum(rewards.values()) / weight_sum

    return SVGRewards(**rewards)


def write_debug_images_dict(
    p: PreprocessedResponse,
    im: ImageComparisonResult,
    output_dir: Path,
    id_prefix: str = "",
) -> dict[str, str]:
    """
    Write debug images to files and return a dictionary mapping logical names to filenames.

    Args:
        p: PreprocessedResponse containing the images
        im: ImageComparisonResult containing comparison images
        output_dir: Directory to write images to (str or Path)
        id_prefix: Optional prefix for filenames (e.g., "001_")

    Returns:
        Dictionary mapping logical names to filenames, e.g., {"svg": "001_svg.png"}
    """
    from PIL import ImageChops

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate diff images
    diff = None
    diff_canny = None
    try:
        diff = ImageChops.difference(p.svg_im, p.svg_im_gt)
        diff_canny = ImageChops.difference(im.canny_im, im.canny_im_ref)
    except Exception as e:
        logger.error(f"Error generating diff images: {e}")
        logger.error(traceback.format_exc())

    # Define image mappings: logical_name -> (image, filename)
    images = {
        "svg": (p.svg_im, f"{id_prefix}svg.png"),
        "svg_gt": (p.svg_im_gt, f"{id_prefix}svg_gt.png"),
        "canny": (im.canny_im, f"{id_prefix}canny.png"),
        "canny_gt": (im.canny_im_ref, f"{id_prefix}canny_gt.png"),
        "diff": (diff, f"{id_prefix}diff.png"),
        "diff_canny": (diff_canny, f"{id_prefix}diff_canny.png"),
    }

    # Save images and build return dictionary
    result = {}
    for logical_name, (image, filename) in images.items():
        filepath = output_dir / filename
        image.save(filepath, format="PNG")
        result[logical_name] = filename

    return result


def write_debug_images(
    p: PreprocessedResponse,
    im: ImageComparisonResult,
    rewards: SVGRewards,
    tempdir: Path,
    markdown=True,
    markdown_title: str | None = "Image comparison",
    log=False,
):
    # Use the new consolidated function
    image_paths = write_debug_images_dict(p, im, tempdir)

    if log and not markdown:
        for logical_name, filename in image_paths.items():
            print(f"Saved {logical_name} to {tempdir}/{filename}")

    if markdown:
        with open(f"{tempdir}/images.md", "w") as f:
            f.write(f"# {markdown_title}\n")
            f.write("\n")
            f.write("## Rewards\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in rewards.__dict__.items():
                f.write(f"| {key.replace('_', ' ').title()} | {value:.3f} |\n")
            f.write("\n\n")
            f.write("## Images\n")
            for logical_name, filename in image_paths.items():
                f.write(f"{logical_name}\n")
                f.write(f"![{logical_name}]({filename})\n")
                f.write("\n")
        print(f"Wrote {tempdir}/images.md")


def render_and_compute_rewards(response_str: str, svg_gt: str) -> SVGRewards | None:
    p = svg_env(response_str, svg_gt)
    image_scores = None
    if p is not None:
        image_scores = get_image_comparator().compare_images(p.svg_im, p.svg_im_gt)
    return compute_rewards(p, image_scores)


def evaluate_and_log_to_wandb(
    response_str: str, svg_gt: str, step: Optional[int] = None, prefix: str = "eval"
) -> Optional[SVGRewards]:
    """
    Evaluate the generated SVG response against the ground truth, log images and metrics to wandb.
    Args:
        response_str: The model's response string (should contain <svg>...)</svg>).
        svg_gt: The ground truth SVG string.
        step: Optional step/global_step for wandb logging.
        prefix: Optional prefix for wandb keys.
    Returns:
        SVGRewards or None if evaluation failed.
    """
    p = svg_env(response_str, svg_gt)
    if p is None:
        print("Failed to parse SVG from response.")
        return None
    image_scores = get_image_comparator().compare_images(p.svg_im, p.svg_im_gt)
    rewards = compute_rewards(p, image_scores)

    # Convert images to wandb.Image
    images = {}
    if image_scores is not None:
        images = {
            f"{prefix}/response": wandb.Image(p.svg_im, caption="Response SVG"),
            f"{prefix}/ground_truth": wandb.Image(
                p.svg_im_gt, caption="Ground Truth SVG"
            ),
            f"{prefix}/response_canny": wandb.Image(
                image_scores.canny_im, caption="Response Canny"
            ),
            f"{prefix}/ground_truth_canny": wandb.Image(
                image_scores.canny_im_ref, caption="Ground Truth Canny"
            ),
        }
    else:
        images = {
            f"{prefix}/response": wandb.Image(p.svg_im, caption="Response SVG"),
            f"{prefix}/ground_truth": wandb.Image(
                p.svg_im_gt, caption="Ground Truth SVG"
            ),
        }

    # Prepare metrics for wandb
    metrics = {f"{prefix}/{k}": float(v) for k, v in rewards.__dict__.items()}
    if step is not None:
        wandb.log({**metrics, **images}, step=step)
    else:
        wandb.log({**metrics, **images})
    return rewards


if __name__ == "__main__":

    def circle_svg(r: float) -> str:
        return f"<svg width='512' height='512'><circle cx='256' cy='256' r='{r:.0f}' stroke='black' stroke-width='3' fill='red' /></svg>"

    def random_path_svg(n: int) -> str:
        p = torch.rand(n, 2) * 512
        p = p.tolist()
        p = [f"{x:.0f} {y:.0f}" for x, y in p]
        p = " ".join(p)
        return f"<svg width='512' height='512'><path d='M 256 256 L {p}' stroke='black' stroke-width='3' fill='none' /></svg>"

    # Test case 1: Identical circles
    response = expected_response(
        thinking="I think this is just a circle, so it should be easy to generate.",
        svg=circle_svg(200),
        add_line_breaks=True,
    )
    p = svg_env(response, circle_svg(200))
    im = get_image_comparator().compare_images(p.svg_im, p.svg_im_gt)

    tempdir = tempfile.mkdtemp()
    thinking = "I think this is just a circle, so it should be easy to generate."
    test_cases = [
        (
            "Identical circles",
            expected_response(
                thinking=thinking,
                svg=circle_svg(200),
                add_line_breaks=True,
            ),
            circle_svg(200),
        ),
        (
            "Slightly different circles",
            expected_response(
                thinking=thinking,
                svg=circle_svg(180),
                add_line_breaks=True,
            ),
            circle_svg(200),
        ),
        (
            "Very different circles",
            expected_response(
                thinking=thinking,
                svg=circle_svg(100),
                add_line_breaks=True,
            ),
            circle_svg(200),
        ),
        (
            "Missing thinking tags",
            f"<answer>{circle_svg(200)}</answer>",
            circle_svg(200),
        ),
        (
            "Can't parse svg",
            "<answer>I don't know how to do this</answer>",
            circle_svg(200),
        ),
        (
            "Random path",
            expected_response(
                thinking=thinking,
                svg=random_path_svg(100),
                add_line_breaks=True,
            ),
            random_path_svg(100),
        ),
    ]

    for idx, (description, response, svg_gt) in enumerate(test_cases, 1):
        print("-" * 80)
        print(f"Case {idx} - {description}:")
        p = svg_env(response, svg_gt)
        if p is None:
            print("  !Failed to parse SVG!")
            print(f"  Response: {response}")
            print(f"  SVG GT: {svg_gt}")
            continue
        im = get_image_comparator().compare_images(p.svg_im, p.svg_im_gt)
        tempdir = tempfile.mkdtemp()
        print(f"Writing comparison images to: {tempdir}")
        rewards = compute_rewards(p, im)
        write_debug_images(
            p,
            im,
            rewards,
            tempdir,
            markdown=True,
            markdown_title=f"Case {idx} - {description}",
            log=True,
        )
        print("\n".join(f"{k}: {v:.3f}" for k, v in rewards.__dict__.items()))
