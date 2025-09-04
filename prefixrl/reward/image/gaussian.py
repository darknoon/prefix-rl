import torch
import torch.nn.functional as F
from PIL import Image

from .laplacian import _gauss_kernel, _downsample
from .util import pil_to_tensor, resize_to_reference_image


def gaussian_pyramid(
    img: torch.Tensor, levels: int = 4, padding_mode: str = "reflect"
) -> list[torch.Tensor]:
    k = _gauss_kernel(img.device, img.dtype)
    pyr = [img]
    curr = img

    for _ in range(levels - 1):
        curr = _downsample(curr, k, padding_mode=padding_mode)
        pyr.append(curr)

    return pyr


def gaussian_pyramid_distance(
    pyr1: list[torch.Tensor],
    pyr2: list[torch.Tensor],
    weights: list[float] | None = None,
) -> float:
    """
    Compute weighted L2 distance between two Gaussian pyramids.

    Args:
        pyr1: First Gaussian pyramid (list of tensors)
        pyr2: Second Gaussian pyramid (list of tensors)
        weights: Optional list of weights for each pyramid level.
                If None, uses equal weights for all levels.

    Returns:
        float: Weighted sum of MSE losses across all pyramid levels
    """
    if weights is None:
        weights = [1.0] * len(pyr1)

    if len(weights) != len(pyr1):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match pyramid levels ({len(pyr1)})"
        )
    if sum(weights) < 1e-6 or any(weight < 0 for weight in weights):
        raise ValueError("Weights must be positive and sum to non-zero")

    total_distance = 0.0
    total_weight = 0.0

    for weight, g1, g2 in zip(weights, pyr1, pyr2):
        total_distance += weight * F.mse_loss(g1, g2, reduction="mean").item()
        total_weight += weight

    return total_distance / total_weight


def compare_images_by_gaussian(
    im: Image.Image,
    im_ref: Image.Image,
    padding_mode: str = "reflect",
    levels: int = 8,
    weights: list[float] | None = None,
) -> float:
    """
    Compare two images by the L2 distance of their Gaussian pyramids.
    The reference image is resized to match the input image before comparison.

    Args:
        im: Input image
        im_ref: Reference image
        padding_mode: Padding mode for convolutions ("zeros", "reflect", "replicate", "circular")
                     Default "reflect" reduces edge artifacts.
        levels: Number of pyramid levels
        weights: Optional list of weights for each pyramid level.
                If None, uses equal weights for all levels.

    Returns:
        float: Gaussian pyramid distance
    """
    # Prepare images for fair comparison
    im, im_ref = resize_to_reference_image(im, im_ref)

    # Convert to tensors and compute Gaussian pyramids
    p = gaussian_pyramid(
        pil_to_tensor(im).unsqueeze(0), levels=levels, padding_mode=padding_mode
    )
    p_ref = gaussian_pyramid(
        pil_to_tensor(im_ref).unsqueeze(0), levels=levels, padding_mode=padding_mode
    )

    if weights is None:
        weights = get_pyramid_weights(levels, weight_strategy="coarse_to_fine")

    return gaussian_pyramid_distance(p, p_ref, weights=weights)


def get_pyramid_weights(
    levels: int = 8,
    weight_strategy: str = "uniform",
) -> float:
    """
    Weighting strategy - "uniform", "coarse_to_fine", "fine_to_coarse"
    """
    if weight_strategy == "uniform":
        weights = None  # Equal weights
    elif weight_strategy == "coarse_to_fine":
        # Give more weight to coarser levels (higher indices)
        weights = [0.5 ** (levels - 1 - i) for i in range(levels)]
    elif weight_strategy == "fine_to_coarse":
        # Give more weight to finer levels (lower indices)
        weights = [0.5**i for i in range(levels)]
    else:
        raise ValueError(f"Unknown weight_strategy: {weight_strategy}")

    return weights
