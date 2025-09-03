import torch
import torch.nn.functional as F
from PIL import Image
import argparse

from .util import pil_to_tensor, resize_to_reference_image


def _gauss_kernel(device, dtype):
    """5x5 Gaussian blur kernel."""
    a = torch.tensor([1, 4, 6, 4, 1], dtype=dtype, device=device)
    k = (a[:, None] * a[None, :]) / 256.0
    return k


def _blur(x, k, padding_mode="zeros"):
    """Apply Gaussian blur with configurable padding mode."""
    N, C, H, W = x.shape  # Destructure: batch, channels, height, width
    w = k.expand(C, 1, 5, 5)  # Expand kernel to match number of channels
    if padding_mode == "zeros":
        return F.conv2d(x, w, padding=2, groups=C)
    else:
        # Use F.pad for non-zero padding modes, then conv without padding
        x_padded = F.pad(x, (2, 2, 2, 2), mode=padding_mode)
        return F.conv2d(x_padded, w, padding=0, groups=C)


def _downsample(x, k, padding_mode="zeros"):
    """Blur + decimate (handles odd sizes) with configurable padding mode."""
    N, C, H, W = x.shape  # Destructure: batch, channels, height, width
    w = k.expand(C, 1, 5, 5)  # Expand kernel to match number of channels
    if padding_mode == "zeros":
        return F.conv2d(x, w, stride=2, padding=2, groups=C)
    else:
        # Use F.pad for non-zero padding modes, then conv without padding
        x_padded = F.pad(x, (2, 2, 2, 2), mode=padding_mode)
        return F.conv2d(x_padded, w, stride=2, padding=0, groups=C)


def _upsample(x, size, k, padding_mode="zeros"):
    """Bilinear upsample + blur with configurable padding mode."""
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return _blur(x, k, padding_mode=padding_mode)


def laplacian_pyramid(
    img: torch.Tensor, levels: int = 4, padding_mode: str = "reflect"
) -> list[torch.Tensor]:
    """
    Compute Laplacian pyramid with configurable padding mode.

    Args:
        img: (N,C,H,W) float tensor in [0,1] or any range. Arbitrary H,W.
        levels: Number of pyramid levels
        padding_mode: Padding mode for convolutions ("zeros", "reflect", "replicate", "circular")
                     Default "reflect" reduces edge artifacts.

    Returns:
        List of tensors: [Laplacian_0, Laplacian_1, ..., Gaussian_residual]
        pyr[0..levels-2] are Laplacians (high-pass), pyr[-1] is the smallest Gaussian residual.
    """
    k = _gauss_kernel(img.device, img.dtype)
    pyr = []
    curr = img
    for _ in range(levels - 1):
        low = _downsample(curr, k, padding_mode=padding_mode)
        up = _upsample(low, curr.shape[-2:], k, padding_mode=padding_mode)
        pyr.append(curr - up)
        curr = low
    pyr.append(curr)  # residual
    return pyr


def laplacian_pyramid_distance(pyr1, pyr2):
    """
    Compute L2 distance between two Laplacian pyramids.

    Args:
        pyr1: First Laplacian pyramid (list of tensors)
        pyr2: Second Laplacian pyramid (list of tensors)

    Returns:
        float: Sum of MSE losses across all pyramid levels
    """
    return sum(
        F.mse_loss(lx, ly, reduction="mean").item() for lx, ly in zip(pyr1, pyr2)
    )


def compare_images_by_laplacian(
    im: Image.Image, im_ref: Image.Image, padding_mode: str = "reflect"
) -> float:
    """
    Compare two images by the L2 distance of their Laplacian pyramids.
    The reference image is resized to match the input image before comparison.

    Args:
        im: Input image
        im_ref: Reference image
        padding_mode: Padding mode for convolutions ("zeros", "reflect", "replicate", "circular")
                     Default "reflect" reduces edge artifacts.

    Returns:
        float: Laplacian pyramid distance
    """
    # Prepare images for fair comparison
    im, im_ref = resize_to_reference_image(im, im_ref)

    # Convert to tensors and compute Laplacian pyramids
    p = laplacian_pyramid(pil_to_tensor(im).unsqueeze(0), padding_mode=padding_mode)
    p_ref = laplacian_pyramid(
        pil_to_tensor(im_ref).unsqueeze(0), padding_mode=padding_mode
    )
    return laplacian_pyramid_distance(p, p_ref)
