"""
L2 distance computation for image comparison.
"""

import torch
from PIL import Image

from .util import pil_to_tensor, resize_to_reference_image


def l2_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute normalized L2 distance between two tensors."""
    h, w, c = x.shape
    den = (h * w * c) ** 0.5
    return (torch.norm(x - y) / den).item()


@torch.no_grad()
def compare_images_by_l2(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute L2 distance between two PIL images.

    Args:
        img1: First image
        img2: Second image (reference)

    Returns:
        L2 distance between the images
    """
    img1, img2 = resize_to_reference_image(img1, img2)

    return l2_distance(pil_to_tensor(img1), pil_to_tensor(img2))
