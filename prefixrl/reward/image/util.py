import torch
from torchvision.transforms import ToTensor
from PIL import Image
from typing import Tuple


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to torch tensor in [0,1], shape [C,H,W]."""
    return ToTensor()(img)


def resize_to_reference_image(
    image: Image.Image, reference: Image.Image, mode: str = "RGB"
) -> Tuple[Image.Image, Image.Image]:
    """
    Prepare two images for fair comparison by converting to RGB and resizing.

    Args:
        image: First image
        reference: Second image (reference - image will be resized to match this)

    Returns:
        Tuple of prepared images (image, reference)
    """
    # Convert to RGB
    image = image.convert(mode)
    reference = reference.convert(mode)

    # Resize img1 to match img2's size
    image = image.resize(reference.size, Image.Resampling.LANCZOS)

    return image, reference
