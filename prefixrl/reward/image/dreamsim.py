"""
DreamSim distance computation for perceptual image comparison.

Based on the DreamSim library: https://github.com/ssundaram21/dreamsim
"""

import os
import logging
import torch
from PIL import Image
from typing import Optional

from .util import resize_to_reference_image

logger = logging.getLogger(__name__)


class DreamSimComparator:
    """
    This caches a DreamSim model and preprocessor for reuse.
    """

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the DreamSim comparator.

        Args:
            device: Device to run the model on ("cpu" or "cuda")
            cache_dir: Directory to cache model weights. If None, uses DREAMSIM_CACHE_DIR
                      environment variable or defaults to "./models"
        """
        self.device = device

        if cache_dir is None:
            cache_dir = os.environ.get("DREAMSIM_CACHE_DIR", "./models")

        logger.info("Loading DreamSim model (from dreamsim library)...")
        try:
            from dreamsim import dreamsim

            self.dreamsim_model, self._dreamsim_preprocess = dreamsim(
                pretrained=True, device=self.device, cache_dir=cache_dir
            )
            self.dreamsim_model.eval()
            logger.info("DreamSim model and preprocessor loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load DreamSim model from library: {e}")
            logger.error(
                "Please ensure the 'dreamsim' library is installed and torch.hub caching is working."
            )
            raise

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for DreamSim model.

        Args:
            image: PIL Image to preprocess

        Returns:
            Preprocessed tensor on the correct device
        """
        return self._dreamsim_preprocess(image).to(self.device)

    def __call__(self, image: Image.Image, reference: Image.Image) -> float:
        return self.compare_images(image, reference)

    @torch.no_grad()
    def compare_images(self, image: Image.Image, reference: Image.Image) -> float:
        # Prepare images for fair comparison
        image, reference = resize_to_reference_image(image, reference, mode="RGB")

        # Resize to 224x224 and convert to tensors
        processed_img1 = self.preprocess(image)
        processed_img2 = self.preprocess(reference)

        # Compute distance
        distance = float(self.dreamsim_model(processed_img1, processed_img2))

        return distance


# Global instance to cache the comparator
_model: Optional[DreamSimComparator] = None


def get_dreamsim_comparator(device: Optional[str] = None) -> DreamSimComparator:
    """
    Get or create a global DreamSim comparator instance.

    Args:
        device: Device to run on. If None, auto-detects CUDA availability
        cache_dir: Cache directory for model weights

    Returns:
        DreamSim comparator instance
    """
    global _model

    if _model is None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        logger.info(f"Initializing DreamSim comparator with device: {device}")
        _model = DreamSimComparator(device=device)

    return _model


@torch.no_grad()
def compare_images_by_dreamsim(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute DreamSim perceptual distance between two PIL images.

    This is a convenience function that uses a global DreamSim comparator instance.

    Args:
        img1: First image
        img2: Second image (reference)

    Returns:
        DreamSim distance (lower values indicate more similar images)
    """
    return get_dreamsim_comparator()(img1, img2)


def dreamsim_reward(dreamsim_distance: float) -> float:
    """
    Convert DreamSim distance to a reward value.

    Args:
        dreamsim_distance: DreamSim distance (lower is more similar)

    Returns:
        Reward value (higher is better)
    """
    return 1.0 - 2.0 * dreamsim_distance
