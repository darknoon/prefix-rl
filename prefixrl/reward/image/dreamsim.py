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
    DreamSim-based image comparator for perceptual similarity.

    This class wraps the DreamSim model for computing perceptual distances
    between images, which correlates better with human perception than
    pixel-based metrics like L2.
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

    @torch.no_grad()
    def compare_images(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Compute DreamSim perceptual distance between two PIL images.

        Args:
            img1: First image
            img2: Second image (reference)

        Returns:
            DreamSim distance (lower values indicate more similar images)
        """
        try:
            # Prepare images for fair comparison
            img1, img2 = resize_to_reference_image(img1, img2, mode="RGB")

            # Preprocess images
            processed_img1 = self.preprocess(img1)
            processed_img2 = self.preprocess(img2)

            # Compute distance
            distance = float(self.dreamsim_model(processed_img1, processed_img2))

            return distance

        except Exception as e:
            logger.error(f"DreamSim comparison failed: {e}")
            logger.error(f"Image sizes: {img1.size}, {img2.size}")
            logger.error(f"Image modes: {img1.mode}, {img2.mode}")
            raise


# Global instance for reuse
_dreamsim_comparator: Optional[DreamSimComparator] = None


def get_dreamsim_comparator(
    device: Optional[str] = None, cache_dir: Optional[str] = None
) -> DreamSimComparator:
    """
    Get or create a global DreamSim comparator instance.

    Args:
        device: Device to run on. If None, auto-detects CUDA availability
        cache_dir: Cache directory for model weights

    Returns:
        DreamSim comparator instance
    """
    global _dreamsim_comparator

    if _dreamsim_comparator is None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing DreamSim comparator with device: {device}")
        _dreamsim_comparator = DreamSimComparator(device=device, cache_dir=cache_dir)

    return _dreamsim_comparator


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
    comparator = get_dreamsim_comparator()
    return comparator.compare_images(img1, img2)


def dreamsim_reward(dreamsim_distance: float) -> float:
    """
    Convert DreamSim distance to a reward value.

    Args:
        dreamsim_distance: DreamSim distance (lower is more similar)

    Returns:
        Reward value (higher is better)
    """
    return 1.0 - 2.0 * dreamsim_distance
