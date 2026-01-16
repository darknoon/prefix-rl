"""
Metrics computation for the agentic evaluation harness.
"""

from PIL import Image

from prefixrl.reward.image.l2 import compare_images_by_l2
from prefixrl.reward.image.dreamsim import compare_images_by_dreamsim


class MetricsComputer:
    """Computes image comparison metrics."""

    def compute(self, generated: Image.Image, target: Image.Image) -> dict:
        """
        Compute metrics comparing generated image to target.

        Args:
            generated: The generated/rasterized image
            target: The target/ground truth image

        Returns:
            Dict with metric values (lower is better for all)
        """
        l2_distance = compare_images_by_l2(generated, target)
        dreamsim_distance = compare_images_by_dreamsim(generated, target)

        return {
            "l2_distance": round(l2_distance, 6),
            "dreamsim_distance": round(dreamsim_distance, 6),
        }
