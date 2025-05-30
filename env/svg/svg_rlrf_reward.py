# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from cairosvg import svg2png
from dataclasses import dataclass
import torch

from transformers import AutoProcessor, AutoModel
from PIL import Image as PILImage
import io
import numpy as np
import torch.nn.functional as F


@dataclass
class ImageComparisonResult:
    l2: float
    clip_vit_b_32: float


class ImageComparator:
    def __init__(self, device: str = "cpu"):
        self.processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", use_fast=True
        )
        self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        self.model.to(self.device)

    def process_image(self, image_bytes: bytes):
        """Process image bytes into RGB PIL Image."""

        image = PILImage.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        return image

    @torch.no_grad()
    def get_image_embedding(self, image, size=(224, 224)):
        """Get normalized vision embedding for an image."""
        image = image.resize(size)
        inputs = self.processor(images=image, return_tensors="pt")
        embedding = self.model.get_image_features(**inputs)
        return embedding / embedding.norm(dim=-1, keepdim=True)

    def compare_images(
        self, image_bytes: bytes, reference_bytes: bytes
    ) -> ImageComparisonResult:
        """Compare two images using L2 distance and vision embedding similarity.

        Args:
            image_bytes: PNG image bytes of the first image
            reference_bytes: PNG image bytes of the reference image to compare against

        Returns:
            Dict containing l2_distance and embedding_similarity metrics
        """

        # Load and process images
        image = self.process_image(image_bytes)
        reference = self.process_image(reference_bytes)

        # Calculate L2 distance
        if image.size != reference.size:
            # resize image to reference size
            image = image.resize(reference.size)
            # warn
            print(
                f"Image size {image.size} does not match reference size {reference.size}, resizing image to reference size. Results may be inaccurate."
            )

        img_array = np.array(image)
        ref_array = np.array(reference)

        l2_distance = np.sqrt(np.sum((img_array - ref_array) ** 2)) / (
            img_array.size * 255
        )

        # Get embeddings and calculate similarity
        emb_image = self.get_image_embedding(image)
        emb_reference = self.get_image_embedding(reference)
        embedding_similarity = F.cosine_similarity(emb_image, emb_reference).item()

        return {
            "l2": float(l2_distance),
            "clip_vit_b_32": float(embedding_similarity),
        }


def dream_sim_reward(svg: str, reference: str) -> float:
    """
    Semantic Similarity Rewards For semantic-level rewards, we use DreamSim [Fu et al., 2023],
    which encodes each image using three ViT-B/16 backbones - CLIP, OpenCLIP [Cherti et al., 2023],
    and DINOv2 [Oquab et al., 2023]. Their feature vectors are concatenated, passed through a linear
    projection, and compared using cosine similarity. This provides a meaningful semantic similarity
    signal for the Im2SVG task. For shape-focused feedback, we apply DreamSim Canny, which uses
    an edge detector on both prediction and target images before computing DreamSim. A comparison
    of edge maps emphasizes crisp contours and geometric accuracy while remaining insensitive to
    variation in color or texture. We convert the similarity score sim ∈ [-1, 1] to a reward using
    Rsim = 1 - 2 * sim, where higher values indicate stronger semantic alignment. For the Text2SVG
    task, we use CLIP as a reward to compute the cosine similarity between the text prompt and the
    rendered SVG image in the embedding space. We also utilize VLM as a judge to assess the quality of
    generation (see more details about Text2SVG rewards in Appendix A.2)."""
    pass


def rasterize_svg(svg_content: str, width=1200, height=800) -> bytes:
    """Rasterize SVG content to PNG image bytes using CairoSVG."""
    return svg2png(
        bytestring=svg_content.encode("utf-8"), output_width=width, output_height=height
    )


def format_reward(predict: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict)
    return 1.0 if format_match else 0.0


def extract_svg_text(full_response: str) -> str:
    pattern = re.compile(r"<svg.*?</svg>", re.DOTALL)
    match = re.search(pattern, full_response)
    return match.group(0).strip() if match else ""


# TODO: support unloading
image_comparator = ImageComparator()


def compute_score(
    predict: str, ground_truth: str, format_weight: float = 0.5
) -> dict[str, float]:
    format_score = format_reward(predict)
    svg = extract_svg_text(predict)
    im = rasterize_svg(svg)
    gt_im = rasterize_svg(ground_truth)
    image_scores = image_comparator.compare_images(im, gt_im)
    score_weights = {
        "format": format_weight,
        "l2": 0.5,
        "clip_vit_b_32": 0.5,
    }
    return {
        "overall": sum(
            [image_scores[k] * score_weights[k] for k in image_scores.keys()]
        )
        / sum(score_weights.values()),
        "format": format_score,
        **image_scores,
    }


if __name__ == "__main__":
    svg = """
    <svg width='512' height='512'><circle cx='256' cy='256' r='192' stroke='black' stroke-width='3' fill='red' /></svg>
    """

    response = """
<think>
I think this is just a circle.
</think>
<answer>
<svg width='512' height='512'><circle cx='256' cy='256' r='200' stroke='black' stroke-width='3' fill='red' /></svg>
</answer>
"""
    print(compute_score(response, svg))
