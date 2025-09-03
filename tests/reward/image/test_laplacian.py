"""Tests for Laplacian pyramid specific functionality."""

import pytest
from prefixrl.reward.image.laplacian import (
    compare_images_by_laplacian,
    laplacian_pyramid,
)
from prefixrl.reward.image.util import pil_to_tensor
from prefixrl.reward.svg.cairosvg import rasterize_svg
from prefixrl.reward.svg.shapes import make_circle


def rasterize(svg: str):
    img, _, _ = rasterize_svg(svg)
    return img


@pytest.mark.parametrize("levels", [3, 4, 6, 8])
def test_pyramid_levels(levels):
    """Test pyramid generates correct number of levels with proper shapes."""
    svg = make_circle(cx=256, cy=256, r=100, fill="blue")
    img = rasterize(svg)
    img_tensor = pil_to_tensor(img).unsqueeze(0)

    pyramid = laplacian_pyramid(img_tensor, levels=levels)
    assert len(pyramid) == levels

    # Check shape progression
    h, w = 512, 512
    for level in pyramid[:-1]:
        assert level.shape == (1, 3, h, w)
        h, w = h // 2, w // 2

    # Check final level
    assert pyramid[-1].shape == (1, 3, h, w)


@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
def test_laplacian_padding_modes(mode):
    """Test different padding modes work correctly for Laplacian."""
    svg = make_circle(cx=256, cy=256, r=100, fill="green")
    img = rasterize(svg)

    distance = compare_images_by_laplacian(img, img, padding_mode=mode)
    assert distance < 1e-6
