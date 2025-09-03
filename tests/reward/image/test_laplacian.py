"""Tests for Laplacian pyramid functionality."""

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


def test_laplacian_identical_images():
    """Test Laplacian pyramid distance for identical images."""
    svg = make_circle(
        cx=256, cy=256, r=100, fill="red", stroke="black", background="gray"
    )
    d = compare_images_by_laplacian(rasterize(svg), rasterize(svg))
    assert d < 1e-6


def test_laplacian_shifted_circles():
    """Test Laplacian pyramid distance for slightly shifted images."""
    svg_center = make_circle(
        cx=256, cy=256, r=100, fill="red", stroke="black", background="gray"
    )
    svg_shifted = make_circle(
        cx=236, cy=256, r=100, fill="red", stroke="black", background="gray"
    )
    d = compare_images_by_laplacian(rasterize(svg_center), rasterize(svg_shifted))
    assert 0.001 < d < 1.0


def test_laplacian_bgcolor_gray_vs_white():
    """Test that changing background color from #eee to white has minimal effect on Laplacian distance."""
    svg_gray = make_circle(
        cx=256, cy=256, r=100, fill="red", stroke="black", background="gray"
    )
    svg_white = make_circle(
        cx=256, cy=256, r=100, fill="red", stroke="black", background="white"
    )
    d = compare_images_by_laplacian(rasterize(svg_gray), rasterize(svg_white))
    # The distance should be small, but not zero (since #eee != white)
    assert 0.01 < d < 1.0


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
def test_padding_modes(mode):
    """Test different padding modes work correctly."""
    svg = make_circle(cx=256, cy=256, r=100, fill="green")
    img = rasterize(svg)

    distance = compare_images_by_laplacian(img, img, padding_mode=mode)
    assert distance < 1e-6
