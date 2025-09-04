"""Tests for general image comparison functionality (L2, Laplacian, Gaussian, and DreamSim)."""

import pytest
from prefixrl.reward.image.laplacian import compare_images_by_laplacian
from prefixrl.reward.image.l2 import compare_images_by_l2
from prefixrl.reward.image.gaussian import compare_images_by_gaussian
from prefixrl.reward.image.dreamsim import compare_images_by_dreamsim
from prefixrl.reward.svg.cairosvg import rasterize_svg
from prefixrl.reward.svg.shapes import make_circle


all_compare_funcs = [
    compare_images_by_l2,
    compare_images_by_laplacian,
    compare_images_by_gaussian,
    compare_images_by_dreamsim,
]


def rasterize(svg: str):
    img, _, _ = rasterize_svg(svg)
    return img


@pytest.mark.parametrize("compare_func", all_compare_funcs)
def test_identical_images(compare_func):
    """Test distance functions for identical images."""
    svg = make_circle(
        cx=256, cy=256, r=100, fill="red", stroke="black", background="gray"
    )
    d = compare_func(rasterize(svg), rasterize(svg))
    assert d < 1e-6


@pytest.mark.parametrize("compare_func", all_compare_funcs)
def test_shifted_circles(compare_func):
    """Test distance functions for slightly shifted images."""
    svg_center = make_circle(
        cx=256, cy=256, r=100, fill="red", stroke="black", background="gray"
    )
    svg_shifted = make_circle(
        cx=236, cy=256, r=100, fill="red", stroke="black", background="gray"
    )
    d = compare_func(rasterize(svg_center), rasterize(svg_shifted))
    assert 0.001 < d < 1.0


@pytest.mark.parametrize("compare_func", all_compare_funcs)
def test_bgcolor_gray_vs_white(compare_func):
    """Test that changing background color from gray to white has measurable effect."""
    svg_gray = make_circle(
        cx=256, cy=256, r=100, fill="red", stroke="black", background="gray"
    )
    svg_white = make_circle(
        cx=256, cy=256, r=100, fill="red", stroke="black", background="white"
    )
    d = compare_func(rasterize(svg_gray), rasterize(svg_white))
    # The distance should be small, but not zero (since gray != white)
    assert 0.01 < d < 1.0
