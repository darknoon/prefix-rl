"""Tests for DreamSim-specific functionality."""

import pytest
from prefixrl.reward.image.dreamsim import compare_images_by_dreamsim, dreamsim_reward
from prefixrl.reward.svg.cairosvg import rasterize_svg
from prefixrl.reward.svg.shapes import make_circle


def rasterize(svg: str):
    img, _, _ = rasterize_svg(svg)
    return img


def test_dreamsim_reward_function():
    """Test that DreamSim reward function converts distances correctly."""
    # Test perfect similarity (distance = 0)
    assert dreamsim_reward(0.0) == 1.0

    # Test moderate similarity (distance = 0.25)
    assert dreamsim_reward(0.25) == 0.5

    # Test high dissimilarity (distance = 0.5)
    assert dreamsim_reward(0.5) == 0.0

    # Test very high dissimilarity (distance = 1.0)
    assert dreamsim_reward(1.0) == -1.0


def test_dreamsim_specific_behavior():
    """Test DreamSim-specific behavior with real images."""
    # Create identical images
    svg_identical = make_circle(
        cx=256, cy=256, r=100, fill="red", stroke="black", background="white"
    )

    # Create slightly different image (different color)
    svg_different_color = make_circle(
        cx=256, cy=256, r=100, fill="blue", stroke="black", background="white"
    )

    # Create very different image (different size and position)
    svg_very_different = make_circle(
        cx=128, cy=128, r=50, fill="green", stroke="red", background="black"
    )

    # Test distances
    d_identical = compare_images_by_dreamsim(
        rasterize(svg_identical), rasterize(svg_identical)
    )
    d_color = compare_images_by_dreamsim(
        rasterize(svg_identical), rasterize(svg_different_color)
    )
    d_very_different = compare_images_by_dreamsim(
        rasterize(svg_identical), rasterize(svg_very_different)
    )

    # DreamSim should show progression: identical < color change < very different
    assert d_identical < 0.01  # Nearly identical
    assert d_color > d_identical  # Color change should be detectable
    assert d_very_different > d_color  # Major changes should have higher distance

    # Test rewards
    r_identical = dreamsim_reward(d_identical)
    r_color = dreamsim_reward(d_color)
    r_very_different = dreamsim_reward(d_very_different)

    # Rewards should be inversely related to distances
    assert r_identical > r_color > r_very_different


def test_dreamsim_identical_images():
    """Test DreamSim distance for identical images."""
    svg = make_circle(
        cx=256, cy=256, r=100, fill="red", stroke="black", background="gray"
    )
    d = compare_images_by_dreamsim(rasterize(svg), rasterize(svg))
    assert d < 1e-6


def test_dreamsim_shifted_circles():
    """Test DreamSim distance for slightly shifted images."""
    svg_center = make_circle(
        cx=256, cy=256, r=100, fill="red", stroke="black", background="gray"
    )
    svg_shifted = make_circle(
        cx=236, cy=256, r=100, fill="red", stroke="black", background="gray"
    )
    d = compare_images_by_dreamsim(rasterize(svg_center), rasterize(svg_shifted))
    assert 0.001 < d < 1.0


def test_dreamsim_bgcolor_gray_vs_white():
    """Test that DreamSim detects background color changes."""
    svg_gray = make_circle(
        cx=256, cy=256, r=100, fill="red", stroke="black", background="gray"
    )
    svg_white = make_circle(
        cx=256, cy=256, r=100, fill="red", stroke="black", background="white"
    )
    d = compare_images_by_dreamsim(rasterize(svg_gray), rasterize(svg_white))
    # The distance should be small, but not zero (since gray != white)
    assert 0.01 < d < 1.0
