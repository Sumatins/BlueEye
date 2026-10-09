"""Unit tests for the optional underwater enhancement pipeline."""

from __future__ import annotations

import numpy as np

from app.processing.enhancement import (
    EnhancementConfig,
    correct_color,
    denoise,
    enhance,
    enhance_contrast,
)


def _gradient_image(height: int = 120, width: int = 160) -> np.ndarray:
    """A dark synthetic underwater-ish image with a coloured blob."""
    rng = np.random.default_rng(42)
    image = rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)
    image[40:80, 60:120] = (20, 160, 200)
    return image


def test_disabled_config_returns_input_unchanged() -> None:
    image = _gradient_image()
    config = EnhancementConfig(enabled=False)
    result = enhance(image, config)
    assert result is image  # passthrough, no copy, no work done


def test_none_config_returns_input_unchanged() -> None:
    image = _gradient_image()
    assert enhance(image, None) is image


def test_enhanced_output_preserves_shape_and_dtype() -> None:
    image = _gradient_image()
    config = EnhancementConfig(enabled=True, color_correction=True, contrast=True, denoise=True)
    result = enhance(image, config)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result is not image
    assert not np.array_equal(result, image)


def test_color_correction_equalizes_channel_means() -> None:
    # Strong blue/green cast, typical for underwater photos.
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 0] = 30   # blue
    image[..., 1] = 90   # green
    image[..., 2] = 180  # red
    corrected = correct_color(image)
    means = corrected.reshape(-1, 3).mean(axis=0)
    assert means.max() - means.min() < 5.0  # channels now nearly equal


def test_contrast_enhancement_increases_variation() -> None:
    image = _gradient_image()
    before = float(image.std())
    after = float(enhance_contrast(image).std())
    assert after > before


def test_denoise_reduces_pixel_noise() -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    smoothed = denoise(image)
    assert smoothed.shape == image.shape
    assert smoothed.dtype == np.uint8


def test_individual_steps_can_be_selected() -> None:
    image = _gradient_image()
    only_color = enhance(image, EnhancementConfig(enabled=True, color_correction=True, contrast=False, denoise=False))
    assert not np.array_equal(only_color, image)

    passthrough = enhance(
        image,
        EnhancementConfig(enabled=True, color_correction=False, contrast=False, denoise=False),
    )
    assert np.array_equal(passthrough, image)


def test_from_flags_constructor() -> None:
    config = EnhancementConfig.from_flags(enabled=True, denoise=True)
    assert config.enabled and config.denoise and config.color_correction and config.contrast
