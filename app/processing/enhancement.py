"""Optional underwater image enhancement.

The BlueEye report identifies low lighting, colour distortion, noise and
blur as the main underwater image problems, so enhancement is provided as a
**modular, opt-in** preprocessing step:

    Input -> colour correction -> contrast enhancement -> denoising -> YOLO

It is disabled by default: the upstream models were trained on raw imagery,
so enhancement is *not* claimed to improve accuracy. Toggle it in the UI (or
pass ``--enhance`` on the CLI) and compare results on your own data.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EnhancementConfig:
    """Configuration for the underwater enhancement pipeline.

    Attributes
    ----------
    enabled:
        Master switch. When ``False``, :func:`enhance` returns the input
        unchanged.
    color_correction:
        Grey-world white balance (per-channel mean matching) to counteract
        the blue/green cast of underwater photography.
    contrast:
        CLAHE on the lightness channel of a LAB conversion, which lifts
        contrast without shifting colours strongly.
    denoise:
        Edge-preserving denoising. Off by default because it is the most
        expensive step.
    """

    enabled: bool = False
    color_correction: bool = True
    contrast: bool = True
    denoise: bool = False

    @classmethod
    def from_flags(
        cls,
        enabled: bool = False,
        color_correction: bool = True,
        contrast: bool = True,
        denoise: bool = False,
    ) -> "EnhancementConfig":
        return cls(
            enabled=bool(enabled),
            color_correction=bool(color_correction),
            contrast=bool(contrast),
            denoise=bool(denoise),
        )


def correct_color(image: np.ndarray) -> np.ndarray:
    """Grey-world white balance: scale each channel towards a common mean."""
    if image is None or image.size == 0:
        return image
    means = image.reshape(-1, 3).mean(axis=0)
    global_mean = float(means.mean())
    if global_mean < 1e-6:
        return image
    # Generous clip range: strong red/blue deficits are normal underwater.
    scales = np.clip(global_mean / np.maximum(means, 1e-6), 0.1, 5.0)
    corrected = image.astype(np.float32) * scales
    return np.clip(corrected, 0, 255).astype(np.uint8)


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast enhancement with CLAHE on the LAB lightness channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab[:, :, 0] = clahe.apply(lightness)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(image: np.ndarray, h: float = 5.0) -> np.ndarray:
    """Mild edge-preserving denoising."""
    return cv2.bilateralFilter(image, d=7, sigmaColor=h * 10, sigmaSpace=h * 10)


def enhance(image: np.ndarray, config: EnhancementConfig | None = None) -> np.ndarray:
    """Run the configured enhancement pipeline.

    Returns a new array (the input is never modified in place). When
    ``config`` is ``None`` or disabled, the input is returned as-is.
    """
    if config is None or not config.enabled:
        return image

    result = image
    if config.color_correction:
        result = correct_color(result)
    if config.contrast:
        result = enhance_contrast(result)
    if config.denoise:
        result = denoise(result)
    return result
