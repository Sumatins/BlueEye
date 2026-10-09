"""Bounding-box rendering shared by the image and video processors.

Adapted from the annotation approach of *marine-detect* by Orange Business
Services SA (AGPL-3.0-only) - https://github.com/Orange-OpenSource/marine-detect

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import zlib
from typing import Iterable, Sequence

import cv2
import numpy as np

from app.detection.results import Detection, display_class_name

#: Distinct BGR colours used for detection boxes (cycled per class name).
PALETTE: tuple[tuple[int, int, int], ...] = (
    (66, 133, 244),   # blue
    (52, 168, 83),    # green
    (251, 188, 5),    # amber
    (234, 67, 53),    # red
    (171, 71, 188),   # purple
    (0, 172, 193),    # cyan
    (255, 145, 0),    # orange
    (240, 240, 240),  # white
)

WHITE = (255, 255, 255)


def class_color(class_name: str) -> tuple[int, int, int]:
    """Stable colour for a class name (same class -> same colour)."""
    index = zlib.crc32(str(class_name).encode("utf-8")) % len(PALETTE)
    return PALETTE[index]


def _scales(height: int, width: int) -> tuple[int, float]:
    """Thickness and font scale that stay readable at any resolution."""
    reference = max(min(height, width), 64)
    thickness = max(1, round(reference / 320))
    font_scale = max(0.4, reference / 700.0)
    return thickness, font_scale


def draw_detections(
    image: np.ndarray,
    detections: Iterable[Detection] | Sequence[Detection],
    font_scale: float | None = None,
) -> np.ndarray:
    """Draw labelled bounding boxes on a copy of ``image``.

    Labels show the friendly class name and the confidence percentage,
    e.g. ``Shark 92.4%``.
    """
    annotated = image.copy()
    if not detections:
        return annotated

    height, width = annotated.shape[:2]
    thickness, auto_scale = _scales(height, width)
    scale = font_scale if font_scale is not None else auto_scale

    for det in detections:
        colour = class_color(det.class_name)
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)

        label = f"{display_class_name(det.class_name)} {det.confidence * 100:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, thickness - 1)
        )
        # Keep the label inside the frame.
        label_x = max(0, min(x1, width - text_w - 4))
        label_y = y1 - 4
        if label_y - text_h - baseline < 0:
            label_y = min(y1 + text_h + baseline + 4, height - baseline - 1)

        cv2.rectangle(
            annotated,
            (label_x, label_y - text_h - baseline - 4),
            (label_x + text_w + 4, label_y + baseline - 2),
            colour,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (label_x + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            WHITE,
            max(1, thickness - 1),
            lineType=cv2.LINE_AA,
        )
    return annotated
