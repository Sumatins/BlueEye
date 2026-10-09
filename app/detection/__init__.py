"""Detection package: model management and inference.

SPDX-License-Identifier: AGPL-3.0-only
"""

from app.detection.detector import MarineDetector
from app.detection.model_manager import (
    ModelDownloadError,
    ModelError,
    ModelManager,
    ModelNotAvailableError,
    ModelSpec,
)
from app.detection.results import Detection, DetectionResult

__all__ = [
    "Detection",
    "DetectionResult",
    "MarineDetector",
    "ModelDownloadError",
    "ModelError",
    "ModelManager",
    "ModelNotAvailableError",
    "ModelSpec",
]
