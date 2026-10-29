"""Detection package: model management and inference.

SPDX-License-Identifier: AGPL-3.0-only
"""

from app.detection.bootstrap import bootstrap_keys, ensure_weights, start_background_bootstrap
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
    "bootstrap_keys",
    "ensure_weights",
    "start_background_bootstrap",
]
