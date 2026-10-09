"""Shared pytest fixtures and test doubles for BlueEye.

The tests never touch the network and never require a GPU. Model inference
is mocked in unit tests; integration tests that use the real weights are
skipped automatically when the weights are not downloaded.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

from app.config import Settings, reset_settings_cache
from app.detection.results import Detection, DetectionResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Settings with isolated temporary directories
# --------------------------------------------------------------------------- #
@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    reset_settings_cache()
    custom = Settings(
        models_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        outputs_dir=tmp_path / "outputs",
    )
    custom.ensure_dirs()
    yield custom
    reset_settings_cache()


# --------------------------------------------------------------------------- #
# Sample media
# --------------------------------------------------------------------------- #
@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    """A small synthetic PNG (a coloured rectangle and a circle)."""
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(image, (60, 60), (180, 160), (0, 200, 255), -1)
    cv2.circle(image, (250, 180), 40, (255, 100, 50), -1)
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), image)
    return path


@pytest.fixture()
def corrupt_image(tmp_path: Path) -> Path:
    """A file with an image extension but invalid content."""
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"this is definitely not an image")
    return path


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    """A tiny synthetic MP4: 8 frames of 160x120 at 10 FPS."""
    path = tmp_path / "clip.mp4"
    width, height, fps, frames = 160, 120, 10.0, 8
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter could not be opened in this environment")
    for index in range(frames):
        frame = np.full((height, width, 3), (30, 60, 90), dtype=np.uint8)
        cv2.rectangle(frame, (10 + 8 * index, 30), (70 + 8 * index, 90), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("Could not create the synthetic video in this environment")
    return path


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class MockDetector:
    """Detector stand-in returning canned detections.

    Behaves like :class:`app.detection.detector.MarineDetector` for the
    methods used by the processors: it applies the confidence threshold
    exactly like the real implementation so filtering can be asserted.
    """

    def __init__(self, predefined: Optional[list[Detection]] = None) -> None:
        self.predefined = predefined
        self.calls: list[dict] = []
        self.models_used = ["mock_model"]

    def _canned(self, image: np.ndarray, confidence: float, source: str, enhanced: bool):
        height, width = image.shape[:2]
        threshold = 0.5 if confidence is None else float(confidence)
        if self.predefined is None:
            detections = [
                Detection(class_name="shark", confidence=0.92, bbox=(10, 10, 100, 90), model="mock_model"),
                Detection(class_name="fish", confidence=0.71, bbox=(120, 40, 200, 120), model="mock_model"),
                Detection(class_name="fish", confidence=0.30, bbox=(5, 150, 60, 220), model="mock_model"),
            ]
        else:
            detections = list(self.predefined)
        kept = [det for det in detections if det.confidence >= threshold]
        return DetectionResult(
            source=source,
            detections=kept,
            models_used=list(self.models_used),
            confidence_threshold=confidence,
            thresholds={key: threshold for key in self.models_used},
            inference_time_ms=1.0,
            image_width=width,
            image_height=height,
            enhanced=enhanced,
        )

    def predict_image(
        self,
        image: np.ndarray,
        model_selection: str | None = None,
        confidence: float | None = None,
        source_name: str = "image",
        enhanced: bool = False,
    ) -> DetectionResult:
        self.calls.append(
            {
                "source": source_name,
                "model_selection": model_selection,
                "confidence": confidence,
                "enhanced": enhanced,
            }
        )
        return self._canned(image, confidence, source_name, enhanced)

    def predict_frame(
        self,
        frame: np.ndarray,
        model_selection: str | None = None,
        confidence: float | None = None,
        frame_index: int | None = None,
        enhanced: bool = False,
    ) -> DetectionResult:
        name = f"frame {frame_index}" if frame_index is not None else "frame"
        self.calls.append(
            {
                "source": name,
                "model_selection": model_selection,
                "confidence": confidence,
                "enhanced": enhanced,
                "frame_index": frame_index,
            }
        )
        return self._canned(frame, confidence, name, enhanced)

    def load_model(self, model_selection: str | None = None) -> list[str]:
        return list(self.models_used)


class FakeBox:
    def __init__(self, cls: int, conf: float, xyxy) -> None:
        self.cls = [cls]
        self.conf = [conf]
        self.xyxy = [list(xyxy)]


class FakeResult:
    def __init__(self, boxes: list[FakeBox], names: dict[int, str]) -> None:
        self.boxes = boxes
        self.names = names


class FakeYOLO:
    """Mimics the slice of the Ultralytics YOLO API used by the detector."""

    def __init__(self, boxes: list[FakeBox], names: dict[int, str]) -> None:
        self._boxes = boxes
        self._names = names
        self.calls: list[dict] = []

    def predict(self, image, conf=0.25, device="cpu", verbose=False, **kwargs):
        self.calls.append({"conf": conf, "device": device, "shape": getattr(image, "shape", None)})
        return [FakeResult(self._boxes, self._names)]


class FakeModelManager:
    """ModelManager double: no disk access, no network, no Ultralytics."""

    def __init__(self, available: list[str], boxes_by_key: dict[str, list[FakeBox]] | None = None):
        from app.detection.model_manager import MODEL_REGISTRY

        self._registry = MODEL_REGISTRY
        self._available = set(available)
        self._boxes = boxes_by_key or {}
        self.loaded: list[str] = []
        self._models: dict[str, FakeYOLO] = {}

    # --- registry API ------------------------------------------------- #
    def get_spec(self, key: str):
        return self._registry[key]

    def resolve_path(self, key: str):
        return Path("fake") / self._registry[key].filename

    def is_available(self, key: str) -> bool:
        return key in self._available

    def available_models(self):
        return [self._registry[key] for key in sorted(self._available)]

    def missing_models(self):
        return [spec for spec in self._registry.values() if spec.key not in self._available]

    def load(self, key: str):
        self.loaded.append(key)
        if key not in self._models:
            names = {idx: name for idx, name in enumerate(self._fake_names(key))}
            self._models[key] = FakeYOLO(self._boxes.get(key, []), names)
        return self._models[key]

    def get_class_names(self, key: str):
        return {idx: name for idx, name in enumerate(self._fake_names(key))}

    @staticmethod
    def _fake_names(key: str) -> list[str]:
        if key == "megafauna":
            return ["ray", "shark", "turtle"]
        return ["fish", "scaridae", "urchin"]

    @property
    def device(self) -> str:
        return "cpu"
