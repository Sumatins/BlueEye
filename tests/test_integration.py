"""Integration tests: real YOLOv8 weights on real media.

These tests run only when the official marine-detect weights have been
downloaded (``python scripts/download_models.py``); otherwise they are
skipped - weights are never fabricated for testing. CPU execution is used,
no GPU required.

Run them explicitly with::

    pytest -m integration

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import pytest

from app.config import PROJECT_ROOT, get_settings
from app.detection.detector import MarineDetector
from app.detection.model_manager import ModelManager
from app.processing.image_processor import ImageProcessor
from app.processing.video_processor import VideoProcessor

pytestmark = pytest.mark.integration

WEIGHTS_PRESENT = all(
    ModelManager().is_available(key) for key in ("fish_inv", "megafauna")
)
SAMPLE_IMAGE = PROJECT_ROOT / "assets" / "images" / "input_folder" / "regq.jpg"


@pytest.fixture()
def real_settings(tmp_path: Path):
    """Real model weights, but outputs confined to a temp directory."""
    base = get_settings()
    return replace(base, outputs_dir=tmp_path / "outputs", data_dir=tmp_path / "data")

skip_no_weights = pytest.mark.skipif(
    not WEIGHTS_PRESENT,
    reason="model weights not downloaded (run: python scripts/download_models.py)",
)
skip_no_sample = pytest.mark.skipif(
    not SAMPLE_IMAGE.is_file(),
    reason="sample image from marine-detect assets not found",
)


@skip_no_weights
@skip_no_sample
def test_real_image_detection_end_to_end(real_settings, tmp_path: Path) -> None:
    """Image -> YOLOv8 -> annotated image + JSON report."""
    detector = MarineDetector(settings=real_settings)
    processor = ImageProcessor(detector, settings=real_settings)
    result = processor.process(SAMPLE_IMAGE, output_dir=tmp_path)

    assert result.output_path.exists()
    assert result.detection.models_used  # at least one model ran
    assert result.detection.inference_time_ms > 0
    # The upstream sample image contains parrotfish (scaridae) - verified
    # manually against the published marine-detect example output.
    assert not result.detection.is_empty
    assert result.report_path is not None
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["statistics"]["total_detections"] == len(result.detection)
    for det in report["detections"]:
        assert 0.0 <= det["confidence"] <= 1.0
        assert len(det["bbox"]) == 4


@skip_no_weights
def test_real_video_detection_end_to_end(real_settings, tmp_path: Path) -> None:
    """Video -> per-frame YOLOv8 -> annotated video + statistics."""
    if not SAMPLE_IMAGE.is_file():
        pytest.skip("sample image not available to build the test video")

    # Build a short real video from the sample image (6 frames).
    image = cv2.imread(str(SAMPLE_IMAGE))
    assert image is not None
    height, width = image.shape[:2]
    scale = 480 / max(width, height)
    small = cv2.resize(image, (int(width * scale), int(height * scale)))
    video_path = tmp_path / "integration.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (small.shape[1], small.shape[0])
    )
    assert writer.isOpened()
    for _ in range(6):
        writer.write(small)
    writer.release()

    detector = MarineDetector(settings=real_settings)
    processor = VideoProcessor(detector, settings=real_settings)
    result = processor.process(video_path, output_path=tmp_path / "annotated.mp4")

    assert result.frames_processed == 6
    assert result.output_path.exists()
    assert result.total_detections >= 1
    assert result.statistics["unique_classes"] >= 1
    assert result.report_path is not None


@skip_no_weights
def test_real_model_class_names_are_complete() -> None:
    detector = MarineDetector()
    names = detector.get_class_names("auto")
    assert len(names["fish_inv"]) == 15
    assert sorted(names["megafauna"].values()) == ["ray", "shark", "turtle"]
