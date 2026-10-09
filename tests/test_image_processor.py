"""Unit tests for the image pipeline (model mocked)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.processing.enhancement import EnhancementConfig
from app.processing.image_processor import ImageProcessor
from tests.conftest import MockDetector


def test_process_saves_annotated_image_and_report(settings, sample_image: Path) -> None:
    detector = MockDetector()
    processor = ImageProcessor(detector, settings=settings)
    result = processor.process(sample_image)

    # Output artefacts exist in the configured directories.
    assert result.output_path.exists()
    assert result.output_path.parent == settings.images_output_dir
    assert result.report_path is not None and result.report_path.exists()
    assert result.report_path.parent == settings.reports_output_dir

    # The annotated image differs from the input (boxes were drawn).
    original = cv2.imread(str(sample_image))
    annotated = cv2.imread(str(result.output_path))
    assert annotated is not None and annotated.shape == original.shape
    assert not np.array_equal(original, annotated)

    # The JSON report matches the schema and the actual detections.
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["input"] == sample_image.name
    assert report["output_image"] == result.output_path.name
    assert report["statistics"]["total_detections"] == len(result.detection)
    assert len(report["detections"]) == len(result.detection)


def test_process_applies_confidence_threshold(settings, sample_image: Path) -> None:
    processor = ImageProcessor(MockDetector(), settings=settings)

    strict = processor.process(sample_image, confidence=0.5, save_json=False)
    assert len(strict.detection) == 2  # the 0.30 detection is filtered out

    permissive = processor.process(sample_image, confidence=0.2, save_json=False)
    assert len(permissive.detection) == 3


def test_process_without_json_report(settings, sample_image: Path) -> None:
    result = ImageProcessor(MockDetector(), settings=settings).process(
        sample_image, save_json=False
    )
    assert result.report_path is None
    assert result.output_path.exists()


def test_process_with_enhancement_flags_result(settings, sample_image: Path) -> None:
    processor = ImageProcessor(MockDetector(), settings=settings)
    result = processor.process(
        sample_image,
        enhancement=EnhancementConfig(enabled=True),
        save_json=True,
    )
    assert result.enhanced is True
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["enhanced"] is True
    assert result.detection.enhanced is True


def test_process_without_enhancement(settings, sample_image: Path) -> None:
    result = ImageProcessor(MockDetector(), settings=settings).process(
        sample_image, enhancement=EnhancementConfig(enabled=False)
    )
    assert result.enhanced is False


def test_process_rejects_corrupt_image(settings, corrupt_image: Path) -> None:
    from app.utils.validation import ValidationError

    processor = ImageProcessor(MockDetector(), settings=settings)
    try:
        processor.process(corrupt_image)
    except ValidationError:
        pass
    else:  # pragma: no cover - guard
        raise AssertionError("ValidationError expected")


def test_output_names_are_unique(settings, sample_image: Path) -> None:
    processor = ImageProcessor(MockDetector(), settings=settings)
    first = processor.process(sample_image, save_json=False)
    second = processor.process(sample_image, save_json=False)
    assert first.output_path != second.output_path


def test_custom_output_dir(settings, sample_image: Path, tmp_path: Path) -> None:
    custom = tmp_path / "elsewhere"
    result = ImageProcessor(MockDetector(), settings=settings).process(
        sample_image, output_dir=custom, save_json=False
    )
    assert result.output_path.parent == custom.resolve() or result.output_path.parent == custom


def test_statistics_reflect_detections(settings, sample_image: Path) -> None:
    result = ImageProcessor(MockDetector(), settings=settings).process(sample_image, confidence=0.5)
    stats = result.stats
    assert stats["total_detections"] == 2
    assert stats["unique_classes"] == 2
    assert stats["detections_per_class"] == {"fish": 1, "shark": 1}
    assert stats["max_confidence"] >= stats["average_confidence"] > 0
