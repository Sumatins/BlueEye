"""Unit tests for the MarineDetector inference wrapper (model mocked)."""

from __future__ import annotations

import numpy as np
import pytest

from app.detection.detector import MarineDetector
from app.detection.model_manager import ModelNotAvailableError
from tests.conftest import FakeBox, FakeModelManager


def _detector(available, boxes_by_key=None) -> MarineDetector:
    manager = FakeModelManager(available, boxes_by_key)
    return MarineDetector(model_manager=manager)


def _image(width: int = 320, height: int = 240) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
def test_auto_runs_every_available_model() -> None:
    detector = _detector(["fish_inv", "megafauna"])
    assert detector.resolve_keys("auto") == ["fish_inv", "megafauna"]
    assert detector.resolve_keys("both") == ["fish_inv", "megafauna"]


def test_auto_runs_single_model_when_only_one_present() -> None:
    detector = _detector(["megafauna"])
    assert detector.resolve_keys("auto") == ["megafauna"]


def test_explicit_selections_and_aliases() -> None:
    detector = _detector(["fish_inv", "megafauna"])
    assert detector.resolve_keys("fish_inv") == ["fish_inv"]
    assert detector.resolve_keys("fish") == ["fish_inv"]
    assert detector.resolve_keys("megafauna") == ["megafauna"]
    assert detector.resolve_keys("mega") == ["megafauna"]


def test_auto_without_any_weights_raises_friendly_error() -> None:
    detector = _detector([])
    with pytest.raises(ModelNotAvailableError, match="download_models"):
        detector.resolve_keys("auto")


def test_missing_selected_model_raises() -> None:
    detector = _detector(["fish_inv"])
    with pytest.raises(ModelNotAvailableError, match="not available"):
        detector.resolve_keys("megafauna")


def test_unknown_selection_raises() -> None:
    detector = _detector(["fish_inv"])
    with pytest.raises(ModelNotAvailableError, match="Unknown model selection"):
        detector.resolve_keys("dolphin")


def test_load_model_preloads_weights() -> None:
    detector = _detector(["fish_inv", "megafauna"])
    keys = detector.load_model("auto")
    assert keys == ["fish_inv", "megafauna"]
    assert detector.model_manager.loaded == ["fish_inv", "megafauna"]


def test_get_class_names_per_model() -> None:
    detector = _detector(["fish_inv", "megafauna"])
    names = detector.get_class_names("auto")
    assert "shark" in names["megafauna"].values()
    assert "fish" in names["fish_inv"].values()


# --------------------------------------------------------------------------- #
# Inference behaviour
# --------------------------------------------------------------------------- #
def test_confidence_filtering_applied() -> None:
    boxes = [
        FakeBox(cls=1, conf=0.9, xyxy=(10, 10, 100, 90)),
        FakeBox(cls=0, conf=0.4, xyxy=(120, 40, 200, 120)),
    ]
    detector = _detector(["fish_inv"], {"fish_inv": boxes})

    strict = detector.predict_image(_image(), confidence=0.5)
    assert len(strict) == 1
    assert strict.detections[0].confidence == 0.9
    assert strict.confidence_threshold == 0.5
    assert strict.thresholds == {"fish_inv": 0.5}

    permissive = detector.predict_image(_image(), confidence=0.2)
    assert len(permissive) == 2


def test_default_threshold_is_model_recommended_value() -> None:
    boxes = [FakeBox(cls=0, conf=0.53, xyxy=(1, 1, 50, 50))]
    detector = _detector(["fish_inv"], {"fish_inv": boxes})
    result = detector.predict_image(_image(), confidence=None)
    assert result.confidence_threshold is None
    assert result.thresholds == {"fish_inv": pytest.approx(0.523)}
    assert len(result) == 1  # 0.53 >= 0.523


def test_inference_receives_resolved_device() -> None:
    detector = _detector(["fish_inv"], {"fish_inv": []})
    detector.predict_image(_image())
    call = detector.model_manager.load("fish_inv").calls[-1]
    assert call["device"] == "cpu"


def test_boxes_are_clamped_to_frame_bounds() -> None:
    boxes = [FakeBox(cls=0, conf=0.99, xyxy=(-50, -10, 9999, 9999))]
    detector = _detector(["fish_inv"], {"fish_inv": boxes})
    result = detector.predict_image(_image(320, 240))
    x1, y1, x2, y2 = result.detections[0].bbox
    assert (x1, y1, x2, y2) == (0, 0, 319, 239)


def test_degenerate_boxes_are_dropped() -> None:
    boxes = [FakeBox(cls=0, conf=0.9, xyxy=(100, 100, 100.2, 100.4))]
    detector = _detector(["fish_inv"], {"fish_inv": boxes})
    assert detector.predict_image(_image()).is_empty


def test_result_metadata_is_populated() -> None:
    detector = _detector(["fish_inv"], {"fish_inv": []})
    result = detector.predict_image(_image(640, 480), source_name="reef.jpg")
    assert result.source == "reef.jpg"
    assert result.models_used == ["fish_inv"]
    assert (result.image_width, result.image_height) == (640, 480)
    assert result.inference_time_ms >= 0


def test_predict_frame_records_frame_index() -> None:
    detector = _detector(["fish_inv"], {"fish_inv": []})
    result = detector.predict_frame(_image(), frame_index=42)
    assert result.source == "frame 42"


def test_invalid_confidence_rejected() -> None:
    detector = _detector(["fish_inv"])
    with pytest.raises(ValueError, match="between"):
        detector.predict_image(_image(), confidence=1.5)
    with pytest.raises(ValueError, match="between"):
        detector.predict_image(_image(), confidence=-0.1)


def test_invalid_image_rejected() -> None:
    detector = _detector(["fish_inv"])
    with pytest.raises(ValueError, match="BGR numpy array"):
        detector.predict_image(None)
    with pytest.raises(ValueError, match="BGR numpy array"):
        detector.predict_image(np.zeros((10, 10), dtype=np.uint8))


def test_model_status_lists_registry(settings) -> None:
    detector = _detector(["fish_inv"])
    status = detector.model_status()
    assert len(status) == 2
    by_key = {entry["key"]: entry for entry in status}
    assert by_key["fish_inv"]["available"] is True
    assert by_key["megafauna"]["available"] is False
    assert by_key["megafauna"]["classes"] == []
    assert by_key["fish_inv"]["classes"] == ["fish", "scaridae", "urchin"]
