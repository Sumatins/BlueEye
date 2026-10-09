"""Unit tests for detection results, statistics and report serialisation."""

from __future__ import annotations

import json

from app.detection.results import (
    Detection,
    DetectionResult,
    compute_statistics,
    display_class_name,
)


def _det(name: str, conf: float, bbox=(0, 0, 10, 10), model: str = "fish_inv") -> Detection:
    return Detection(class_name=name, confidence=conf, bbox=bbox, model=model)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def test_detection_from_xyxy_rounds_and_converts() -> None:
    det = Detection.from_xyxy((10.4, 20.6, 30.7, 40.49), "shark", 0.91, model="megafauna")
    assert det.bbox == (10, 21, 31, 40)
    assert det.confidence == 0.91
    assert det.width == 21 and det.height == 19


def test_detection_to_dict_schema() -> None:
    det = _det("shark", 0.924, (100, 120, 500, 400), "megafauna")
    payload = det.to_dict()
    assert payload["class_name"] == "shark"
    assert payload["confidence"] == 0.924
    assert payload["bounding_box"] == {"x1": 100, "y1": 120, "x2": 500, "y2": 400}
    assert payload["model"] == "megafauna"


def test_detection_to_report_dict_uses_list_bbox() -> None:
    det = _det("fish", 0.5, (1, 2, 3, 4))
    payload = det.to_report_dict()
    assert payload == {"class": "fish", "confidence": 0.5, "bbox": [1, 2, 3, 4], "model": "fish_inv"}


def test_confidence_percent_formatting() -> None:
    assert _det("shark", 0.924).confidence_percent == "92.4%"


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_stats_are_computed_correctly() -> None:
    result = DetectionResult(
        source="img.jpg",
        detections=[
            _det("shark", 0.90),
            _det("fish", 0.85),
            _det("fish", 0.80),
            _det("ray", 0.70),
        ],
    )
    stats = result.stats()
    assert stats["total_detections"] == 4
    assert stats["unique_classes"] == 3
    assert stats["detections_per_class"] == {"fish": 2, "ray": 1, "shark": 1}
    assert abs(stats["average_confidence"] - 0.8125) < 1e-9
    assert stats["max_confidence"] == 0.90


def test_stats_on_empty_result_are_zero() -> None:
    stats = DetectionResult(source="img.jpg").stats()
    assert stats["total_detections"] == 0
    assert stats["unique_classes"] == 0
    assert stats["average_confidence"] == 0.0
    assert stats["max_confidence"] == 0.0
    assert stats["detections_per_class"] == {}


def test_compute_statistics_directly() -> None:
    stats = compute_statistics([0.5, 1.0], {"turtle": 2})
    assert stats["total_detections"] == 2
    assert stats["unique_classes"] == 1
    assert stats["average_confidence"] == 0.75
    assert stats["max_confidence"] == 1.0


def test_sorted_detections_orders_by_confidence() -> None:
    result = DetectionResult(
        source="x",
        detections=[_det("a", 0.4), _det("b", 0.9), _det("c", 0.6)],
    )
    assert [d.class_name for d in result.sorted_detections()] == ["b", "c", "a"]


# --------------------------------------------------------------------------- #
# Result / report
# --------------------------------------------------------------------------- #
def test_detection_result_len_iter_empty() -> None:
    result = DetectionResult(source="x", detections=[_det("shark", 0.9)])
    assert len(result) == 1
    assert not result.is_empty
    assert [d.class_name for d in result] == ["shark"]
    assert DetectionResult(source="x").is_empty


def test_report_is_json_serialisable_and_has_required_keys() -> None:
    result = DetectionResult(
        source="sample.jpg",
        detections=[_det("shark", 0.924, (10, 20, 30, 40))],
        models_used=["megafauna"],
        confidence_threshold=0.5,
        thresholds={"megafauna": 0.5},
        inference_time_ms=123.4,
        image_width=640,
        image_height=480,
    )
    report = result.to_report()
    # Round-trip through JSON proves serialisability.
    decoded = json.loads(json.dumps(report))
    for key in ("input", "timestamp", "confidence_threshold", "detections", "statistics"):
        assert key in decoded
    assert decoded["input"] == "sample.jpg"
    assert decoded["models"] == ["megafauna"]
    assert decoded["detections"][0]["bbox"] == [10, 20, 30, 40]
    assert decoded["statistics"]["total_detections"] == 1


# --------------------------------------------------------------------------- #
# Display names
# --------------------------------------------------------------------------- #
def test_display_name_for_known_classes() -> None:
    assert display_class_name("shark") == "Shark"
    assert display_class_name("scaridae") == "Parrotfish (Scaridae)"
    assert display_class_name("SEA_CUCUMBER") == "Sea cucumber"


def test_display_name_falls_back_for_unknown_classes() -> None:
    # Unknown ids are only formatted, never invented.
    assert display_class_name("new_species_v2") == "New species v2"
