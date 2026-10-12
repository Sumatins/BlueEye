"""Tests for the model-comparison logic and its export helper.

The comparison orchestrator only needs a small detector-shaped object, so the
tests run with a stub - no weights, no network, no GPU. They verify that each
model is run on the same image, that a failing model never aborts the others,
and that differing class coverage is reported honestly.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.detection.comparison import ModelRun, class_coverage, run_comparison
from app.detection.results import Detection
from app.ui.pages.compare import comparison_json


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _Spec:
    def __init__(self, key: str, display_name: str) -> None:
        self.key = key
        self.display_name = display_name


class _ModelManager:
    def __init__(self, specs, classes) -> None:
        self._specs = specs
        self._classes = classes

    def get_spec(self, key: str) -> _Spec:
        return self._specs[key]

    def get_class_names(self, key: str) -> dict[int, str]:
        return {index: name for index, name in enumerate(self._classes.get(key, []))}


class _Result:
    def __init__(self, detections) -> None:
        self.detections = detections


class _Detector:
    def __init__(self, specs, classes, detections, fail=()) -> None:
        self.model_manager = _ModelManager(specs, classes)
        self._detections = detections
        self._fail = set(fail)
        self.calls: list[str] = []

    def predict_image(self, image, model_selection=None, confidence=None, source_name=None):
        self.calls.append(model_selection)
        if model_selection not in self.model_manager._specs:
            raise KeyError(f"unknown model: {model_selection}")
        if model_selection in self._fail:
            raise RuntimeError("weights exploded")
        return _Result(list(self._detections.get(model_selection, [])))


def _det(name: str, conf: float, model: str = "") -> Detection:
    return Detection(class_name=name, confidence=conf, bbox=(0, 0, 10, 10), model=model)


def _detector() -> _Detector:
    specs = {"fish_inv": _Spec("fish_inv", "Fish & Invertebrates"),
             "megafauna": _Spec("megafauna", "MegaFauna")}
    classes = {
        "fish_inv": ["fish", "scaridae", "turtle"],
        "megafauna": ["shark", "turtle", "ray"],
    }
    detections = {
        "fish_inv": [_det("scaridae", 0.9), _det("scaridae", 0.6), _det("turtle", 0.8)],
        "megafauna": [_det("turtle", 0.7)],
    }
    return _Detector(specs, classes, detections)


IMAGE = np.zeros((32, 32, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# run_comparison
# --------------------------------------------------------------------------- #
def test_run_comparison_runs_every_model_on_the_same_image() -> None:
    detector = _detector()
    runs = run_comparison(detector, IMAGE, ["fish_inv", "megafauna"], confidence=0.5)

    assert detector.calls == ["fish_inv", "megafauna"]
    assert [run.key for run in runs] == ["fish_inv", "megafauna"]
    assert all(isinstance(run, ModelRun) for run in runs)
    assert all(run.error is None for run in runs)


def test_run_comparison_reports_real_counts_and_confidences() -> None:
    fish = run_comparison(_detector(), IMAGE, ["fish_inv"], confidence=0.5)[0]
    assert fish.total_detections == 3
    assert fish.detected_classes == {"scaridae": 2, "turtle": 1}
    assert fish.max_confidence == pytest.approx(0.9)
    assert fish.average_confidence == pytest.approx((0.9 + 0.6 + 0.8) / 3)
    assert fish.inference_ms >= 0.0
    assert set(fish.class_names) == {"fish", "scaridae", "turtle"}


def test_a_failing_model_does_not_abort_the_others() -> None:
    specs = {"good": _Spec("good", "Good"), "bad": _Spec("bad", "Broken")}
    classes = {"good": ["fish"], "bad": ["fish"]}
    detections = {"good": [_det("fish", 0.7)]}
    detector = _Detector(specs, classes, detections, fail={"bad"})

    runs = run_comparison(detector, IMAGE, ["bad", "good"], confidence=0.5)

    assert runs[0].key == "bad"
    assert runs[0].error and "weights exploded" in runs[0].error
    assert runs[0].total_detections == 0
    # The healthy model still produced its real result.
    assert runs[1].error is None
    assert runs[1].total_detections == 1


def test_run_comparison_handles_unknown_model_key() -> None:
    runs = run_comparison(_detector(), IMAGE, ["does_not_exist"], confidence=0.5)
    assert len(runs) == 1
    assert runs[0].error is not None
    assert runs[0].display_name == "does_not_exist"


# --------------------------------------------------------------------------- #
# class_coverage
# --------------------------------------------------------------------------- #
def test_class_coverage_separates_shared_and_unique_classes() -> None:
    runs = run_comparison(_detector(), IMAGE, ["fish_inv", "megafauna"], confidence=0.5)
    coverage = class_coverage(runs)

    assert coverage["shared"] == ["turtle"]
    assert coverage["unique"]["fish_inv"] == ["fish", "scaridae"]
    assert coverage["unique"]["megafauna"] == ["ray", "shark"]
    assert coverage["all_equal"] is False


def test_class_coverage_reports_equal_sets() -> None:
    specs = {"a": _Spec("a", "A"), "b": _Spec("b", "B")}
    classes = {"a": ["fish"], "b": ["fish"]}
    detector = _Detector(specs, classes, {})
    coverage = class_coverage(run_comparison(detector, IMAGE, ["a", "b"]))
    assert coverage["all_equal"] is True
    assert coverage["shared"] == ["fish"]


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def test_comparison_json_excludes_inline_images() -> None:
    result = {
        "input": "reef.jpg",
        "confidence": 0.5,
        "created": "2026-10-12T10:00:00",
        "original": b"\x00\x01",
        "annotated": {"fish_inv": b"\x02\x03"},
        "coverage": {"shared": [], "unique": {}, "all_equal": True},
        "runs": [{"key": "fish_inv", "total_detections": 0}],
    }
    payload = json.loads(comparison_json(result))
    assert "original" not in payload
    assert "annotated" not in payload
    assert payload["input"] == "reef.jpg"
    assert payload["runs"][0]["key"] == "fish_inv"
