"""Run several models on one image and report their (honest) differences.

This is the backend for the Model Comparison page. It runs the *existing*
:class:`~app.detection.detector.MarineDetector` once per selected model with a
fixed confidence threshold and records exactly what the application would
produce - no inference logic is duplicated here.

The class sets of BlueEye's models are **not** the same, so the helper
:func:`class_coverage` reports the shared classes and each model's unique
classes. A comparison never pretends two models have equivalent coverage.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.detection.results import Detection


@dataclass
class ModelRun:
    """One model's result on one image."""

    key: str
    display_name: str
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    class_names: tuple[str, ...] = ()
    error: str | None = None

    @property
    def total_detections(self) -> int:
        return len(self.detections)

    @property
    def detected_classes(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    @property
    def average_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return sum(det.confidence for det in self.detections) / len(self.detections)

    @property
    def max_confidence(self) -> float:
        return max((det.confidence for det in self.detections), default=0.0)


def run_comparison(
    detector,
    image: np.ndarray,
    model_keys: list[str],
    confidence: float | None = None,
) -> list[ModelRun]:
    """Run each model key on ``image`` and capture its result or error.

    One model failing (missing weights, inference error) never aborts the
    comparison: the failure is recorded on that model's :class:`ModelRun` and
    the others still run.
    """
    runs: list[ModelRun] = []
    for key in model_keys:
        try:
            spec = detector.model_manager.get_spec(key)
            display_name = spec.display_name
        except Exception:  # noqa: BLE001 - unknown key must not crash the page
            display_name = key

        try:
            started = time.perf_counter()
            result = detector.predict_image(
                image,
                model_selection=key,
                confidence=confidence,
                source_name=key,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            names = tuple(detector.model_manager.get_class_names(key).values())
            runs.append(
                ModelRun(
                    key=key,
                    display_name=display_name,
                    detections=list(result.detections),
                    inference_ms=elapsed_ms,
                    class_names=names,
                )
            )
        except Exception as exc:  # noqa: BLE001 - surfaced on the result object
            runs.append(
                ModelRun(key=key, display_name=display_name, error=str(exc))
            )
    return runs


def class_coverage(runs: list[ModelRun]) -> dict:
    """Compare the class sets of the runs.

    Returns ``{"shared": [...], "unique": {key: [...]}, "counts": {key: n},
    "all_equal": bool}``. ``unique`` lists the classes a model can emit that
    no other selected model shares, which makes differing coverage explicit.
    """
    sets = {run.key: set(run.class_names) for run in runs if run.class_names}
    if sets:
        shared = set.intersection(*sets.values())
    else:
        shared = set()
    unique = {
        key: sorted(values - shared)
        for key, values in sets.items()
    }
    counts = {key: len(values) for key, values in sets.items()}
    return {
        "shared": sorted(shared),
        "unique": unique,
        "counts": counts,
        "all_equal": bool(sets) and all(values == shared for values in sets.values()),
    }


__all__ = ["ModelRun", "class_coverage", "run_comparison"]
