"""Run a detector over a dataset split and compute detection metrics.

The evaluation core is intentionally independent of the detection stack: a
*predictor* is any callable ``path -> [(class_name, confidence, xyxy), ...]``.
That makes the metric logic unit-testable with a stub predictor (no weights,
no network) while the CLI plugs in the real
:class:`~app.detection.detector.MarineDetector` through
:func:`detector_predictor`.

The pretrained weights are only ever read, never modified. If the model and
the dataset share no class name the run is still recorded, but it is marked
*not comparable* rather than scored.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from app.evaluation.compat import ClassCompat, _normalise, compare_classes
from app.evaluation.datasets import DatasetSplit
from app.evaluation.metrics import DetectionBox, EvaluationResult, evaluate

#: ``(class_name, confidence, (x1, y1, x2, y2))`` in pixel coordinates.
Prediction = tuple[str, float, tuple[float, float, float, float]]
Predictor = Callable[[Path], list[Prediction]]
#: ``(stage, images_done, images_total)`` progress notification for one model.
#: Called from the worker thread with *genuine* counters only; it may be called
#: often, so a handler should throttle any expensive work (e.g. a DB write).
ProgressHook = Callable[[str, int, int], None]


class InferenceError(RuntimeError):
    """Raised when a predictor fails on an image."""


@dataclass
class ModelEvaluation:
    """The full outcome of evaluating one model on one dataset split."""

    model: str
    dataset: str
    split: str
    model_classes: tuple[str, ...]
    compat: ClassCompat
    metrics: EvaluationResult
    num_images: int
    images_evaluated: int
    num_predictions: int
    inference_seconds: float
    status: str = "complete"
    notes: tuple[str, ...] = ()
    #: Predictions in dataset-class space, kept for qualitative figures only;
    #: deliberately excluded from :meth:`as_dict` (JSON stays compact).
    predictions: dict[str, list[DetectionBox]] = field(
        default_factory=dict, repr=False, compare=False
    )
    #: Ground truth in the same space, kept for curve/confusion figures only.
    ground_truth: dict[str, list[DetectionBox]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def seconds_per_image(self) -> float:
        if self.images_evaluated <= 0:
            return 0.0
        return self.inference_seconds / self.images_evaluated

    @property
    def comparable_classes(self) -> tuple[str, ...]:
        return self.compat.supported

    def as_dict(self) -> dict:
        """JSON-serialisable view of the whole evaluation."""
        return {
            "model": self.model,
            "dataset": self.dataset,
            "split": self.split,
            "status": self.status,
            "model_classes": list(self.model_classes),
            "comparison": {
                "supported": list(self.compat.supported),
                "unsupported_dataset_classes": list(self.compat.unsupported),
                "extra_model_classes": list(self.compat.extra_model),
                "comparable": self.compat.comparable,
            },
            "metrics": self.metrics.as_dict(),
            "timing": {
                "images": self.images_evaluated,
                "total_images": self.num_images,
                "predictions": self.num_predictions,
                "inference_seconds": round(self.inference_seconds, 4),
                "seconds_per_image": round(self.seconds_per_image, 4),
            },
            "notes": list(self.notes),
        }


def _aggregate(metrics: EvaluationResult) -> tuple[float, float]:
    """Mean per-class precision / recall over classes with ground truth."""
    scored = [item for item in metrics.classes if item.num_gt > 0]
    if not scored:
        return 0.0, 0.0
    precision = sum(item.precision for item in scored) / len(scored)
    recall = sum(item.recall for item in scored) / len(scored)
    return precision, recall


def run_evaluation(
    model_name: str,
    predictor: Predictor,
    dataset: DatasetSplit,
    model_classes: Sequence[str],
    *,
    on_progress: ProgressHook | None = None,
) -> ModelEvaluation:
    """Evaluate ``predictor`` on ``dataset`` and return the metrics.

    Only class names present in both the model and the dataset are scored;
    predictions of unknown classes are dropped (an unknown name can never be
    mapped to a dataset label by guessing).

    ``on_progress``, when given, is called as ``(stage, images_done,
    images_total)`` with real counters: ``"inference"`` before the loop, after
    every image, and ``"metric calculation"`` once the loop has finished and the
    metrics are about to be computed. It is never called with invented values.
    """
    compat = compare_classes(dataset.class_names, model_classes)
    name_to_id = {_normalise(name): index for index, name in enumerate(dataset.class_names)}

    predictions: dict[str, list[DetectionBox]] = {}
    num_predictions = 0
    evaluated = 0
    total_images = dataset.num_images
    if on_progress is not None:
        on_progress("inference", 0, total_images)
    started = time.perf_counter()
    for image_path in dataset.images:
        try:
            raw = predictor(image_path)
        except Exception as exc:  # noqa: BLE001 - surfaced as InferenceError
            raise InferenceError(
                f"Prediction failed on {image_path.name}: {exc}"
            ) from exc
        boxes: list[DetectionBox] = []
        for class_name, confidence, xyxy in raw:
            class_id = name_to_id.get(_normalise(class_name))
            if class_id is None:
                continue
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            boxes.append(
                DetectionBox(
                    class_id=class_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=float(confidence),
                )
            )
        predictions[image_path.name] = boxes
        num_predictions += len(boxes)
        evaluated += 1
        if on_progress is not None:
            on_progress("inference", evaluated, total_images)
    elapsed = time.perf_counter() - started

    if on_progress is not None:
        on_progress("metric calculation", evaluated, total_images)
    metrics = evaluate(
        dataset.ground_truth,
        predictions,
        class_names={index: name for index, name in enumerate(dataset.class_names)},
    )

    status = "complete" if compat.comparable else "not_comparable"
    notes: list[str] = list(metrics.notes)
    if not compat.comparable:
        notes.append(
            "No class names are shared between the model and the dataset; "
            "the metrics below are not a valid comparison."
        )

    return ModelEvaluation(
        model=model_name,
        dataset=dataset.name,
        split=dataset.split,
        model_classes=tuple(str(name) for name in model_classes),
        compat=compat,
        metrics=metrics,
        num_images=dataset.num_images,
        images_evaluated=evaluated,
        num_predictions=num_predictions,
        inference_seconds=elapsed,
        status=status,
        notes=tuple(notes),
        predictions=predictions,
        ground_truth=dict(dataset.ground_truth),
    )


def summarise(evaluation: ModelEvaluation) -> dict:
    """Convenience view: headline metrics plus timing for one evaluation."""
    precision, recall = _aggregate(evaluation.metrics)
    return {
        "model": evaluation.model,
        "dataset": evaluation.dataset,
        "split": evaluation.split,
        "status": evaluation.status,
        "images": evaluation.images_evaluated,
        "objects": sum(item.num_gt for item in evaluation.metrics.classes),
        "mAP50": evaluation.metrics.map50,
        "mAP50_95": evaluation.metrics.map50_95,
        "precision": precision,
        "recall": recall,
        "seconds_per_image": evaluation.seconds_per_image,
    }


def detector_predictor(
    detector,
    model_selection: str | None = None,
    confidence: float | None = None,
) -> Predictor:
    """Adapt a :class:`~app.detection.detector.MarineDetector` to a predictor.

    The detector's own post-processing (per-model thresholds, clamping,
    duplicate suppression) is used unchanged, so the evaluation measures the
    exact pipeline the application runs.
    """
    import cv2
    import numpy as np

    def _predict(image_path: Path) -> list[Prediction]:
        image = cv2.imread(str(image_path))
        if image is None or not isinstance(image, np.ndarray):
            raise InferenceError(f"Could not read image: {image_path}")
        result = detector.predict_image(
            image,
            model_selection=model_selection,
            confidence=confidence,
            source_name=image_path.name,
        )
        return [
            (det.class_name, float(det.confidence), tuple(float(v) for v in det.bbox))
            for det in result.detections
        ]

    return _predict


def model_class_names(detector, model_selection: str | None = None) -> tuple[str, ...]:
    """All class names the selected model(s) can emit (union, order-preserving)."""
    names_by_model: dict[str, dict[int, str]] = detector.get_class_names(model_selection)
    ordered: list[str] = []
    for mapping in names_by_model.values():
        for _, name in sorted(mapping.items()):
            if name not in ordered:
                ordered.append(name)
    return tuple(ordered)


def run_many(
    models: Iterable[tuple[str, Predictor, Sequence[str]]],
    dataset: DatasetSplit,
) -> list[ModelEvaluation]:
    """Evaluate several models on the same split (fresh predictions each)."""
    return [
        run_evaluation(name, predictor, dataset, classes)
        for name, predictor, classes in models
    ]
