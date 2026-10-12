"""Object-detection metrics: IoU, AP, mAP@0.5 and mAP@0.5:0.95.

This is a small, dependency-light NumPy implementation used by the BlueEye
evaluation pipeline. The matching recipe is the standard COCO one:

* predictions are ordered by descending confidence;
* each prediction is greedily matched to the best still-unmatched
  ground-truth box of the same class when ``IoU >= threshold``;
* average precision is the area under the interpolated
  precision-recall curve.

Metrics are *computed* from real predictions and real annotations. With no
ground truth, or with a class that has no predictions, the value is zero and
an explicit note is attached rather than an invented number.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

#: IoU thresholds for the COCO-style ``mAP@0.5:0.95`` (0.50 .. 0.95).
IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * index, 2) for index in range(10))


@dataclass(frozen=True)
class DetectionBox:
    """One predicted or ground-truth box, in pixel coordinates."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class ClassMetrics:
    """Precision / recall / AP for a single class."""

    class_id: int
    name: str
    num_gt: int
    num_pred: int
    ap50: float
    ap50_95: float
    precision: float
    recall: float


@dataclass(frozen=True)
class EvaluationResult:
    """The complete metric block for one model on one dataset split."""

    map50: float
    map50_95: float
    iou_thresholds: tuple[float, ...]
    classes: tuple[ClassMetrics, ...]
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        """JSON-serialisable view."""
        return {
            "mAP50": round(self.map50, 6),
            "mAP50_95": round(self.map50_95, 6),
            "iou_thresholds": list(self.iou_thresholds),
            "classes": [
                {
                    "class_id": item.class_id,
                    "name": item.name,
                    "gt": item.num_gt,
                    "predictions": item.num_pred,
                    "precision": round(item.precision, 6),
                    "recall": round(item.recall, 6),
                    "AP50": round(item.ap50, 6),
                    "AP50_95": round(item.ap50_95, 6),
                }
                for item in self.classes
            ],
            "notes": list(self.notes),
        }


def box_iou(box_a: DetectionBox, box_b: DetectionBox) -> float:
    """Intersection-over-union of two axis-aligned boxes (0.0 .. 1.0)."""
    ix1 = max(box_a.x1, box_b.x1)
    iy1 = max(box_a.y1, box_b.y1)
    ix2 = min(box_a.x2, box_b.x2)
    iy2 = min(box_a.y2, box_b.y2)
    inter_width = max(0.0, ix2 - ix1)
    inter_height = max(0.0, iy2 - iy1)
    intersection = inter_width * inter_height
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, box_a.x2 - box_a.x1) * max(0.0, box_a.y2 - box_a.y1)
    area_b = max(0.0, box_b.x2 - box_b.x1) * max(0.0, box_b.y2 - box_b.y1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _pr_curve(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> tuple[np.ndarray, np.ndarray]:
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / float(n_gt)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    return recall, precision


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the (monotonically enveloped) precision-recall curve."""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def _best_f1(recall: np.ndarray, precision: np.ndarray) -> tuple[float, float]:
    """Operating (precision, recall) at the maximum-F1 point."""
    if recall.size == 0:
        return 0.0, 0.0
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-9)
    index = int(np.argmax(f1))
    return float(precision[index]), float(recall[index])


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _match_threshold(
    preds: Sequence[tuple[str, DetectionBox]],
    gt_by_image: Mapping[str, list[DetectionBox]],
    iou_threshold: float,
) -> tuple[list[int], list[int]]:
    """Greedy match for one IoU threshold; returns (tp_flags, fp_flags)."""
    matched = {image: [False] * len(boxes) for image, boxes in gt_by_image.items()}
    tp: list[int] = []
    fp: list[int] = []
    for image, box in preds:
        boxes = gt_by_image.get(image, [])
        flags = matched.setdefault(image, [False] * len(boxes))
        best_iou = 0.0
        best_index = -1
        for index, gt_box in enumerate(boxes):
            value = box_iou(box, gt_box)
            if value > best_iou:
                best_iou = value
                best_index = index
        if best_index >= 0 and not flags[best_index] and best_iou >= iou_threshold:
            flags[best_index] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    return tp, fp


def _evaluate_class(
    class_id: int,
    name: str,
    ground_truth: Mapping[str, list[DetectionBox]],
    predictions: Mapping[str, list[DetectionBox]],
    iou_thresholds: Sequence[float],
) -> ClassMetrics:
    gt_by_image = {
        image: [box for box in boxes if box.class_id == class_id]
        for image, boxes in ground_truth.items()
    }
    num_gt = sum(len(boxes) for boxes in gt_by_image.values())

    preds: list[tuple[str, DetectionBox]] = [
        (image, box)
        for image, boxes in predictions.items()
        for box in boxes
        if box.class_id == class_id
    ]
    preds.sort(key=lambda item: item[1].score, reverse=True)
    num_pred = len(preds)

    ap_by_threshold: list[float] = []
    precision = 0.0
    recall = 0.0
    for threshold in iou_thresholds:
        tp, fp = _match_threshold(preds, gt_by_image, threshold)
        if num_gt <= 0:
            ap_by_threshold.append(0.0)
            continue
        curve_recall, curve_precision = _pr_curve(np.array(tp), np.array(fp), num_gt)
        ap_by_threshold.append(_average_precision(curve_recall, curve_precision))
        if abs(threshold - 0.5) < 1e-9:
            precision, recall = _best_f1(curve_recall, curve_precision)

    ap50 = ap_by_threshold[0] if ap_by_threshold else 0.0
    return ClassMetrics(
        class_id=class_id,
        name=name,
        num_gt=num_gt,
        num_pred=num_pred,
        ap50=ap50,
        ap50_95=_mean(ap_by_threshold),
        precision=precision,
        recall=recall,
    )


def evaluate(
    ground_truth: Mapping[str, list[DetectionBox]],
    predictions: Mapping[str, list[DetectionBox]],
    *,
    class_names: Mapping[int, str] | None = None,
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
) -> EvaluationResult:
    """Compute precision / recall / mAP for one model on one split.

    Parameters
    ----------
    ground_truth, predictions:
        ``{image_key: [DetectionBox, ...]}``. Ground truth carries the
        dataset's annotation boxes; predictions carry the model's output
        mapped onto the dataset classes (see :mod:`app.evaluation.compat`).
    class_names:
        Optional ``{class_id: name}`` used for the per-class report.
    """
    names = dict(class_names or {})
    class_ids = sorted(
        {box.class_id for boxes in ground_truth.values() for box in boxes}
        | {box.class_id for boxes in predictions.values() for box in boxes}
    )

    notes: list[str] = []
    total_gt = sum(len(boxes) for boxes in ground_truth.values())
    total_pred = sum(len(boxes) for boxes in predictions.values())
    if total_gt == 0:
        notes.append(
            "No ground-truth boxes were provided, so AP is undefined and reported as 0."
        )
    if total_pred == 0:
        notes.append("The model produced no predictions above the confidence threshold.")

    classes = tuple(
        _evaluate_class(class_id, names.get(class_id, str(class_id)), ground_truth, predictions, iou_thresholds)
        for class_id in class_ids
    )
    return EvaluationResult(
        map50=_mean([item.ap50 for item in classes]),
        map50_95=_mean([item.ap50_95 for item in classes]),
        iou_thresholds=tuple(float(threshold) for threshold in iou_thresholds),
        classes=classes,
        notes=tuple(notes),
    )

