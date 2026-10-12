"""Optional matplotlib visualisations for evaluation results.

Plotting is *optional*: if matplotlib is not installed the functions return
``None`` and the caller records that the figure was skipped. Matplotlib is a
transitive dependency of Ultralytics, but BlueEye degrades gracefully rather
than making a hard requirement of it.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.evaluation.metrics import (
    ClassCurve,
    box_iou,
    class_curves,
    detection_confusion,
)
from app.evaluation.runner import ModelEvaluation, summarise


def _pyplot():
    """Import matplotlib lazily; return ``None`` when unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception:  # noqa: BLE001 - plotting is optional
        return None
    return plt


def plot_metric_comparison(
    evaluations: Sequence[ModelEvaluation],
    output_path: str | Path,
) -> Path | None:
    """Bar chart comparing mAP50 and mAP50:95 across models."""
    plt = _pyplot()
    if plt is None or not evaluations:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = [item.model for item in evaluations]
    map50 = [summarise(item)["mAP50"] for item in evaluations]
    map5095 = [summarise(item)["mAP50_95"] for item in evaluations]

    positions = range(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4.5))
    axis.bar([p - width / 2 for p in positions], map50, width, label="mAP@0.5")
    axis.bar([p + width / 2 for p in positions], map5095, width, label="mAP@0.5:0.95")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_ylabel("mAP")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Local BlueEye evaluation (same dataset/split per group)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_per_class(
    evaluation: ModelEvaluation,
    output_path: str | Path,
) -> Path | None:
    """Bar chart of AP50 per class for a single model evaluation."""
    plt = _pyplot()
    classes = [metric for metric in evaluation.metrics.classes if metric.num_gt > 0]
    if plt is None or not classes:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    names = [metric.name for metric in classes]
    ap50 = [metric.ap50 for metric in classes]
    positions = range(len(names))
    figure, axis = plt.subplots(figsize=(max(6, len(names) * 1.1), 4.5))
    axis.bar(list(positions), ap50, color="#2c7fb8")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(names, rotation=40, ha="right")
    axis.set_ylabel("AP@0.5")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(f"Per-class AP@0.5 - {evaluation.model} on {evaluation.dataset}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_all(
    evaluations: Sequence[ModelEvaluation],
    output_dir: str | Path,
) -> list[Path]:
    """Write the comparison chart plus one per-class chart per model."""
    out_dir = Path(output_dir)
    created: list[Path] = []
    comparison = plot_metric_comparison(evaluations, out_dir / "metric_comparison.png")
    if comparison is not None:
        created.append(comparison)
    for index, evaluation in enumerate(evaluations):
        safe = "".join(
            character if character.isalnum() else "_" for character in evaluation.model
        )
        figure = plot_per_class(
            evaluation, out_dir / f"per_class_{index:02d}_{safe}.png"
        )
        if figure is not None:
            created.append(figure)
    return created


def _class_names_for(evaluation: ModelEvaluation) -> dict[int, str]:
    """``{class_id: name}`` from whatever classes the evaluation touched."""
    names: dict[int, str] = {
        metric.class_id: metric.name for metric in evaluation.metrics.classes
    }
    for boxes in (*evaluation.ground_truth.values(), *evaluation.predictions.values()):
        for box in boxes:
            names.setdefault(box.class_id, str(box.class_id))
    return names


def plot_precision_recall(
    evaluation: ModelEvaluation,
    output_path: str | Path,
    *,
    iou_threshold: float = 0.5,
) -> Path | None:
    """Per-class precision-recall curves at one IoU threshold."""
    plt = _pyplot()
    names = _class_names_for(evaluation)
    curves: tuple[ClassCurve, ...] = class_curves(
        evaluation.ground_truth,
        evaluation.predictions,
        class_names=names,
        iou_threshold=iou_threshold,
    )
    if plt is None or not curves:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6.5, 5))
    for curve in curves:
        axis.plot(
            list(curve.recall),
            list(curve.precision),
            linewidth=1.8,
            label=f"{curve.name} (AP@0.5={curve.ap:.3f}, n={curve.num_gt})",
        )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_title(
        f"Precision-recall @ IoU {iou_threshold:g} - {evaluation.model}"
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="lower left")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_f1_confidence(
    evaluation: ModelEvaluation,
    output_path: str | Path,
    *,
    iou_threshold: float = 0.5,
) -> Path | None:
    """Per-class F1 as a function of the confidence threshold."""
    plt = _pyplot()
    names = _class_names_for(evaluation)
    curves: tuple[ClassCurve, ...] = class_curves(
        evaluation.ground_truth,
        evaluation.predictions,
        class_names=names,
        iou_threshold=iou_threshold,
    )
    if plt is None or not curves:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6.5, 5))
    for curve in curves:
        axis.plot(
            list(curve.confidence),
            list(curve.f1),
            linewidth=1.8,
            label=curve.name,
        )
    axis.set_xlabel("Confidence")
    axis.set_ylabel("F1")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_title(f"F1 vs confidence @ IoU {iou_threshold:g} - {evaluation.model}")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_confusion_matrix(
    evaluation: ModelEvaluation,
    output_path: str | Path,
    *,
    iou_threshold: float = 0.5,
) -> Path | None:
    """Detections confusion matrix (last row/column = background)."""
    plt = _pyplot()
    names = _class_names_for(evaluation)
    if plt is None or not names:
        return None
    num_classes = max(names) + 1
    matrix = detection_confusion(
        evaluation.ground_truth,
        evaluation.predictions,
        num_classes,
        iou_threshold=iou_threshold,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = [names.get(index, str(index)) for index in range(num_classes)] + [
        "background"
    ]
    figure, axis = plt.subplots(figsize=(max(6, num_classes * 0.9), max(5, num_classes * 0.8)))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("Ground truth")
    axis.set_ylabel("Predicted")
    axis.set_title(f"Detections confusion @ IoU {iou_threshold:g} - {evaluation.model}")
    normaliser = matrix.max() or 1.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            if value:
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > normaliser * 0.6 else "black",
                    fontsize=8,
                )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_evaluation_figures(
    evaluation: ModelEvaluation,
    output_dir: str | Path,
    *,
    iou_threshold: float = 0.5,
) -> list[Path]:
    """All supported per-model figures: per-class AP, PR, F1 and confusion."""
    out_dir = Path(output_dir)
    safe = "".join(
        character if character.isalnum() else "_" for character in evaluation.model
    )
    written: list[Path] = []
    for path in (
        plot_per_class(evaluation, out_dir / f"per_class_{safe}.png"),
        plot_precision_recall(
            evaluation, out_dir / f"pr_curve_{safe}.png", iou_threshold=iou_threshold
        ),
        plot_f1_confidence(
            evaluation, out_dir / f"f1_curve_{safe}.png", iou_threshold=iou_threshold
        ),
        plot_confusion_matrix(
            evaluation, out_dir / f"confusion_{safe}.png", iou_threshold=iou_threshold
        ),
    ):
        if path is not None:
            written.append(path)
    return written


def _match_stats(ground_truth, predictions):
    """Count (true positives, misses, false positives) for one image."""
    matched = [False] * len(ground_truth)
    true_positives = 0
    false_positives = 0
    for prediction in predictions:
        best = 0.0
        best_index = -1
        for index, target in enumerate(ground_truth):
            if target.class_id != prediction.class_id:
                continue
            value = box_iou(prediction, target)
            if value > best:
                best = value
                best_index = index
        if best_index >= 0 and best >= 0.5 and not matched[best_index]:
            matched[best_index] = True
            true_positives += 1
        else:
            false_positives += 1
    misses = sum(1 for flag in matched if not flag)
    return true_positives, misses, false_positives


def plot_examples(
    evaluation: ModelEvaluation,
    dataset,
    output_dir: str | Path,
    limit: int = 4,
) -> list[Path]:
    """Draw ground truth (green) and predictions (red) for a few images.

    A mix of successes, misses and false positives is chosen so the figures
    show both correct detections and failure cases.
    """
    plt = _pyplot()
    if plt is None or not evaluation.predictions:
        return []

    ranked: list[tuple[int, str]] = []
    for image_path in dataset.images:
        key = image_path.name
        ground_truth = dataset.ground_truth.get(key, [])
        predictions = evaluation.predictions.get(key, [])
        if not ground_truth and not predictions:
            continue
        true_positives, misses, _ = _match_stats(ground_truth, predictions)
        category = 0 if true_positives else (1 if misses else 2)
        ranked.append((category, key))
    ranked.sort()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from matplotlib.patches import Rectangle

    names = dataset.class_names
    written: list[Path] = []
    for index, (category, key) in enumerate(ranked[:limit]):
        image_path = next((p for p in dataset.images if p.name == key), None)
        if image_path is None:
            continue
        image = plt.imread(str(image_path))
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.imshow(image)
        axis.axis("off")
        for box in dataset.ground_truth.get(key, []):
            axis.add_patch(
                Rectangle(
                    (box.x1, box.y1),
                    box.x2 - box.x1,
                    box.y2 - box.y1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
            )
            label = names[box.class_id] if box.class_id < len(names) else str(box.class_id)
            axis.text(box.x1, max(0, box.y1 - 4), f"GT:{label}", color="lime", fontsize=8)
        for box in evaluation.predictions.get(key, []):
            axis.add_patch(
                Rectangle(
                    (box.x1, box.y1),
                    box.x2 - box.x1,
                    box.y2 - box.y1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
            )
            label = names[box.class_id] if box.class_id < len(names) else str(box.class_id)
            axis.text(box.x1, box.y1 + 10, f"{label} {box.score:.2f}", color="red", fontsize=8)
        outcome = {0: "success", 1: "miss", 2: "false positive"}[category]
        axis.set_title(f"{evaluation.model} - {key} ({outcome})")
        figure.tight_layout()
        path = out_dir / f"example_{index:02d}_{outcome.replace(' ', '_')}_{key}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        written.append(path)
    return written
