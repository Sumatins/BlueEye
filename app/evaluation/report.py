"""Write evaluation results as JSON, CSV and a readable Markdown summary.

Everything BlueEye writes here is *reproduced locally* from an actual run.
The Markdown summary states the exact conditions (device, package versions,
dataset, split) so a reader can never mistake a local measurement for a
number copied from a paper. Reference values published by a model's authors
are a separate, clearly-labelled category and are never mixed in.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from app.evaluation.runner import ModelEvaluation, summarise

METRICS_HEADER = (
    "model",
    "dataset",
    "split",
    "status",
    "images",
    "objects",
    "mAP50",
    "mAP50_95",
    "precision",
    "recall",
    "seconds_per_image",
    "comparable_classes",
    "unsupported_dataset_classes",
)

PER_CLASS_HEADER = (
    "model",
    "class",
    "gt",
    "predictions",
    "precision",
    "recall",
    "AP50",
    "AP50_95",
)


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def _metrics_rows(evaluations: Sequence[ModelEvaluation]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in evaluations:
        headline = summarise(item)
        rows.append(
            [
                item.model,
                item.dataset,
                item.split,
                item.status,
                str(item.images_evaluated),
                str(headline["objects"]),
                f"{headline['mAP50']:.6f}",
                f"{headline['mAP50_95']:.6f}",
                f"{headline['precision']:.6f}",
                f"{headline['recall']:.6f}",
                f"{headline['seconds_per_image']:.4f}",
                str(len(item.compat.supported)),
                "; ".join(item.compat.unsupported),
            ]
        )
    return rows


def _per_class_rows(evaluations: Sequence[ModelEvaluation]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in evaluations:
        for metric in item.metrics.classes:
            rows.append(
                [
                    item.model,
                    metric.name,
                    str(metric.num_gt),
                    str(metric.num_pred),
                    f"{metric.precision:.6f}",
                    f"{metric.recall:.6f}",
                    f"{metric.ap50:.6f}",
                    f"{metric.ap50_95:.6f}",
                ]
            )
    return rows


def write_report(
    output_dir: str | Path,
    evaluations: Sequence[ModelEvaluation],
    environment: dict | None = None,
    config: dict | None = None,
) -> dict[str, Path]:
    """Write ``results.json``, ``metrics.csv``, ``per_class.csv`` and ``summary.md``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now().isoformat(timespec="seconds")

    payload = {
        "created": created,
        "config": config or {},
        "environment": environment or {},
        "evaluations": [item.as_dict() for item in evaluations],
        "summary": [summarise(item) for item in evaluations],
    }
    written: dict[str, Path] = {}
    written["results_json"] = out_dir / "results.json"
    written["results_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    written["environment_json"] = out_dir / "environment.json"
    written["environment_json"].write_text(
        json.dumps(environment or {}, indent=2), encoding="utf-8"
    )

    written["metrics_csv"] = out_dir / "metrics.csv"
    _write_csv(written["metrics_csv"], METRICS_HEADER, _metrics_rows(evaluations))

    written["per_class_csv"] = out_dir / "per_class.csv"
    _write_csv(written["per_class_csv"], PER_CLASS_HEADER, _per_class_rows(evaluations))

    written["summary_md"] = out_dir / "summary.md"
    written["summary_md"].write_text(
        render_summary(
            evaluations, environment=environment, config=config, created=created
        ),
        encoding="utf-8",
    )
    return written


def render_summary(
    evaluations: Sequence[ModelEvaluation],
    environment: dict | None = None,
    config: dict | None = None,
    created: str | None = None,
) -> str:
    """Render the human-readable Markdown summary."""
    environment = environment or {}
    lines: list[str] = []
    lines.append("# BlueEye evaluation summary")
    lines.append("")
    lines.append(f"Generated: {created or datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(
        "All numbers below were **reproduced locally** by BlueEye on the dataset "
        "and split named in each row. They are not copied from a paper and must "
        "not be compared across different datasets as if they were equivalent."
    )
    lines.append("")

    if environment:
        hw = environment.get("hardware", {})
        packages = environment.get("packages", {})
        lines.append("## Test conditions")
        lines.append("")
        lines.append(f"- Platform: `{environment.get('platform', 'unknown')}`")
        lines.append(f"- Python: `{environment.get('python', 'unknown')}`")
        lines.append(f"- Device: `{hw.get('gpu') or 'CPU'}`")
        lines.append(f"- CUDA available: `{hw.get('cuda_available', False)}`")
        for name, version in packages.items():
            lines.append(f"- {name}: `{version}`")
        lines.append("")

    lines.append("## Model comparison")
    lines.append("")
    lines.append(
        "| Model | Dataset | Split | Status | Images | mAP50 | mAP50:95 | "
        "Precision | Recall | s/image |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for item in evaluations:
        headline = summarise(item)
        lines.append(
            f"| {item.model} | {item.dataset} | {item.split} | {item.status} | "
            f"{item.images_evaluated} | {headline['mAP50']:.3f} | "
            f"{headline['mAP50_95']:.3f} | {headline['precision']:.3f} | "
            f"{headline['recall']:.3f} | {headline['seconds_per_image']:.3f} |"
        )
    lines.append("")

    for item in evaluations:
        lines.append(f"## {item.model} on {item.dataset} ({item.split})")
        lines.append("")
        lines.append(f"- Status: **{item.status}**")
        lines.append(
            f"- Comparable classes: {', '.join(item.compat.supported) or 'none'}"
        )
        if item.compat.unsupported:
            lines.append(
                "- Dataset classes this model cannot detect (excluded from "
                f"per-class scoring): {', '.join(item.compat.unsupported)}"
            )
        if item.compat.extra_model:
            lines.append(
                f"- Model classes absent from the dataset (ignored): "
                f"{', '.join(item.compat.extra_model)}"
            )
        lines.append("")
        if item.metrics.classes:
            lines.append("| Class | GT | Pred | Precision | Recall | AP50 | AP50:95 |")
            lines.append("|---|---|---|---|---|---|---|")
            for metric in item.metrics.classes:
                lines.append(
                    f"| {metric.name} | {metric.num_gt} | {metric.num_pred} | "
                    f"{metric.precision:.3f} | {metric.recall:.3f} | "
                    f"{metric.ap50:.3f} | {metric.ap50_95:.3f} |"
                )
            lines.append("")
        for note in item.notes:
            lines.append(f"> {note}")
        lines.append("")

    if config:
        lines.append("## Configuration")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(config, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)
