"""Side-by-side comparison of saved observation records.

This compares what the **model predicted** across several saved analyses. It
is a study aid for reviewing saved detections - it is explicitly *not* a claim
about biodiversity change and never implies that two images show the same
animal. The wording returned by :data:`COMPARISON_NOTE` is included in every
export so the distinction travels with the data.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Sequence

from app.detection.results import display_class_name
from app.observations.models import Analysis

#: Plain-language note attached to every comparison.
COMPARISON_NOTE = (
    "This table compares model *predictions* across separately saved images or "
    "videos. Differences do not establish biodiversity change, and identical "
    "labels do not mean the images show the same individual animal. Species-"
    "level names appear only where a model supports that class."
)

COMPARISON_COLUMNS = (
    "analysis_id",
    "source_filename",
    "detected_at",
    "kind",
    "project",
    "review_status",
    "models",
    "total_detections",
    "unique_classes",
    "average_confidence",
    "max_confidence",
    "labels",
)


def label_counts(analysis: Analysis) -> dict[str, int]:
    """Counts of each *effective* label in one analysis (corrections applied)."""
    counter: Counter[str] = Counter()
    for det in analysis.detections:
        counter[det.effective_label] += 1
    return {label: count for label, count in counter.most_common()}


def _row(analysis: Analysis) -> dict[str, Any]:
    counts = label_counts(analysis)
    return {
        "analysis_id": analysis.id,
        "source_filename": analysis.source_filename,
        "detected_at": analysis.detected_at,
        "kind": analysis.kind,
        "project": analysis.project,
        "review_status": analysis.review_status,
        "models": list(analysis.models),
        "total_detections": analysis.total_detections,
        "unique_classes": analysis.unique_classes,
        "average_confidence": round(analysis.average_confidence, 4),
        "max_confidence": round(analysis.max_confidence, 4),
        "labels": counts,
    }


def compare_analyses(analyses: Sequence[Analysis]) -> dict[str, Any]:
    """Structure for comparing several saved observations.

    ``label_matrix`` maps every label seen anywhere to a per-analysis count, so
    the UI can render a compact matrix without recomputing anything. ``labels``
    is the ordered union of every label present across the selection.
    """
    rows = [_row(item) for item in analyses]
    labels: list[str] = []
    for row in rows:
        for label in row["labels"]:
            if label not in labels:
                labels.append(label)
    matrix: dict[str, dict[str, int]] = {
        label: {row["analysis_id"]: row["labels"].get(label, 0) for row in rows}
        for label in labels
    }
    shared = [label for label in labels if sum(matrix[label].values()) == len(rows)]
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": COMPARISON_NOTE,
        "count": len(rows),
        "observations": rows,
        "labels": labels,
        "label_matrix": matrix,
        "labels_in_every_observation": shared,
    }


def comparison_rows(analyses: Sequence[Analysis]) -> list[dict[str, Any]]:
    """Flat display rows (label counts rendered with friendly class names)."""
    rows = []
    for row in (_row(item) for item in analyses):
        rows.append(
            {
                "source": row["source_filename"],
                "detected_at": row["detected_at"],
                "kind": row["kind"],
                "project": row["project"] or "unassigned",
                "review_status": row["review_status"],
                "objects": row["total_detections"],
                "classes": row["unique_classes"],
                "average_confidence": f"{row['average_confidence'] * 100:.1f}%",
                "max_confidence": f"{row['max_confidence'] * 100:.1f}%",
                "labels": ", ".join(
                    f"{display_class_name(label)} ×{count}"
                    for label, count in row["labels"].items()
                )
                or "none",
            }
        )
    return rows


def comparison_csv(analyses: Sequence[Analysis]) -> str:
    """CSV of the comparison table (one row per saved observation)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(COMPARISON_COLUMNS)
    for row in (_row(item) for item in analyses):
        writer.writerow(
            [
                row["analysis_id"],
                row["source_filename"],
                row["detected_at"],
                row["kind"],
                row["project"],
                row["review_status"],
                "|".join(row["models"]),
                row["total_detections"],
                row["unique_classes"],
                f"{row['average_confidence']:.4f}",
                f"{row['max_confidence']:.4f}",
                "|".join(f"{label}:{count}" for label, count in row["labels"].items()),
            ]
        )
    return buffer.getvalue()


__all__ = [
    "COMPARISON_COLUMNS",
    "COMPARISON_NOTE",
    "compare_analyses",
    "comparison_csv",
    "comparison_rows",
    "label_counts",
]
