"""Build persistent observation records from real detection reports.

This module is deliberately inference-free: it takes the JSON report the
existing pipeline already wrote (plus the paths it produced) and turns it into
an :class:`~app.observations.models.Analysis` with its individual detections.
Any field the pipeline does not expose is stored as ``None`` - nothing is
invented.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.observations.models import Analysis, DetectionRecord, ObservationError

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bbox(raw: Any) -> tuple[int, int, int, int] | None:
    """Coerce a report bbox (list/tuple of four numbers) to int pixels."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in raw)
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2)


def _detection(
    payload: dict[str, Any],
    *,
    analysis_id: str,
    user_id: str,
    frame_index: int | None = None,
    frame_time: float | None = None,
) -> DetectionRecord:
    """One detection row from a report entry (original model prediction)."""
    class_name = str(payload.get("class_name") or payload.get("class") or "unknown")
    return DetectionRecord(
        id=uuid.uuid4().hex,
        analysis_id=analysis_id,
        user_id=user_id,
        class_name=class_name,
        confidence=_as_float(payload.get("confidence")),
        bbox=_bbox(payload.get("bbox") or payload.get("bounding_box")),
        model=str(payload.get("model") or ""),
        frame_index=frame_index,
        frame_time=frame_time,
        created_at=_utcnow(),
    )


def detections_from_report(
    kind: str, report: dict[str, Any], *, analysis_id: str, user_id: str
) -> list[DetectionRecord]:
    """Extract every detection the report recorded for this run.

    Images carry a flat ``detections`` list; videos carry ``frame_detections``
    (one entry per frame that had at least one object). Frame timestamps are
    derived from the video's real FPS when the report provides it.
    """
    if kind == "video":
        fps = _as_float((report.get("video") or {}).get("fps"))
        results: list[DetectionRecord] = []
        for frame in report.get("frame_detections") or []:
            if not isinstance(frame, dict):
                continue
            index = _as_int(frame.get("frame"))
            frame_time = (index / fps) if (index is not None and fps > 0) else None
            for payload in frame.get("detections") or []:
                if isinstance(payload, dict):
                    results.append(
                        _detection(
                            payload,
                            analysis_id=analysis_id,
                            user_id=user_id,
                            frame_index=index,
                            frame_time=frame_time,
                        )
                    )
        return results

    return [
        _detection(payload, analysis_id=analysis_id, user_id=user_id)
        for payload in (report.get("detections") or [])
        if isinstance(payload, dict)
    ]


def build_analysis(
    *,
    user_id: str,
    kind: str,
    report: dict[str, Any] | None,
    original_path: str | Path,
    annotated_path: str | Path | None = None,
    report_path: str | Path | None = None,
    source_filename: str | None = None,
    notes: str = "",
    project: str = "",
    location: str = "",
    habitat: str = "",
    review_status: str = "predicted",
) -> Analysis:
    """Assemble an :class:`Analysis` from a finished run.

    Raises :class:`ObservationError` when there is no report to persist (the
    pipeline always writes one; a missing report means there is nothing real
    to store).
    """
    if kind not in ("image", "video"):
        raise ObservationError(f"Unsupported analysis kind: {kind!r}")
    if not report:
        raise ObservationError("There is no detection report to save.")

    analysis_id = uuid.uuid4().hex
    detections = detections_from_report(
        kind, report, analysis_id=analysis_id, user_id=user_id
    )
    stats = report.get("statistics") or {}
    processing = report.get("processing") or {}

    total = _as_int(stats.get("total_detections"))
    unique = _as_int(stats.get("unique_classes"))
    average = stats.get("average_confidence")
    maximum = stats.get("max_confidence")

    if total is None:
        total = len(detections)
    if unique is None:
        unique = len({det.class_name for det in detections})
    if average is None:
        average = (
            sum(det.confidence for det in detections) / len(detections) if detections else 0.0
        )
    if maximum is None:
        maximum = max((det.confidence for det in detections), default=0.0)

    elapsed = _as_float(processing.get("elapsed_seconds"), default=0.0) or _as_float(
        report.get("processing_time_seconds"), default=0.0
    )
    frames = _as_int(processing.get("frames_processed"))
    source = source_filename or Path(str(report.get("input") or original_path)).name

    return Analysis(
        id=analysis_id,
        user_id=user_id,
        kind=kind,
        source_filename=source,
        original_path=str(original_path),
        annotated_path=str(annotated_path) if annotated_path else None,
        report_path=str(report_path) if report_path else None,
        models=[str(m) for m in (report.get("models") or [])],
        confidence_threshold=(
            None
            if report.get("confidence_threshold") is None
            else _as_float(report.get("confidence_threshold"))
        ),
        enhanced=bool(report.get("enhanced", False)),
        total_detections=int(total or 0),
        unique_classes=int(unique or 0),
        average_confidence=float(average or 0.0),
        max_confidence=float(maximum or 0.0),
        frames_processed=frames,
        elapsed_seconds=elapsed or None,
        # The pipeline does not expose a model version, so it is recorded as
        # unavailable rather than guessed.
        model_version=None,
        detected_at=str(report.get("timestamp") or ""),
        created_at=_utcnow(),
        updated_at=_utcnow(),
        notes=notes.strip(),
        project=project.strip(),
        location=location.strip(),
        habitat=habitat.strip(),
        review_status=review_status if review_status else "predicted",
        detections=detections,
    )


__all__ = ["build_analysis", "detections_from_report"]
