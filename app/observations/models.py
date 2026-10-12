"""Data types for persistent BlueEye observation records.

An *analysis* is one saved detection run (one uploaded image or video). It
owns zero or more *detections* - the individual objects the model found in
that run. A *correction* records a user's label change separately from the
original model prediction, which is never mutated. A *collection* groups
saved analyses into a named survey / research project.

Nothing here invents values: fields the pipeline does not expose (for example
a model version) are stored as ``None`` and rendered as "unavailable".

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: Allowed review states for a saved analysis.
#:
#: ``predicted``    - AI predicted, not reviewed yet
#: ``needs_review`` - the owner flagged it for a closer look
#: ``corrected``    - the owner changed at least one label
#: ``verified``     - confirmed by an authorised reviewer
REVIEW_STATUSES: tuple[str, ...] = ("predicted", "needs_review", "corrected", "verified")

#: Human label + badge kind for each review state (display only).
REVIEW_STATUS_LABELS: dict[str, tuple[str, str]] = {
    "predicted": ("AI predicted - not reviewed", "info"),
    "needs_review": ("Needs review", "warn"),
    "corrected": ("Corrected by user", "warn"),
    "verified": ("Verified identification", "ok"),
}


class ObservationError(RuntimeError):
    """Base error for observation storage problems."""


class DuplicateObservationError(ObservationError):
    """Raised when the same saved analysis would be stored twice."""


class NotOwnerError(ObservationError):
    """Raised when an operation targets a record the account does not own."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_timestamp(value: str) -> str:
    """Render a stored ISO timestamp for people, without inventing a value.

    Returns ``"unavailable"`` for an empty value and echoes unparseable text
    unchanged, so a record can never show a made-up date or time.
    """
    if not value:
        return "unavailable"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return str(value)
    return parsed.strftime("%d %b %Y, %H:%M")


@dataclass
class Correction:
    """One user label correction, kept separate from the original prediction."""

    id: str
    detection_id: str
    analysis_id: str
    reviewer_id: str
    original_label: str
    corrected_label: str
    note: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "correction_id": self.id,
            "detection_id": self.detection_id,
            "original_label": self.original_label,
            "corrected_label": self.corrected_label,
            "note": self.note,
            "reviewer_id": self.reviewer_id,
            "created_at": self.created_at,
        }


@dataclass
class DetectionRecord:
    """One object detected within a saved analysis (the original prediction)."""

    id: str
    analysis_id: str
    user_id: str
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int] | None = None
    model: str = ""
    frame_index: int | None = None
    frame_time: float | None = None
    created_at: str = ""
    corrections: list[Correction] = field(default_factory=list)

    @property
    def latest_correction(self) -> Correction | None:
        """The most recent correction, if the label was ever changed."""
        if not self.corrections:
            return None
        return max(self.corrections, key=lambda item: item.created_at or "")

    @property
    def effective_label(self) -> str:
        """The corrected label when one exists, else the original prediction."""
        latest = self.latest_correction
        return latest.corrected_label if latest else self.class_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox) if self.bbox else None,
            "model": self.model,
            "frame_index": self.frame_index,
            "frame_time": self.frame_time,
            "is_corrected": bool(self.corrections),
            "corrections": [item.to_dict() for item in self.corrections],
        }


@dataclass
class Analysis:
    """A saved detection analysis (one image or video run)."""

    id: str
    user_id: str
    kind: str  # "image" | "video"
    source_filename: str
    original_path: str
    annotated_path: str | None = None
    report_path: str | None = None
    models: list[str] = field(default_factory=list)
    confidence_threshold: float | None = None
    enhanced: bool = False
    total_detections: int = 0
    unique_classes: int = 0
    average_confidence: float = 0.0
    max_confidence: float = 0.0
    frames_processed: int | None = None
    elapsed_seconds: float | None = None
    model_version: str | None = None
    detected_at: str = ""
    created_at: str = ""
    updated_at: str = ""
    notes: str = ""
    project: str = ""
    location: str = ""
    habitat: str = ""
    review_status: str = "predicted"
    detections: list[DetectionRecord] = field(default_factory=list)
    collection_ids: list[str] = field(default_factory=list)
    collection_names: list[str] = field(default_factory=list)

    @property
    def status_label(self) -> str:
        return REVIEW_STATUS_LABELS.get(self.review_status, ("Unknown", "info"))[0]

    @property
    def status_kind(self) -> str:
        return REVIEW_STATUS_LABELS.get(self.review_status, ("Unknown", "info"))[1]

    @property
    def display_time(self) -> str:
        """Human-readable detected timestamp (``08 Oct 2026, 19:42``)."""
        return format_timestamp(self.detected_at)

    @property
    def display_created(self) -> str:
        return format_timestamp(self.created_at)

    def to_dict(self, *, include_detections: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "analysis_id": self.id,
            "kind": self.kind,
            "source_filename": self.source_filename,
            "detected_at": self.detected_at,
            "saved_at": self.created_at,
            "updated_at": self.updated_at,
            "models": list(self.models),
            "confidence_threshold": self.confidence_threshold,
            "enhanced": self.enhanced,
            "model_version": self.model_version or "unavailable",
            "project": self.project,
            "notes": self.notes,
            "location": self.location,
            "habitat": self.habitat,
            "review_status": self.review_status,
            "collections": list(self.collection_names),
            "statistics": {
                "total_detections": self.total_detections,
                "unique_classes": self.unique_classes,
                "average_confidence": round(self.average_confidence, 4),
                "max_confidence": round(self.max_confidence, 4),
                "frames_processed": self.frames_processed,
                "elapsed_seconds": self.elapsed_seconds,
            },
        }
        if self.detections:
            payload["detections"] = [det.to_dict() for det in self.detections]
        return payload


@dataclass
class Collection:
    """A named survey / research project grouping saved analyses."""

    id: str
    user_id: str
    name: str
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    item_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "item_count": self.item_count,
        }


__all__ = [
    "REVIEW_STATUSES",
    "REVIEW_STATUS_LABELS",
    "Analysis",
    "Collection",
    "Correction",
    "DetectionRecord",
    "DuplicateObservationError",
    "NotOwnerError",
    "ObservationError",
    "format_timestamp",
    "_utcnow",
]
