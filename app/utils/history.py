"""Persistent detection history for BlueEye.

Every completed detection can be appended here, giving the web UI a real
"History" and real "Analytics" page. Nothing is fabricated: entries are
built from the JSON reports that the detection pipeline already writes.

The store is a single JSON file (``outputs/history/detections.json`` by
default) written atomically, newest entry first, and capped so it can never
grow without bound.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Only these media kinds are accepted (guards against junk entries).
VALID_KINDS = ("image", "video")

#: Default cap so the file stays small and the UI stays fast.
DEFAULT_MAX_ENTRIES = 500


@dataclass
class HistoryEntry:
    """One completed detection run (image or video)."""

    timestamp: str
    kind: str  # "image" | "video"
    filename: str
    models: list[str] = field(default_factory=list)
    total_detections: int = 0
    unique_classes: int = 0
    average_confidence: float = 0.0
    max_confidence: float = 0.0
    confidence_threshold: float | None = None
    enhanced: bool = False
    output: str = ""
    report: str | None = None
    elapsed_seconds: float | None = None
    frames_processed: int | None = None
    detections_per_class: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_report(
        cls,
        kind: str,
        report: dict[str, Any],
        output: str,
        report_path: str | None = None,
    ) -> "HistoryEntry":
        """Build an entry from a pipeline JSON report payload.

        Works for both the image report and the video report because both
        share the ``statistics`` block written by
        :func:`app.detection.results.compute_statistics`.
        """
        if kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}, got {kind!r}")
        stats = report.get("statistics") or {}
        processing = report.get("processing") or {}
        return cls(
            timestamp=str(report.get("timestamp") or datetime.now().isoformat(timespec="seconds")),
            kind=kind,
            filename=str(report.get("input") or "unknown"),
            models=[str(m) for m in (report.get("models") or [])],
            total_detections=int(stats.get("total_detections", 0) or 0),
            unique_classes=int(stats.get("unique_classes", 0) or 0),
            average_confidence=float(stats.get("average_confidence", 0.0) or 0.0),
            max_confidence=float(stats.get("max_confidence", 0.0) or 0.0),
            confidence_threshold=report.get("confidence_threshold"),
            enhanced=bool(report.get("enhanced", False)),
            output=str(output or ""),
            report=str(report_path) if report_path else None,
            elapsed_seconds=_first_float(
                processing.get("elapsed_seconds"), report.get("processing_time_seconds")
            ),
            frames_processed=_as_int_or_none(processing.get("frames_processed")),
            detections_per_class={
                str(k): int(v) for k, v in (stats.get("detections_per_class") or {}).items()
            },
        )

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "kind": self.kind,
            "filename": self.filename,
            "models": list(self.models),
            "total_detections": self.total_detections,
            "unique_classes": self.unique_classes,
            "average_confidence": round(self.average_confidence, 4),
            "max_confidence": round(self.max_confidence, 4),
            "confidence_threshold": self.confidence_threshold,
            "enhanced": self.enhanced,
            "output": self.output,
            "report": self.report,
            "elapsed_seconds": self.elapsed_seconds,
            "frames_processed": self.frames_processed,
            "detections_per_class": dict(self.detections_per_class),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HistoryEntry":
        """Rebuild an entry, tolerating missing/odd fields from older files."""
        kind = str(payload.get("kind") or "image")
        return cls(
            timestamp=str(payload.get("timestamp") or ""),
            kind=kind if kind in VALID_KINDS else "image",
            filename=str(payload.get("filename") or "unknown"),
            models=[str(m) for m in (payload.get("models") or [])],
            total_detections=_as_int(payload.get("total_detections")),
            unique_classes=_as_int(payload.get("unique_classes")),
            average_confidence=_as_float(payload.get("average_confidence")),
            max_confidence=_as_float(payload.get("max_confidence")),
            confidence_threshold=_as_float_or_none(payload.get("confidence_threshold")),
            enhanced=bool(payload.get("enhanced", False)),
            output=str(payload.get("output") or ""),
            report=str(payload["report"]) if payload.get("report") else None,
            elapsed_seconds=_as_float_or_none(payload.get("elapsed_seconds")),
            frames_processed=_as_int_or_none(payload.get("frames_processed")),
            detections_per_class={
                str(k): _as_int(v) for k, v in (payload.get("detections_per_class") or {}).items()
            },
        )

    @property
    def display_time(self) -> str:
        """Human-readable timestamp (``08 Oct 2026, 19:42``); raw if odd."""
        try:
            return datetime.fromisoformat(self.timestamp).strftime("%d %b %Y, %H:%M")
        except (ValueError, TypeError):
            return self.timestamp or "—"


# --------------------------------------------------------------------------- #
# Store operations
# --------------------------------------------------------------------------- #
def history_path(settings: Settings | None = None) -> Path:
    """Location of the history file (inside the outputs directory)."""
    settings = settings or get_settings()
    return settings.outputs_dir / "history" / "detections.json"


def load_entries(path: str | Path) -> list[HistoryEntry]:
    """Read the history, newest first. Never raises on bad/missing files."""
    import json

    file_path = Path(path)
    if not file_path.is_file():
        return []
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable history file %s: %s", file_path, exc)
        return []
    if not isinstance(payload, list):
        logger.warning("Ignoring history file %s: expected a JSON array.", file_path)
        return []

    entries: list[HistoryEntry] = []
    for item in payload:
        if isinstance(item, dict):
            entries.append(HistoryEntry.from_dict(item))
    return entries


def add_entry(
    entry: HistoryEntry,
    path: str | Path | None = None,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> list[HistoryEntry]:
    """Prepend ``entry`` to the store and persist it atomically.

    Returns the resulting list (newest first). A failed write is logged but
    never breaks the detection flow that triggered it.
    """
    import json

    file_path = Path(path) if path else history_path()
    entries = [entry, *load_entries(file_path)][:max_entries]
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = file_path.with_suffix(file_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps([e.to_dict() for e in entries], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, file_path)
    except OSError as exc:  # pragma: no cover - disk full / permissions
        logger.warning("Could not persist detection history to %s: %s", file_path, exc)
    return entries


def clear_history(path: str | Path | None = None) -> None:
    """Delete the stored history (used by the UI's "clear" action)."""
    file_path = Path(path) if path else history_path()
    try:
        file_path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - permissions
        logger.warning("Could not clear detection history %s: %s", file_path, exc)


def summarize(entries: Sequence[HistoryEntry]) -> dict[str, Any]:
    """Aggregate statistics over the stored runs (drives the Analytics page).

    All values are computed from real stored entries; nothing is inferred.
    """
    species_counts: dict[str, int] = {}
    model_keys: set[str] = set()
    total_objects = 0
    weighted_confidence = 0.0
    frames = 0
    images = videos = 0

    for entry in entries:
        total_objects += entry.total_detections
        weighted_confidence += entry.average_confidence * entry.total_detections
        frames += entry.frames_processed or 0
        model_keys.update(entry.models)
        for name, count in entry.detections_per_class.items():
            species_counts[name] = species_counts.get(name, 0) + count
        if entry.kind == "video":
            videos += 1
        else:
            images += 1

    top = sorted(species_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "runs": len(entries),
        "images": images,
        "videos": videos,
        "total_objects": total_objects,
        "unique_species": len(species_counts),
        "average_confidence": round(weighted_confidence / total_objects, 4) if total_objects else 0.0,
        "frames_processed": frames,
        "models_used": sorted(model_keys),
        "species_counts": dict(top),
        "top_species": top[:5],
        "first_run": entries[-1].display_time if entries else None,
        "last_run": entries[0].display_time if entries else None,
    }


def confidence_histogram(
    entries: Sequence[HistoryEntry], bins: int = 10
) -> list[tuple[str, int]]:
    """Bucketed *average* confidence of the stored runs, for a bar chart.

    Returns ``[(label, count), ...]``. A run contributes to the bucket of
    its own average confidence - we never invent per-detection values that
    the pipeline did not store.
    """
    buckets = [0] * max(1, bins)
    step = 1.0 / bins
    for entry in entries:
        index = min(int(entry.average_confidence / step), bins - 1)
        buckets[index] += 1
    return [
        (f"{int(i * step * 100)}–{int((i + 1) * step * 100)}%", count)
        for i, count in enumerate(buckets)
    ]


# --------------------------------------------------------------------------- #
# Small coercion helpers (tolerant parsing of older/newer files)
# --------------------------------------------------------------------------- #
def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_int_or_none(value: Any) -> int | None:
    return None if value is None else _as_int(value)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_float_or_none(value: Any) -> float | None:
    return None if value is None else _as_float(value)


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is not None:
            parsed = _as_float_or_none(value)
            if parsed is not None:
                return parsed
    return None


def extend(entries: Iterable[HistoryEntry]) -> list[dict[str, Any]]:
    """Convenience for JSON export of a list of entries."""
    return [entry.to_dict() for entry in entries]
