"""Unit tests for the persistent detection history store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.utils.history import (
    HistoryEntry,
    add_entry,
    clear_history,
    confidence_histogram,
    history_path,
    load_entries,
    summarize,
)

IMAGE_REPORT = {
    "input": "reef.jpg",
    "timestamp": "2026-10-08T19:42:00",
    "models": ["fish_inv", "megafauna"],
    "confidence_threshold": 0.5,
    "enhanced": False,
    "inference_time_ms": 812.4,
    "output_image": "detection_abc.jpg",
    "processing_time_seconds": 0.91,
    "detections": [],
    "statistics": {
        "total_detections": 4,
        "unique_classes": 3,
        "detections_per_class": {"scaridae": 2, "turtle": 1, "chaetodontidae": 1},
        "average_confidence": 0.739,
        "max_confidence": 0.879,
    },
}

VIDEO_REPORT = {
    "input": "dive.mp4",
    "timestamp": "2026-10-08T20:00:00",
    "models": ["fish_inv"],
    "confidence_threshold": 0.6,
    "enhanced": True,
    "output_video": "detection_xyz.mp4",
    "processing": {
        "frames_processed": 12,
        "frames_with_detections": 12,
        "processing_fps": 0.5,
        "elapsed_seconds": 23.5,
    },
    "statistics": {
        "total_detections": 24,
        "unique_classes": 2,
        "detections_per_class": {"scaridae": 12, "turtle": 12},
        "average_confidence": 0.698,
        "max_confidence": 0.76,
    },
}


# --------------------------------------------------------------------------- #
# Entry construction
# --------------------------------------------------------------------------- #
def test_entry_from_image_report() -> None:
    entry = HistoryEntry.from_report("image", IMAGE_REPORT, "detection_abc.jpg", "r.json")
    assert entry.kind == "image"
    assert entry.filename == "reef.jpg"
    assert entry.models == ["fish_inv", "megafauna"]
    assert entry.total_detections == 4
    assert entry.unique_classes == 3
    assert entry.average_confidence == pytest.approx(0.739)
    assert entry.max_confidence == pytest.approx(0.879)
    assert entry.confidence_threshold == pytest.approx(0.5)
    assert entry.frames_processed is None
    assert entry.detections_per_class["scaridae"] == 2
    assert entry.display_time == "08 Oct 2026, 19:42"


def test_entry_from_video_report() -> None:
    entry = HistoryEntry.from_report("video", VIDEO_REPORT, "detection_xyz.mp4")
    assert entry.kind == "video"
    assert entry.frames_processed == 12
    assert entry.elapsed_seconds == pytest.approx(23.5)
    assert entry.enhanced is True
    assert entry.total_detections == 24


def test_entry_from_report_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        HistoryEntry.from_report("audio", IMAGE_REPORT, "x")


def test_entry_round_trip_preserves_fields() -> None:
    entry = HistoryEntry.from_report("image", IMAGE_REPORT, "out.jpg")
    restored = HistoryEntry.from_dict(entry.to_dict())
    assert restored.to_dict() == entry.to_dict()


def test_entry_from_dict_tolerates_junk() -> None:
    restored = HistoryEntry.from_dict(
        {"kind": "audio", "total_detections": "many", "average_confidence": "x"}
    )
    assert restored.kind == "image"  # sanitised to a valid kind
    assert restored.total_detections == 0
    assert restored.average_confidence == 0.0
    assert restored.filename == "unknown"


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
def test_add_entry_prepends_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    first = HistoryEntry.from_report("image", IMAGE_REPORT, "a.jpg")
    second = HistoryEntry.from_report("video", VIDEO_REPORT, "b.mp4")

    add_entry(first, path)
    entries = add_entry(second, path)

    assert [e.filename for e in entries] == ["dive.mp4", "reef.jpg"]
    assert [e.filename for e in load_entries(path)] == ["dive.mp4", "reef.jpg"]
    assert path.is_file()


def test_add_entry_caps_history(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    for index in range(5):
        report = dict(IMAGE_REPORT, input=f"img{index}.jpg")
        add_entry(HistoryEntry.from_report("image", report, "o.jpg"), path, max_entries=3)
    entries = load_entries(path)
    assert len(entries) == 3
    assert entries[0].filename == "img4.jpg"  # newest kept


def test_history_path_lives_under_outputs(settings: Settings) -> None:
    assert settings.outputs_dir in history_path(settings).parents


def test_load_entries_tolerates_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text("{oops", encoding="utf-8")
    assert load_entries(path) == []

    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    assert load_entries(path) == []

    assert load_entries(tmp_path / "missing.json") == []


def test_clear_history(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    add_entry(HistoryEntry.from_report("image", IMAGE_REPORT, "a.jpg"), path)
    assert load_entries(path)
    clear_history(path)
    assert load_entries(path) == []
    clear_history(path)  # idempotent


# --------------------------------------------------------------------------- #
# Aggregation (Analytics)
# --------------------------------------------------------------------------- #
def _sample_entries() -> list[HistoryEntry]:
    """Newest first, matching the store's on-disk ordering."""
    return [
        HistoryEntry.from_report("video", VIDEO_REPORT, "b.mp4"),
        HistoryEntry.from_report("image", IMAGE_REPORT, "a.jpg"),
    ]


def test_summarize_matches_stored_entries() -> None:
    stats = summarize(_sample_entries())
    assert stats["runs"] == 2
    assert stats["images"] == 1
    assert stats["videos"] == 1
    assert stats["total_objects"] == 28
    assert stats["unique_species"] == 3  # scaridae, turtle, chaetodontidae
    assert stats["frames_processed"] == 12
    assert stats["models_used"] == ["fish_inv", "megafauna"]
    assert stats["species_counts"]["scaridae"] == 14
    assert stats["top_species"][0] == ("scaridae", 14)
    # weighted by detections: (0.739*4 + 0.698*24) / 28
    assert stats["average_confidence"] == pytest.approx((0.739 * 4 + 0.698 * 24) / 28, abs=1e-3)
    assert stats["last_run"] == "08 Oct 2026, 20:00"


def test_summarize_empty_history() -> None:
    stats = summarize([])
    assert stats["runs"] == 0
    assert stats["total_objects"] == 0
    assert stats["average_confidence"] == 0.0
    assert stats["top_species"] == []
    assert stats["first_run"] is None


def test_confidence_histogram_buckets_real_values() -> None:
    buckets = dict(confidence_histogram(_sample_entries(), bins=10))
    # 0.739 -> 70-80%, 0.698 -> 60-70%
    assert buckets["70–80%"] == 1
    assert buckets["60–70%"] == 1
    assert sum(buckets.values()) == 2
