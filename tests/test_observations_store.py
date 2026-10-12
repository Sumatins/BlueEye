"""Unit tests for the persistent observation-records layer.

These cover the guarantees that matter most: a saved observation belongs to the
account that created it, cannot be seen or changed by another account, and a
user correction never rewrites the original model prediction.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from pathlib import Path

from app.observations import compare, export as obs_export
from app.observations.builder import build_analysis, detections_from_report
from app.observations.models import (
    DuplicateObservationError,
    NotOwnerError,
    ObservationError,
)
from app.observations.store import ObservationStore


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _image_report() -> dict:
    return {
        "input": "reef.jpg",
        "timestamp": "2026-10-08T19:42:00",
        "models": ["fish_inv"],
        "confidence_threshold": 0.5,
        "enhanced": False,
        "statistics": {
            "total_detections": 2,
            "unique_classes": 2,
            "detections_per_class": {"fish": 1, "serranidae": 1},
            "average_confidence": 0.6,
            "max_confidence": 0.8,
        },
        "detections": [
            {"class_name": "fish", "confidence": 0.8, "bbox": [1, 2, 30, 40], "model": "fish_inv"},
            {"class_name": "serranidae", "confidence": 0.4, "bbox": [5, 6, 70, 80], "model": "fish_inv"},
        ],
    }


def _video_report() -> dict:
    return {
        "input": "dive.mp4",
        "timestamp": "2026-10-08T20:00:00",
        "models": ["megafauna"],
        "confidence_threshold": 0.4,
        "statistics": {
            "total_detections": 2,
            "unique_classes": 2,
            "average_confidence": 0.7,
            "max_confidence": 0.9,
        },
        "video": {"fps": 25.0},
        "processing": {"frames_processed": 100, "elapsed_seconds": 12.5},
        "frame_detections": [
            {"frame": 10, "detections": [{"class_name": "shark", "confidence": 0.9, "bbox": [0, 0, 5, 5]}]},
            {"frame": 40, "detections": [{"class_name": "ray", "confidence": 0.5, "bbox": [1, 1, 6, 6]}]},
        ],
    }


def _store(tmp_path) -> ObservationStore:
    return ObservationStore(tmp_path / "observations" / "test.db")


def _analysis(tmp_path, user_id="u1", kind="image", report=None, name="reef.jpg"):
    return build_analysis(
        user_id=user_id,
        kind=kind,
        report=report or (_image_report() if kind == "image" else _video_report()),
        original_path=tmp_path / "media" / name,
        annotated_path=tmp_path / "media" / f"annotated_{name}",
        report_path=tmp_path / "media" / f"{name}.json",
    )


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def test_builder_reads_image_report_fields() -> None:
    analysis = build_analysis(
        user_id="u1",
        kind="image",
        report=_image_report(),
        original_path=Path("reef.jpg"),
        annotated_path=Path("reef_annotated.jpg"),
        report_path=Path("reef.json"),
    )
    assert analysis.source_filename == "reef.jpg"
    assert analysis.total_detections == 2
    assert analysis.unique_classes == 2
    assert analysis.average_confidence == 0.6
    assert analysis.detected_at == "2026-10-08T19:42:00"
    assert analysis.model_version is None  # never invented
    assert {det.class_name for det in analysis.detections} == {"fish", "serranidae"}


def test_builder_reads_video_frames_and_fps() -> None:
    detections = detections_from_report(
        "video", _video_report(), analysis_id="a1", user_id="u1"
    )
    assert len(detections) == 2
    assert detections[0].frame_index == 10
    assert detections[0].frame_time == 10 / 25.0
    assert detections[1].frame_index == 40


def test_builder_rejects_missing_report() -> None:
    try:
        build_analysis(user_id="u1", kind="image", report=None, original_path=Path("x.jpg"))
    except ObservationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ObservationError for a missing report")


# --------------------------------------------------------------------------- #
# Save / load / ownership
# --------------------------------------------------------------------------- #
def test_save_and_load_round_trip(tmp_path) -> None:
    store = _store(tmp_path)
    analysis = store.save_analysis(_analysis(tmp_path))
    loaded = store.get_analysis(analysis.id, "u1")
    assert loaded is not None
    assert loaded.source_filename == "reef.jpg"
    assert len(loaded.detections) == 2


def test_records_are_isolated_by_account(tmp_path) -> None:
    store = _store(tmp_path)
    analysis = store.save_analysis(_analysis(tmp_path, user_id="u1"))
    assert store.get_analysis(analysis.id, "u2") is None
    assert store.list_analyses("u2") == []
    assert store.count_analyses("u2") == 0


def test_duplicate_save_is_rejected(tmp_path) -> None:
    store = _store(tmp_path)
    store.save_analysis(_analysis(tmp_path))
    try:
        store.save_analysis(_analysis(tmp_path))
    except DuplicateObservationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected DuplicateObservationError")


def test_find_existing_returns_the_saved_record(tmp_path) -> None:
    store = _store(tmp_path)
    saved = store.save_analysis(_analysis(tmp_path))
    found = store.find_existing("u1", saved.original_path, saved.annotated_path)
    assert found is not None and found.id == saved.id
    assert store.find_existing("u2", saved.original_path, saved.annotated_path) is None


# --------------------------------------------------------------------------- #
# Corrections
# --------------------------------------------------------------------------- #
def test_correction_preserves_the_original_prediction(tmp_path) -> None:
    store = _store(tmp_path)
    analysis = store.save_analysis(_analysis(tmp_path))
    detection = store.get_analysis(analysis.id, "u1").detections[0]
    store.add_correction(detection.id, "u1", "Grouper (Serranidae)", note="visual")

    reloaded = store.get_analysis(analysis.id, "u1")
    corrected = next(det for det in reloaded.detections if det.id == detection.id)
    assert corrected.class_name == detection.class_name  # original untouched
    assert corrected.effective_label == "Grouper (Serranidae)"
    assert reloaded.review_status == "corrected"


def test_correction_requires_ownership(tmp_path) -> None:
    store = _store(tmp_path)
    analysis = store.save_analysis(_analysis(tmp_path, user_id="u1"))
    detection = store.get_analysis(analysis.id, "u1").detections[0]
    try:
        store.add_correction(detection.id, "u2", "anything")
    except NotOwnerError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected NotOwnerError")


# --------------------------------------------------------------------------- #
# Collections
# --------------------------------------------------------------------------- #
def test_collections_link_and_unlink_owned_records(tmp_path) -> None:
    store = _store(tmp_path)
    analysis = store.save_analysis(_analysis(tmp_path))
    collection = store.create_collection("u1", "Reef survey", "2026 transects")
    store.add_to_collection(collection.id, analysis.id, "u1")
    store.add_to_collection(collection.id, analysis.id, "u1")  # idempotent
    assert store.list_collections("u1")[0].item_count == 1
    assert [item.id for item in store.list_collection_analyses(collection.id, "u1")] == [
        analysis.id
    ]

    store.remove_from_collection(collection.id, analysis.id, "u1")
    assert store.list_collection_analyses(collection.id, "u1") == []


def test_collection_cannot_link_another_accounts_record(tmp_path) -> None:
    store = _store(tmp_path)
    analysis = store.save_analysis(_analysis(tmp_path, user_id="u2"))
    collection = store.create_collection("u1", "Mine")
    try:
        store.add_to_collection(collection.id, analysis.id, "u1")
    except NotOwnerError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected NotOwnerError")


# --------------------------------------------------------------------------- #
# Delete
# --------------------------------------------------------------------------- #
def test_delete_requires_ownership_and_cascades(tmp_path) -> None:
    store = _store(tmp_path)
    analysis = store.save_analysis(_analysis(tmp_path, user_id="u1"))
    assert store.delete_analysis(analysis.id, "u2") is False
    assert store.get_analysis(analysis.id, "u1") is not None
    assert store.delete_analysis(analysis.id, "u1") is True
    assert store.get_analysis(analysis.id, "u1") is None


# --------------------------------------------------------------------------- #
# Stats / filters
# --------------------------------------------------------------------------- #
def test_stats_are_user_scoped_and_real(tmp_path) -> None:
    store = _store(tmp_path)
    store.save_analysis(_analysis(tmp_path, user_id="u1"))
    store.create_collection("u1", "Survey")
    stats = store.stats("u1")
    assert stats["analyses"] == 1
    assert stats["observations"] == 2
    assert stats["collections"] == 1
    assert stats["awaiting_review"] == 1  # predicted and not yet reviewed
    assert store.stats("u2")["analyses"] == 0


def test_search_and_filters(tmp_path) -> None:
    store = _store(tmp_path)
    first = store.save_analysis(_analysis(tmp_path, name="reef.jpg"))
    second = _analysis(tmp_path, name="boat.jpg")
    second.source_filename = "boat.jpg"
    store.save_analysis(second)

    assert [item.id for item in store.list_analyses("u1", query="reef")] == [first.id]
    assert store.count_analyses("u1", kind="image") == 2
    assert store.count_analyses("u1", kind="video") == 0


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
def test_exports_carry_the_data_limitation(tmp_path) -> None:
    store = _store(tmp_path)
    analysis = store.get_analysis(store.save_analysis(_analysis(tmp_path)).id, "u1")

    payload = obs_export.analysis_to_dict(analysis)
    assert "data_limitations" in payload
    assert "not" in payload["data_limitations"].lower()

    csv_text = obs_export.observations_csv([analysis])
    assert "analysis_id" in csv_text.splitlines()[0]
    assert "reef.jpg" in csv_text

    html = obs_export.analysis_report_html(analysis, embed_images=False)
    assert "observation report" in html.lower()
    assert "data limitations" in html.lower()


def test_collection_export_mentions_the_survey(tmp_path) -> None:
    store = _store(tmp_path)
    analysis = store.save_analysis(_analysis(tmp_path))
    collection = store.create_collection("u1", "Reef survey")
    store.add_to_collection(collection.id, analysis.id, "u1")
    analyses = store.list_collection_analyses(collection.id, "u1")

    payload = obs_export.collection_to_dict(collection, analyses)
    assert payload["collection"]["name"] == "Reef survey"
    assert payload["count"] == 1
    html = obs_export.collection_report_html(collection, analyses, embed_images=False)
    assert "Reef survey" in html


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def test_comparison_reports_predictions_not_identity(tmp_path) -> None:
    store = _store(tmp_path)
    first = store.save_analysis(_analysis(tmp_path, name="one.jpg"))
    second = store.save_analysis(_analysis(tmp_path, name="two.jpg"))
    full = [store.get_analysis(first.id, "u1"), store.get_analysis(second.id, "u1")]

    result = compare.compare_analyses(full)
    assert result["count"] == 2
    assert result["note"]  # the caveat always travels with the data
    assert result["labels"]
    rows = compare.comparison_rows(full)
    assert len(rows) == 2
    assert "Fish" in rows[0]["labels"]


def test_stats_counts_only_real_rows(tmp_path) -> None:
    store = _store(tmp_path)
    store.save_analysis(_analysis(tmp_path, kind="video", report=_video_report()))
    stats = store.stats("u1")
    assert stats["observations"] == 2
    assert stats["analyses"] == 1
