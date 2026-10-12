"""Tests for continuous (scheduled) evaluation and publication export.

These exercise the *real* code paths with a stub detector - no weights and no
network - exactly like the rest of the evaluation suite:

* :mod:`app.evaluation.schedules` - persistence and the pure due decision;
* :mod:`app.evaluation.scheduler` - one tick actually launching a run, cleaning
  up a stale ``running`` row, and refusing a dataset that no longer validates;
* :mod:`app.evaluation.publication` - paper-ready tables built from stored,
  genuinely measured results.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

from app.evaluation.experiments import EvaluationStore
from app.evaluation.jobs import is_running
from app.evaluation import publication as pub
from app.evaluation import scheduler as sched
from app.evaluation.schedules import (
    DEFAULT_POLL_SECONDS,
    MIN_POLL_SECONDS,
    SchedulePreferences,
    ScheduleStore,
    due_reason,
    poll_due,
)

USER = "u" * 32


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_dataset(root: Path, *, label: str = "0 0.5 0.5 0.2 0.2\n") -> Path:
    """A tiny one-image YOLO dataset (test split only), like the workspace tests."""
    (root / "images" / "test").mkdir(parents=True)
    (root / "labels" / "test").mkdir(parents=True)
    Image.new("RGB", (32, 32), (10, 40, 90)).save(root / "images" / "test" / "a.jpg")
    (root / "labels" / "test" / "a.txt").write_text(label, encoding="utf-8")
    (root / "data.yaml").write_text(
        "path: .\ntest: images/test\nnames:\n  0: fish\n", encoding="utf-8"
    )
    return root / "data.yaml"


class _Detection:
    def __init__(self, class_name, confidence, bbox):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox


class _Result:
    def __init__(self, detections):
        self.detections = detections


class _FakeDetector:
    """A detector stand-in with real predictor plumbing but no weights."""

    device = "cpu"

    def get_class_names(self, model_selection=None):
        return {(model_selection or "fake"): {0: "fish"}}

    def predict_image(self, image, model_selection=None, confidence=None, source_name=""):
        # A box that overlaps the ground-truth box enough to score, proving the
        # scheduled run measures something real rather than returning zeros.
        return _Result([_Detection("fish", 0.9, (12, 12, 20, 20))])


def _wait_until_done(experiment_id: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while is_running(experiment_id) and time.time() < deadline:
        time.sleep(0.05)


def _new_schedule(sched_store: ScheduleStore, data_yaml: Path, **overrides):
    params = {
        "confidence": None,
        "use_recommended_threshold": True,
        "iou_threshold_for_figures": 0.5,
        "max_images": 0,
        "plots": False,
    }
    params.update(overrides.pop("parameters", {}))
    return sched_store.create(
        USER,
        overrides.pop("name", "Nightly reef check"),
        data_yaml,
        dataset_name="reef",
        split="test",
        class_names=("fish",),
        model_keys=overrides.pop("model_keys", ("fake",)),
        parameters=params,
        **overrides,
    )


# --------------------------------------------------------------------------- #
# Schedule store
# --------------------------------------------------------------------------- #
def test_schedule_store_crud_and_owner_scope(tmp_path: Path) -> None:
    store = ScheduleStore(tmp_path / "eval.db")
    schedule = store.create(
        USER,
        "Watch reef",
        tmp_path / "data.yaml",
        dataset_name="reef",
        class_names=("fish",),
        model_keys=("fish_inv",),
        trigger_mode="both",
        interval_minutes=30,
    )
    assert schedule.id.startswith("sch_")
    assert schedule.enabled
    assert schedule.trigger_label == "On change or interval"

    assert schedule.next_due_hint().startswith("Due now")
    assert schedule.as_dict()["model_keys"] == ["fish_inv"]

    updated = store.update(schedule.id, USER, name="Renamed", interval_minutes=90)
    assert updated is not None and updated.name == "Renamed"
    assert updated.interval_minutes == 90

    assert store.list_enabled()[0].id == schedule.id
    assert store.set_enabled(schedule.id, USER, False)
    assert store.list_enabled() == []

    # A different owner can never see or delete it.
    assert store.get(schedule.id, "other") is None
    assert store.delete(schedule.id, "other") is False
    assert store.delete(schedule.id, USER) is True
    assert store.list(USER) == []


# --------------------------------------------------------------------------- #
# The due decision
# --------------------------------------------------------------------------- #
def test_due_reason_first_run_interval_and_change(tmp_path: Path) -> None:
    store = ScheduleStore(tmp_path / "eval.db")
    schedule = _new_schedule(store, tmp_path / "data.yaml", trigger_mode="both", interval_minutes=30)

    now = datetime.now(timezone.utc)
    first = due_reason(schedule, now, "fp1", {"fake": {"size": 1}})
    assert first is not None and first.reason == "first run"

    store.record_run(
        schedule.id,
        USER,
        experiment_id="exp_1",
        status="queued",
        dataset_fingerprint="fp1",
        model_fingerprints={"fake": {"size": 1}},
        ran_at=now.isoformat(),
    )
    schedule = store.get(schedule.id, USER)
    assert schedule is not None

    # Nothing changed and the interval has not elapsed -> not due.
    assert due_reason(schedule, now, "fp1", {"fake": {"size": 1}}) is None

    changed = due_reason(schedule, now, "fp2", {"fake": {"size": 1}})
    assert changed is not None and changed.reason == "change"

    weights_changed = due_reason(schedule, now, "fp1", {"fake": {"size": 2}})
    assert weights_changed is not None and weights_changed.reason == "change"

    later = now + timedelta(minutes=31)
    elapsed = due_reason(schedule, now=later, dataset_fingerprint="fp1", model_fingerprints={"fake": {"size": 1}})
    assert elapsed is not None and elapsed.reason == "interval"


def test_due_reason_respects_disabled_and_unknown_fingerprints(tmp_path: Path) -> None:
    store = ScheduleStore(tmp_path / "eval.db")
    schedule = _new_schedule(store, tmp_path / "data.yaml", trigger_mode="on_change")
    now = datetime.now(timezone.utc)

    store.record_run(
        schedule.id,
        USER,
        experiment_id="exp_1",
        status="queued",
        dataset_fingerprint="fp1",
        model_fingerprints={},
        ran_at=now.isoformat(),
    )
    store.update(schedule.id, USER, enabled=False)
    schedule = store.get(schedule.id, USER)
    assert due_reason(schedule, now, "fp2", {"fake": {"size": 9}}) is None

    # An unknown (empty) fingerprint can never trigger a change run.
    store.update(schedule.id, USER, enabled=True)
    schedule = store.get(schedule.id, USER)
    assert due_reason(schedule, now, "", {}) is None


# --------------------------------------------------------------------------- #
# Per-account continuous-evaluation preferences
# --------------------------------------------------------------------------- #
def test_schedule_preferences_defaults_and_roundtrip(tmp_path: Path) -> None:
    store = ScheduleStore(tmp_path / "eval.db")

    # An account that never touched the settings gets sensible defaults.
    defaults = store.get_preferences(USER)
    assert defaults.enabled is True
    assert defaults.poll_seconds == DEFAULT_POLL_SECONDS
    assert defaults.last_checked_at is None

    saved = store.set_preferences(USER, enabled=False, poll_seconds=120)
    assert saved.enabled is False
    assert saved.poll_seconds == 120

    # A second call with only one field leaves the other untouched.
    partial = store.set_preferences(USER, enabled=True)
    assert partial.enabled is True
    assert partial.poll_seconds == 120

    # The poll interval is clamped into the supported range.
    assert store.set_preferences(USER, poll_seconds=1).poll_seconds == MIN_POLL_SECONDS

    store.mark_checked(USER, "2026-01-01T00:00:00+00:00")
    assert store.get_preferences(USER).last_checked_at == "2026-01-01T00:00:00+00:00"


def test_poll_due_is_timing_only() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    fresh = SchedulePreferences(user_id=USER, poll_seconds=60, last_checked_at=None)
    assert poll_due(fresh, now) is True

    recent = SchedulePreferences(
        user_id=USER,
        poll_seconds=60,
        last_checked_at=(now - timedelta(seconds=30)).isoformat(),
    )
    assert poll_due(recent, now) is False

    elapsed = SchedulePreferences(
        user_id=USER,
        poll_seconds=60,
        last_checked_at=(now - timedelta(seconds=61)).isoformat(),
    )
    assert poll_due(elapsed, now) is True


def test_tick_respects_a_disabled_account(settings, tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "ds")
    store = EvaluationStore(settings.evaluation_db_path)
    sched_store = ScheduleStore(settings.evaluation_db_path)
    _new_schedule(sched_store, data_yaml)
    sched_store.set_preferences(USER, enabled=False)

    result = sched.tick_once(
        settings, _FakeDetector(), store=store, sched_store=sched_store
    )
    assert result.started == []
    assert result.checked == 1
    assert any("switched off" in item["reason"] for item in result.skipped)
    assert store.count_experiments(USER) == 0


def test_tick_respects_the_poll_gate_and_force_overrides_it(
    settings, tmp_path: Path
) -> None:
    data_yaml = make_dataset(tmp_path / "ds")
    store = EvaluationStore(settings.evaluation_db_path)
    sched_store = ScheduleStore(settings.evaluation_db_path)
    _new_schedule(sched_store, data_yaml)
    now = datetime.now(timezone.utc)
    sched_store.set_preferences(USER, enabled=True, poll_seconds=600)
    sched_store.mark_checked(USER, now.isoformat())

    gated = sched.tick_once(
        settings,
        _FakeDetector(),
        store=store,
        sched_store=sched_store,
        now=now,
    )
    assert gated.started == []
    assert any("next scheduled check" in item["reason"] for item in gated.skipped)

    # force=True (the Account page's manual "check now") ignores the poll gate.
    forced = sched.tick_once(
        settings,
        _FakeDetector(),
        store=store,
        sched_store=sched_store,
        now=now,
        force=True,
    )
    assert len(forced.started) == 1
    experiment_id = forced.started[0]["experiment_id"]
    _wait_until_done(experiment_id)
    assert store.get_experiment(experiment_id, USER) is not None


def test_tick_only_user_filters_other_accounts(settings, tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "ds")
    store = EvaluationStore(settings.evaluation_db_path)
    sched_store = ScheduleStore(settings.evaluation_db_path)
    _new_schedule(sched_store, data_yaml)
    _new_schedule(sched_store, data_yaml, name="Other account")

    result = sched.tick_once(
        settings,
        _FakeDetector(),
        store=store,
        sched_store=sched_store,
        only_user="someone-else",
    )
    assert result.checked == 0
    assert result.started == []


# --------------------------------------------------------------------------- #
# The worker tick (integration with the real job launcher)
# --------------------------------------------------------------------------- #
def test_tick_starts_a_real_scheduled_run(settings, tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "ds")
    store = EvaluationStore(settings.evaluation_db_path)
    sched_store = ScheduleStore(settings.evaluation_db_path)
    schedule = _new_schedule(sched_store, data_yaml)

    detector = _FakeDetector()
    result = sched.tick_once(settings, detector, store=store, sched_store=sched_store)

    assert result.checked == 1
    assert len(result.started) == 1
    assert result.started[0]["reason"] == "first run"
    experiment_id = result.started[0]["experiment_id"]

    _wait_until_done(experiment_id)
    record = store.get_experiment(experiment_id, USER)
    assert record is not None
    assert record.status in ("completed", "partially_completed")
    assert record.results, "a scheduled run must store genuinely measured results"
    assert record.parameters["schedule_id"] == schedule.id

    stored = sched_store.get(schedule.id, USER)
    assert stored.last_experiment_id == experiment_id
    assert stored.run_count == 1
    assert stored.last_dataset_fingerprint

    # A second immediate tick does not re-run an unchanged schedule.
    again = sched.tick_once(settings, detector, store=store, sched_store=sched_store)
    assert again.started == []


def test_tick_cleans_up_a_stale_running_row(settings, tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "ds")
    store = EvaluationStore(settings.evaluation_db_path)
    sched_store = ScheduleStore(settings.evaluation_db_path)
    _new_schedule(sched_store, data_yaml)

    stale = store.create_experiment(
        USER, "Interrupted run", data_yaml=str(data_yaml), split="test", status="running"
    )
    assert is_running(stale.id) is False

    sched.tick_once(settings, _FakeDetector(), store=store, sched_store=sched_store)

    refreshed = store.get_experiment(stale.id, USER)
    assert refreshed is not None and refreshed.status == "failed"
    assert "no live worker" in refreshed.error.lower()


def test_scheduled_run_refuses_an_invalid_dataset(settings, tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "ds", label="0 0.5 0.5\n")  # malformed row
    store = EvaluationStore(settings.evaluation_db_path)
    sched_store = ScheduleStore(settings.evaluation_db_path)
    schedule = _new_schedule(sched_store, data_yaml)

    with pytest.raises(sched.ScheduleRunError):
        sched.start_scheduled_run(
            store=store,
            settings=settings,
            sched_store=sched_store,
            schedule=schedule,
            detector=_FakeDetector(),
        )
    # Nothing was stored and no score was invented.
    assert store.count_experiments(USER) == 0


# --------------------------------------------------------------------------- #
# Publication export
# --------------------------------------------------------------------------- #
def _stored_experiment(settings, tmp_path: Path):
    """Run one real (stubbed-detector) evaluation and return its record."""
    data_yaml = make_dataset(tmp_path / "ds")
    store = EvaluationStore(settings.evaluation_db_path)
    sched_store = ScheduleStore(settings.evaluation_db_path)
    schedule = _new_schedule(sched_store, data_yaml)
    result = sched.tick_once(settings, _FakeDetector(), store=store, sched_store=sched_store)
    experiment_id = result.started[0]["experiment_id"]
    _wait_until_done(experiment_id)
    return store, store.get_experiment(experiment_id, USER)


def test_publication_tables_from_stored_results(settings, tmp_path: Path) -> None:
    _store, record = _stored_experiment(settings, tmp_path)

    rows = pub.headline_rows([record])
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "fake"
    assert row["classes_scored"] == 1
    assert row["classes_unsupported"] == ""
    assert 0.0 <= row["mAP50"] <= 1.0

    per_class = pub.per_class_rows([record])
    assert [item["class_name"] for item in per_class] == ["fish"]

    markdown = pub.render_markdown([record])
    assert "mAP@0.5" in markdown and "Provenance sheet" in markdown

    latex = pub.render_latex([record])
    assert r"\begin{table}" in latex and r"\toprule" in latex and r"\bottomrule" in latex

    html = pub.render_html([record], embed_plots=False)
    assert "<table>" in html and record.dataset_name in html


def test_build_bundle_writes_every_file(settings, tmp_path: Path) -> None:
    _store, record = _stored_experiment(settings, tmp_path)
    written = pub.build_bundle([record], tmp_path / "bundle")

    expected = {
        "publication_metrics.csv",
        "publication_per_class.csv",
        "publication_tables.md",
        "publication_tables.tex",
        "dataset_cards.md",
        "provenance.md",
        "publication_report.html",
    }
    assert expected <= set(written)
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0


def test_latex_escapes_special_characters() -> None:
    text = pub.to_latex(
        [{"experiment": "a_b & c%"}],
        (("experiment", "Experiment", False),),
    )
    assert r"\_" in text and r"\&" in text and r"\%" in text
