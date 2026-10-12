"""Execute *due* evaluation schedules while the application is running.

This is the *real-time* half of the continuous research evaluation: a single
daemon thread wakes periodically, asks :mod:`app.evaluation.schedules` which
saved schedules are due (a first run, an elapsed interval, or a genuine change
in the dataset / model weights) and, for those, launches the **same**
background experiment the UI launches.

Design notes
------------
* Nothing here invents a result. A due schedule is turned into a normal
  experiment through :func:`app.evaluation.jobs.start_experiment` and every
  metric it produces comes from the real detector and the real annotations.
* A schedule whose dataset no longer validates is *not* run; it is reported as
  blocked and retried later, never scored with stale or missing data.
* The worker never runs when a user already has an active experiment, so a
  scheduled run can never compete with the interactive workspace for the same
  account. At most ``max_runs`` schedules are started per tick.
* :func:`configure` is idempotent: the Streamlit entry point may call it on
  every rerun and only one worker thread is ever created.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.environment import (
    collect_environment,
    evaluator_version,
    git_commit,
)
from app.evaluation.experiments import (
    ACTIVE_STATUSES,
    EvaluationStore,
    file_fingerprint,
)
from app.evaluation.jobs import ExperimentRequest, is_running, start_experiment
from app.evaluation.schedules import (
    DueDecision,
    Schedule,
    SchedulePreferences,
    ScheduleStore,
    due_reason,
    poll_due,
)
from app.evaluation.validation import DatasetValidationError, validate_dataset

logger = logging.getLogger(__name__)


class ScheduleRunError(RuntimeError):
    """Raised when a schedule cannot be turned into a run right now."""


# --------------------------------------------------------------------------- #
# Fingerprints (from the real files on disk)
# --------------------------------------------------------------------------- #
def compute_dataset_fingerprint(data_yaml: str | Path) -> str:
    """Fingerprint the dataset config + inventory, or ``""`` if unreadable."""
    try:
        return validate_dataset(data_yaml).fingerprint
    except DatasetValidationError:
        return ""
    except Exception:  # noqa: BLE001 - a broken dataset cannot trigger a run
        logger.debug("Could not fingerprint dataset %s", data_yaml, exc_info=True)
        return ""


def compute_model_fingerprints(detector: Any, model_keys: tuple[str, ...]) -> dict:
    """Fingerprint each model's weights so a swapped checkpoint is detected."""
    fingerprints: dict[str, dict] = {}
    manager = getattr(detector, "model_manager", None)
    if manager is None:
        return fingerprints
    for key in model_keys:
        try:
            fingerprints[key] = file_fingerprint(manager.resolve_path(key))
        except Exception:  # noqa: BLE001 - an unavailable model is not a reason to run
            continue
    return fingerprints


def evaluate_schedule(
    schedule: Schedule, detector: Any, now: datetime
) -> tuple[DueDecision | None, str, dict]:
    """Return ``(decision, dataset_fingerprint, model_fingerprints)``.

    The fingerprints are computed once and returned so the caller records
    exactly what the run was based on.
    """
    dataset_fp = compute_dataset_fingerprint(schedule.data_yaml)
    model_fps = compute_model_fingerprints(detector, schedule.model_keys)
    decision = due_reason(schedule, now, dataset_fp, model_fps)
    return decision, dataset_fp, model_fps


# --------------------------------------------------------------------------- #
# Starting one run
# --------------------------------------------------------------------------- #
def start_scheduled_run(
    *,
    store: EvaluationStore,
    settings: Any,
    sched_store: ScheduleStore,
    schedule: Schedule,
    detector: Any,
    trigger: str = "scheduled",
    dataset_fingerprint: str = "",
    model_fingerprints: dict | None = None,
) -> str:
    """Create and launch a real experiment for ``schedule``; return its id.

    Raises :class:`ScheduleRunError` when the schedule cannot be run (no
    detector, dataset no longer valid, no models). In that case nothing is
    stored and no score is invented.
    """
    if detector is None:
        raise ScheduleRunError("No detector is available to run this schedule.")
    if not schedule.model_keys:
        raise ScheduleRunError("The schedule selects no models.")

    try:
        validation = validate_dataset(schedule.data_yaml, split=schedule.split)
    except DatasetValidationError as exc:
        raise ScheduleRunError(f"Dataset config could not be read: {exc}") from exc
    if not validation.ready:
        problems = "; ".join(issue.message for issue in validation.errors) or (
            "the dataset did not validate"
        )
        raise ScheduleRunError(f"Dataset is not ready: {problems}")

    record = store.register_dataset(
        schedule.user_id,
        schedule.dataset_name or schedule.name,
        schedule.data_yaml,
        class_names=validation.class_names,
        splits=validation.splits_with_images,
        fingerprint=validation.fingerprint,
        validation=validation.as_dict(),
    )

    parameters = dict(schedule.parameters or {})
    parameters.update(
        {
            "split": schedule.split,
            "schedule_id": schedule.id,
            "schedule_name": schedule.name,
            "trigger": trigger,
        }
    )
    dataset_fingerprint = dataset_fingerprint or validation.fingerprint
    model_fingerprints = (
        model_fingerprints
        if model_fingerprints is not None
        else compute_model_fingerprints(detector, schedule.model_keys)
    )

    experiment = store.create_experiment(
        schedule.user_id,
        f"[scheduled] {schedule.name}",
        notes=str(parameters.get("notes", "")),
        purpose=str(parameters.get("purpose", "")) or (
            "Scheduled re-evaluation: " + schedule.trigger_label
        ),
        dataset_id=record.id,
        dataset_name=record.name,
        data_yaml=str(schedule.data_yaml),
        split=schedule.split,
        class_names=validation.class_names,
        dataset_fingerprint=dataset_fingerprint,
        model_keys=tuple(schedule.model_keys),
        model_fingerprints=model_fingerprints,
        evaluator_version=evaluator_version(),
        parameters=parameters,
        environment=collect_environment(),
        git_commit=git_commit(),
        status="queued",
    )

    confidence = parameters.get("confidence")
    try:
        confidence = None if confidence is None else float(confidence)
    except (TypeError, ValueError):
        confidence = None
    try:
        iou = float(parameters.get("iou_threshold_for_figures", 0.5))
    except (TypeError, ValueError):
        iou = 0.5
    try:
        limit = int(parameters.get("max_images", 0) or 0)
    except (TypeError, ValueError):
        limit = 0
    make_plots = bool(parameters.get("plots", True))

    output_dir = (
        Path(settings.outputs_dir) / "evaluation" / schedule.user_id / experiment.id
    )
    request = ExperimentRequest(
        experiment_id=experiment.id,
        user_id=schedule.user_id,
        data_yaml=str(schedule.data_yaml),
        split=schedule.split,
        model_keys=tuple(schedule.model_keys),
        confidence=confidence,
        iou_threshold=iou,
        limit=limit,
        output_dir=output_dir,
        make_plots=make_plots,
        parameters=parameters,
    )
    start_experiment(store, request, detector=detector)
    sched_store.record_run(
        schedule.id,
        schedule.user_id,
        experiment_id=experiment.id,
        status="queued",
        dataset_fingerprint=dataset_fingerprint,
        model_fingerprints=model_fingerprints,
    )
    logger.info(
        "Scheduled run %s started for schedule %s (%s)",
        experiment.id,
        schedule.id,
        trigger,
    )
    return experiment.id


# --------------------------------------------------------------------------- #
# The tick
# --------------------------------------------------------------------------- #
@dataclass
class TickResult:
    """What one pass of the worker did (also handy in tests and the UI)."""

    checked: int = 0
    started: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    blocked: list[dict] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.checked:
            return "No enabled schedules to check."
        parts = []
        if self.started:
            parts.append(f"started {len(self.started)}")
        if self.blocked:
            parts.append(f"blocked {len(self.blocked)}")
        parts.append(f"idle {len(self.skipped)}")
        return f"Checked {self.checked} schedule(s): " + ", ".join(parts) + "."


def _has_active_run(store: EvaluationStore, user_id: str) -> bool:
    """True when the account already has a live experiment.

    An experiment that is marked active but has no live worker (the process was
    restarted) is marked *failed* with an explicit reason so it can no longer
    block new work - the same rule the workspace UI applies.
    """
    active = store.list_experiments(user_id, statuses=ACTIVE_STATUSES, limit=20)
    live = False
    for record in active:
        if is_running(record.id):
            live = True
        else:
            store.set_status(
                record.id,
                user_id,
                "failed",
                error=(
                    "Interrupted: no live worker (the application restarted). "
                    "Re-run it or let a schedule create a fresh one."
                ),
            )
    return live


def _skip(schedule: Schedule, reason: str) -> dict:
    return {"id": schedule.id, "name": schedule.name, "reason": reason}


def tick_once(
    settings: Any,
    detector: Any,
    *,
    now: datetime | None = None,
    store: EvaluationStore | None = None,
    sched_store: ScheduleStore | None = None,
    max_runs: int = 1,
    only_user: str | None = None,
    force: bool = False,
) -> TickResult:
    """Check every enabled schedule once and start the due ones.

    ``only_user`` restricts the pass to one account (the Account page's
    "check now"). ``force`` ignores the account's on/off and poll settings for
    this single pass - it is how an explicit manual check runs immediately
    without waiting for the next poll. Dataset validation and the honesty
    rules are never bypassed.

    Pure enough to test: with a stub ``detector`` and a temporary database this
    runs the real due logic and the real job launcher without any weights.
    """
    store = store or EvaluationStore(settings.evaluation_db_path)
    sched_store = sched_store or ScheduleStore(settings.evaluation_db_path)
    now = now or datetime.now(timezone.utc)
    result = TickResult()

    schedules = sched_store.list_enabled()
    if only_user is not None:
        schedules = [item for item in schedules if item.user_id == only_user]
    result.checked = len(schedules)
    if not schedules:
        return result

    default_poll = max(5, int(getattr(settings, "scheduler_tick_seconds", 20)))
    preferences: dict[str, SchedulePreferences] = {}
    active_by_user: dict[str, bool] = {}
    polled_users: set[str] = set()
    started = 0

    for schedule in schedules:
        user_id = schedule.user_id
        prefs = preferences.get(user_id)
        if prefs is None:
            prefs = sched_store.get_preferences(user_id)
            preferences[user_id] = prefs

        if detector is None:
            result.skipped.append(_skip(schedule, "no detector"))
            continue
        if not force and not prefs.enabled:
            result.skipped.append(
                _skip(schedule, "continuous evaluation is switched off for this account")
            )
            continue
        if not force and not poll_due(prefs, now, default_poll):
            result.skipped.append(_skip(schedule, "waiting for the next scheduled check"))
            continue
        polled_users.add(user_id)

        active = active_by_user.get(user_id)
        if active is None:
            active = _has_active_run(store, user_id)
            active_by_user[user_id] = active
        if active:
            result.skipped.append(_skip(schedule, "another run is active"))
            continue

        decision, dataset_fp, model_fps = evaluate_schedule(schedule, detector, now)
        if decision is None:
            continue
        if started >= max_runs:
            result.skipped.append(_skip(schedule, "tick budget reached"))
            continue

        try:
            experiment_id = start_scheduled_run(
                store=store,
                settings=settings,
                sched_store=sched_store,
                schedule=schedule,
                detector=detector,
                trigger=f"{decision.reason}",
                dataset_fingerprint=dataset_fp,
                model_fingerprints=model_fps,
            )
        except ScheduleRunError as exc:
            sched_store.set_last_status(schedule.id, schedule.user_id, "blocked")
            result.blocked.append(
                {"id": schedule.id, "name": schedule.name, "reason": str(exc)}
            )
            logger.warning("Schedule %s blocked: %s", schedule.id, exc)
            continue
        except Exception as exc:  # noqa: BLE001 - one bad schedule must not stop the worker
            sched_store.set_last_status(schedule.id, schedule.user_id, "failed")
            result.blocked.append(
                {"id": schedule.id, "name": schedule.name, "reason": str(exc)}
            )
            logger.exception("Schedule %s failed to start", schedule.id)
            continue

        started += 1
        result.started.append(
            {
                "id": schedule.id,
                "name": schedule.name,
                "experiment_id": experiment_id,
                "reason": decision.reason,
                "detail": decision.detail,
            }
        )

    for user_id in polled_users:
        sched_store.mark_checked(user_id, now.isoformat())
    return result


# --------------------------------------------------------------------------- #
# The worker thread (single, process-wide)
# --------------------------------------------------------------------------- #
_STATE_LOCK = threading.Lock()
_STOP = threading.Event()
_THREAD: threading.Thread | None = None
_SETTINGS: Any = None
_DETECTOR: Any = None
_LAST_TICK: str | None = None
_LAST_SUMMARY: str = "The continuous worker has not checked yet."


def configure(settings: Any, detector: Any, *, start_thread: bool = True) -> None:
    """Register the settings/detector and ensure the worker is running.

    Safe to call on every rerun: the settings and detector references are
    refreshed but only one thread is ever created.
    """
    global _SETTINGS, _DETECTOR, _THREAD
    if not getattr(settings, "scheduler_enabled", True):
        return
    with _STATE_LOCK:
        _SETTINGS = settings
        _DETECTOR = detector
        if not start_thread:
            return
        if _THREAD is not None and _THREAD.is_alive():
            return
        _STOP.clear()
        _THREAD = threading.Thread(
            target=_loop, name="blueeye-evaluation-scheduler", daemon=True
        )
        _THREAD.start()


def shutdown() -> None:
    """Stop the worker (used by tests and clean shutdowns)."""
    global _THREAD
    _STOP.set()
    with _STATE_LOCK:
        thread = _THREAD
        _THREAD = None
    if thread is not None and thread.is_alive():
        thread.join(timeout=2)


def status() -> dict:
    """Live status for the workspace UI (real, never simulated)."""
    with _STATE_LOCK:
        alive = bool(_THREAD is not None and _THREAD.is_alive())
        last = _LAST_TICK
        summary = _LAST_SUMMARY
    return {"running": alive, "last_tick": last, "summary": summary}


def _loop() -> None:
    global _LAST_TICK, _LAST_SUMMARY
    while not _STOP.is_set():
        settings = _SETTINGS
        detector = _DETECTOR
        if settings is None:
            _STOP.wait(5)
            continue
        try:
            result = tick_once(settings, detector)
            _LAST_TICK = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _LAST_SUMMARY = result.summary
        except Exception:  # noqa: BLE001 - the worker must survive any tick
            logger.exception("Evaluation scheduler tick failed")
            _LAST_SUMMARY = "The last check failed; see the application log."
        interval = max(5, int(getattr(settings, "scheduler_tick_seconds", 20)))
        _STOP.wait(interval)


__all__ = [
    "ScheduleRunError",
    "TickResult",
    "compute_dataset_fingerprint",
    "compute_model_fingerprints",
    "configure",
    "evaluate_schedule",
    "shutdown",
    "start_scheduled_run",
    "status",
    "tick_once",
]
