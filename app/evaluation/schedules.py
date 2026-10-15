"""Continuous / scheduled research evaluation: persisted schedules.

A *schedule* is a saved evaluation recipe plus a trigger. The Research
Evaluation Center uses it to re-run an experiment automatically while the
application process is alive:

* ``on_change`` - re-run when the dataset content or the model weights
  change (detected from real fingerprints, never by guessing);
* ``interval``  - re-run every N minutes;
* ``both``      - either of the above, whichever happens first.

This module owns only the persistence and the *pure* due decision. It never
runs a model and never invents a result: :func:`due_reason` is a function of
stored state plus freshly computed fingerprints, so it is fully unit-testable
without weights. Execution lives in :mod:`app.evaluation.scheduler`.

The schedules live in their own table in the evaluation database so an
existing database is extended in place (``CREATE TABLE IF NOT EXISTS``) and no
migration of the existing experiment history is required.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

#: Supported trigger modes, with a human label for the UI.
TRIGGER_MODES: tuple[str, ...] = ("on_change", "interval", "both")
TRIGGER_LABELS = {
    "on_change": "On dataset / model change",
    "interval": "On a fixed interval",
    "both": "On change or interval",
}

#: Guard rails for the interval (minutes).
MIN_INTERVAL_MINUTES = 1
MAX_INTERVAL_MINUTES = 60 * 24 * 30  # 30 days

#: Guard rails for a per-account "how often to check" poll (seconds).
MIN_POLL_SECONDS = 5
MAX_POLL_SECONDS = 3600
DEFAULT_POLL_SECONDS = 20

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluation_schedules (
    id                        TEXT PRIMARY KEY,
    user_id                   TEXT NOT NULL,
    name                      TEXT NOT NULL,
    data_yaml                 TEXT NOT NULL,
    dataset_name              TEXT NOT NULL DEFAULT '',
    split                     TEXT NOT NULL DEFAULT 'test',
    class_names               TEXT NOT NULL DEFAULT '[]',
    model_keys                TEXT NOT NULL DEFAULT '[]',
    parameters                TEXT NOT NULL DEFAULT '{}',
    trigger_mode              TEXT NOT NULL DEFAULT 'on_change',
    interval_minutes          INTEGER NOT NULL DEFAULT 60,
    enabled                   INTEGER NOT NULL DEFAULT 1,
    last_dataset_fingerprint  TEXT NOT NULL DEFAULT '',
    last_model_fingerprints   TEXT NOT NULL DEFAULT '{}',
    last_run_at               TEXT,
    last_experiment_id        TEXT,
    last_status               TEXT NOT NULL DEFAULT '',
    last_error                TEXT NOT NULL DEFAULT '',
    run_count                 INTEGER NOT NULL DEFAULT 0,
    created_at                TEXT NOT NULL,
    updated_at                TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_schedules_user
    ON evaluation_schedules(user_id);
CREATE INDEX IF NOT EXISTS idx_eval_schedules_enabled
    ON evaluation_schedules(enabled);
"""

#: Per-account settings for the continuous worker: whether the account's
#: schedules may run on their own, how often they are checked, and when they
#: were last checked. Absent rows fall back to the defaults.
_PREFERENCES_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedule_preferences (
    user_id         TEXT PRIMARY KEY,
    enabled         INTEGER NOT NULL DEFAULT 1,
    poll_seconds    INTEGER NOT NULL DEFAULT 20,
    last_checked_at TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""

#: Editable columns (never the id, owner or bookkeeping fields directly).
_EDITABLE = {
    "name",
    "data_yaml",
    "dataset_name",
    "split",
    "class_names",
    "model_keys",
    "parameters",
    "trigger_mode",
    "interval_minutes",
    "enabled",
}


class ScheduleError(RuntimeError):
    """Raised for an invalid schedule operation."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse a stored ISO timestamp, tolerating the ``...Z`` suffix."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _canonical(value: Any) -> str:
    """Stable JSON for fingerprint comparison (order-independent)."""
    return json.dumps(value or {}, sort_keys=True, default=str)


def _clamp_interval(minutes: int) -> int:
    try:
        minutes = int(minutes)
    except (TypeError, ValueError):
        minutes = 60
    return max(MIN_INTERVAL_MINUTES, min(MAX_INTERVAL_MINUTES, minutes))


def _clamp_poll(seconds: int) -> int:
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        seconds = DEFAULT_POLL_SECONDS
    return max(MIN_POLL_SECONDS, min(MAX_POLL_SECONDS, seconds))


@dataclass(frozen=True)
class DueDecision:
    """Why a schedule should run now (or ``None`` when it should not)."""

    reason: str  # "first run" | "interval" | "change"
    detail: str = ""


@dataclass
class Schedule:
    """A saved, possibly recurring evaluation recipe."""

    id: str
    user_id: str
    name: str
    data_yaml: str
    dataset_name: str = ""
    split: str = "test"
    class_names: tuple[str, ...] = ()
    model_keys: tuple[str, ...] = ()
    parameters: dict = field(default_factory=dict)
    trigger_mode: str = "on_change"
    interval_minutes: int = 60
    enabled: bool = True
    last_dataset_fingerprint: str = ""
    last_model_fingerprints: dict = field(default_factory=dict)
    last_run_at: str | None = None
    last_experiment_id: str | None = None
    last_status: str = ""
    last_error: str = ""
    run_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    @property
    def trigger_label(self) -> str:
        return TRIGGER_LABELS.get(self.trigger_mode, self.trigger_mode)

    def next_due_hint(self, now: datetime | None = None) -> str:
        """A human sentence describing when this schedule will next fire."""
        if not self.enabled:
            return "Paused"
        if self.last_run_at is None:
            return "Due now - has not run yet"
        if self.trigger_mode in ("interval", "both"):
            last = parse_timestamp(self.last_run_at)
            if last is not None:
                now = now or datetime.now(timezone.utc)
                elapsed = (now - last).total_seconds()
                remaining = self.interval_minutes * 60 - elapsed
                if remaining <= 0:
                    return "Due now - interval elapsed"
                minutes = int(remaining // 60)
                if minutes >= 60:
                    return f"Next check in ~{minutes // 60}h {minutes % 60}m"
                return f"Next check in ~{max(minutes, 1)}m"
        if self.trigger_mode == "on_change":
            return "Waiting for a dataset / model change"
        return "Waiting"

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "data_yaml": self.data_yaml,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "class_names": list(self.class_names),
            "model_keys": list(self.model_keys),
            "parameters": self.parameters,
            "trigger_mode": self.trigger_mode,
            "interval_minutes": self.interval_minutes,
            "enabled": self.enabled,
            "last_run_at": self.last_run_at,
            "last_experiment_id": self.last_experiment_id,
            "last_status": self.last_status,
            "last_error": self.last_error,
            "run_count": self.run_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class SchedulePreferences:
    """Per-account settings for the continuous worker.

    ``enabled`` is the account's master switch (when off, the worker ignores
    the account's schedules). ``poll_seconds`` is the minimum time between
    checks of this account's schedules; ``last_checked_at`` records when a
    check actually happened, so the poll is honoured across restarts.
    """

    user_id: str
    enabled: bool = True
    poll_seconds: int = DEFAULT_POLL_SECONDS
    last_checked_at: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "enabled": self.enabled,
            "poll_seconds": self.poll_seconds,
            "last_checked_at": self.last_checked_at,
        }


def poll_due(
    preferences: SchedulePreferences,
    now: datetime,
    default_seconds: int = DEFAULT_POLL_SECONDS,
) -> bool:
    """True when enough time has passed since this account was last checked.

    Timing only - the account's ``enabled`` flag is checked separately by the
    caller so the two gates stay independent and easy to reason about.
    """
    interval = _clamp_poll(preferences.poll_seconds or default_seconds)
    last = parse_timestamp(preferences.last_checked_at)
    if last is None:
        return True
    return (now - last).total_seconds() >= interval


def due_reason(
    schedule: Schedule,
    now: datetime,
    dataset_fingerprint: str,
    model_fingerprints: dict,
) -> DueDecision | None:
    """Decide whether ``schedule`` should run now.

    ``dataset_fingerprint`` and ``model_fingerprints`` are computed from the
    real files on disk by the caller. Passing empty values simply means
    "unknown", in which case that signal cannot trigger a run.
    """
    if not schedule.enabled:
        return None

    if schedule.last_run_at is None:
        return DueDecision("first run", "This schedule has not run yet.")

    changed: list[str] = []
    if dataset_fingerprint and dataset_fingerprint != schedule.last_dataset_fingerprint:
        changed.append("the dataset content changed")
    if (
        model_fingerprints
        and _canonical(model_fingerprints) != _canonical(schedule.last_model_fingerprints)
    ):
        changed.append("the model weights changed")

    last = parse_timestamp(schedule.last_run_at)
    interval_due = False
    if last is not None and schedule.trigger_mode in ("interval", "both"):
        if (now - last).total_seconds() >= schedule.interval_minutes * 60:
            interval_due = True

    if schedule.trigger_mode in ("interval", "both") and interval_due:
        detail = (
            f"Interval of {schedule.interval_minutes} minute(s) elapsed since "
            "the last run."
        )
        if changed:
            detail += " (" + " and ".join(changed) + ")"
        return DueDecision("interval", detail)

    if schedule.trigger_mode in ("on_change", "both") and changed:
        return DueDecision("change", "Since the last run, " + " and ".join(changed) + ".")

    return None


class ScheduleStore:
    """SQLite-backed store for continuous-evaluation schedules."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=15000")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.executescript(_PREFERENCES_SCHEMA)
            try:
                conn.execute(
                    "ALTER TABLE evaluation_schedules ADD COLUMN last_error TEXT NOT NULL DEFAULT ''"
                )
            except sqlite3.OperationalError:
                pass

    # -- create / read --------------------------------------------------- #
    def create(
        self,
        user_id: str,
        name: str,
        data_yaml: str | Path,
        *,
        dataset_name: str = "",
        split: str = "test",
        class_names: tuple[str, ...] = (),
        model_keys: tuple[str, ...] = (),
        parameters: dict | None = None,
        trigger_mode: str = "on_change",
        interval_minutes: int = 60,
        enabled: bool = True,
    ) -> Schedule:
        if trigger_mode not in TRIGGER_MODES:
            raise ScheduleError(f"Unsupported trigger mode: {trigger_mode!r}")
        schedule_id = f"sch_{uuid.uuid4().hex}"
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO evaluation_schedules (id, user_id, name, data_yaml, "
                "dataset_name, split, class_names, model_keys, parameters, "
                "trigger_mode, interval_minutes, enabled, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    schedule_id,
                    user_id,
                    name,
                    str(data_yaml),
                    dataset_name,
                    split,
                    json.dumps(list(class_names)),
                    json.dumps(list(model_keys)),
                    json.dumps(parameters or {}),
                    trigger_mode,
                    _clamp_interval(interval_minutes),
                    int(bool(enabled)),
                    now,
                    now,
                ),
            )
        record = self.get(schedule_id, user_id)
        if record is None:  # pragma: no cover - defensive
            raise ScheduleError("Schedule was not persisted.")
        return record

    def get(self, schedule_id: str, user_id: str) -> Schedule | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evaluation_schedules WHERE id = ? AND user_id = ?",
                (schedule_id, user_id),
            ).fetchone()
        return _row(row) if row else None

    def list(self, user_id: str) -> list[Schedule]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evaluation_schedules WHERE user_id = ? "
                "ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [_row(row) for row in rows]

    def list_enabled(self) -> list[Schedule]:
        """Every enabled schedule across all accounts (used by the worker)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evaluation_schedules WHERE enabled = 1 "
                "ORDER BY created_at"
            ).fetchall()
        return [_row(row) for row in rows]

    # -- update / delete ------------------------------------------------- #
    def update(self, schedule_id: str, user_id: str, **fields) -> Schedule | None:
        updates: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            if key not in _EDITABLE:
                raise ScheduleError(f"Cannot edit schedule field {key!r}")
            if key in ("class_names", "model_keys"):
                value = json.dumps(list(value))
            elif key == "parameters":
                value = json.dumps(value or {})
            elif key == "enabled":
                value = int(bool(value))
            elif key == "interval_minutes":
                value = _clamp_interval(value)
            elif key == "trigger_mode" and value not in TRIGGER_MODES:
                raise ScheduleError(f"Unsupported trigger mode: {value!r}")
            updates.append(f"{key} = ?")
            values.append(value)
        if not updates:
            return self.get(schedule_id, user_id)
        updates.append("updated_at = ?")
        values.append(_utcnow())
        values.extend([schedule_id, user_id])
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE evaluation_schedules SET {', '.join(updates)} "
                "WHERE id = ? AND user_id = ?",
                tuple(values),
            )
        return self.get(schedule_id, user_id) if cursor.rowcount else None

    def set_enabled(self, schedule_id: str, user_id: str, enabled: bool) -> bool:
        return self.update(schedule_id, user_id, enabled=enabled) is not None

    def delete(self, schedule_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM evaluation_schedules WHERE id = ? AND user_id = ?",
                (schedule_id, user_id),
            )
            return cursor.rowcount > 0

    # -- run bookkeeping ------------------------------------------------- #
    def record_run(
        self,
        schedule_id: str,
        user_id: str,
        *,
        experiment_id: str,
        status: str,
        dataset_fingerprint: str,
        model_fingerprints: dict,
        ran_at: str | None = None,
        error: str = "",
    ) -> None:
        """Record that a run was started, with the fingerprints it used.

        The fingerprints are stored from the *actual* files at run time so a
        later change is detected honestly rather than by assuming nothing
        changed.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE evaluation_schedules SET last_run_at = ?, "
                "last_experiment_id = ?, last_status = ?, last_error = ?, "
                "last_dataset_fingerprint = ?, last_model_fingerprints = ?, "
                "run_count = run_count + 1, updated_at = ? "
                "WHERE id = ? AND user_id = ?",
                (
                    ran_at or _utcnow(),
                    experiment_id,
                    status,
                    error,
                    dataset_fingerprint,
                    _canonical(model_fingerprints),
                    _utcnow(),
                    schedule_id,
                    user_id,
                ),
            )

    def set_last_status(
        self, schedule_id: str, user_id: str, status: str, error: str = ""
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE evaluation_schedules SET last_status = ?, last_error = ?, updated_at = ? "
                "WHERE id = ? AND user_id = ?",
                (status, error, _utcnow(), schedule_id, user_id),
            )

    # -- per-account continuous-evaluation preferences ------------------- #
    def get_preferences(self, user_id: str) -> SchedulePreferences:
        """The account's worker settings, or the defaults when never set."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM schedule_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return _prefs_row(row) if row else SchedulePreferences(user_id=user_id)

    def set_preferences(
        self,
        user_id: str,
        *,
        enabled: bool | None = None,
        poll_seconds: int | None = None,
    ) -> SchedulePreferences:
        """Create or update the account's worker settings.

        ``None`` leaves a value unchanged. The poll interval is clamped to the
        supported range so the worker can never be asked to spin.
        """
        existing = self.get_preferences(user_id)
        new_enabled = existing.enabled if enabled is None else bool(enabled)
        new_poll = (
            existing.poll_seconds
            if poll_seconds is None
            else _clamp_poll(poll_seconds)
        )
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO schedule_preferences "
                "(user_id, enabled, poll_seconds, last_checked_at, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET enabled = excluded.enabled, "
                "poll_seconds = excluded.poll_seconds, updated_at = excluded.updated_at",
                (
                    user_id,
                    int(new_enabled),
                    new_poll,
                    existing.last_checked_at,
                    existing.created_at or now,
                    now,
                ),
            )
        return self.get_preferences(user_id)

    def mark_checked(self, user_id: str, when: str | None = None) -> None:
        """Record that this account's schedules were checked (poll bookkeeping)."""
        now = when or _utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO schedule_preferences "
                "(user_id, enabled, poll_seconds, last_checked_at, created_at, updated_at) "
                "VALUES (?, 1, ?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "last_checked_at = excluded.last_checked_at, updated_at = excluded.updated_at",
                (user_id, DEFAULT_POLL_SECONDS, now, now, now),
            )


def _row(row: sqlite3.Row) -> Schedule:
    def loads(value, default):
        if value in (None, ""):
            return default
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default

    return Schedule(
        id=row["id"],
        user_id=row["user_id"],
        name=row["name"],
        data_yaml=row["data_yaml"],
        dataset_name=row["dataset_name"],
        split=row["split"],
        class_names=tuple(loads(row["class_names"], [])),
        model_keys=tuple(loads(row["model_keys"], [])),
        parameters=loads(row["parameters"], {}),
        trigger_mode=row["trigger_mode"],
        interval_minutes=row["interval_minutes"],
        enabled=bool(row["enabled"]),
        last_dataset_fingerprint=row["last_dataset_fingerprint"],
        last_model_fingerprints=loads(row["last_model_fingerprints"], {}),
        last_run_at=row["last_run_at"],
        last_experiment_id=row["last_experiment_id"],
        last_status=row["last_status"],
        last_error=str(row["last_error"] if "last_error" in row.keys() else ""),
        run_count=row["run_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _prefs_row(row: sqlite3.Row) -> SchedulePreferences:
    return SchedulePreferences(
        user_id=row["user_id"],
        enabled=bool(row["enabled"]),
        poll_seconds=row["poll_seconds"],
        last_checked_at=row["last_checked_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


__all__ = [
    "DEFAULT_POLL_SECONDS",
    "DueDecision",
    "MAX_INTERVAL_MINUTES",
    "MAX_POLL_SECONDS",
    "MIN_INTERVAL_MINUTES",
    "MIN_POLL_SECONDS",
    "Schedule",
    "ScheduleError",
    "SchedulePreferences",
    "ScheduleStore",
    "TRIGGER_LABELS",
    "TRIGGER_MODES",
    "due_reason",
    "parse_timestamp",
    "poll_due",
]
