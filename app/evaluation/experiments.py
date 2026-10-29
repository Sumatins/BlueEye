"""Persistent storage for research-evaluation datasets and experiments.

Experiments must survive an application restart, be attributable to the
authenticated owner, and be reproducibly re-openable. This module keeps all of
that in its **own** SQLite database (``outputs/evaluation/``), separate from
the authentication and observations databases, so its schema migrations can
never touch accounts, sessions or saved observations.

Ownership is enforced *inside* every SQL statement: a browser-supplied
experiment id is never trusted on its own. A fresh, short-lived connection is
opened per call (like :mod:`app.auth.store` and :mod:`app.observations.store`)
so the store is safe under Streamlit's concurrent per-browser runners.

Nothing here invents a number: a stored experiment contains exactly the
metrics an evaluator actually produced, plus the provenance needed to judge
them.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

# --------------------------------------------------------------------------- #
# Statuses
# --------------------------------------------------------------------------- #
STATUS_CONFIGURED = "configured"
STATUS_VALIDATING = "validating"
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_PARTIAL = "partially_completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

#: Every legal experiment status, in lifecycle order.
EXPERIMENT_STATUSES: tuple[str, ...] = (
    STATUS_CONFIGURED,
    "validating",
    "queued",
    "running",
    "completed",
    "partially_completed",
    "failed",
    "cancelled",
)

#: Statuses that mean "the run is still in flight".
ACTIVE_STATUSES: tuple[str, ...] = ("validating", "queued", "running")
#: Statuses that will never change again.
TERMINAL_STATUSES: tuple[str, ...] = (
    "completed",
    "partially_completed",
    "failed",
    "cancelled",
)

#: Schema version for the evaluation database (its own ``user_version``).
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    data_yaml   TEXT NOT NULL,
    class_names TEXT NOT NULL DEFAULT '[]',
    splits      TEXT NOT NULL DEFAULT '[]',
    fingerprint TEXT NOT NULL DEFAULT '',
    validation  TEXT NOT NULL DEFAULT '{}',
    notes       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_datasets_user ON datasets(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_datasets_unique
    ON datasets(user_id, data_yaml);

CREATE TABLE IF NOT EXISTS experiments (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL,
    name                TEXT NOT NULL,
    notes               TEXT NOT NULL DEFAULT '',
    purpose             TEXT NOT NULL DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'configured',
    dataset_id          TEXT,
    dataset_name        TEXT NOT NULL DEFAULT '',
    data_yaml           TEXT NOT NULL DEFAULT '',
    split               TEXT NOT NULL DEFAULT '',
    class_names         TEXT NOT NULL DEFAULT '[]',
    dataset_fingerprint TEXT NOT NULL DEFAULT '',
    model_keys          TEXT NOT NULL DEFAULT '[]',
    model_fingerprints  TEXT NOT NULL DEFAULT '{}',
    evaluator_version   TEXT NOT NULL DEFAULT '',
    parameters          TEXT NOT NULL DEFAULT '{}',
    environment         TEXT NOT NULL DEFAULT '{}',
    git_commit          TEXT NOT NULL DEFAULT '',
    progress            TEXT NOT NULL DEFAULT '{}',
    error               TEXT NOT NULL DEFAULT '',
    warnings            TEXT NOT NULL DEFAULT '[]',
    cancel_requested    INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    started_at          TEXT,
    completed_at        TEXT,
    updated_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_exp_user ON experiments(user_id);
CREATE INDEX IF NOT EXISTS idx_eval_exp_created
    ON experiments(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_exp_status
    ON experiments(user_id, status);

CREATE TABLE IF NOT EXISTS experiment_results (
    id            TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    user_id       TEXT NOT NULL,
    model         TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'complete',
    comparable    INTEGER NOT NULL DEFAULT 1,
    summary       TEXT NOT NULL DEFAULT '{}',
    metrics       TEXT NOT NULL DEFAULT '{}',
    model_classes TEXT NOT NULL DEFAULT '[]',
    notes         TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_results_exp
    ON experiment_results(experiment_id);

CREATE TABLE IF NOT EXISTS experiment_artifacts (
    id            TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    user_id       TEXT NOT NULL,
    name          TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'artifact',
    path          TEXT NOT NULL,
    size          INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_artifacts_exp
    ON experiment_artifacts(experiment_id);
"""


class ExperimentError(RuntimeError):
    """Raised for an invalid evaluation-store operation."""


class NotOwnerError(ExperimentError):
    """Raised when a record exists but belongs to a different account."""


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%z")


# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #
def file_fingerprint(path: str | Path) -> dict:
    """A cheap, change-detecting fingerprint of a weights file.

    Uses size + mtime + a hash of the first megabyte: strong enough to notice a
    swapped checkpoint without reading ~90 MB on every run.
    """
    candidate = Path(path)
    info = {"path": candidate.name, "exists": candidate.is_file()}
    if not candidate.is_file():
        return info
    try:
        stat = candidate.stat()
        info["size"] = stat.st_size
        info["mtime"] = int(stat.st_mtime)
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
        info["digest"] = digest.hexdigest()[:32]
    except OSError:
        info["exists"] = False
    return info


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class DatasetRecord:
    """A dataset configuration saved for reuse."""

    id: str
    user_id: str
    name: str
    data_yaml: str
    class_names: tuple[str, ...] = ()
    splits: tuple[str, ...] = ()
    fingerprint: str = ""
    validation: dict = field(default_factory=dict)
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def is_synthetic(self) -> bool:
        return bool(
            (self.validation or {}).get("is_synthetic")
            or any(
                token in self.name.lower() or token in self.data_yaml.lower()
                for token in ("synthetic", "toy", "mock", "dummy")
            )
        )

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "data_yaml": self.data_yaml,
            "class_names": list(self.class_names),
            "splits": list(self.splits),
            "fingerprint": self.fingerprint,
            "validation": self.validation,
            "notes": self.notes,
            "is_synthetic": self.is_synthetic,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ModelResultRecord:
    """The measured outcome of evaluating one model in an experiment."""

    id: str
    experiment_id: str
    user_id: str
    model: str
    status: str = "complete"
    comparable: bool = True
    summary: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    model_classes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    created_at: str = ""

    @property
    def per_class(self) -> tuple[dict, ...]:
        return tuple(self.metrics.get("classes", ()))

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "status": self.status,
            "comparable": self.comparable,
            "summary": self.summary,
            "metrics": self.metrics,
            "model_classes": list(self.model_classes),
            "notes": list(self.notes),
        }


@dataclass
class ArtifactRecord:
    """A generated file (report, plot, CSV, log) tied to an experiment."""

    id: str
    experiment_id: str
    user_id: str
    name: str
    role: str
    path: str
    size: int = 0
    created_at: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "path": self.path,
            "size": self.size,
            "created_at": self.created_at,
        }


@dataclass
class ExperimentRecord:
    """One saved evaluation experiment with its results and artifacts."""

    id: str
    user_id: str
    name: str
    status: str = "configured"
    notes: str = ""
    purpose: str = ""
    dataset_id: str | None = None
    dataset_name: str = ""
    data_yaml: str = ""
    split: str = ""
    class_names: tuple[str, ...] = ()
    dataset_fingerprint: str = ""
    model_keys: tuple[str, ...] = ()
    model_fingerprints: dict = field(default_factory=dict)
    evaluator_version: str = ""
    parameters: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    git_commit: str = ""
    progress: dict = field(default_factory=dict)
    error: str = ""
    warnings: tuple[str, ...] = ()
    cancel_requested: bool = False
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str = ""
    results: tuple[ModelResultRecord, ...] = ()
    artifacts: tuple[ArtifactRecord, ...] = ()

    @property
    def is_active(self) -> bool:
        return self.status in ("validating", "queued", "running")

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "partially_completed", "failed", "cancelled")

    @property
    def mAP50(self) -> float | None:
        values = [
            result.summary.get("mAP50")
            for result in self.results
            if isinstance(result.summary.get("mAP50"), (int, float))
        ]
        evaluated = [value for value in values if value is not None]
        return float(max(evaluated)) if evaluated else None

    @property
    def duration_seconds(self) -> float | None:
        if not self.started_at or not self.completed_at:
            return None
        try:
            from app.evaluation.schedules import parse_timestamp
            start = parse_timestamp(self.started_at)
            end = parse_timestamp(self.completed_at)
            if start and end:
                return max(0.0, (end - start).total_seconds())
        except Exception:
            return None
        return None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "notes": self.notes,
            "purpose": self.purpose,
            "dataset": {
                "id": self.dataset_id,
                "name": self.dataset_name,
                "data_yaml": self.data_yaml,
                "split": self.split,
                "class_names": list(self.class_names),
                "fingerprint": self.dataset_fingerprint,
            },
            "models": list(self.model_keys),
            "model_fingerprints": self.model_fingerprints,
            "evaluator_version": self.evaluator_version,
            "parameters": self.parameters,
            "environment": self.environment,
            "git_commit": self.git_commit,
            "progress": self.progress,
            "error": self.error,
            "warnings": list(self.warnings),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "updated_at": self.updated_at,
            "results": [result.as_dict() for result in self.results],
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
        }


# --------------------------------------------------------------------------- #
# Row conversion
# --------------------------------------------------------------------------- #
def _loads(value, default):
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _dataset_row(row: sqlite3.Row) -> DatasetRecord:
    return DatasetRecord(
        id=row["id"],
        user_id=row["user_id"],
        name=row["name"],
        data_yaml=row["data_yaml"],
        class_names=tuple(_loads(row["class_names"], [])),
        splits=tuple(_loads(row["splits"], [])),
        fingerprint=row["fingerprint"],
        validation=_loads(row["validation"], {}),
        notes=row["notes"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _result_row(row: sqlite3.Row) -> ModelResultRecord:
    return ModelResultRecord(
        id=row["id"],
        experiment_id=row["experiment_id"],
        user_id=row["user_id"],
        model=row["model"],
        status=row["status"],
        comparable=bool(row["comparable"]),
        summary=_loads(row["summary"], {}),
        metrics=_loads(row["metrics"], {}),
        model_classes=tuple(_loads(row["model_classes"], [])),
        notes=tuple(_loads(row["notes"], [])),
        created_at=row["created_at"],
    )


def _artifact_row(row: sqlite3.Row) -> ArtifactRecord:
    return ArtifactRecord(
        id=row["id"],
        experiment_id=row["experiment_id"],
        user_id=row["user_id"],
        name=row["name"],
        role=row["role"],
        path=row["path"],
        size=row["size"],
        created_at=row["created_at"],
    )


def _experiment_row(
    row: sqlite3.Row,
    results: tuple[ModelResultRecord, ...] = (),
    artifacts: tuple[ArtifactRecord, ...] = (),
) -> ExperimentRecord:
    return ExperimentRecord(
        id=row["id"],
        user_id=row["user_id"],
        name=row["name"],
        status=row["status"],
        notes=row["notes"],
        purpose=row["purpose"],
        dataset_id=row["dataset_id"],
        dataset_name=row["dataset_name"],
        data_yaml=row["data_yaml"],
        split=row["split"],
        class_names=tuple(_loads(row["class_names"], [])),
        dataset_fingerprint=row["dataset_fingerprint"],
        model_keys=tuple(_loads(row["model_keys"], [])),
        model_fingerprints=_loads(row["model_fingerprints"], {}),
        evaluator_version=row["evaluator_version"],
        parameters=_loads(row["parameters"], {}),
        environment=_loads(row["environment"], {}),
        git_commit=row["git_commit"],
        progress=_loads(row["progress"], {}),
        error=row["error"],
        warnings=tuple(_loads(row["warnings"], [])),
        cancel_requested=bool(row["cancel_requested"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        updated_at=row["updated_at"],
        results=results,
        artifacts=artifacts,
    )


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class EvaluationStore:
    """SQLite-backed store for evaluation datasets, experiments and artifacts."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # -- connection handling -------------------------------------------- #
    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=15000")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version < SCHEMA_VERSION:
                conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    # -- datasets -------------------------------------------------------- #
    def register_dataset(
        self,
        user_id: str,
        name: str,
        data_yaml: str | Path,
        *,
        class_names: tuple[str, ...] = (),
        splits: tuple[str, ...] = (),
        fingerprint: str = "",
        validation: dict | None = None,
        notes: str = "",
    ) -> DatasetRecord:
        """Create or update this account's config for ``data_yaml``."""
        now = _utcnow()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM datasets WHERE user_id = ? AND data_yaml = ?",
                (user_id, str(data_yaml)),
            ).fetchone()
            if existing is not None:
                record_id = existing["id"]
                conn.execute(
                    "UPDATE datasets SET name=?, class_names=?, splits=?, "
                    "fingerprint=?, validation=?, notes=?, updated_at=? "
                    "WHERE id=? AND user_id=?",
                    (
                        name,
                        json.dumps(list(class_names)),
                        json.dumps(list(splits)),
                        fingerprint,
                        json.dumps(validation or {}),
                        notes,
                        now,
                        record_id,
                        user_id,
                    ),
                )
            else:
                record_id = f"ds_{uuid.uuid4().hex}"
                conn.execute(
                    "INSERT INTO datasets (id, user_id, name, data_yaml, "
                    "class_names, splits, fingerprint, validation, notes, "
                    "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        record_id,
                        user_id,
                        name,
                        str(data_yaml),
                        json.dumps(list(class_names)),
                        json.dumps(list(splits)),
                        fingerprint,
                        json.dumps(validation or {}),
                        notes,
                        now,
                        now,
                    ),
                )
            row = conn.execute(
                "SELECT * FROM datasets WHERE id = ?", (record_id,)
            ).fetchone()
        return _dataset_row(row)

    def list_datasets(self, user_id: str) -> list[DatasetRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM datasets WHERE user_id = ? ORDER BY name COLLATE NOCASE",
                (user_id,),
            ).fetchall()
        return [_dataset_row(row) for row in rows]

    def get_dataset(self, dataset_id: str, user_id: str) -> DatasetRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE id = ? AND user_id = ?",
                (dataset_id, user_id),
            ).fetchone()
        return _dataset_row(row) if row else None

    def delete_dataset(self, dataset_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM datasets WHERE id = ? AND user_id = ?",
                (dataset_id, user_id),
            )
            return cursor.rowcount > 0

    # -- experiments ---------------------------------------------------- #
    def create_experiment(
        self,
        user_id: str,
        name: str,
        *,
        notes: str = "",
        purpose: str = "",
        dataset_id: str | None = None,
        dataset_name: str = "",
        data_yaml: str = "",
        split: str = "",
        class_names: tuple[str, ...] = (),
        dataset_fingerprint: str = "",
        model_keys: tuple[str, ...] = (),
        model_fingerprints: dict | None = None,
        evaluator_version: str = "",
        parameters: dict | None = None,
        environment: dict | None = None,
        git_commit: str = "",
        status: str = "configured",
    ) -> ExperimentRecord:
        if status not in (
            "configured",
            "validating",
            "queued",
            "running",
            "completed",
            "partially_completed",
            "failed",
            "cancelled",
        ):
            raise ExperimentError(f"Unsupported experiment status: {status!r}")
        exp_id = f"exp_{uuid.uuid4().hex}"
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO experiments (id, user_id, name, notes, purpose, "
                "status, dataset_id, dataset_name, data_yaml, split, class_names, "
                "dataset_fingerprint, model_keys, model_fingerprints, "
                "evaluator_version, parameters, environment, git_commit, "
                "progress, error, warnings, cancel_requested, created_at, "
                "started_at, completed_at, updated_at) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    exp_id,
                    user_id,
                    name,
                    notes,
                    purpose,
                    status,
                    dataset_id,
                    dataset_name,
                    data_yaml,
                    split,
                    json.dumps(list(class_names)),
                    dataset_fingerprint,
                    json.dumps(list(model_keys)),
                    json.dumps(model_fingerprints or {}),
                    evaluator_version,
                    json.dumps(parameters or {}),
                    json.dumps(environment or {}),
                    git_commit,
                    "{}",
                    "",
                    "[]",
                    0,
                    now,
                    now if status in ("validating", "queued", "running") else None,
                    None,
                    now,
                ),
            )
        record = self.get_experiment(exp_id, user_id)
        if record is None:  # pragma: no cover - defensive
            raise ExperimentError("Experiment was not persisted.")
        return record

    def set_status(
        self,
        experiment_id: str,
        user_id: str,
        status: str,
        *,
        error: str | None = None,
        progress: dict | None = None,
        warnings: tuple[str, ...] | None = None,
    ) -> bool:
        """Transition an experiment's status (owner-checked)."""
        if status not in (
            "configured",
            "validating",
            "queued",
            "running",
            "completed",
            "partially_completed",
            "failed",
            "cancelled",
        ):
            raise ExperimentError(f"Unknown status: {status!r}")
        fields = ["status = ?", "updated_at = ?"]
        values: list = [status, _utcnow()]
        if status in ("validating", "queued", "running"):
            fields.append("started_at = COALESCE(started_at, ?)")
            values.append(_utcnow())
        if status in ("completed", "partially_completed", "failed", "cancelled"):
            fields.append("completed_at = ?")
            values.append(_utcnow())
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if progress is not None:
            fields.append("progress = ?")
            values.append(json.dumps(progress))
        if warnings is not None:
            fields.append("warnings = ?")
            values.append(json.dumps(list(warnings)))
        values.extend([experiment_id, user_id])
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE experiments SET {', '.join(fields)} "
                "WHERE id = ? AND user_id = ?",
                tuple(values),
            )
            return cursor.rowcount > 0

    def update_parameters(
        self, experiment_id: str, user_id: str, parameters: dict
    ) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE experiments SET parameters = ?, updated_at = ? "
                "WHERE id = ? AND user_id = ?",
                (json.dumps(parameters), _utcnow(), experiment_id, user_id),
            )
            return cursor.rowcount > 0

    def add_model_result(self, experiment_id: str, user_id: str, evaluation) -> None:
        """Persist one :class:`~app.evaluation.runner.ModelEvaluation`."""
        from app.evaluation.runner import summarise

        try:
            summary = summarise(evaluation)
        except Exception:  # noqa: BLE001 - keep storage best-effort but honest
            summary = {"model": getattr(evaluation, "model", "")}
        try:
            metrics = evaluation.metrics.as_dict()
        except Exception:  # noqa: BLE001
            metrics = {}
        record = ModelResultRecord(
            id=f"res_{uuid.uuid4().hex}",
            experiment_id=experiment_id,
            user_id=user_id,
            model=str(getattr(evaluation, "model", "")),
            status=str(getattr(evaluation, "status", "complete")),
            comparable=bool(getattr(getattr(evaluation, "compat", None), "comparable", True)),
            summary=summary,
            metrics=metrics,
            model_classes=tuple(str(v) for v in getattr(evaluation, "model_classes", ())),
            notes=tuple(str(v) for v in getattr(evaluation, "notes", ())),
            created_at=_utcnow(),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO experiment_results (id, experiment_id, user_id, "
                "model, status, comparable, summary, metrics, model_classes, "
                "notes, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.id,
                    experiment_id,
                    user_id,
                    record.model,
                    record.status,
                    int(record.comparable),
                    json.dumps(record.summary),
                    json.dumps(record.metrics),
                    json.dumps(list(record.model_classes)),
                    json.dumps(list(record.notes)),
                    record.created_at,
                ),
            )

    def add_artifact(
        self,
        experiment_id: str,
        user_id: str,
        path: str | Path,
        *,
        name: str = "",
        role: str = "artifact",
    ) -> ArtifactRecord:
        candidate = Path(path)
        size = candidate.stat().st_size if candidate.is_file() else 0
        record = ArtifactRecord(
            id=f"art_{uuid.uuid4().hex}",
            experiment_id=experiment_id,
            user_id=user_id,
            name=name or candidate.name,
            role=role,
            path=str(candidate),
            size=size,
            created_at=_utcnow(),
        )
        with self._connect() as conn:
            # Only the owner may attach an artifact to the experiment.
            owned = conn.execute(
                "SELECT 1 FROM experiments WHERE id = ? AND user_id = ?",
                (experiment_id, user_id),
            ).fetchone()
            if owned is None:
                raise NotOwnerError("No such experiment for this account.")
            conn.execute(
                "INSERT INTO experiment_artifacts (id, experiment_id, user_id, "
                "name, role, path, size, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    record.id,
                    record.experiment_id,
                    record.user_id,
                    record.name,
                    record.role,
                    record.path,
                    record.size,
                    record.created_at,
                ),
            )
        return record

    def get_experiment(self, experiment_id: str, user_id: str) -> ExperimentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ? AND user_id = ?",
                (experiment_id, user_id),
            ).fetchone()
            if row is None:
                return None
            results = tuple(
                _result_row(item)
                for item in conn.execute(
                    "SELECT * FROM experiment_results WHERE experiment_id = ? "
                    "ORDER BY model",
                    (experiment_id,),
                ).fetchall()
            )
            artifacts = tuple(
                _artifact_row(item)
                for item in conn.execute(
                    "SELECT * FROM experiment_artifacts WHERE experiment_id = ? "
                    "ORDER BY created_at",
                    (experiment_id,),
                ).fetchall()
            )
        return _experiment_row(row, results, artifacts)

    def list_experiments(
        self,
        user_id: str,
        *,
        statuses: tuple[str, ...] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[ExperimentRecord, ...]:
        query = "SELECT * FROM experiments WHERE user_id = ?"
        params: list = [user_id]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
        params.extend([max(1, int(limit)), max(0, int(offset))])
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            experiments = []
            for row in rows:
                results = tuple(
                    _result_row(item)
                    for item in conn.execute(
                        "SELECT * FROM experiment_results WHERE experiment_id = ? "
                        "ORDER BY model",
                        (row["id"],),
                    ).fetchall()
                )
                artifacts = tuple(
                    _artifact_row(item)
                    for item in conn.execute(
                        "SELECT * FROM experiment_artifacts WHERE experiment_id = ? "
                        "ORDER BY created_at",
                        (row["id"],),
                    ).fetchall()
                )
                experiments.append(_experiment_row(row, results, artifacts))
        return tuple(experiments)

    def count_experiments(
        self, user_id: str, *, statuses: tuple[str, ...] | None = None
    ) -> int:
        query = "SELECT COUNT(*) FROM experiments WHERE user_id = ?"
        params: list = [user_id]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(statuses)
        with self._connect() as conn:
            return int(conn.execute(query, params).fetchone()[0])

    def stats(self, user_id: str) -> dict:
        """Real, user-scoped counts for the overview panel."""
        with self._connect() as conn:
            def _count(where: str, extra: tuple = ()) -> int:
                return int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM experiments WHERE user_id = ? {where}",
                        (user_id, *extra),
                    ).fetchone()[0]
                )

            datasets = int(
                conn.execute(
                    "SELECT COUNT(*) FROM datasets WHERE user_id = ?", (user_id,)
                ).fetchone()[0]
            )
            experiments = _count("")
            active = int(
                conn.execute(
                    "SELECT COUNT(*) FROM experiments WHERE user_id = ? AND status "
                    "IN ('validating','queued','running')",
                    (user_id,),
                ).fetchone()[0]
            )
            completed = _count("AND status = 'completed'")
            partial = _count("AND status = 'partially_completed'")
            failed = _count("AND status = 'failed'")
            cancelled = _count("AND status = 'cancelled'")
            last_row = conn.execute(
                "SELECT * FROM experiments WHERE user_id = ? AND status IN "
                "('completed','partially_completed') ORDER BY completed_at DESC, "
                "rowid DESC LIMIT 1",
                (user_id,),
            ).fetchone()
            last_completed = _experiment_row(last_row) if last_row else None
        return {
            "datasets": datasets,
            "experiments": experiments,
            "active": active,
            "completed": completed,
            "partial": partial,
            "failed": failed,
            "cancelled": cancelled,
            "last_completed": last_completed,
        }

    def delete_experiment(self, experiment_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM experiments WHERE id = ? AND user_id = ?",
                (experiment_id, user_id),
            )
            return cursor.rowcount > 0

    def request_cancel(self, experiment_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE experiments SET cancel_requested = 1, updated_at = ? "
                "WHERE id = ? AND user_id = ?",
                (_utcnow(), experiment_id, user_id),
            )
            return cursor.rowcount > 0

    def is_cancel_requested(self, experiment_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cancel_requested FROM experiments WHERE id = ? AND user_id = ?",
                (experiment_id, user_id),
            ).fetchone()
        return bool(row and row["cancel_requested"])
