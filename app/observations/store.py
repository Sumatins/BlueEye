"""SQLite persistence for saved observation records.

Every operation takes the authenticated account id and constrains its SQL to
that id: ownership is enforced in the query itself, never by a browser-supplied
value. A fresh connection is opened per call (like :mod:`app.auth.store`) so
the store is safe under Streamlit's concurrent per-browser runners; WAL
journaling and ``foreign_keys`` are enabled on every connection.

The observations live in their own database file (``outputs/observations/``),
separate from the authentication database, so schema migrations here can never
interfere with accounts or sessions.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.observations.models import (
    REVIEW_STATUSES,
    Analysis,
    Collection,
    Correction,
    DetectionRecord,
    DuplicateObservationError,
    NotOwnerError,
    ObservationError,
    _utcnow,
)

logger = logging.getLogger(__name__)

#: Schema version for the observations database (its own ``user_version``).
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id                   TEXT PRIMARY KEY,
    user_id              TEXT NOT NULL,
    kind                 TEXT NOT NULL,
    source_filename      TEXT NOT NULL,
    original_path        TEXT NOT NULL,
    annotated_path       TEXT,
    report_path          TEXT,
    models               TEXT NOT NULL DEFAULT '[]',
    confidence_threshold REAL,
    enhanced             INTEGER NOT NULL DEFAULT 0,
    total_detections     INTEGER NOT NULL DEFAULT 0,
    unique_classes       INTEGER NOT NULL DEFAULT 0,
    average_confidence   REAL NOT NULL DEFAULT 0,
    max_confidence       REAL NOT NULL DEFAULT 0,
    frames_processed     INTEGER,
    elapsed_seconds      REAL,
    model_version        TEXT,
    detected_at          TEXT NOT NULL,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    notes                TEXT NOT NULL DEFAULT '',
    project              TEXT NOT NULL DEFAULT '',
    location             TEXT NOT NULL DEFAULT '',
    habitat              TEXT NOT NULL DEFAULT '',
    review_status        TEXT NOT NULL DEFAULT 'predicted'
);
CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(user_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_analyses_dedupe
    ON analyses(user_id, original_path, COALESCE(annotated_path, ''));

CREATE TABLE IF NOT EXISTS detections (
    id           TEXT PRIMARY KEY,
    analysis_id  TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    user_id      TEXT NOT NULL,
    class_name   TEXT NOT NULL,
    confidence   REAL NOT NULL,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    model        TEXT NOT NULL DEFAULT '',
    frame_index  INTEGER,
    frame_time   REAL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_analysis ON detections(analysis_id);
CREATE INDEX IF NOT EXISTS idx_detections_user ON detections(user_id);

CREATE TABLE IF NOT EXISTS corrections (
    id               TEXT PRIMARY KEY,
    detection_id     TEXT NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
    analysis_id      TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    user_id          TEXT NOT NULL,
    original_label   TEXT NOT NULL,
    corrected_label  TEXT NOT NULL,
    note             TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corrections_detection ON corrections(detection_id);
CREATE INDEX IF NOT EXISTS idx_corrections_analysis ON corrections(analysis_id);

CREATE TABLE IF NOT EXISTS collections (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_collections_user ON collections(user_id);

CREATE TABLE IF NOT EXISTS collection_items (
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    analysis_id   TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    added_at      TEXT NOT NULL,
    PRIMARY KEY (collection_id, analysis_id)
);
CREATE INDEX IF NOT EXISTS idx_collection_items_analysis ON collection_items(analysis_id);
"""


class ObservationStore:
    """SQLite-backed store for analyses, detections, collections and review."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ------------------------------------------------------------------ #
    # Connection handling
    # ------------------------------------------------------------------ #
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
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
            if version < 1:
                conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    # ------------------------------------------------------------------ #
    # Analyses
    # ------------------------------------------------------------------ #
    def find_existing(
        self, user_id: str, original_path: str, annotated_path: str | None
    ) -> Analysis | None:
        """The already-saved analysis for this exact media, if any.

        Used to guarantee that saving never runs inference again and that the
        same run is not stored twice for one account.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM analyses WHERE user_id = ? AND original_path = ? "
                "AND COALESCE(annotated_path, '') = COALESCE(?, '')",
                (user_id, original_path, annotated_path),
            ).fetchone()
        return _analysis_from_row(row) if row else None

    def save_analysis(self, analysis: Analysis) -> Analysis:
        """Insert a new saved analysis and its detections.

        Raises :class:`DuplicateObservationError` when the same media has
        already been saved for the account.
        """
        if analysis.kind not in ("image", "video"):
            raise ObservationError(f"Unsupported analysis kind: {analysis.kind!r}")
        now = _utcnow()
        analysis.created_at = analysis.created_at or now
        analysis.updated_at = now
        if not analysis.detected_at:
            analysis.detected_at = now
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO analyses ("
                    "id, user_id, kind, source_filename, original_path, annotated_path, "
                    "report_path, models, confidence_threshold, enhanced, total_detections, "
                    "unique_classes, average_confidence, max_confidence, frames_processed, "
                    "elapsed_seconds, model_version, detected_at, created_at, updated_at, "
                    "notes, project, location, habitat, review_status"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        analysis.id,
                        analysis.user_id,
                        analysis.kind,
                        analysis.source_filename,
                        analysis.original_path,
                        analysis.annotated_path,
                        analysis.report_path,
                        json.dumps(list(analysis.models)),
                        analysis.confidence_threshold,
                        int(bool(analysis.enhanced)),
                        int(analysis.total_detections),
                        int(analysis.unique_classes),
                        float(analysis.average_confidence),
                        float(analysis.max_confidence),
                        analysis.frames_processed,
                        analysis.elapsed_seconds,
                        analysis.model_version,
                        analysis.detected_at,
                        analysis.created_at,
                        analysis.updated_at,
                        analysis.notes,
                        analysis.project,
                        analysis.location,
                        analysis.habitat,
                        analysis.review_status,
                    ),
                )
                _insert_detections(conn, analysis)
        except sqlite3.IntegrityError as exc:
            raise DuplicateObservationError(
                "This analysis has already been saved."
            ) from exc
        return analysis

    def get_analysis(
        self, analysis_id: str, user_id: str, *, with_children: bool = True
    ) -> Analysis | None:
        """Load one analysis that ``user_id`` owns (``None`` otherwise)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM analyses WHERE id = ? AND user_id = ?",
                (analysis_id, user_id),
            ).fetchone()
            if row is None:
                return None
            analysis = _analysis_from_row(row)
            if with_children:
                analysis.detections = _load_detections(conn, analysis.id, user_id)
                analysis.collection_ids, analysis.collection_names = _load_collections(
                    conn, analysis.id, user_id
                )
        return analysis

    def list_analyses(
        self,
        user_id: str,
        *,
        query: str = "",
        kind: str | None = None,
        model: str | None = None,
        review_status: str | None = None,
        project: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Analysis]:
        """Saved analyses for the account, newest first, with optional filters."""
        clauses, params = _filters(
            user_id,
            query=query,
            kind=kind,
            model=model,
            review_status=review_status,
            project=project,
        )
        sql = f"SELECT * FROM analyses {clauses} ORDER BY created_at DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([int(limit), max(0, int(offset))])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_analysis_from_row(row) for row in rows]

    def count_analyses(
        self,
        user_id: str,
        *,
        query: str = "",
        kind: str | None = None,
        model: str | None = None,
        review_status: str | None = None,
        project: str | None = None,
    ) -> int:
        clauses, params = _filters(
            user_id,
            query=query,
            kind=kind,
            model=model,
            review_status=review_status,
            project=project,
        )
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) AS n FROM analyses {clauses}", params
            ).fetchone()
        return int(row["n"])

    def update_analysis(
        self,
        analysis_id: str,
        user_id: str,
        *,
        notes: str | None = None,
        project: str | None = None,
        location: str | None = None,
        habitat: str | None = None,
        review_status: str | None = None,
    ) -> bool:
        """Update the mutable metadata of an owned analysis.

        The original model output (detections, confidences) is never touched -
        only user-owned metadata and the review status.
        """
        if review_status is not None and review_status not in REVIEW_STATUSES:
            raise ObservationError(f"Unknown review status: {review_status!r}")
        fields: list[str] = []
        values: list[Any] = []
        for column, value in (
            ("notes", notes),
            ("project", project),
            ("location", location),
            ("habitat", habitat),
            ("review_status", review_status),
        ):
            if value is not None:
                fields.append(f"{column} = ?")
                values.append(value)
        if not fields:
            return self.owns_analysis(analysis_id, user_id)
        fields.append("updated_at = ?")
        values.append(_utcnow())
        values.extend([analysis_id, user_id])
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE analyses SET {', '.join(fields)} WHERE id = ? AND user_id = ?",
                values,
            )
        return cursor.rowcount > 0

    def owns_analysis(self, analysis_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM analyses WHERE id = ? AND user_id = ?",
                (analysis_id, user_id),
            ).fetchone()
        return row is not None

    def delete_analysis(self, analysis_id: str, user_id: str) -> bool:
        """Delete an owned analysis (cascades to detections and collection links)."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM analyses WHERE id = ? AND user_id = ?",
                (analysis_id, user_id),
            )
        return cursor.rowcount > 0

    def distinct_projects(self, user_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT project FROM analyses "
                "WHERE user_id = ? AND project <> '' ORDER BY project",
                (user_id,),
            ).fetchall()
        return [row["project"] for row in rows]

    def distinct_models(self, user_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT models FROM analyses WHERE user_id = ?", (user_id,)
            ).fetchall()
        seen: set[str] = set()
        for row in rows:
            seen.update(_json_list(row["models"]))
        return sorted(seen)

    # ------------------------------------------------------------------ #
    # Corrections (label review)
    # ------------------------------------------------------------------ #
    def add_correction(
        self,
        detection_id: str,
        user_id: str,
        corrected_label: str,
        note: str = "",
        *,
        verified: bool = False,
    ) -> Correction:
        """Record a label correction, leaving the original prediction intact.

        ``verified`` may only be true for an authorised reviewer (checked by
        the caller before this is reached); a self-correction sets the analysis
        to ``corrected`` and can never claim scientific verification.
        """
        label = (corrected_label or "").strip()
        if not label:
            raise ObservationError("A corrected label is required.")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT d.* FROM detections d JOIN analyses a ON a.id = d.analysis_id "
                "WHERE d.id = ? AND d.user_id = ? AND a.user_id = ?",
                (detection_id, user_id, user_id),
            ).fetchone()
            if row is None:
                raise NotOwnerError("That detection does not belong to this account.")
            correction = Correction(
                id=uuid.uuid4().hex,
                detection_id=detection_id,
                analysis_id=row["analysis_id"],
                reviewer_id=user_id,
                original_label=row["class_name"],
                corrected_label=label,
                note=(note or "").strip(),
                created_at=_utcnow(),
            )
            conn.execute(
                "INSERT INTO corrections (id, detection_id, analysis_id, user_id, "
                "original_label, corrected_label, note, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    correction.id,
                    correction.detection_id,
                    correction.analysis_id,
                    user_id,
                    correction.original_label,
                    correction.corrected_label,
                    correction.note,
                    correction.created_at,
                ),
            )
            status = "verified" if verified else "corrected"
            conn.execute(
                "UPDATE analyses SET review_status = ?, updated_at = ? "
                "WHERE id = ? AND user_id = ? AND review_status <> 'verified'",
                (status, _utcnow(), correction.analysis_id, user_id),
            )
        return correction

    def list_corrections(self, analysis_id: str, user_id: str) -> list[Correction]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM corrections WHERE analysis_id = ? AND user_id = ? "
                "ORDER BY created_at ASC",
                (analysis_id, user_id),
            ).fetchall()
        return [_correction_from_row(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Collections
    # ------------------------------------------------------------------ #
    def create_collection(
        self, user_id: str, name: str, description: str = ""
    ) -> Collection:
        clean = (name or "").strip()
        if not clean:
            raise ObservationError("A collection name is required.")
        now = _utcnow()
        collection = Collection(
            id=uuid.uuid4().hex,
            user_id=user_id,
            name=clean,
            description=(description or "").strip(),
            created_at=now,
            updated_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO collections (id, user_id, name, description, created_at, "
                "updated_at) VALUES (?,?,?,?,?,?)",
                (
                    collection.id,
                    collection.user_id,
                    collection.name,
                    collection.description,
                    collection.created_at,
                    collection.updated_at,
                ),
            )
        return collection

    def list_collections(self, user_id: str) -> list[Collection]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT c.*, COUNT(ci.analysis_id) AS item_count FROM collections c "
                "LEFT JOIN collection_items ci ON ci.collection_id = c.id "
                "WHERE c.user_id = ? GROUP BY c.id ORDER BY c.updated_at DESC",
                (user_id,),
            ).fetchall()
        return [_collection_from_row(row) for row in rows]

    def get_collection(self, collection_id: str, user_id: str) -> Collection | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT c.*, COUNT(ci.analysis_id) AS item_count FROM collections c "
                "LEFT JOIN collection_items ci ON ci.collection_id = c.id "
                "WHERE c.id = ? AND c.user_id = ? GROUP BY c.id",
                (collection_id, user_id),
            ).fetchone()
        return _collection_from_row(row) if row else None

    def update_collection(
        self,
        collection_id: str,
        user_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> bool:
        fields: list[str] = []
        values: list[Any] = []
        if name is not None:
            clean = name.strip()
            if not clean:
                raise ObservationError("A collection name is required.")
            fields.append("name = ?")
            values.append(clean)
        if description is not None:
            fields.append("description = ?")
            values.append(description.strip())
        if not fields:
            return self.get_collection(collection_id, user_id) is not None
        fields.append("updated_at = ?")
        values.append(_utcnow())
        values.extend([collection_id, user_id])
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE collections SET {', '.join(fields)} WHERE id = ? AND user_id = ?",
                values,
            )
        return cursor.rowcount > 0

    def delete_collection(self, collection_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM collections WHERE id = ? AND user_id = ?",
                (collection_id, user_id),
            )
        return cursor.rowcount > 0

    def add_to_collection(self, collection_id: str, analysis_id: str, user_id: str) -> bool:
        """Link an owned analysis into an owned collection (idempotent)."""
        with self._connect() as conn:
            owned = conn.execute(
                "SELECT 1 FROM collections WHERE id = ? AND user_id = ?",
                (collection_id, user_id),
            ).fetchone()
            analysed = conn.execute(
                "SELECT 1 FROM analyses WHERE id = ? AND user_id = ?",
                (analysis_id, user_id),
            ).fetchone()
            if owned is None or analysed is None:
                raise NotOwnerError(
                    "The collection or the analysis does not belong to this account."
                )
            conn.execute(
                "INSERT OR IGNORE INTO collection_items (collection_id, analysis_id, added_at) "
                "VALUES (?,?,?)",
                (collection_id, analysis_id, _utcnow()),
            )
            conn.execute(
                "UPDATE collections SET updated_at = ? WHERE id = ?",
                (_utcnow(), collection_id),
            )
        return True

    def remove_from_collection(
        self, collection_id: str, analysis_id: str, user_id: str
    ) -> bool:
        with self._connect() as conn:
            owned = conn.execute(
                "SELECT 1 FROM collections WHERE id = ? AND user_id = ?",
                (collection_id, user_id),
            ).fetchone()
            if owned is None:
                raise NotOwnerError("That collection does not belong to this account.")
            cursor = conn.execute(
                "DELETE FROM collection_items WHERE collection_id = ? AND analysis_id = ?",
                (collection_id, analysis_id),
            )
            conn.execute(
                "UPDATE collections SET updated_at = ? WHERE id = ?",
                (_utcnow(), collection_id),
            )
        return cursor.rowcount > 0

    def list_collection_analyses(self, collection_id: str, user_id: str) -> list[Analysis]:
        with self._connect() as conn:
            owned = conn.execute(
                "SELECT 1 FROM collections WHERE id = ? AND user_id = ?",
                (collection_id, user_id),
            ).fetchone()
            if owned is None:
                raise NotOwnerError("That collection does not belong to this account.")
            rows = conn.execute(
                "SELECT a.* FROM analyses a JOIN collection_items ci "
                "ON ci.analysis_id = a.id WHERE ci.collection_id = ? AND a.user_id = ? "
                "ORDER BY ci.added_at DESC",
                (collection_id, user_id),
            ).fetchall()
        return [_analysis_from_row(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Dashboard statistics
    # ------------------------------------------------------------------ #
    def stats(self, user_id: str) -> dict[str, Any]:
        """Real, account-scoped counts for the dashboard (never fabricated)."""
        with self._connect() as conn:
            analyses = conn.execute(
                "SELECT COUNT(*) AS n FROM analyses WHERE user_id = ?", (user_id,)
            ).fetchone()["n"]
            detections = conn.execute(
                "SELECT COUNT(*) AS n FROM detections WHERE user_id = ?", (user_id,)
            ).fetchone()["n"]
            awaiting = conn.execute(
                "SELECT COUNT(*) AS n FROM analyses WHERE user_id = ? "
                "AND review_status IN ('predicted', 'needs_review')",
                (user_id,),
            ).fetchone()["n"]
            collections = conn.execute(
                "SELECT COUNT(*) AS n FROM collections WHERE user_id = ?", (user_id,)
            ).fetchone()["n"]
            recent_rows = conn.execute(
                "SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT 5",
                (user_id,),
            ).fetchall()
            collection_rows = conn.execute(
                "SELECT c.*, COUNT(ci.analysis_id) AS item_count FROM collections c "
                "LEFT JOIN collection_items ci ON ci.collection_id = c.id "
                "WHERE c.user_id = ? GROUP BY c.id ORDER BY c.updated_at DESC LIMIT 4",
                (user_id,),
            ).fetchall()
        return {
            "analyses": int(analyses),
            "observations": int(detections),
            "awaiting_review": int(awaiting),
            "collections": int(collections),
            "recent": [_analysis_from_row(row) for row in recent_rows],
            "recent_collections": [_collection_from_row(row) for row in collection_rows],
        }


# --------------------------------------------------------------------------- #
# Row helpers
# --------------------------------------------------------------------------- #
def _insert_detections(conn: sqlite3.Connection, analysis: Analysis) -> None:
    for det in analysis.detections:
        det.analysis_id = analysis.id
        det.user_id = analysis.user_id
        det.created_at = det.created_at or _utcnow()
        x1, y1, x2, y2 = det.bbox if det.bbox else (None, None, None, None)
        conn.execute(
            "INSERT INTO detections (id, analysis_id, user_id, class_name, confidence, "
            "x1, y1, x2, y2, model, frame_index, frame_time, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                det.id,
                det.analysis_id,
                det.user_id,
                det.class_name,
                float(det.confidence),
                x1,
                y1,
                x2,
                y2,
                det.model,
                det.frame_index,
                det.frame_time,
                det.created_at,
            ),
        )


def _json_list(value: Any) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _analysis_from_row(row: sqlite3.Row) -> Analysis:
    return Analysis(
        id=row["id"],
        user_id=row["user_id"],
        kind=row["kind"],
        source_filename=row["source_filename"],
        original_path=row["original_path"],
        annotated_path=row["annotated_path"],
        report_path=row["report_path"],
        models=_json_list(row["models"]),
        confidence_threshold=row["confidence_threshold"],
        enhanced=bool(row["enhanced"]),
        total_detections=int(row["total_detections"] or 0),
        unique_classes=int(row["unique_classes"] or 0),
        average_confidence=float(row["average_confidence"] or 0.0),
        max_confidence=float(row["max_confidence"] or 0.0),
        frames_processed=row["frames_processed"],
        elapsed_seconds=row["elapsed_seconds"],
        model_version=row["model_version"],
        detected_at=row["detected_at"] or "",
        created_at=row["created_at"] or "",
        updated_at=row["updated_at"] or "",
        notes=row["notes"] or "",
        project=row["project"] or "",
        location=row["location"] or "",
        habitat=row["habitat"] or "",
        review_status=row["review_status"] or "predicted",
    )


def _detection_from_row(row: sqlite3.Row) -> DetectionRecord:
    bbox: tuple[int, int, int, int] | None = None
    if row["x1"] is not None and row["y1"] is not None:
        bbox = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
    return DetectionRecord(
        id=row["id"],
        analysis_id=row["analysis_id"],
        user_id=row["user_id"],
        class_name=row["class_name"],
        confidence=float(row["confidence"]),
        bbox=bbox,
        model=row["model"] or "",
        frame_index=row["frame_index"],
        frame_time=row["frame_time"],
        created_at=row["created_at"] or "",
    )


def _correction_from_row(row: sqlite3.Row) -> Correction:
    return Correction(
        id=row["id"],
        detection_id=row["detection_id"],
        analysis_id=row["analysis_id"],
        reviewer_id=row["user_id"],
        original_label=row["original_label"],
        corrected_label=row["corrected_label"],
        note=row["note"] or "",
        created_at=row["created_at"] or "",
    )


def _collection_from_row(row: sqlite3.Row) -> Collection:
    keys = row.keys()
    return Collection(
        id=row["id"],
        user_id=row["user_id"],
        name=row["name"],
        description=row["description"] or "",
        created_at=row["created_at"] or "",
        updated_at=row["updated_at"] or "",
        item_count=int(row["item_count"]) if "item_count" in keys else 0,
    )


def _load_detections(
    conn: sqlite3.Connection, analysis_id: str, user_id: str
) -> list[DetectionRecord]:
    rows = conn.execute(
        "SELECT * FROM detections WHERE analysis_id = ? AND user_id = ? "
        "ORDER BY confidence DESC, rowid ASC",
        (analysis_id, user_id),
    ).fetchall()
    detections = [_detection_from_row(row) for row in rows]
    if not detections:
        return detections
    ids = [det.id for det in detections]
    placeholders = ",".join("?" for _ in ids)
    correction_rows = conn.execute(
        f"SELECT * FROM corrections WHERE analysis_id = ? AND user_id = ? "
        f"AND detection_id IN ({placeholders}) ORDER BY created_at ASC",
        [analysis_id, user_id, *ids],
    ).fetchall()
    by_detection: dict[str, list[Correction]] = {}
    for row in correction_rows:
        by_detection.setdefault(row["detection_id"], []).append(_correction_from_row(row))
    for det in detections:
        det.corrections = by_detection.get(det.id, [])
    return detections


def _load_collections(
    conn: sqlite3.Connection, analysis_id: str, user_id: str
) -> tuple[list[str], list[str]]:
    rows = conn.execute(
        "SELECT c.id, c.name FROM collections c JOIN collection_items ci "
        "ON ci.collection_id = c.id WHERE ci.analysis_id = ? AND c.user_id = ? "
        "ORDER BY c.name",
        (analysis_id, user_id),
    ).fetchall()
    return [row["id"] for row in rows], [row["name"] for row in rows]


def _filters(
    user_id: str,
    *,
    query: str,
    kind: str | None,
    model: str | None,
    review_status: str | None,
    project: str | None,
) -> tuple[str, list[Any]]:
    clauses = ["WHERE user_id = ?"]
    params: list[Any] = [user_id]
    if kind:
        clauses.append("AND kind = ?")
        params.append(kind)
    if review_status:
        clauses.append("AND review_status = ?")
        params.append(review_status)
    if project:
        clauses.append("AND project = ?")
        params.append(project)
    if model:
        clauses.append("AND models LIKE ?")
        params.append(f'%"{model}"%')
    text = (query or "").strip()
    if text:
        like = f"%{text}%"
        clauses.append(
            "AND (source_filename LIKE ? OR project LIKE ? OR notes LIKE ? "
            "OR EXISTS (SELECT 1 FROM detections d WHERE d.analysis_id = analyses.id "
            "AND d.class_name LIKE ?))"
        )
        params.extend([like, like, like, like])
    return " ".join(clauses), params


__all__ = ["SCHEMA_VERSION", "ObservationStore"]
