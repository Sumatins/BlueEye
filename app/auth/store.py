"""SQLite persistence for users, profiles and sessions.

The store opens a fresh database connection for every operation, which keeps
it safe under Streamlit's concurrent per-browser script runners without any
cross-thread connection sharing. WAL journaling keeps readers/writers from
blocking each other.

Schema migrations are applied from :data:`SCHEMA_VERSION` using SQLite's
``PRAGMA user_version`` so future changes can be added as ordered steps.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.auth import SCHEMA_VERSION
from app.auth.errors import DuplicateEmailError
from app.auth.models import Session, User

logger = logging.getLogger(__name__)

#: User-facing table of records. ``email`` is unique after normalisation
#: (see security.normalize_email); passwords are stored as bcrypt hashes.
_SCHEMA_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    full_name     TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    last_login_at TEXT
);
"""

#: One row per issued session token. Only the SHA-256 digest of the raw
#: token is stored; revocation flips ``revoked_at`` from NULL.
_SCHEMA_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_digest TEXT NOT NULL UNIQUE,
    created_at   TEXT NOT NULL,
    expires_at   TEXT NOT NULL,
    revoked_at   TEXT,
    last_seen_at TEXT,
    user_agent   TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user   ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_digest ON sessions(token_digest);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


class AuthStore:
    """SQLite-backed store for users and sessions."""

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
                conn.executescript(_SCHEMA_USERS)
                conn.executescript(_SCHEMA_SESSIONS)
            # Future migrations become ordered `elif version < N:` steps.
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    # ------------------------------------------------------------------ #
    # Users / profiles
    # ------------------------------------------------------------------ #
    def create_user(self, email: str, full_name: str, password_hash: str) -> User:
        """Insert a new account.

        Raises :class:`DuplicateEmailError` when the (normalised) email is
        already registered. The caller hashes the password; this method
        never sees a plaintext.
        """
        user = User(
            id=uuid.uuid4().hex,
            email=email,
            full_name=full_name,
            password_hash=password_hash,
            created_at=_utcnow(),
        )
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users (id, email, full_name, password_hash, "
                    "created_at, last_login_at) VALUES (?, ?, ?, ?, ?, NULL)",
                    (user.id, user.email, user.full_name, user.password_hash, user.created_at),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateEmailError() from exc
        return user

    def get_user_by_email(self, email: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
        return _user_from_row(row)

    def get_user_by_id(self, user_id: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return _user_from_row(row)

    def update_profile(self, user_id: str, full_name: str) -> User | None:
        """Rename a profile (identity/email are never changeable here)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET full_name = ? WHERE id = ?", (full_name, user_id)
            )
        return self.get_user_by_id(user_id)

    def set_password_hash(self, user_id: str, password_hash: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id)
            )

    def touch_last_login(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE users SET last_login_at = ? WHERE id = ?", (_utcnow(), user_id)
            )

    # ------------------------------------------------------------------ #
    # Sessions
    # ------------------------------------------------------------------ #
    def create_session(
        self,
        user_id: str,
        token_digest: str,
        ttl_seconds: int,
        user_agent: str | None = None,
    ) -> Session:
        """Insert a session row; returns it with a fresh server-side expiry."""
        now = _utcnow()
        session = Session(
            id=uuid.uuid4().hex,
            user_id=user_id,
            token_digest=token_digest,
            created_at=now,
            expires_at=datetime.fromtimestamp(
                datetime.now(timezone.utc).timestamp() + ttl_seconds, tz=timezone.utc
            ).isoformat(timespec="seconds"),
            user_agent=user_agent,
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, user_id, token_digest, created_at, "
                "expires_at, revoked_at, last_seen_at, user_agent) "
                "VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)",
                (
                    session.id,
                    session.user_id,
                    session.token_digest,
                    session.created_at,
                    session.expires_at,
                    session.user_agent,
                ),
            )
        return session

    def get_session_by_digest(self, token_digest: str) -> Session | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE token_digest = ?", (token_digest,)
            ).fetchone()
        return _session_from_row(row)

    def get_session_by_id(self, session_id: str) -> Session | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return _session_from_row(row)

    def mark_seen(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE id = ?", (_utcnow(), session_id)
            )

    def revoke_session(self, session_id: str) -> bool:
        """Revoke one session by id; returns whether a row changed."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                (_utcnow(), session_id),
            )
        return cursor.rowcount > 0  # type: ignore[union-attr]

    def revoke_all_sessions(self, user_id: str) -> int:
        """Revoke every session belonging to ``user_id``; returns the count."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
                (_utcnow(), user_id),
            )
        return cursor.rowcount  # type: ignore[union-attr]

    def list_sessions(self, user_id: str) -> list[Session]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? "
                "ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [_session_from_row(row) for row in rows]  # type: ignore[arg-type]

    def purge_expired(self) -> int:
        """Delete expired sessions; returns how many were removed."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE expires_at < ?", (_utcnow(),)
            )
        return cursor.rowcount  # type: ignore[union-attr]

    def touch_stale_expired(self, user_id: str) -> int:  # pragma: no cover - helper
        """Count of expired-but-unrevoked sessions (used by tests/debug)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM sessions WHERE user_id = ? AND expires_at < ?",
                (user_id, _utcnow()),
            ).fetchone()
        return int(row["n"])  # type: ignore[index]


# --------------------------------------------------------------------------- #
# Row mappers
# --------------------------------------------------------------------------- #
def _user_from_row(row: Any) -> User | None:
    if row is None:
        return None
    return User(
        id=row["id"],
        email=row["email"],
        full_name=row["full_name"],
        password_hash=row["password_hash"],
        created_at=row["created_at"],
        last_login_at=row["last_login_at"],
    )


def _session_from_row(row: Any) -> Session | None:
    if row is None:
        return None
    return Session(
        id=row["id"],
        user_id=row["user_id"],
        token_digest=row["token_digest"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        revoked_at=row["revoked_at"],
        last_seen_at=row["last_seen_at"],
        user_agent=row["user_agent"],
    )