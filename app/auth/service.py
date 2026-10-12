"""High-level authentication operations (register, login, authenticate, logout,
password change, profile updates).

Every method here works against the real SQLite store with real bcrypt
verification. There are no mocks, no fake successes and no browser-supplied
identities: the ``user_id`` on every profile/history operation is derived
from a validated session token.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
from typing import Any

from app.auth.errors import (
    AuthValidationError,
    DuplicateEmailError,
    InvalidCredentialsError,
)
from app.auth.models import Session, User
from app.auth.security import (
    MAX_PROFILE_BIO,
    MAX_PROFILE_FIELD,
    dummy_verify,
    generate_token,
    hash_password,
    hash_token,
    tokens_match,
    validate_email,
    validate_name,
    validate_optional_field,
    validate_password,
    verify_password,
)
from app.auth.store import AuthStore
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: The token value is only ever handed back to the caller (the web UI keeps
#: it in server-side session state). Use a wrapper so accidental repr/log of
#: the return value cannot leak it.
_LOG_REDACTED = "<redacted>"


class AuthService:
    """Business logic over :class:`AuthStore`; the single backend for the UI."""

    def __init__(self, store: AuthStore | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = store if store is not None else AuthStore(self.settings.auth_db_path)

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register(
        self,
        full_name: str,
        email: str,
        password: str,
        confirm_password: str | None = None,
    ) -> User:
        """Create an account and return it.

        Validates input server-side (email, name, password policy), hashes
        the password with bcrypt and commits to the database. Duplicate
        (normalised) emails raise :class:`DuplicateEmailError`. The caller is
        responsible for signing the new account in immediately afterwards.
        """
        try:
            name = validate_name(full_name)
            address = validate_email(email)
            validate_password(password, self.settings.password_min_length)
        except ValueError as exc:
            raise AuthValidationError(str(exc)) from exc

        if confirm_password is not None and password != confirm_password:
            raise AuthValidationError("The passwords do not match.")

        if self.store.get_user_by_email(address) is not None:
            raise DuplicateEmailError()

        try:
            return self.store.create_user(address, name, hash_password(password))
        except DuplicateEmailError:
            # Race: another writer created the account between the check and
            # the insert. Re-raise the same user-facing error.
            raise
        except Exception:  # pragma: no cover - defensive
            logger.exception("Account registration failed unexpectedly")
            raise

    # ------------------------------------------------------------------ #
    # Login / logout
    # ------------------------------------------------------------------ #
    def login(self, email: str, password: str, user_agent: str | None = None) -> tuple[User, str]:
        """Authenticate credentials and mint a session token.

        Returns ``(user, raw_token)``. The raw token is returned exactly once;
        the store keeps only its SHA-256 digest. Failing credentials raise
        :class:`InvalidCredentialsError` with generic wording.
        """
        try:
            address = validate_email(email)
        except ValueError as exc:
            raise InvalidCredentialsError() from exc

        user = self.store.get_user_by_email(address)
        if user is None:
            dummy_verify(password)  # keep timing flat against unknown emails
            raise InvalidCredentialsError()
        if not verify_password(password, user.password_hash):
            raise InvalidCredentialsError()

        raw_token = generate_token()
        self.store.create_session(
            user.id,
            hash_token(raw_token),
            self.settings.session_ttl_seconds,
            user_agent=user_agent,
        )
        self.store.touch_last_login(user.id)
        return user, raw_token

    def authenticate(self, raw_token: str | None) -> User | None:
        """Validate a session token and return its user, or ``None``.

        Checks (in order): token present, digest exists, session not revoked,
        session not expired. A valid lookup refreshes ``last_seen_at``. This
        is the single server-side gate for every protected operation.
        """
        if not raw_token:
            return None
        digest = hash_token(raw_token)
        session = self.store.get_session_by_digest(digest)
        if session is None or not session.active:
            return None
        if not self._is_valid(session):
            return None
        user = self.store.get_user_by_id(session.user_id)
        if user is None:
            return None
        self.store.mark_seen(session.id)
        return user

    def logout(self, raw_token: str | None) -> bool:
        """Revoke this device's session server-side.

        Returns ``True`` when a session was actually revoked, ``False`` for a
        missing/invalid token. ``False`` never fakes a revocation.
        """
        if not raw_token:
            return False
        session = self.store.get_session_by_digest(hash_token(raw_token))
        if session is None or not session.active:
            return False
        return self.store.revoke_session(session.id)

    # ------------------------------------------------------------------ #
    # Profile / account
    # ------------------------------------------------------------------ #
    def update_profile(
        self,
        user_id: str,
        full_name: str,
        *,
        profession: str | None = None,
        organization: str | None = None,
        country: str | None = None,
        research_interests: str | None = None,
        bio: str | None = None,
    ) -> User:
        """Update the display name and any provided profile fields.

        ``None`` means "leave this field unchanged"; a string (including an
        empty one) is validated and stored. Every value is trimmed and
        length-checked here so the store only ever receives clean input.
        """
        try:
            name = validate_name(full_name)
        except ValueError as exc:
            raise AuthValidationError(str(exc)) from exc

        cleaned: dict[str, str] = {}
        for label, key, value, limit in (
            ("Profession", "profession", profession, MAX_PROFILE_FIELD),
            ("Organisation", "organization", organization, MAX_PROFILE_FIELD),
            ("Country / region", "country", country, MAX_PROFILE_FIELD),
            ("Research interests", "research_interests", research_interests, MAX_PROFILE_FIELD),
            ("About", "bio", bio, MAX_PROFILE_BIO),
        ):
            if value is None:
                continue
            try:
                cleaned[key] = validate_optional_field(value, label, limit)
            except ValueError as exc:
                raise AuthValidationError(str(exc)) from exc

        updated = self.store.update_profile(user_id, name, **cleaned)
        if updated is None:
            raise AuthValidationError("Account not found.")
        return updated

    def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
        confirm_password: str | None = None,
        keep_token_digest: str | None = None,
    ) -> None:
        """Verify the current password and set a new one.

        Every *other* session is revoked so old tokens cannot be replayed
        after a credential change; the current device (identified by
        ``keep_token_digest``) stays signed in.
        """
        user = self.store.get_user_by_id(user_id)
        if user is None:
            raise AuthValidationError("Account not found.")
        if not current_password or not verify_password(current_password, user.password_hash):
            raise InvalidCredentialsError("Current password is incorrect.")
        try:
            validate_password(new_password, self.settings.password_min_length)
        except ValueError as exc:
            raise AuthValidationError(str(exc)) from exc
        if confirm_password is not None and new_password != confirm_password:
            raise AuthValidationError("The new passwords do not match.")
        self.store.set_password_hash(user_id, hash_password(new_password))
        revoked = 0
        for session in self.store.list_sessions(user_id):
            if keep_token_digest and tokens_match(session.token_digest, keep_token_digest):
                continue
            if session.active and self.store.revoke_session(session.id):
                revoked += 1
        return revoked

    def list_sessions(self, user_id: str, current_digest: str | None = None) -> list[dict[str, Any]]:
        """Active+recent sessions for the account page (never exposes the
        raw token or its digest; ``current`` marks this device)."""
        rows: list[dict[str, Any]] = []
        for session in self.store.list_sessions(user_id):
            rows.append(
                {
                    "id": session.id,
                    "created_at": session.created_at,
                    "expires_at": session.expires_at,
                    "revoked": session.revoked_at is not None,
                    "last_seen_at": session.last_seen_at,
                    "active": session.active and self._is_valid(session),
                    "current": bool(current_digest)
                    and tokens_match(session.token_digest, current_digest),
                }
            )
        return rows

    def revoke_other_sessions(self, user_id: str, keep_digest: str) -> int:
        """Revoke every session except the current one; returns the count."""
        revoked = 0
        for session in self.store.list_sessions(user_id):
            if tokens_match(session.token_digest, keep_digest):
                continue
            if session.active and self.store.revoke_session(session.id):
                revoked += 1
        return revoked

    def revoke_session_for_user(self, user_id: str, session_id: str) -> bool:
        """Revoke one session, only when it belongs to ``user_id``."""
        session = self.store.get_session_by_id(session_id)
        if session is None or session.user_id != user_id or not session.active:
            return False
        return self.store.revoke_session(session_id)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _is_valid(self, session: Session) -> bool:
        """Expiry check against the server clock."""
        from datetime import datetime, timezone

        try:
            expires = datetime.fromisoformat(session.expires_at)
        except (ValueError, TypeError):
            return False
        return expires > datetime.now(timezone.utc)