"""Integration tests for the real authentication backend (app.auth).

These tests exercise :class:`app.auth.service.AuthService` against a real
SQLite database and real bcrypt hashing in a temporary directory. Nothing is
mocked: registration writes rows, login verifies hashes, sessions are stored
only as digests and sign-out really revokes server-side.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.auth import (
    AuthService,
    AuthValidationError,
    DuplicateEmailError,
    InvalidCredentialsError,
)
from app.auth.errors import AuthError
from app.auth.security import hash_token
from app.config import Settings, reset_settings_cache


@pytest.fixture()
def auth_settings(tmp_path: Path) -> Settings:
    """Isolated settings whose auth database lives in the temp directory."""
    reset_settings_cache()
    custom = Settings(
        models_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
        outputs_dir=tmp_path / "outputs",
        auth_db=tmp_path / "auth" / "blueeye_auth.db",
    )
    custom.ensure_dirs()
    yield custom
    reset_settings_cache()


@pytest.fixture()
def service(auth_settings: Settings) -> AuthService:
    return AuthService(settings=auth_settings)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def test_register_creates_a_real_hashed_account(service: AuthService, auth_settings: Settings) -> None:
    user = service.register("Ada Lovelace", "Ada@Example.ORG", "secreT123")

    assert user.id
    assert user.email == "ada@example.org"  # normalised (trim + lowercase)
    assert user.full_name == "Ada Lovelace"
    assert user.password_hash != "secreT123"
    assert user.password_hash.startswith("$2")  # bcrypt marker

    # The plaintext password must never reach the database file.
    raw = auth_settings.auth_db_path.read_bytes()
    assert b"secreT123" not in raw

    # The account can immediately authenticate with the original password.
    _user, token = service.login("ada@example.org", "secreT123")
    assert service.authenticate(token).id == user.id


def test_register_is_case_insensitive_for_duplicates(service: AuthService) -> None:
    service.register("Ada", "ada@example.org", "secreT123")
    with pytest.raises(DuplicateEmailError):
        service.register("Ada Two", "ADA@Example.org", "secreT123")


def test_register_rejects_invalid_input(service: AuthService) -> None:
    with pytest.raises(AuthValidationError):
        service.register("Ada", "not-an-email", "secreT123")
    with pytest.raises(AuthValidationError):
        service.register("   ", "ada@example.org", "secreT123")
    with pytest.raises(AuthValidationError):
        service.register("Ada", "ada@example.org", "short1")  # too short
    with pytest.raises(AuthValidationError):
        service.register("Ada", "ada@example.org", "allletters")  # no digit
    with pytest.raises(AuthValidationError):
        service.register("Ada", "ada@example.org", "12345678")  # no letter
    with pytest.raises(AuthValidationError):
        service.register("Ada", "ada@example.org", "secreT123", "secreT124")


def test_register_honours_password_minimum_length(auth_settings: Settings) -> None:
    auth_settings.password_min_length = 12
    service = AuthService(settings=auth_settings)
    with pytest.raises(AuthValidationError):
        service.register("Ada", "ada@example.org", "secreT123")  # 9 chars
    # A 12+ character password with a letter and a digit passes.
    assert service.register("Ada", "ada@example.org", "secreT1234567").id


def test_register_reports_non_auth_failures_honestly(service: AuthService, monkeypatch) -> None:
    """A genuine database error is not silently swallowed."""

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(service.store, "create_user", boom)
    with pytest.raises(sqlite3.OperationalError):
        service.register("Ada", "ada@example.org", "secreT123")


# --------------------------------------------------------------------------- #
# Login
# --------------------------------------------------------------------------- #
def test_login_rejects_wrong_password_and_unknown_email(service: AuthService) -> None:
    service.register("Ada", "ada@example.org", "secreT123")
    with pytest.raises(InvalidCredentialsError) as wrong_pw:
        service.login("ada@example.org", "wrongPass1")
    with pytest.raises(InvalidCredentialsError) as unknown:
        service.login("nobody@example.org", "secreT123")
    # The message must not reveal whether the email exists.
    assert "invalid email or password" in str(wrong_pw.value).lower()
    assert str(wrong_pw.value) == str(unknown.value)


def test_login_rejects_malformed_email(service: AuthService) -> None:
    with pytest.raises(InvalidCredentialsError):
        service.login("not-an-email", "secreT123")


def test_session_token_is_only_stored_as_a_digest(service: AuthService, auth_settings: Settings) -> None:
    service.register("Ada", "ada@example.org", "secreT123")
    _user, token = service.login("ada@example.org", "secreT123")

    with sqlite3.connect(auth_settings.auth_db_path) as conn:
        rows = conn.execute("SELECT token_digest FROM sessions").fetchall()

    digests = [row[0] for row in rows]
    assert token not in digests  # raw token never stored
    assert hash_token(token) in digests


def test_authenticate_returns_none_for_missing_or_junk_tokens(service: AuthService) -> None:
    assert service.authenticate(None) is None
    assert service.authenticate("") is None
    assert service.authenticate("not-a-real-token") is None


# --------------------------------------------------------------------------- #
# Logout / revocation
# --------------------------------------------------------------------------- #
def test_logout_revokes_the_session_server_side(service: AuthService) -> None:
    service.register("Ada", "ada@example.org", "secreT123")
    _user, token = service.login("ada@example.org", "secreT123")
    assert service.authenticate(token) is not None

    assert service.logout(token) is True
    assert service.authenticate(token) is None  # revoked immediately

    # A second logout must not pretend to revoke anything.
    assert service.logout(token) is False
    assert service.logout("nonsense") is False
    assert service.logout(None) is False


def test_expired_session_is_rejected(service: AuthService) -> None:
    user = service.register("Ada", "ada@example.org", "secreT123")
    stale = service.store.create_session(user.id, hash_token("stale-token"), -60)
    assert stale.id
    assert service.authenticate("stale-token") is None


def test_list_sessions_marks_the_current_device(service: AuthService) -> None:
    service.register("Ada", "ada@example.org", "secreT123")
    _user, token = service.login("ada@example.org", "secreT123")
    _user2, other = service.login("ada@example.org", "secreT123")

    rows = service.list_sessions(service.authenticate(token).id, current_digest=hash_token(token))
    current = [row for row in rows if row["current"]]
    assert len(current) == 1
    assert all(not row["revoked"] for row in rows)
    assert len(rows) == 2
    del other


def test_revoke_other_sessions_keeps_only_the_current_one(service: AuthService) -> None:
    service.register("Ada", "ada@example.org", "secreT123")
    user, token_a = service.login("ada@example.org", "secreT123")
    service.login("ada@example.org", "secreT123")

    count = service.revoke_other_sessions(user.id, keep_digest=hash_token(token_a))
    assert count == 1
    assert service.authenticate(token_a) is not None


# --------------------------------------------------------------------------- #
# Profile / password
# --------------------------------------------------------------------------- #
def test_update_profile_renames_the_account(service: AuthService) -> None:
    user = service.register("Ada", "ada@example.org", "secreT123")
    updated = service.update_profile(user.id, "  Ada Byron  ")
    assert updated.full_name == "Ada Byron"
    with pytest.raises(AuthValidationError):
        service.update_profile(user.id, "   ")


def test_change_password_keeps_current_session_and_retires_others(service: AuthService) -> None:
    service.register("Ada", "ada@example.org", "secreT123")
    user, token_a = service.login("ada@example.org", "secreT123")
    _, token_b = service.login("ada@example.org", "secreT123")

    service.change_password(
        user.id,
        "secreT123",
        "newPass456",
        "newPass456",
        keep_token_digest=hash_token(token_a),
    )

    assert service.authenticate(token_a) is not None  # this device stays in
    assert service.authenticate(token_b) is None  # other devices signed out
    with pytest.raises(InvalidCredentialsError):
        service.login("ada@example.org", "secreT123")
    assert service.login("ada@example.org", "newPass456")[1]


def test_change_password_requires_the_current_password(service: AuthService) -> None:
    user = service.register("Ada", "ada@example.org", "secreT123")
    with pytest.raises(InvalidCredentialsError):
        service.change_password(user.id, "wrongPass1", "newPass456")
    with pytest.raises(AuthValidationError):
        service.change_password(user.id, "secreT123", "short1")
    with pytest.raises(AuthValidationError):
        service.change_password(user.id, "secreT123", "newPass456", "different7")


def test_revoke_session_for_user_only_touches_own_sessions(service: AuthService) -> None:
    ada = service.register("Ada", "ada@example.org", "secreT123")
    bob = service.register("Bob", "bob@example.org", "secreT123")
    _, bob_token = service.login("bob@example.org", "secreT123")
    bob_session = service.store.get_session_by_digest(hash_token(bob_token))

    # Ada may not revoke Bob's session.
    assert service.revoke_session_for_user(ada.id, bob_session.id) is False
    assert service.authenticate(bob_token) is not None
    # Bob can revoke his own session, and doing it twice reports the truth.
    assert service.revoke_session_for_user(bob.id, bob_session.id) is True
    assert service.revoke_session_for_user(bob.id, bob_session.id) is False


# --------------------------------------------------------------------------- #
# Two independent accounts
# --------------------------------------------------------------------------- #
def test_two_accounts_have_distinct_identities_and_sessions(service: AuthService) -> None:
    ada = service.register("Ada", "ada@example.org", "secreT123")
    bob = service.register("Bob", "bob@example.org", "otheR456")

    assert ada.id != bob.id
    _, ada_token = service.login("ada@example.org", "secreT123")
    _, bob_token = service.login("bob@example.org", "otheR456")

    assert service.authenticate(ada_token).id == ada.id
    assert service.authenticate(bob_token).id == bob.id

    # Revoking Ada's session must not affect Bob's.
    service.logout(ada_token)
    assert service.authenticate(ada_token) is None
    assert service.authenticate(bob_token).id == bob.id


def test_store_persists_across_service_instances(auth_settings: Settings) -> None:
    first = AuthService(settings=auth_settings)
    user = first.register("Ada", "ada@example.org", "secreT123")
    _u, token = first.login("ada@example.org", "secreT123")

    # A brand-new service over the same database validates the same session.
    second = AuthService(settings=auth_settings)
    assert second.authenticate(token).id == user.id


def test_errors_are_user_facing(auth_settings: Settings) -> None:
    # Sanity: every public error type carries a safe, non-empty message.
    for exc in (AuthValidationError("x"), DuplicateEmailError(), InvalidCredentialsError()):
        assert isinstance(exc, AuthError)
        assert exc.message
