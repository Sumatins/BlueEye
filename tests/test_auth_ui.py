"""End-to-end UI tests for the authentication gate using Streamlit's AppTest.

These run the *real* ``app/ui/streamlit_app.py`` script against a real SQLite
database and real bcrypt hashing in a temporary directory - the same code the
browser executes. Streamlit's AppTest preserves session state across runs, so
a second ``run()`` (with no widget changes) models a browser refresh: the
server-side token must keep the session alive.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.auth import AuthService
from app.config import Settings, reset_settings_cache

APP = str(Path(__file__).resolve().parents[1] / "app" / "ui" / "streamlit_app.py")


def _by_key(elements, key):
    for element in elements:
        if getattr(element, "key", None) == key:
            return element
    available = [getattr(element, "key", None) for element in elements]
    raise AssertionError(f"no widget with key {key!r}; available: {available}")


@pytest.fixture()
def app_env(tmp_path: Path, monkeypatch):
    """Point the app at temporary dirs + a throwaway auth database."""
    monkeypatch.setenv("BLUEEYE_AUTH_DB", str(tmp_path / "auth" / "blueeye_auth.db"))
    monkeypatch.setenv("BLUEEYE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BLUEEYE_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("BLUEEYE_MODELS_DIR", str(tmp_path / "models"))
    reset_settings_cache()
    st.cache_resource.clear()
    st.cache_data.clear()
    yield tmp_path
    reset_settings_cache()
    st.cache_resource.clear()
    st.cache_data.clear()
    shutil.rmtree(tmp_path, ignore_errors=True)


def _app(page: str = "dashboard") -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["nav_sel"] = page
    return at


def _register(at: AppTest, name: str, email: str, password: str) -> AppTest:
    """Drive the real create-account form."""
    at.run()
    _by_key(at.radio, "auth_mode").set_value("Create account")
    at.run()
    _by_key(at.text_input, "auth_register_name").set_value(name)
    _by_key(at.text_input, "auth_register_email").set_value(email)
    _by_key(at.text_input, "auth_register_password").set_value(password)
    _by_key(at.text_input, "auth_register_confirm").set_value(password)
    _by_key(at.button, "auth_register_submit").click()
    return at.run()


# --------------------------------------------------------------------------- #
# Gate / anonymous access
# --------------------------------------------------------------------------- #
def test_gate_is_shown_for_anonymous_visitors_on_private_pages(app_env) -> None:
    at = _app("dashboard").run()

    assert not at.exception, at.exception
    # The sign-in / create-account radio is present (the gate), and no token.
    _by_key(at.radio, "auth_mode")
    assert "be_auth_token" not in at.session_state


@pytest.mark.parametrize("page", ["dashboard", "detect", "history", "analytics", "account"])
def test_every_private_page_gates_anonymous_visitors(app_env, page: str) -> None:
    at = _app(page).run()
    assert not at.exception, at.exception
    _by_key(at.radio, "auth_mode")
    assert "be_auth_token" not in at.session_state


@pytest.mark.parametrize("page", ["species", "models", "about", "help"])
def test_public_pages_stay_browsable_for_guests(app_env, page: str) -> None:
    at = _app(page).run()
    assert not at.exception, at.exception
    # No sign-in gate widget is rendered for public pages.
    assert not any(getattr(r, "key", None) == "auth_mode" for r in at.radio)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def test_registration_creates_a_real_account_and_signs_in(app_env) -> None:
    at = _register(_app("dashboard"), "Ada Lovelace", "ada@example.org", "secreT123")

    assert not at.exception, at.exception
    token = at.session_state["be_auth_token"]
    assert token  # a real server-side token was issued
    assert at.session_state["be_auth_user_name"] == "Ada Lovelace"

    # The account exists in the real database and the password is hashed.
    settings = Settings(
        outputs_dir=app_env / "outputs",
        data_dir=app_env / "data",
        auth_db=app_env / "auth" / "blueeye_auth.db",
    )
    stored = AuthService(settings=settings).store.get_user_by_email("ada@example.org")
    assert stored is not None
    assert stored.password_hash.startswith("$2")
    assert (app_env / "auth" / "blueeye_auth.db").read_bytes().find(b"secreT123") == -1


def test_registration_then_refresh_keeps_the_session(app_env) -> None:
    at = _register(_app("dashboard"), "Ada", "ada@example.org", "secreT123")
    token_before = at.session_state["be_auth_token"]

    # A plain re-run (no widget change) is Streamlit's session restoring.
    at.run()

    assert not at.exception, at.exception
    assert at.session_state["be_auth_token"] == token_before
    assert not any(getattr(r, "key", None) == "auth_mode" for r in at.radio)


def test_sign_in_emits_a_session_cookie_script(app_env, monkeypatch) -> None:
    """A signed-in run writes the opaque token into a browser cookie.

    Streamlit session state alone does not survive a full page reload, so the
    token is also mirrored into a SameSite cookie that ``bootstrap`` reads back
    via ``st.context.cookies``. AppTest cannot execute JavaScript, so we assert
    on the script the app emits.
    """
    captured: list[str] = []

    def _capture(body, *args, **kwargs):
        captured.append(str(body))

    monkeypatch.setattr("app.ui.auth_ui.st.html", _capture)

    at = _register(_app("dashboard"), "Ada", "ada@example.org", "secreT123")
    assert not at.exception, at.exception
    token = at.session_state["be_auth_token"]

    joined = "\n".join(captured)
    assert "document.cookie" in joined
    assert f"blueeye_session={token}" in joined


def test_sign_out_emits_a_cookie_clear_script(app_env, monkeypatch) -> None:
    captured: list[str] = []

    def _capture(body, *args, **kwargs):
        captured.append(str(body))

    monkeypatch.setattr("app.ui.auth_ui.st.html", _capture)

    at = _register(_app("dashboard"), "Ada", "ada@example.org", "secreT123")
    captured.clear()

    _by_key(at.button, "auth_signout").click()
    at.run()

    assert not at.exception, at.exception
    joined = "\n".join(captured)
    assert "document.cookie" in joined
    assert "max-age=0" in joined


def test_anonymous_visitor_gets_no_session_cookie(app_env, monkeypatch) -> None:
    captured: list[str] = []

    def _capture(body, *args, **kwargs):
        captured.append(str(body))

    monkeypatch.setattr("app.ui.auth_ui.st.html", _capture)

    at = _app("dashboard").run()

    assert not at.exception, at.exception
    assert not any(f"blueeye_session={''}" in script for script in captured)
    assert "be_auth_token" not in at.session_state


def test_duplicate_registration_is_rejected(app_env) -> None:
    _register(_app("dashboard"), "Ada", "ada@example.org", "secreT123")

    at = _app("dashboard").run()
    _by_key(at.radio, "auth_mode").set_value("Create account")
    at.run()
    _by_key(at.text_input, "auth_register_name").set_value("Impostor")
    _by_key(at.text_input, "auth_register_email").set_value("ADA@example.org")
    _by_key(at.text_input, "auth_register_password").set_value("secreT123")
    _by_key(at.text_input, "auth_register_confirm").set_value("secreT123")
    _by_key(at.button, "auth_register_submit").click()
    at.run()

    assert "be_auth_token" not in at.session_state
    assert any("already exists" in (e.value or "").lower() for e in at.error)


def test_registration_rejects_a_weak_password(app_env) -> None:
    at = _app("dashboard").run()
    _by_key(at.radio, "auth_mode").set_value("Create account")
    at.run()
    _by_key(at.text_input, "auth_register_name").set_value("Ada")
    _by_key(at.text_input, "auth_register_email").set_value("ada@example.org")
    _by_key(at.text_input, "auth_register_password").set_value("weak")
    _by_key(at.text_input, "auth_register_confirm").set_value("weak")
    _by_key(at.button, "auth_register_submit").click()
    at.run()

    assert "be_auth_token" not in at.session_state
    assert at.error


# --------------------------------------------------------------------------- #
# Login
# --------------------------------------------------------------------------- #
def test_login_with_valid_credentials_succeeds(app_env) -> None:
    settings = Settings(
        outputs_dir=app_env / "outputs",
        data_dir=app_env / "data",
        auth_db=app_env / "auth" / "blueeye_auth.db",
    )
    AuthService(settings=settings).register("Bob", "bob@example.org", "secreT123")

    at = _app("dashboard").run()
    _by_key(at.text_input, "auth_login_email").set_value("bob@example.org")
    _by_key(at.text_input, "auth_login_password").set_value("secreT123")
    _by_key(at.button, "auth_login_submit").click()
    at.run()

    assert not at.exception, at.exception
    assert at.session_state["be_auth_token"]
    assert at.session_state["be_auth_user_name"] == "Bob"


def test_login_with_invalid_credentials_is_rejected(app_env) -> None:
    settings = Settings(
        outputs_dir=app_env / "outputs",
        data_dir=app_env / "data",
        auth_db=app_env / "auth" / "blueeye_auth.db",
    )
    AuthService(settings=settings).register("Bob", "bob@example.org", "secreT123")

    at = _app("dashboard").run()
    _by_key(at.text_input, "auth_login_email").set_value("bob@example.org")
    _by_key(at.text_input, "auth_login_password").set_value("wrongPass1")
    _by_key(at.button, "auth_login_submit").click()
    at.run()

    assert "be_auth_token" not in at.session_state
    assert any("invalid email or password" in (e.value or "").lower() for e in at.error)


# --------------------------------------------------------------------------- #
# Logout
# --------------------------------------------------------------------------- #
def test_logout_revokes_the_session_and_returns_to_the_gate(app_env) -> None:
    at = _register(_app("dashboard"), "Ada", "ada@example.org", "secreT123")
    token = at.session_state["be_auth_token"]
    assert token

    _by_key(at.button, "auth_signout").click()
    at.run()

    # Local state is cleared and the gate is showing again.
    assert "be_auth_token" not in at.session_state
    _by_key(at.radio, "auth_mode")

    # The token is genuinely revoked server-side: a fresh validation fails.
    settings = Settings(
        outputs_dir=app_env / "outputs",
        data_dir=app_env / "data",
        auth_db=app_env / "auth" / "blueeye_auth.db",
    )
    assert AuthService(settings=settings).authenticate(token) is None


def test_revoked_session_cannot_access_private_pages(app_env) -> None:
    """A token revoked elsewhere degrades to the gate on the next run."""
    at = _register(_app("dashboard"), "Ada", "ada@example.org", "secreT123")
    token = at.session_state["be_auth_token"]

    settings = Settings(
        outputs_dir=app_env / "outputs",
        data_dir=app_env / "data",
        auth_db=app_env / "auth" / "blueeye_auth.db",
    )
    service = AuthService(settings=settings)
    assert service.logout(token) is True

    at.run()  # next run re-validates against the database

    assert not at.exception, at.exception
    assert "be_auth_token" not in at.session_state  # stale token dropped
    _by_key(at.radio, "auth_mode")


# --------------------------------------------------------------------------- #
# Per-account scoping
# --------------------------------------------------------------------------- #
def test_two_accounts_are_isolated_in_the_ui(app_env) -> None:
    at = _register(_app("dashboard"), "Ada", "ada@example.org", "secreT123")
    ada_id = at.session_state["be_auth_user_id"]
    _by_key(at.button, "auth_signout").click()
    at.run()

    at = _register(_app("dashboard"), "Bob", "bob@example.org", "otheR456")
    bob_id = at.session_state["be_auth_user_id"]

    assert ada_id and bob_id and ada_id != bob_id

    settings = Settings(
        outputs_dir=app_env / "outputs",
        data_dir=app_env / "data",
        auth_db=app_env / "auth" / "blueeye_auth.db",
    )
    service = AuthService(settings=settings)
    assert service.store.get_user_by_id(ada_id).email == "ada@example.org"
    assert service.store.get_user_by_id(bob_id).email == "bob@example.org"


# --------------------------------------------------------------------------- #
# Account page (private)
# --------------------------------------------------------------------------- #
def _goto_account(at: AppTest) -> AppTest:
    _by_key(at.radio, "nav_sel").set_value("account")
    return at.run()


def _settings_for(app_env) -> Settings:
    return Settings(
        outputs_dir=app_env / "outputs",
        data_dir=app_env / "data",
        auth_db=app_env / "auth" / "blueeye_auth.db",
    )


def test_account_page_renders_for_a_signed_in_user(app_env) -> None:
    at = _goto_account(_register(_app("dashboard"), "Ada", "ada@example.org", "secreT123"))
    assert not at.exception, at.exception
    _by_key(at.button, "acct_pw_change")
    _by_key(at.text_input, "acct_name")


def test_account_page_hides_internal_implementation_details(app_env) -> None:
    """The Account page never exposes storage paths, ids or hashes."""
    at = _goto_account(_register(_app("dashboard"), "Ada", "ada@example.org", "secreT123"))
    assert not at.exception, at.exception
    parts = [m.value for m in at.markdown]
    parts += [c.value for c in getattr(at, "caption", [])]
    blob = "\n".join(parts)
    assert "blueeye_auth.db" not in blob
    assert "Account id" not in blob
    assert str(app_env) not in blob
    assert "$2" not in blob  # no bcrypt hashes
    # The professional identity banner is shown instead.
    assert "be-identity" in blob


def test_account_page_profile_update_persists(app_env) -> None:
    at = _goto_account(_register(_app("dashboard"), "Ada", "ada@example.org", "secreT123"))
    _by_key(at.text_input, "acct_name").set_value("Ada Byron")
    _by_key(at.button, "acct_save_name").click()
    at.run()

    assert not at.exception, at.exception
    assert at.session_state["be_auth_user_name"] == "Ada Byron"
    stored = AuthService(settings=_settings_for(app_env)).store.get_user_by_email(
        "ada@example.org"
    )
    assert stored.full_name == "Ada Byron"


def test_account_page_password_change_persists(app_env) -> None:
    at = _goto_account(_register(_app("dashboard"), "Ada", "ada@example.org", "secreT123"))
    _by_key(at.text_input, "acct_pw_current").set_value("secreT123")
    _by_key(at.text_input, "acct_pw_new").set_value("newPass456")
    _by_key(at.text_input, "acct_pw_confirm").set_value("newPass456")
    _by_key(at.button, "acct_pw_change").click()
    at.run()

    assert not at.exception, at.exception
    # This device stays signed in after changing the password.
    assert at.session_state["be_auth_token"]
    # The new password really works; the old one no longer does.
    service = AuthService(settings=_settings_for(app_env))
    user, _token = service.login("ada@example.org", "newPass456")
    assert user.email == "ada@example.org"
    from app.auth import InvalidCredentialsError

    with pytest.raises(InvalidCredentialsError):
        service.login("ada@example.org", "secreT123")


def test_sign_out_everywhere_else_keeps_this_device(app_env) -> None:
    at = _register(_app("dashboard"), "Ada", "ada@example.org", "secreT123")
    token = at.session_state["be_auth_token"]
    at = _goto_account(at)

    _by_key(at.button, "acct_revoke_all").click()
    at.run()

    assert not at.exception, at.exception
    # The current device remains signed in and its token is still valid.
    assert at.session_state["be_auth_token"] == token
    assert AuthService(settings=_settings_for(app_env)).authenticate(token) is not None


# --------------------------------------------------------------------------- #
# Navigation (every remaining sidebar section)
# --------------------------------------------------------------------------- #
def test_every_sidebar_section_renders_for_a_signed_in_user(app_env) -> None:
    """Every remaining navigation entry renders without error, and the
    removed Map entry is not offered anywhere."""
    from app.ui.theme import PAGES

    at = _register(_app("dashboard"), "Ada", "ada@example.org", "secreT123")
    for page_id, _label, _icon in PAGES:
        _by_key(at.radio, "nav_sel").set_value(page_id)
        at.run()
        assert not at.exception, f"{page_id}: {at.exception}"

    offered = set(_by_key(at.radio, "nav_sel").options)
    assert "Map" not in offered
    assert offered == {label for _, label, _ in PAGES}

