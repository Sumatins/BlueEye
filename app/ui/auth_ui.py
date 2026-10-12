"""Streamlit glue for authentication.

Responsibilities
----------------
* Server-side session bootstrap: validate the token cached in
  ``st.session_state`` against the real database on every run.
* The sign-in / create-account gate shown to anonymous visitors on private
  pages (and the guest banner on public pages).
* The sidebar account block (who is signed in / sign out).
* The Account page (profile, password change, session management).

Security model
--------------
* Identity is *server-side only*. The browser cannot choose a user id; every
  protected operation receives the :class:`app.auth.models.User` produced by
  validating a token that only the server ever created.
* The token is re-validated against the database on every run (expiry +
  revocation checks). Streamlit session state alone does not survive a full
  page reload, so the token is also kept in a SameSite=Lax browser cookie;
  on load it is read back with ``st.context.cookies`` and validated exactly
  like a session-state token. The cookie holds no user data and is deleted on
  sign-out; a revoked session fails validation and the cookie is cleared.
* Signing out revokes the session in the database *before* anything is
  cleared locally; if the server-side revocation fails the UI says so instead
  of claiming success.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from urllib.parse import quote

import streamlit as st

from app.auth import (
    AuthService,
    AuthValidationError,
    DuplicateEmailError,
    InvalidCredentialsError,
)
from app.auth.models import User
from app.auth.security import hash_token
from app.ui import state
from app.ui.components import badge, hero, notice, section_title
from app.ui.icons import span
from app.ui.theme import esc

logger = logging.getLogger(__name__)

#: st.session_state keys owned by the auth layer.
TOKEN_KEY = "be_auth_token"
USER_ID_KEY = "be_auth_user_id"
USER_NAME_KEY = "be_auth_user_name"

#: Browser cookie that carries the opaque session token so a signed-in session
#: survives a full page reload / reconnection. The value is the same random
#: token that is stored (hashed) server-side; it carries no user data and is
#: revoked server-side on sign-out, so a stale cookie is worthless.
COOKIE_NAME = "blueeye_session"

#: Session-state flag recording which token the cookie was last synced with,
#: so the cookie-writing script is not re-emitted on every rerun.
_COOKIE_SYNC_KEY = "be_auth_cookie_synced"

#: One-shot account-page notice (kind, message) shown after the next rerun.
_FLASH_KEY = "be_account_flash"

#: Private workspace pages: anonymous visitors see the sign-in gate here.
#: Species / map / models / about / help stay public.
PRIVATE_PAGES = frozenset(
    {
        "dashboard",
        "detect",
        "observations",
        "history",
        "analytics",
        "compare",
        "evaluation",
        "account",
    }
)

#: One-time notice flags surfaced to the user after a partially failed logout.
_REVOCATION_WARN_KEY = "be_auth_revocation_warning"


def _service() -> AuthService:
    """Build the auth backend for this run (cheap; connects on demand)."""
    return AuthService(settings=state.get_settings_cached())


# --------------------------------------------------------------------------- #
# Session bootstrap
# --------------------------------------------------------------------------- #
def bootstrap() -> User | None:
    """Resolve the authenticated user for this run, or ``None``.

    The token from server-side session state (or, after a reload, the session
    cookie) is validated against the real database (existence, revocation,
    expiry). A dead token is dropped immediately so a revoked/expired session
    degrades to the sign-in screen instead of appearing to persist.
    """
    cookie_token = _read_session_cookie()
    token = st.session_state.get(TOKEN_KEY) or cookie_token
    if not token:
        return None
    user = _service().authenticate(token)
    if user is None:
        st.session_state.pop(TOKEN_KEY, None)
        st.session_state.pop(USER_ID_KEY, None)
        if cookie_token:
            _clear_session_cookie()
            st.session_state.pop(_COOKIE_SYNC_KEY, None)
        return None
    st.session_state[TOKEN_KEY] = token
    st.session_state[USER_ID_KEY] = user.id
    st.session_state[USER_NAME_KEY] = user.full_name
    # Keep the browser cookie in step with the validated session so a reload
    # reconnects to the same account (and a signed-out session is cleaned up).
    # The check avoids re-emitting the script on every rerun.
    if cookie_token != token and st.session_state.get(_COOKIE_SYNC_KEY) != token:
        _set_session_cookie(token)
        st.session_state[_COOKIE_SYNC_KEY] = token
    return user


def _current_token() -> str | None:
    return st.session_state.get(TOKEN_KEY)


# --------------------------------------------------------------------------- #
# Session cookie (survives a full page reload)
# --------------------------------------------------------------------------- #
def _read_session_cookie() -> str | None:
    """Return the session token from the browser cookie, if any."""
    try:
        value = st.context.cookies.get(COOKIE_NAME)
    except Exception:  # noqa: BLE001 - no request context (e.g. AppTest)
        return None
    return value if isinstance(value, str) and value else None


def _is_https() -> bool:
    try:
        return st.context.url.lower().startswith("https")
    except Exception:  # noqa: BLE001 - no request context
        return False


def _set_session_cookie(token: str) -> None:
    """Persist the opaque session token in a SameSite=Lax cookie.

    The value is URL-encoded and only ever a random token, so it is safe to
    interpolate into the tiny inline script. ``Secure`` is added when the app
    is served over HTTPS. The cookie is intentionally not ``HttpOnly`` because
    it has to be written from the page (Streamlit cannot set response headers);
    it holds no personal data and is useless once the server-side session is
    revoked.
    """
    settings = state.get_settings_cached()
    attrs = f"path=/; max-age={settings.session_ttl_seconds}; SameSite=Lax"
    if _is_https():
        attrs += "; Secure"
    cookie = f"{COOKIE_NAME}={quote(token, safe='')}; {attrs}"
    st.html(
        f"<script>document.cookie={cookie!r};</script>",
        unsafe_allow_javascript=True,
    )


def _clear_session_cookie() -> None:
    """Delete the session cookie (sign-out / dead session)."""
    cookie = f"{COOKIE_NAME}=; path=/; max-age=0; SameSite=Lax"
    st.html(
        f"<script>document.cookie={cookie!r};</script>",
        unsafe_allow_javascript=True,
    )


# --------------------------------------------------------------------------- #
# Logout
# --------------------------------------------------------------------------- #
def logout_flow() -> None:
    """Sign out: revoke server-side, then clear local/private state.

    The server-side revocation happens first. If it genuinely cannot be
    performed the local session is still cleared (so private data stops
    rendering) but the UI tells the user that server-side revocation failed
    instead of claiming success.
    """
    service = _service()
    token = _current_token()
    revoked = False
    try:
        revoked = service.logout(token) if token else False
    except Exception:  # noqa: BLE001 - never let logout crash the app
        logger.exception("Session revocation failed during logout")
    state.clear_user_state()
    st.session_state.pop(TOKEN_KEY, None)
    st.session_state.pop(USER_ID_KEY, None)
    st.session_state.pop(_COOKIE_SYNC_KEY, None)
    _clear_session_cookie()
    if token and not revoked:
        st.session_state[_REVOCATION_WARN_KEY] = True
    st.rerun()


def _consume_revocation_warning() -> None:
    if st.session_state.pop(_REVOCATION_WARN_KEY, False):
        notice(
            st,
            "warning",
            "Signed out on this device",
            "The server-side session could not be revoked right now, so a "
            "previously issued session may still be valid on another device. "
            "If you suspect unauthorised access, change your password from a "
            "trusted device.",
        )


# --------------------------------------------------------------------------- #
# Sidebar account block
# --------------------------------------------------------------------------- #
def sidebar_account(user: User | None) -> None:
    """Small identity / sign-out block at the bottom of the sidebar."""
    with st.sidebar:
        if user is None:
            if st.button(
                "Sign in / Create account",
                key="auth_guest_cta",
                width="stretch",
            ):
                state.navigate("dashboard")
            return
        st.markdown(
            '<div class="be-side-acct">'
            f'<div class="be-side-acct-head">{span("account_circle")}'
            f"<div><p class=\"n\">{esc(user.full_name)}</p>"
            f'<p class="e">{esc(user.email)}</p></div></div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Sign out", key="auth_signout", width="stretch"):
            logout_flow()


# --------------------------------------------------------------------------- #
# Gate (sign in / create account)
# --------------------------------------------------------------------------- #
def render_gate() -> None:
    """Full sign-in / registration screen for anonymous visitors."""
    _consume_revocation_warning()
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown(
            '<div class="be-auth">'
            '<div class="be-auth-mark">' + span("visibility") + "</div>"
            "<h1>BlueEye</h1>"
            '<p class="be-auth-sub">See beneath the surface — your workspace '
            "for aquatic life detection</p></div>",
            unsafe_allow_html=True,
        )
        mode = st.radio(
            "Account",
            ["Sign in", "Create account"],
            horizontal=True,
            key="auth_mode",
            index=0,
            label_visibility="collapsed",
        )
        if mode == "Sign in":
            _render_login()
        else:
            _render_register()


def _render_login() -> None:
    settings = state.get_settings_cached()
    col = st.container(border=True)
    col.markdown(
        '<div class="be-auth-title">'
        + span("login")
        + "<span>Welcome back</span></div>",
        unsafe_allow_html=True,
    )
    email = col.text_input("Email", key="auth_login_email", placeholder="you@example.org")
    password = col.text_input(
        "Password",
        type="password",
        key="auth_login_password",
        placeholder="Your password",
    )
    if col.button("Sign in", key="auth_login_submit", width="stretch"):
        if not email.strip() or not password:
            col.error("Enter your email and password.")
        else:
            try:
                _user, token = _service().login(email, password)
            except InvalidCredentialsError:
                col.error("Invalid email or password. Please try again.")
            except Exception:  # noqa: BLE001 - DB/network failures surfaced honestly
                logger.exception("Login failed unexpectedly")
                col.error(
                    "Could not reach the authentication service. Please try again."
                )
            else:
                _establish_session(token)
                _signed_in_as(_user)
                st.rerun()
    col.caption(
        f"Sessions last {settings.session_ttl_days} days and are stored "
        "server-side. A revocable session cookie keeps you signed in across "
        "page reloads."
    )


def _render_register() -> None:
    settings = state.get_settings_cached()
    col = st.container(border=True)
    col.markdown(
        '<div class="be-auth-title">'
        + span("person_add")
        + "<span>Create your account</span></div>",
        unsafe_allow_html=True,
    )
    full_name = col.text_input("Full name", key="auth_register_name")
    email = col.text_input("Email", key="auth_register_email", placeholder="you@example.org")
    col.caption("A working email address that identifies your account.")
    password = col.text_input(
        "Password",
        type="password",
        key="auth_register_password",
        help=f"At least {settings.password_min_length} characters with a "
        "letter and a number.",
    )
    confirm = col.text_input(
        "Confirm password", type="password", key="auth_register_confirm"
    )
    if col.button(
        "Create account", key="auth_register_submit", width="stretch"
    ):
        service = _service()
        try:
            user = service.register(full_name, email, password, confirm)
        except DuplicateEmailError:
            col.error(
                "An account with this email already exists — sign in instead."
            )
        except AuthValidationError as exc:
            col.error(str(exc))
        except Exception:  # noqa: BLE001 - DB failures surfaced honestly
            logger.exception("Registration failed unexpectedly")
            col.error("Could not create the account. Please try again.")
        else:
            # Account created against the real backend: sign in immediately.
            try:
                _user, token = service.login(email, password)
            except Exception:  # pragma: no cover - just-created account must work
                logger.exception("Auto sign-in after registration failed")
                col.error("Account created — please sign in.")
                return
            _establish_session(token)
            _signed_in_as(_user)
            state.navigate("dashboard")


def _establish_session(token: str) -> None:
    # Persist the token in session state; the durable browser cookie is written
    # by :func:`bootstrap` on the next authenticated run (writing it here would
    # be discarded by the immediate rerun after sign-in).
    st.session_state[TOKEN_KEY] = token


def _signed_in_as(user: User) -> None:
    st.session_state[USER_ID_KEY] = user.id
    st.session_state[USER_NAME_KEY] = user.full_name
    # A different account must never inherit the previous account's cached
    # media or result state.
    state.clear_user_state()


# --------------------------------------------------------------------------- #
# Guest banner on public pages
# --------------------------------------------------------------------------- #
def render_guest_banner() -> None:
    """Small call-to-action shown to anonymous visitors on public pages."""
    notice(
        st,
        "info",
        "You are browsing as a guest",
        "Sign in to run detections and keep your history, uploads and "
        "results private to your account.",
        icon="lock",
    )
    if st.button("Sign in / Create account", key="auth_guest_banner_cta"):
        state.navigate("dashboard")


# --------------------------------------------------------------------------- #
# Account page (profile + sessions)
# --------------------------------------------------------------------------- #
def render_account(settings: Any, user: User | None) -> None:
    """Private Account page: identity, profile updates, password change and
    session management."""
    if user is None:
        render_gate()
        return

    hero(
        st,
        "Private to your account",
        "Account",
        "Your identity, password and active sessions.",
        icon="account_circle",
    )
    _consume_revocation_warning()
    _consume_flash()

    _profile_section(user)
    _password_section(settings, user)
    _sessions_section(user)


def _profile_section(user: User) -> None:
    """Identity overview + profile edit form.

    Only user-facing account fields are shown. Internal implementation
    details (database paths, account ids, hashes, tokens) are never rendered.
    """
    section_title(st, "Profile", "The details stored with your account.")
    st.markdown(_identity_html(user), unsafe_allow_html=True)
    st.markdown(
        '<div class="be-infogrid">'
        + _info_card("badge", "Account status", badge("Active", "ok", dot=True))
        + _info_card("calendar_month", "Member since", esc(_fmt_date(user.created_at)))
        + _info_card("login", "Last sign-in", esc(_fmt_ts(user.last_login_at)))
        + "</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        section_title(st, "Edit profile")
        st.caption(
            "Update the name shown across BlueEye. Your email address "
            "identifies your account and cannot be changed here."
        )
        name_input = st.text_input(
            "Full name",
            value=user.full_name,
            key="acct_name",
            max_chars=120,
        )
        if st.button("Save changes", key="acct_save_name", icon=":material/save:"):
            try:
                _service().update_profile(user.id, name_input)
            except AuthValidationError as exc:
                st.error(str(exc))
            except Exception:  # noqa: BLE001
                logger.exception("Profile update failed")
                st.error("Could not update your profile right now. Please try again.")
            else:
                _flash("success", "Your profile has been updated.")
                st.rerun()


def _password_section(settings: Any, user: User) -> None:
    """Change-password form.

    The current password is verified server-side and the new one is validated
    and bcrypt-hashed before it is stored; on success the account's *other*
    devices are signed out. This device stays signed in.
    """
    section_title(st, "Password", "Change the password you sign in with.")
    with st.container(border=True):
        current_pw = st.text_input(
            "Current password", type="password", key="acct_pw_current"
        )
        new_pw = st.text_input(
            "New password",
            type="password",
            key="acct_pw_new",
            help=f"At least {settings.password_min_length} characters with "
            "a letter and a number.",
        )
        confirm_pw = st.text_input(
            "Confirm new password", type="password", key="acct_pw_confirm"
        )
        st.caption(
            f"Minimum {settings.password_min_length} characters, including at "
            "least one letter and one number. Changing your password signs out "
            "every other device."
        )
        if st.button("Change password", key="acct_pw_change", icon=":material/key:"):
            try:
                _service().change_password(
                    user.id,
                    current_pw,
                    new_pw,
                    confirm_pw,
                    keep_token_digest=_current_digest(),
                )
            except (AuthValidationError, InvalidCredentialsError) as exc:
                st.error(str(exc))
            except Exception:  # noqa: BLE001
                logger.exception("Password change failed")
                st.error("Could not change your password right now. Please try again.")
            else:
                _flash(
                    "success",
                    "Your password was changed. Other devices were signed out.",
                )
                st.rerun()


def _sessions_section(user: User) -> None:
    """List the account's sessions; allow revoking any other device.

    Revocation is a real server-side update; the button only reports success
    when the store actually revoked the row.
    """
    section_title(st, "Active sessions", "Devices currently signed in to your account.")
    service = _service()
    current = _current_digest()
    sessions = service.list_sessions(user.id, current_digest=current)

    with st.container(border=True):
        if not sessions:
            st.caption("No active sessions were found.")
        for session in sessions:
            columns = st.columns([2, 3, 3, 1.4, 1.2], vertical_alignment="center")
            if session["current"]:
                columns[0].markdown(
                    badge("This device", "ok", dot=True), unsafe_allow_html=True
                )
            else:
                columns[0].markdown(
                    f'{span("devices")} Other device', unsafe_allow_html=True
                )
            columns[1].caption(f"Signed in {_fmt_ts(session['created_at'])}")
            columns[2].caption(f"Expires {_fmt_ts(session['expires_at'])}")
            columns[3].markdown(
                badge(
                    "Active" if session["active"] else "Revoked",
                    "ok" if session["active"] else "off",
                ),
                unsafe_allow_html=True,
            )
            if session["current"]:
                columns[4].caption("—")
            elif columns[4].button(
                "Revoke", key=f"acct_revoke_{session['id']}", width="stretch"
            ):
                if service.revoke_session_for_user(user.id, session["id"]):
                    _flash("success", "That session has been revoked.")
                    st.rerun()
                else:
                    st.error("That session could not be revoked.")

    if st.button(
        "Sign out everywhere else", key="acct_revoke_all", icon=":material/logout:"
    ):
        try:
            count = service.revoke_other_sessions(user.id, keep_digest=current)
        except Exception:  # noqa: BLE001
            logger.exception("Bulk session revocation failed")
            st.error("Could not revoke the other sessions. Please try again.")
        else:
            state.clear_user_state()
            message = (
                f"Signed out {count} other device(s)."
                if count
                else "No other devices were signed in."
            )
            _flash("success" if count else "info", message)
            st.rerun()


def _current_digest() -> str:
    """SHA-256 digest of the current session token (never the token itself)."""
    return hash_token(_current_token() or "")


def _fmt_ts(iso: str | None) -> str:
    """Human-readable timestamp (``08 Oct 2026, 19:42``); raw if odd."""
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y, %H:%M")
    except (ValueError, TypeError):
        return iso or "—"


def _fmt_date(iso: str | None) -> str:
    """Calendar date only (``08 Oct 2026``); raw if odd."""
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y")
    except (ValueError, TypeError):
        return iso or "—"


def _identity_html(user: User) -> str:
    """Identity banner for the account page (name, email, status)."""
    return (
        '<div class="be-identity">'
        f'<div class="avatar">{span("account_circle")}</div>'
        '<div class="who">'
        f"<h4>{esc(user.full_name)}</h4>"
        f"<p>{esc(user.email)}</p>"
        "</div>"
        f'<div class="state">{badge("Active", "ok", dot=True)}</div>'
        "</div>"
    )


def _info_card(icon: str, label: str, value_html: str) -> str:
    """One aligned account-summary card (``value_html`` is trusted HTML)."""
    return (
        '<div class="be-card">'
        f'<div class="be-card-title">{span(icon)}{esc(label)}</div>'
        f'<p class="be-value">{value_html}</p></div>'
    )


def _flash(kind: str, message: str) -> None:
    """Queue a one-shot account notice to display after the next rerun."""
    st.session_state[_FLASH_KEY] = (kind, message)


def _consume_flash() -> None:
    """Show and clear any queued account-page notice."""
    data = st.session_state.pop(_FLASH_KEY, None)
    if not data:
        return
    kind, message = data
    if kind == "success":
        st.success(message)
    elif kind == "error":
        st.error(message)
    elif kind == "warning":
        st.warning(message)
    else:
        st.info(message)