"""BlueEye authentication.

A self-contained, database-backed authentication provider for the Streamlit
application. There is no separate API server in this project, so the auth
"backend" lives here and is exercised by the web UI through real SQLite
operations, bcrypt password hashing and server-side session tokens.

Design notes
------------
* Passwords are hashed with bcrypt (``app/auth/security.py``). Plaintext is
  never stored, logged or returned.
* Sessions are opaque 256-bit random tokens. Only their SHA-256 digest is
  stored in the database, so a leaked database cannot be replayed directly.
  Tokens expire on a server-side deadline and can be revoked immediately.
* The web UI keeps a validated token in Streamlit's server-side session
  state and re-validates it against the database on every run (see
  ``app/ui/auth_ui.py``). Because Streamlit session state does not survive a
  full page reload, the same opaque token is mirrored in a SameSite=Lax
  browser cookie and re-validated on load; the cookie carries no user data and
  is cleared on sign-out. No browser-supplied value is ever treated as
  identity.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

#: The database schema version understood by this code (see store._ensure_schema).
#: Defined before the submodule imports below so that ``app.auth.store`` can
#: import it without hitting a partially-initialised package.
#: v2 adds the optional professional profile fields to ``users``.
SCHEMA_VERSION = 2

from app.auth.errors import (  # noqa: E402 - public error surface
    AuthError,
    AuthValidationError,
    DuplicateEmailError,
    InvalidCredentialsError,
)
from app.auth.models import Session, User  # noqa: E402 - public data types
from app.auth.service import AuthService  # noqa: E402 - public API

__all__ = [
    "SCHEMA_VERSION",
    "AuthError",
    "AuthValidationError",
    "DuplicateEmailError",
    "InvalidCredentialsError",
    "Session",
    "User",
    "AuthService",
]