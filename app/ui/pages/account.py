"""Account page: profile, real-time settings and session management.

Everything here is private to the authenticated user - a guest hitting the
page (or the page id directly) is sent to the sign-in gate instead. Updates
run through :mod:`app.ui.auth_ui` / :mod:`app.auth.service`, which validate
every operation against the server-side session token.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from app.ui import auth_ui


def render(detector=None, settings=None, user=None) -> None:
    """Render the account workspace for the current user."""
    auth_ui.render_account(settings, user, detector=detector)