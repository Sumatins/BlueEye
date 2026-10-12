"""Authentication data types (users and sessions).

Dates are stored as ISO-8601 UTC strings (``timespec="seconds"``) so they
serialise cleanly into SQLite and JSON and compare lexicographically.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class User:
    """An authenticated BlueEye account.

    ``id`` is a stable, server-generated UUID (hex) that the browser never
    controls. It is the single identity used to scope every piece of
    per-user data (detection history, uploads, outputs, reports).
    """

    id: str
    email: str
    full_name: str
    password_hash: str = field(repr=False)  # never render in logs/reprs
    created_at: str = ""
    last_login_at: str | None = None
    # Optional professional profile. All empty until the user sets them; they
    # are free text, trimmed and length-checked, and always escaped on render.
    profession: str = ""
    organization: str = ""
    country: str = ""
    research_interests: str = ""
    bio: str = ""

    @property
    def has_profile_details(self) -> bool:
        """True when any optional professional detail has been provided."""
        return any(
            (self.profession, self.organization, self.country, self.research_interests, self.bio)
        )


@dataclass
class Session:
    """One server-side login session.

    ``token_digest`` is the SHA-256 of the raw token; the raw value exists
    only in the caller's session state, never in the database.
    """

    id: str
    user_id: str
    token_digest: str = field(repr=False)
    created_at: str = ""
    expires_at: str = ""
    revoked_at: str | None = None
    last_seen_at: str | None = None
    user_agent: str | None = None

    @property
    def active(self) -> bool:
        """True while not revoked (callers also check the expiry deadline)."""
        return self.revoked_at is None