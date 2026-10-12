"""Cryptographic primitives for authentication.

* Password hashing: bcrypt (OWASP-recommended, maintained, already part of
  the installed environment).
* Session tokens: 256-bit random values; passwords and tokens are never
  logged. Only the SHA-256 digest of a token is stored, exactly like the
  "token hashing" pattern recommended for API sessions.

Constant-time comparisons (``hmac.compare_digest``) are used wherever a
secret is compared with user-supplied data.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

import bcrypt

#: bcrypt only uses the first 72 bytes of a password; rejecting longer
#: inputs avoids surprising truncation while keeping the policy simple.
MAX_PASSWORD_BYTES = 72

#: bcrypt cost factor. 12 is the current OWASP recommendation and gives a
#: sub-second hash on modern hardware, which also throttles brute force.
BCRYPT_ROUNDS = 12

#: A real bcrypt hash of a random secret, verified against on failed logins
#: so that "unknown email" and "wrong password" take the same time.
_DUMMY_HASH = bcrypt.hashpw(b"blueeye-dummy-comparison-secret", bcrypt.gensalt(rounds=BCRYPT_ROUNDS))

#: Simple structural email check: local part + domain + dot + TLD, ASCII.
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

#: Reasonable boundaries for a display name.
MAX_NAME_LENGTH = 120
MIN_NAME_LENGTH = 1

#: Bounds for the optional, free-text professional profile fields.
MAX_PROFILE_FIELD = 160
MAX_PROFILE_BIO = 600


def generate_token() -> str:
    """A new opaque session token (32 random bytes, URL-safe base64)."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """SHA-256 digest of a session token (the only form stored in the DB)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_match(digest_a: str, digest_b: str) -> bool:
    """Constant-time digest comparison."""
    return hmac.compare_digest(digest_a.encode("utf-8"), digest_b.encode("utf-8"))


def hash_password(password: str) -> str:
    """Hash a password with bcrypt; returns an ASCII string for storage."""
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    ).decode("ascii")


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored bcrypt hash.

    Unreadable/stale hashes return ``False`` (never raise), and malformed
    inputs still perform a dummy comparison to keep timing flat.
    """
    try:
        return bool(
            bcrypt.checkpw(
                password.encode("utf-8"), stored_hash.encode("ascii")
            )
        )
    except (ValueError, TypeError):  # malformed stored hash -> login fails safely
        # Perform a dummy check so timing does not hint at the failure.
        bcrypt.checkpw(password.encode("utf-8"), _DUMMY_HASH)
        return False


def dummy_verify(password: str) -> None:
    """Burn roughly the same time as a real hash check (anti-enumeration)."""
    bcrypt.checkpw(password.encode("utf-8"), _DUMMY_HASH)


def normalize_email(email: str) -> str:
    """Normalise an email for storage and lookup (trim + lowercase)."""
    return (email or "").strip().lower()


def validate_email(email: str) -> str:
    """Validate and normalise an email address.

    Returns the normalised address or raises ``ValueError`` with a
    user-facing message.
    """
    value = normalize_email(email)
    if not value:
        raise ValueError("Enter your email address.")
    if len(value) > 254:
        raise ValueError("That email address is too long.")
    if not _EMAIL_RE.fullmatch(value):
        raise ValueError("Enter a valid email address (for example name@example.org).")
    return value


def validate_password(password: str, min_length: int) -> None:
    """Enforce the password policy; raises ``ValueError`` on failure."""
    if not password:
        raise ValueError("Choose a password.")
    if len(password) < min_length:
        raise ValueError(
            f"Password must be at least {min_length} characters long."
        )
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValueError(
            "Password is too long (maximum 72 bytes; avoid very long passphrases)."
        )
    if not any(ch.isalpha() for ch in password):
        raise ValueError("Password must contain at least one letter.")
    if not any(ch.isdigit() for ch in password):
        raise ValueError("Password must contain at least one number.")


def validate_name(full_name: str) -> str:
    """Validate a display name; returns the stripped value."""
    value = (full_name or "").strip()
    if not value:
        raise ValueError("Enter your full name.")
    if len(value) > MAX_NAME_LENGTH:
        raise ValueError(f"Name must be at most {MAX_NAME_LENGTH} characters.")
    return value


def validate_optional_field(value: str | None, label: str, max_length: int) -> str:
    """Trim an optional profile field; raises ``ValueError`` when too long.

    Empty is always allowed (it clears the field). No format is imposed - the
    value is free text and is HTML-escaped wherever it is rendered.
    """
    text = (value or "").strip()
    if len(text) > max_length:
        raise ValueError(f"{label} must be at most {max_length} characters.")
    return text