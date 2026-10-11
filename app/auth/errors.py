"""Authentication error types.

Errors carry a user-facing ``message`` that is safe to render in the UI.
Account-security failures use deliberately generic wording so the login form
never reveals whether an email address is registered.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations


class AuthError(Exception):
    """Base class for every authentication failure."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class AuthValidationError(AuthError):
    """User input that failed a validation rule (email, password, name)."""


class DuplicateEmailError(AuthError):
    """An account with the normalised email address already exists."""

    def __init__(self, message: str = "An account with this email already exists.") -> None:
        super().__init__(message)


class InvalidCredentialsError(AuthError):
    """Email/password combination did not match an existing account."""

    def __init__(
        self, message: str = "Invalid email or password. Please try again."
    ) -> None:
        super().__init__(message)