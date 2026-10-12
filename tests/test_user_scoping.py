"""Tests for per-account data scoping (history, uploads, outputs, ownership).

These are pure-function tests: they verify that a server-verified user id is
what routes data into per-account directories and that the ownership guard
refuses paths belonging to another account.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings
from app.ui import state
from app.utils.history import HistoryEntry, add_entry, history_path, load_entries
from app.utils.validation import ValidationError

USER_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
USER_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

REPORT = {
    "input": "reef.jpg",
    "timestamp": "2026-10-08T19:42:00",
    "models": ["fish_inv"],
    "statistics": {"total_detections": 1, "unique_classes": 1, "detections_per_class": {"fish": 1}},
}


# --------------------------------------------------------------------------- #
# History scoping
# --------------------------------------------------------------------------- #
def test_history_path_is_per_user(settings: Settings) -> None:
    path_a = history_path(settings, user_id=USER_A)
    path_b = history_path(settings, user_id=USER_B)
    legacy = history_path(settings)

    assert path_a != path_b != legacy
    assert USER_A in path_a.parts
    assert USER_B in path_b.parts
    assert path_a.parent.parent.name == "users"


def test_two_users_never_share_history(settings: Settings) -> None:
    entry_a = HistoryEntry.from_report("image", REPORT, "out.jpg", user_id=USER_A)
    entry_b = HistoryEntry.from_report("image", REPORT, "out.jpg", user_id=USER_B)
    add_entry(entry_a, history_path(settings, user_id=USER_A))
    add_entry(entry_b, history_path(settings, user_id=USER_B))

    assert [e.user_id for e in load_entries(history_path(settings, user_id=USER_A))] == [USER_A]
    assert [e.user_id for e in load_entries(history_path(settings, user_id=USER_B))] == [USER_B]
    # The legacy (unowned) store is untouched.
    assert load_entries(history_path(settings)) == []


def test_history_entry_user_id_round_trips() -> None:
    entry = HistoryEntry.from_report("image", REPORT, "out.jpg", user_id=USER_A)
    restored = HistoryEntry.from_dict(entry.to_dict())
    assert restored.user_id == USER_A


# --------------------------------------------------------------------------- #
# Upload / output directories
# --------------------------------------------------------------------------- #
def test_user_dir_creates_scoped_folders(settings: Settings) -> None:
    for area in ("input", "images", "videos", "reports"):
        directory = state.user_dir(settings, USER_A, area)
        assert directory.is_dir()
        assert directory.name == USER_A
    assert state.user_dir(settings, USER_A, "input") != state.user_dir(settings, USER_B, "input")


def test_user_dir_rejects_unknown_area(settings: Settings) -> None:
    try:
        state.user_dir(settings, USER_A, "secrets")
    except KeyError:
        return
    raise AssertionError("unknown area should raise KeyError")


# --------------------------------------------------------------------------- #
# Ownership guard
# --------------------------------------------------------------------------- #
def test_is_owned_path_accepts_only_own_files(settings: Settings) -> None:
    own = state.user_dir(settings, USER_A, "images") / "detection_1.jpg"
    own.write_bytes(b"x")
    other = state.user_dir(settings, USER_B, "images") / "detection_2.jpg"
    other.write_bytes(b"x")

    assert state.is_owned_path(settings, USER_A, own) is True
    assert state.is_owned_path(settings, USER_A, other) is False
    assert state.is_owned_path(settings, USER_A, settings.images_output_dir / "orphan.jpg") is False


def test_is_owned_path_handles_missing_identity_and_bad_paths(settings: Settings) -> None:
    assert state.is_owned_path(settings, None, "anything") is False
    assert state.is_owned_path(settings, USER_A, None) is False
    assert state.is_owned_path(settings, USER_A, "") is False
    assert state.is_owned_path(settings, USER_A, "https://example.org/file.jpg") is False


def test_history_files_are_owned(settings: Settings) -> None:
    path = history_path(settings, user_id=USER_A)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")
    assert state.is_owned_path(settings, USER_A, path) is True
    assert state.is_owned_path(settings, USER_B, path) is False


# --------------------------------------------------------------------------- #
# Original-input resolution (history viewer)
# --------------------------------------------------------------------------- #
def test_resolve_input_path_finds_only_the_owners_upload(settings: Settings) -> None:
    own = state.user_dir(settings, USER_A, "input") / "photo.png"
    own.write_bytes(b"x")

    assert state.resolve_input_path(settings, USER_A, "photo.png") == own
    # Another account cannot resolve the file, even by name.
    assert state.resolve_input_path(settings, USER_B, "photo.png") is None


def test_resolve_input_path_tolerates_blank_and_missing(settings: Settings) -> None:
    assert state.resolve_input_path(settings, USER_A, None) is None
    assert state.resolve_input_path(settings, USER_A, "") is None
    assert state.resolve_input_path(settings, USER_A, "unknown") is None
    assert state.resolve_input_path(settings, USER_A, "does_not_exist.png") is None


def test_resolve_input_path_accepts_an_absolute_file(settings: Settings, tmp_path: Path) -> None:
    external = tmp_path / "outside.jpg"
    external.write_bytes(b"x")
    assert state.resolve_input_path(settings, USER_A, str(external)) == external


def test_resolve_owned_input_never_escapes_the_account(settings: Settings, tmp_path: Path) -> None:
    external = tmp_path / "elsewhere.jpg"
    external.write_bytes(b"x")
    # The insecure resolver can find it, but the owned one refuses it.
    assert state.resolve_input_path(settings, USER_A, str(external)) == external
    assert state.resolve_owned_input(settings, USER_A, str(external)) is None

    own = settings.input_dir / USER_A / "mine.png"
    own.parent.mkdir(parents=True, exist_ok=True)
    own.write_bytes(b"x")
    assert state.resolve_owned_input(settings, USER_A, "mine.png") == own


# --------------------------------------------------------------------------- #
# Upload intake cleanliness
# --------------------------------------------------------------------------- #
class _FakeUpload:
    """Minimal stand-in for a Streamlit ``UploadedFile`` in pure unit tests."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self.size = len(data)
        self._data = data

    def getvalue(self) -> bytes:
        return self._data

    def getbuffer(self) -> bytes:
        return self._data


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (12, 34, 56)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_save_upload_stores_a_valid_image_for_the_account(settings: Settings) -> None:
    upload = _FakeUpload("reef.png", _png_bytes())

    target = state.save_upload(upload, settings, "image", user_id=USER_A)

    assert target.parent == settings.input_dir / USER_A
    assert target.read_bytes() == upload.getvalue()


def test_save_upload_discards_an_undecodable_file(settings: Settings) -> None:
    """A rejected upload must not be left behind in the input directory."""
    upload = _FakeUpload("broken.png", b"definitely not an image")

    with pytest.raises(ValidationError):
        state.save_upload(upload, settings, "image", user_id=USER_A)

    account_input = settings.input_dir / USER_A
    assert list(account_input.glob("*.png")) == []


def test_save_upload_rejects_unknown_kind(settings: Settings) -> None:
    upload = _FakeUpload("reef.png", _png_bytes())

    with pytest.raises(ValidationError):
        state.save_upload(upload, settings, "audio", user_id=USER_A)
