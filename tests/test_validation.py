"""Unit tests for input validation and filename sanitisation."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.utils.validation import (
    ValidationError,
    ensure_extension,
    ensure_safe_path,
    ensure_size,
    sanitize_filename,
    validate_image_file,
    validate_input_file,
    validate_video_file,
)
from app.config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


# --------------------------------------------------------------------------- #
# Filename sanitisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("photo.jpg", "photo.jpg"),
        ("my photo 1.png", "my_photo_1.png"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\evil.exe.jpg", "evil.exe.jpg"),
        ("C:\\Users\\diver\\dive.png", "dive.png"),
        ("..hidden.png", "hidden.png"),
        ("", "upload"),
        ("...", "upload"),
        ("con.jpg", "upload_con.jpg"),
        ("a" * 300 + ".jpg", "a" * 120 + ".jpg"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_sanitized_filename_has_no_path_separators() -> None:
    name = sanitize_filename("../../secret/data.txt")
    assert "/" not in name and "\\" not in name and name == "data.txt"


# --------------------------------------------------------------------------- #
# Extension / size / path guards
# --------------------------------------------------------------------------- #
def test_ensure_extension_accepts_supported_formats() -> None:
    for ext in IMAGE_EXTENSIONS:
        assert ensure_extension(f"a{ext}", IMAGE_EXTENSIONS, "image") == ext


def test_ensure_extension_rejects_unknown() -> None:
    with pytest.raises(ValidationError, match="Unsupported image format"):
        ensure_extension("malware.exe", IMAGE_EXTENSIONS, "image")
    with pytest.raises(ValidationError, match="Unsupported video format"):
        ensure_extension("notes.txt", VIDEO_EXTENSIONS, "video")


def test_ensure_extension_rejects_missing_extension() -> None:
    with pytest.raises(ValidationError, match="extension"):
        ensure_extension("noextension", IMAGE_EXTENSIONS, "image")


def test_ensure_size_rejects_empty_and_oversized() -> None:
    with pytest.raises(ValidationError, match="empty"):
        ensure_size(0, 1024, "image")
    with pytest.raises(ValidationError, match="too large"):
        ensure_size(2048, 1024, "video")
    ensure_size(512, 1024, "image")  # no exception


def test_ensure_safe_path_blocks_traversal(tmp_path: Path) -> None:
    inside = tmp_path / "upload.png"
    ensure_safe_path(tmp_path, inside)  # ok
    with pytest.raises(ValidationError, match="outside"):
        ensure_safe_path(tmp_path, tmp_path.parent / "elsewhere.png")


# --------------------------------------------------------------------------- #
# File validation
# --------------------------------------------------------------------------- #
def test_validate_image_file_accepts_real_image(sample_image: Path) -> None:
    assert validate_image_file(sample_image) == sample_image


def test_validate_image_file_rejects_corrupt(corrupt_image: Path) -> None:
    with pytest.raises(ValidationError, match="not a valid image"):
        validate_image_file(corrupt_image)


def test_validate_image_file_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not found"):
        validate_image_file(tmp_path / "ghost.jpg")


def test_validate_image_file_rejects_wrong_extension(sample_image: Path, tmp_path: Path) -> None:
    bad = tmp_path / "clip.mp4"
    bad.write_bytes(sample_image.read_bytes())
    with pytest.raises(ValidationError, match="Unsupported image format"):
        validate_image_file(bad)


def test_validate_video_file_accepts_real_video(sample_video: Path) -> None:
    assert validate_video_file(sample_video) == sample_video


def test_validate_video_file_rejects_corrupt(tmp_path: Path) -> None:
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"not a video at all")
    with pytest.raises(ValidationError):
        validate_video_file(fake)


def test_validate_video_file_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not found"):
        validate_video_file(tmp_path / "ghost.mp4")


def test_validate_input_file_dispatches_by_extension(sample_image: Path, sample_video: Path) -> None:
    assert validate_input_file(sample_image) == sample_image
    assert validate_input_file(sample_video, kind="video") == sample_video
    with pytest.raises(ValidationError, match="Unsupported file type"):
        validate_input_file(sample_image.with_suffix(".xyz"))
    with pytest.raises(ValidationError, match="Unknown input kind"):
        validate_input_file(sample_image, kind="audio")
