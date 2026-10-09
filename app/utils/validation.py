"""Input validation and filename sanitisation for BlueEye.

Every file that enters the system (web upload or CLI path) is validated
here: supported extension, non-empty payload, size limits and actual
decodability. Filenames are sanitised so that path traversal and unsafe
names cannot reach the file system.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import re
from pathlib import Path

from app.config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, get_settings

#: User-facing validation failure. The CLI exits with code 2 and the UI
#: shows the message verbatim when this is raised.
class ValidationError(ValueError):
    """Raised when an input file does not pass validation."""


_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_FILENAME_LEN = 120


def sanitize_filename(filename: str, fallback: str = "upload") -> str:
    """Reduce an arbitrary client-supplied name to a safe basename.

    Removes directory components (``../../etc/passwd`` -> ``passwd``),
    replaces unsafe characters, blocks Windows reserved device names and
    truncates overly long names while preserving the extension.
    """
    raw = str(filename or "").replace("\\", "/")
    name = Path(raw).name  # strips any directory component
    name = _UNSAFE_CHARS.sub("_", name)
    name = name.lstrip(".")  # no hidden files / leading dots
    name = name.strip("_")
    if not name:
        return fallback

    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    if stem.lower() in _WINDOWS_RESERVED_NAMES:
        stem = f"upload_{stem}"
    if len(stem) > _MAX_FILENAME_LEN:
        stem = stem[:_MAX_FILENAME_LEN]
    return f"{stem}.{ext}" if ext else stem


def ensure_extension(filename: str, allowed: set[str], kind: str) -> str:
    """Validate the file extension of ``filename`` and return it lower-case."""
    ext = Path(str(filename)).suffix.lower()
    if not ext:
        raise ValidationError(f"{kind} file must have an extension ({', '.join(sorted(allowed))}).")
    if ext not in allowed:
        raise ValidationError(
            f"Unsupported {kind} format '{ext}'. Allowed formats: {', '.join(sorted(allowed))}."
        )
    return ext


def ensure_size(size_bytes: int, limit_bytes: int, kind: str) -> None:
    """Reject empty or oversized payloads."""
    if size_bytes <= 0:
        raise ValidationError(f"The uploaded {kind} file is empty.")
    if size_bytes > limit_bytes:
        limit_mb = limit_bytes // (1024 * 1024)
        raise ValidationError(
            f"The {kind} file is too large ({size_bytes / (1024 * 1024):.1f} MB). "
            f"Maximum allowed size is {limit_mb} MB."
        )


def ensure_safe_path(directory: Path, candidate: Path) -> Path:
    """Make sure ``candidate`` resolves inside ``directory``."""
    directory = Path(directory).resolve()
    candidate = Path(candidate).resolve()
    try:
        candidate.relative_to(directory)
    except ValueError as exc:
        raise ValidationError("Refusing to access a path outside the upload directory.") from exc
    return candidate


#: Magic-byte headers of the formats accepted by :data:`~app.config.IMAGE_EXTENSIONS`.
#: Each entry is one acceptable signature (any of them matches).
_IMAGE_SIGNATURES: tuple[tuple[bytes, ...], ...] = (
    (b"\xff\xd8\xff",),  # JPEG
    (b"\x89PNG\r\n\x1a\n",),  # PNG
    (b"RIFF",),  # WEBP (RIFF....WEBP; PIL checks the rest)
)


def _has_known_image_signature(file_path: Path) -> bool:
    """True when the file starts with a known JPEG/PNG/WEBP magic number."""
    try:
        with file_path.open("rb") as handle:
            header = handle.read(8)
    except OSError:
        return False
    if not header:
        return False
    return any(
        header.startswith(signature) for signatures in _IMAGE_SIGNATURES for signature in signatures
    )


def validate_image_file(path: str | Path) -> Path:
    """Validate an image file on disk (extension, size, decodability)."""
    from PIL import Image, UnidentifiedImageError

    settings = get_settings()
    file_path = Path(path)
    ensure_extension(file_path.name, IMAGE_EXTENSIONS, "image")
    if not file_path.is_file():
        raise ValidationError(f"Image file not found: {file_path.name}")
    ensure_size(file_path.stat().st_size, settings.max_image_mb * 1024 * 1024, "image")
    if not _has_known_image_signature(file_path):
        # Reject garbage before PIL sees it. Besides being faster, this avoids
        # a side effect of the Ultralytics build: it monkey-patches
        # PIL.Image.open so that *any* decode failure triggers a probe for
        # HEIC/HEIF support, which attempts to pip-install "pi-heif" and takes
        # seconds (or crashes) on files that were never images.
        raise ValidationError(f"'{file_path.name}' is not a valid image file.")
    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:  # verify() invalidates the file handle
            img.load()
            if img.format is None:
                raise ValidationError(f"'{file_path.name}' is not a readable image.")
    except ValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValidationError(f"'{file_path.name}' is not a valid image file.") from exc
    except Exception as exc:  # noqa: BLE001 - any decode failure means invalid input
        # Ultralytics monkey-patches PIL.Image.open to probe for HEIC/HEIF
        # support; when the file cannot be identified at all it may raise
        # ModuleNotFoundError (missing pi-heif) instead of UnidentifiedImageError.
        # The contract of this function is "decodable image, or ValidationError",
        # so every failure is normalised here. The original error is chained for
        # the logs.
        raise ValidationError(f"'{file_path.name}' is not a valid image file.") from exc
    return file_path


def validate_video_file(path: str | Path) -> Path:
    """Validate a video file on disk (extension, size, decodability)."""
    import cv2

    settings = get_settings()
    file_path = Path(path)
    ensure_extension(file_path.name, VIDEO_EXTENSIONS, "video")
    if not file_path.is_file():
        raise ValidationError(f"Video file not found: {file_path.name}")
    ensure_size(file_path.stat().st_size, settings.max_video_mb * 1024 * 1024, "video")

    cap = cv2.VideoCapture(str(file_path))
    try:
        if not cap.isOpened():
            raise ValidationError(
                f"'{file_path.name}' could not be opened - the file may be corrupt."
            )
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width <= 0 or height <= 0:
            raise ValidationError(f"'{file_path.name}' has no readable video stream.")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValidationError(f"'{file_path.name}' contains no decodable frames.")
    finally:
        cap.release()
    return file_path


def validate_input_file(path: str | Path, kind: str = "auto") -> Path:
    """Dispatch validation by explicit ``kind`` or by file extension.

    ``kind`` may be ``"image"``, ``"video"`` or ``"auto"`` (infer from the
    extension).
    """
    file_path = Path(path)
    if kind == "image":
        return validate_image_file(file_path)
    if kind == "video":
        return validate_video_file(file_path)
    if kind != "auto":
        raise ValidationError(f"Unknown input kind '{kind}'. Use 'image', 'video' or 'auto'.")

    ext = file_path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return validate_image_file(file_path)
    if ext in VIDEO_EXTENSIONS:
        return validate_video_file(file_path)
    raise ValidationError(
        f"Unsupported file type '{ext or file_path.name}'. "
        f"Images: {', '.join(sorted(IMAGE_EXTENSIONS))}; "
        f"videos: {', '.join(sorted(VIDEO_EXTENSIONS))}."
    )
