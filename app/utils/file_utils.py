"""File-system helpers: safe paths, unique output names, image and JSON I/O.

Reading/writing goes through Pillow where possible so that unicode paths
(e.g. ``D:\\uploads\\sea_photo.jpg``) and EXIF orientation are handled
correctly on every platform.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.config import Settings, get_settings


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_output_dirs(settings: Settings | None = None) -> Settings:
    """Create the standard output directories and return the settings."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    return settings


def unique_output_path(directory: str | Path, prefix: str, extension: str) -> Path:
    """Build a unique, timestamped output path such as
    ``outputs/images/detection_20260101_120000_ab12cd.jpg``.

    A short random suffix guarantees that two detections started within the
    same second never overwrite each other.
    """
    directory = ensure_dir(directory)
    extension = extension if extension.startswith(".") else f".{extension}"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for _ in range(50):
        candidate = directory / f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}{extension}"
        if not candidate.exists():
            return candidate
    raise OSError(f"Could not allocate a unique output name in {directory}")


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Read an image file and return it as a BGR ``uint8`` array.

    EXIF orientation is applied first (same behaviour as the reference
    implementation) so that phone/camera photos are not detected sideways.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image not found: {file_path}")
    with Image.open(file_path) as img:
        img = ImageOps.exif_transpose(img)
        rgb = np.asarray(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_image_bgr(path: str | Path, image: np.ndarray) -> Path:
    """Write a BGR ``uint8`` array to ``path`` (format from the extension)."""
    file_path = Path(path)
    ensure_dir(file_path.parent)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an HxWx3 BGR image array.")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(file_path)
    return file_path


def save_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as pretty-printed UTF-8 JSON."""
    file_path = Path(path)
    ensure_dir(file_path.parent)
    file_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return file_path


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def human_size(size_bytes: int) -> str:
    """Format a byte count for display (``1.5 MB``)."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover - unreachable


def read_file_bytes(path: str | Path) -> bytes:
    """Read a whole file as bytes (used for download buttons)."""
    return Path(path).read_bytes()
