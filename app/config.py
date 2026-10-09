"""BlueEye application configuration and logging setup.

All runtime paths and defaults are resolved here so that no other module
hard-codes file system locations. Values can be overridden through
environment variables or an optional ``.env`` file in the project root
(see ``.env.example``).

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Supported input formats.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

#: Default confidence used by the UI slider / CLI when the user does not
#: pick a value explicitly. The *recommended* per-model thresholds
#: (FishInv 0.523, MegaFauna 0.546) live with the model registry in
#: ``app.detection.model_manager`` and are used when confidence is ``None``.
DEFAULT_CONFIDENCE = 0.5

#: Default model selection: run every available model (combined mode).
DEFAULT_MODEL_SELECTION = "auto"

#: "auto" picks CUDA when available, otherwise CPU. CUDA is never mandatory.
DEFAULT_DEVICE = "auto"

#: Upload safety limits (megabytes).
DEFAULT_MAX_IMAGE_MB = 25
DEFAULT_MAX_VIDEO_MB = 400

ENV_PREFIX = "BLUEEYE_"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def load_env_file(path: Path | None = None) -> None:
    """Load a simple ``KEY=VALUE`` .env file into ``os.environ``.

    Existing environment variables are never overwritten. Quotes around
    values are stripped; blank lines and ``#`` comments are ignored.
    """
    env_path = path or PROJECT_ROOT / ".env"
    try:
        if not env_path.is_file():
            return
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:  # pragma: no cover - defensive
        logging.getLogger(__name__).warning("Could not read %s: %s", env_path, exc)


def _env_str(key: str, default: str) -> str:
    return os.environ.get(ENV_PREFIX + key, default)


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(ENV_PREFIX + key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logging.getLogger(__name__).warning("Invalid float for %s%s: %r", ENV_PREFIX, key, raw)
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(ENV_PREFIX + key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).warning("Invalid int for %s%s: %r", ENV_PREFIX, key, raw)
        return default


def _resolve_path(value: str | Path, default: Path) -> Path:
    """Resolve a possibly relative path against the project root."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve() if path.exists() else path


@dataclass
class Settings:
    """Runtime settings for the BlueEye application."""

    # Directories (defaults below are relative to the project root).
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    outputs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")

    # Behaviour defaults.
    default_model_selection: str = DEFAULT_MODEL_SELECTION
    default_confidence: float = DEFAULT_CONFIDENCE
    default_device: str = DEFAULT_DEVICE
    enhancement_enabled: bool = False

    # Upload safety limits.
    max_image_mb: int = DEFAULT_MAX_IMAGE_MB
    max_video_mb: int = DEFAULT_MAX_VIDEO_MB

    # Logging.
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    # Derived directories
    # ------------------------------------------------------------------ #
    @property
    def input_dir(self) -> Path:
        """Directory where user uploads are stored."""
        return self.data_dir / "input"

    @property
    def images_output_dir(self) -> Path:
        return self.outputs_dir / "images"

    @property
    def videos_output_dir(self) -> Path:
        return self.outputs_dir / "videos"

    @property
    def reports_output_dir(self) -> Path:
        return self.outputs_dir / "reports"

    def ensure_dirs(self) -> None:
        """Create every directory the application writes to."""
        for directory in (
            self.models_dir,
            self.input_dir,
            self.images_output_dir,
            self.videos_output_dir,
            self.reports_output_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Device selection
    # ------------------------------------------------------------------ #
    def resolve_device(self) -> str:
        """Return the compute device to use.

        ``CUDA`` is used automatically when it is available and no explicit
        device was requested; otherwise the code falls back to CPU. A CUDA
        request on a machine without CUDA logs a warning and falls back to
        CPU instead of crashing.
        """
        requested = (self.default_device or "auto").strip().lower()
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a hard dependency
            if requested.startswith("cuda"):
                logging.getLogger(__name__).warning(
                    "CUDA requested but PyTorch is not available; using CPU."
                )
            return "cpu"

        if requested == "cpu":
            return "cpu"

        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                return requested
            logging.getLogger(__name__).warning(
                "CUDA device %r requested but not available; using CPU.", requested
            )
            return "cpu"

        # auto
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        return device


def get_settings() -> Settings:
    """Build (and cache) application settings from the environment."""
    cached = getattr(get_settings, "_cache", None)
    if cached is not None:
        return cached

    load_env_file()

    settings = Settings(
        models_dir=_resolve_path(
            _env_str("MODELS_DIR", str(PROJECT_ROOT / "models")), PROJECT_ROOT / "models"
        ),
        data_dir=_resolve_path(
            _env_str("DATA_DIR", str(PROJECT_ROOT / "data")), PROJECT_ROOT / "data"
        ),
        outputs_dir=_resolve_path(
            _env_str("OUTPUTS_DIR", str(PROJECT_ROOT / "outputs")), PROJECT_ROOT / "outputs"
        ),
        default_model_selection=_env_str("DEFAULT_MODEL", DEFAULT_MODEL_SELECTION),
        default_confidence=_env_float("DEFAULT_CONFIDENCE", DEFAULT_CONFIDENCE),
        default_device=_env_str("DEVICE", DEFAULT_DEVICE),
        enhancement_enabled=_env_str("ENHANCEMENT", "off").lower() in {"on", "true", "1", "yes"},
        max_image_mb=_env_int("MAX_IMAGE_MB", DEFAULT_MAX_IMAGE_MB),
        max_video_mb=_env_int("MAX_VIDEO_MB", DEFAULT_MAX_VIDEO_MB),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
    )
    get_settings._cache = settings  # type: ignore[attr-defined]
    return settings


def reset_settings_cache() -> None:
    """Drop the cached settings (used by tests)."""
    if hasattr(get_settings, "_cache"):
        delattr(get_settings, "_cache")


def setup_logging(level: str | int = logging.INFO) -> None:
    """Configure structured application-wide logging (idempotent)."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        root.addHandler(handler)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(level)
    # Keep third-party loggers quieter by default.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
