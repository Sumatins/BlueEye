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


def load_streamlit_secrets() -> None:
    """Merge ``BLUEEYE_*`` keys from Streamlit secrets into ``os.environ``.

    Streamlit Community Cloud (and ``streamlit run`` with a secrets file)
    expose deployment configuration through ``st.secrets`` rather than process
    environment variables, and BlueEye reads ``BLUEEYE_*`` from the
    environment, so the two are bridged here. Existing environment variables
    are never overwritten, non-``BLUEEYE_`` keys are ignored (a secrets file
    may hold unrelated credentials), and when no secrets file exists this is
    a no-op - local development and the CLI are unaffected.
    """
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.is_file():
        return
    logger = logging.getLogger(__name__)
    try:
        import streamlit as st

        for key in st.secrets.keys():
            if not key.startswith(ENV_PREFIX):
                continue
            value = st.secrets[key]
            if not isinstance(value, str):
                logger.warning("Ignoring non-string secret %s (expected a string).", key)
                continue
            os.environ.setdefault(key, value)
    except Exception as exc:  # noqa: BLE001 - secrets are optional; never block startup
        logger.warning("Could not read %s: %s", secrets_path, exc)


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


def _env_list(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma-separated environment variable into a tuple of strings."""
    raw = os.environ.get(ENV_PREFIX + key)
    if raw is None:
        return default
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    return items


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
    #: Automatically fetch the core model weights when they are missing
    #: (ephemeral deployments lose ``models/`` on every restart). See
    #: :mod:`app.detection.bootstrap`; disable with
    #: ``BLUEEYE_AUTO_DOWNLOAD_WEIGHTS=off``.
    auto_download_weights: bool = True
    #: How many loaded models are kept in memory at once (least recently
    #: used evicted first). The Community Cloud tier has ~1 GB RAM, so a
    #: "run every installed model" pass must not keep every checkpoint
    #: resident. ``1`` = strictly one at a time;
    #: tune with ``BLUEEYE_MAX_LOADED_MODELS``.
    max_loaded_models: int = 2

    # Upload safety limits.
    max_image_mb: int = DEFAULT_MAX_IMAGE_MB
    max_video_mb: int = DEFAULT_MAX_VIDEO_MB

    # Authentication (see app/auth). The database lives under ``outputs/``
    # unless overridden, so it is never committed and survives container
    # restarts when the outputs volume is mounted.
    auth_db: Path | None = None
    #: Persistent observation records database (see app/observations). Kept in
    #: a *separate* SQLite file from authentication so observation migrations
    #: can never touch accounts or sessions.
    observations_db: Path | None = None
    #: Persistent research-evaluation experiments database (see app.evaluation).
    #: Another *separate* SQLite file so evaluation migrations can never touch
    #: accounts, sessions or saved observations.
    evaluation_db: Path | None = None
    #: How long a session token stays valid (days, server-side expiry).
    session_ttl_days: int = 7
    #: Minimum password length enforced at registration / password change.
    password_min_length: int = 8
    #: Emails allowed to *formally verify* an identification. Empty by default:
    #: a user may correct their own records, but verified status is only ever
    #: granted to an explicitly configured reviewer. Never guessed or inferred.
    reviewer_emails: tuple[str, ...] = ()

    # Continuous / scheduled evaluation (see app.evaluation.scheduler).
    #: When enabled, a background worker re-runs *due* schedules while the
    #: application process is alive. Set ``BLUEEYE_SCHEDULER=off`` to disable.
    scheduler_enabled: bool = True
    #: How often the worker checks for due schedules (seconds, floored at 5).
    scheduler_tick_seconds: int = 20

    # Logging.
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    # Authentication paths
    # ------------------------------------------------------------------ #
    @property
    def auth_db_path(self) -> Path:
        """Location of the SQLite authentication database.

        Defaults to ``<outputs>/auth/blueeye_auth.db`` so that user accounts
        and sessions live with the rest of the runtime data and are covered
        by the same backup / volume mount as detection history.
        """
        if self.auth_db is None:
            return self.outputs_dir / "auth" / "blueeye_auth.db"
        path = self.auth_db
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def observations_db_path(self) -> Path:
        """Location of the SQLite observation-records database.

        Defaults to ``<outputs>/observations/blueeye_observations.db`` so saved
        observations live alongside all other runtime data and are covered by
        the same backup / volume mount. This file is independent of the auth
        database, so the two migration timelines can never interfere.
        """
        if self.observations_db is None:
            return self.outputs_dir / "observations" / "blueeye_observations.db"
        path = self.observations_db
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def evaluation_db_path(self) -> Path:
        """Path of the persistent evaluation-experiments database.

        Defaults to ``outputs/evaluation/`` and is overridable with
        ``BLUEEYE_EVALUATION_DB``. It is deliberately separate from both the
        authentication database and the observations database.
        """
        if self.evaluation_db is None:
            return self.outputs_dir / "evaluation" / "blueeye_evaluation.db"
        path = self.evaluation_db
        return path if path.is_absolute() else PROJECT_ROOT / path

    def is_reviewer(self, email: str | None) -> bool:
        """Whether ``email`` may formally verify an identification.

        Returns ``False`` unless the account is listed in
        ``BLUEEYE_REVIEWER_EMAILS``. This is the only path to ``verified``
        status: a user can always correct their own records, but cannot
        self-certify a species identification.
        """
        if not email:
            return False
        wanted = email.strip().lower()
        return any(wanted == candidate.strip().lower() for candidate in self.reviewer_emails)

    @property
    def session_ttl_seconds(self) -> int:
        """Session lifetime in seconds (never below one day)."""
        return max(1, self.session_ttl_days) * 86400

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
            self.observations_db_path.parent,
            self.evaluation_db_path.parent,
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
    load_streamlit_secrets()

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
        auto_download_weights=_env_str("AUTO_DOWNLOAD_WEIGHTS", "on").lower()
        in {"on", "true", "1", "yes"},
        max_loaded_models=max(1, _env_int("MAX_LOADED_MODELS", 2)),
        max_image_mb=_env_int("MAX_IMAGE_MB", DEFAULT_MAX_IMAGE_MB),
        max_video_mb=_env_int("MAX_VIDEO_MB", DEFAULT_MAX_VIDEO_MB),
        auth_db=_resolve_path(_env_str("AUTH_DB", ""), PROJECT_ROOT / "outputs" / "auth" / "blueeye_auth.db")
        if _env_str("AUTH_DB", "")
        else None,
        observations_db=_resolve_path(
            _env_str("OBSERVATIONS_DB", ""),
            PROJECT_ROOT / "outputs" / "observations" / "blueeye_observations.db",
        )
        if _env_str("OBSERVATIONS_DB", "")
        else None,
        evaluation_db=_resolve_path(
            _env_str("EVALUATION_DB", ""),
            PROJECT_ROOT / "outputs" / "evaluation" / "blueeye_evaluation.db",
        )
        if _env_str("EVALUATION_DB", "")
        else None,
        session_ttl_days=_env_int("SESSION_TTL_DAYS", 7),
        password_min_length=_env_int("PASSWORD_MIN_LENGTH", 8),
        reviewer_emails=_env_list("REVIEWER_EMAILS", ()),
        scheduler_enabled=_env_str("SCHEDULER", "on").lower() in {"on", "true", "1", "yes"},
        scheduler_tick_seconds=max(5, _env_int("SCHEDULER_TICK", 20)),
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
