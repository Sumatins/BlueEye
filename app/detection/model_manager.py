"""Model registry, download and cached loading of YOLOv8 weights.

BlueEye ships with the two pretrained models published by the upstream
*marine-detect* project (Orange OpenSource, AGPL-3.0-only):

* ``fish_inv``  - Fish & Invertebrates species (15 classes)
* ``megafauna`` - MegaFauna / rare species: shark, ray, turtle (3 classes)

Weights are **not** bundled with the repository. They are fetched on demand
from the official download links published in the upstream README via
``scripts/download_models.py`` (or the button in the web UI). This avoids
redistributing third-party model files and never fabricates weights.

The registry is **extensible without code changes**: additional YOLOv8
models (for example one trained on regional species) can be registered in
``models/custom/registry.json`` and are then picked up by the CLI, the web
UI and ``auto`` mode. See ``models/custom/README.md``.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Progress callback: (downloaded_bytes, total_bytes_or_None)
ProgressCallback = Callable[[int, Optional[int]], None]

#: Anything below this size after a download is treated as a failed transfer
#: (the real weights are ~87 MB each).
_MIN_WEIGHT_BYTES = 1_000_000

#: Maps :attr:`ModelSpec.loader` to the Ultralytics class that opens it.
LOADERS: dict[str, str] = {"yolo": "YOLO", "rtdetr": "RTDETR"}


class ModelError(Exception):
    """Base class for model-related failures with user-friendly messages."""


class ModelNotAvailableError(ModelError):
    """A required model weight file is missing or the selection is invalid."""


class ModelDownloadError(ModelError):
    """Downloading a model weight file failed or produced an invalid file."""


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a registered model.

    Only ``key``/``display_name``/``filename``/``recommended_confidence`` are
    required; everything else is optional presentation metadata consumed by
    the UI and the CLI. ``url`` is empty for user-registered models (see
    :func:`load_custom_specs`), which are never downloaded automatically.
    """

    key: str
    display_name: str
    filename: str  # relative to the models directory, e.g. "fish_inv/FishInv.pt"
    url: str
    recommended_confidence: float
    description: str = ""
    #: Free-form grouping used by the UI ("marine", "regional", "custom").
    category: str = "marine"
    #: Short, human-readable summary shown on model cards.
    summary: str = ""
    #: Version of the weights ("" when unknown - never guessed).
    version: str = ""
    #: Provenance of the weights, e.g. "Orange OpenSource marine-detect".
    source: str = ""
    #: License of the weights ("" when unknown - never guessed).
    license: str = ""
    #: Class names known ahead of time (empty until weights are loaded).
    classes: tuple[str, ...] = ()
    #: Emoji used on model cards.
    icon: str = "🐟"
    #: ``True`` for user-registered models loaded from ``models/custom``.
    custom: bool = False
    #: Project / dataset homepage ("" when unknown).
    homepage: str = ""
    #: Architecture label shown in the UI (informational only).
    architecture: str = "YOLOv8 (Ultralytics)"
    #: Ultralytics loader used to open these weights: "yolo" or "rtdetr".
    loader: str = "yolo"
    #: Whether the model takes part in the ``auto`` selection. Curated
    #: additions default to opt-in (``False``) so ``auto`` keeps the exact
    #: behaviour of the original two models.
    auto: bool = True
    #: Habitats this model targets, e.g. "Freshwater", "Brackish / estuarine".
    habitat: str = ""
    #: Explicit readiness override for models without usable weights:
    #: ``"needs_training"`` or ``"incompatible"`` ("" derives it from disk).
    readiness: str = ""
    #: Honest one-line note about provenance / availability limits.
    status_note: str = ""
    #: Evaluation metrics measured on a held-out test set, as ordered
    #: ``(label, value)`` pairs (e.g. ``("mAP@50", "0.83")``). Empty when no
    #: metrics were measured - BlueEye never invents evaluation numbers.
    metrics: tuple[tuple[str, str], ...] = ()

    @property
    def short_description(self) -> str:
        return f"{self.display_name} ({self.description})" if self.description else self.display_name

    @property
    def is_downloadable(self) -> bool:
        """True when BlueEye knows an official URL for these weights."""
        return bool(self.url)

    def readiness_status(self, available: bool) -> str:
        """One of ``ready`` / ``not_installed`` / ``needs_training`` / ``incompatible``.

        ``available`` is whether the weight file exists on disk. An explicit
        :attr:`readiness` override always wins, so a model that BlueEye knows
        cannot run yet is never reported as ready.
        """
        if self.readiness:
            return self.readiness
        return "ready" if available else "not_installed"


#: Official weights published by the upstream marine-detect project.
#: The class lists are the species scope documented in the upstream README
#: (they are also re-read from the weights once loaded).
MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            key="fish_inv",
            display_name="Fish & Invertebrates",
            filename="fish_inv/FishInv.pt",
            url=(
                "https://stpubtenakanclyw.blob.core.windows.net/marine-detect/"
                "models2025/FishInv.pt?sv=2022-11-02&ss=bf&srt=co&sp=rltf"
                "&se=2099-12-31T18:55:46Z&st=2025-02-03T10:55:46Z&spr=https,http"
                "&sig=w%2FTQzrECsYsjtkBXNnnuFtn%2BC06PkjgLxDgRw%2FaUUKI%3D"
            ),
            recommended_confidence=0.523,
            description="15 classes: fish families and invertebrates",
            category="marine",
            summary="Reef fish families and invertebrates - the broad workhorse model.",
            source="Orange OpenSource marine-detect",
            license="AGPL-3.0-only (weights as published upstream)",
            homepage="https://github.com/Orange-OpenSource/marine-detect",
            icon="🐟",
            habitat="Marine",
            classes=(
                "fish",
                "serranidae",
                "scaridae",
                "chaetodontidae",
                "lutjanidae",
                "muraenidae",
                "haemulidae",
                "cromileptes_altivelis",
                "cheilinus_undulatus",
                "bolbometopon_muricatum",
                "giant_clam",
                "urchin",
                "sea_cucumber",
                "crown_of_thorns",
                "lobster",
            ),
        ),
        ModelSpec(
            key="megafauna",
            display_name="MegaFauna",
            filename="megafauna/MegaFauna.pt",
            url=(
                "https://stpubtenakanclyw.blob.core.windows.net/marine-detect/"
                "models2025/MegaFauna.pt?sv=2022-11-02&ss=bf&srt=co&sp=rltf"
                "&se=2099-12-31T18:55:46Z&st=2025-02-03T10:55:46Z&spr=https,http"
                "&sig=w%2FTQzrECsYsjtkBXNnnuFtn%2BC06PkjgLxDgRw%2FaUUKI%3D"
            ),
            recommended_confidence=0.546,
            description="3 classes: shark, ray, turtle",
            category="marine",
            summary="Larger and rare marine animals: sharks, rays and sea turtles.",
            source="Orange OpenSource marine-detect",
            license="AGPL-3.0-only (weights as published upstream)",
            homepage="https://github.com/Orange-OpenSource/marine-detect",
            icon="🦈",
            habitat="Marine",
            classes=("shark", "ray", "turtle"),
        ),
    )
}


#: Sub-path (inside the models directory) where user-registered models are
#: described. See ``models/custom/README.md``.
CUSTOM_REGISTRY_PATH = ("custom", "registry.json")

#: Guard rail: a side-loaded registry is never allowed to define this many
#: classes (protects against a nonsense file).
_MAX_CUSTOM_CLASSES = 1000

_CUSTOM_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _custom_registry_file(models_dir: Path) -> Path:
    return models_dir.joinpath(*CUSTOM_REGISTRY_PATH)


def load_custom_specs(models_dir: Path) -> dict[str, ModelSpec]:
    """Read user-registered models (custom / regional) from ``models_dir``.

    This is BlueEye's model-extensibility seam: place a YOLOv8 ``.pt`` file
    anywhere under ``models/`` and describe it in
    ``models/custom/registry.json``; no application code has to change and
    the model is picked up by the UI, the CLI and ``auto`` mode. See
    ``models/custom/README.md`` for the schema and a worked example.

    A malformed entry is **logged and skipped, never fatal** - a broken
    side-loaded file must not stop the built-in models from working.
    Custom models are never downloaded automatically (``url`` is empty).
    """
    registry_file = _custom_registry_file(models_dir)
    logger = logging.getLogger(__name__)
    if not registry_file.is_file():
        return {}

    try:
        payload = json.loads(registry_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring custom model registry %s: %s", registry_file, exc)
        return {}

    entries = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        logger.warning(
            "Ignoring custom model registry %s: expected a JSON array or "
            'an object with a "models" list.',
            registry_file,
        )
        return {}

    specs: dict[str, ModelSpec] = {}
    for index, entry in enumerate(entries):
        spec = _parse_custom_entry(entry, models_dir, index)
        if spec is None:
            continue
        if spec.key in MODEL_REGISTRY or spec.key in specs:
            logger.warning(
                "Custom model '%s' is ignored: the id is already taken.",
                spec.key,
            )
            continue
        specs[spec.key] = spec
    return specs


def _parse_custom_entry(
    entry: object, models_dir: Path, index: int
) -> ModelSpec | None:
    """Validate one custom registry entry; ``None`` when unusable."""
    log = logging.getLogger(__name__)

    def _reject(reason: str) -> None:
        log.warning("Ignoring custom model #%d: %s", index, reason)

    if not isinstance(entry, dict):
        _reject("each entry must be a JSON object.")
        return None

    raw_key = entry.get("id")
    if not isinstance(raw_key, str):
        _reject("id must be a string.")
        return None
    key = raw_key.strip()
    if not _CUSTOM_ID_PATTERN.match(key):
        _reject(f"id {key!r} must be 1-64 characters of [A-Za-z0-9_-].")
        return None

    raw_name = entry.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        _reject(f"model '{key}' has no display name.")
        return None
    name = raw_name.strip()

    raw_weights = entry.get("weights")
    if not isinstance(raw_weights, str) or not raw_weights.strip():
        _reject(f"model '{key}' has no 'weights' path.")
        return None
    weights = raw_weights.strip()
    weight_path = Path(weights)
    if weight_path.is_absolute() or ".." in weight_path.parts:
        _reject(f"model '{key}' has an unsafe weights path {weights!r}.")
        return None
    resolved = (models_dir / weight_path).resolve()
    try:
        resolved.relative_to(Path(models_dir).resolve())
    except ValueError:
        _reject(f"model '{key}' points outside the models directory.")
        return None

    model_type = str(entry.get("type") or "yolov8").strip().lower()
    loader_by_type = {"yolov8": "yolo", "yolo": "yolo", "rtdetr": "rtdetr"}
    if model_type not in loader_by_type:
        _reject(
            f"model '{key}' has unsupported type {model_type!r} "
            "(use 'yolov8' or 'rtdetr')."
        )
        return None
    loader = loader_by_type[model_type]

    raw_confidence = entry.get("recommended_confidence", 0.5)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.5
    if not 0.0 <= confidence <= 1.0:
        _reject(f"model '{key}' has a confidence outside [0, 1].")
        return None

    raw_classes = entry.get("classes") or []
    if not isinstance(raw_classes, list) or len(raw_classes) > _MAX_CUSTOM_CLASSES:
        _reject(f"model '{key}' has an invalid 'classes' list.")
        return None
    classes = tuple(str(c) for c in raw_classes)

    readiness = str(entry.get("readiness") or "").strip().lower()
    if readiness not in {"", "ready", "not_installed", "needs_training", "incompatible"}:
        _reject(f"model '{key}' has an invalid readiness {readiness!r}.")
        return None

    architecture = str(entry.get("architecture") or "").strip()
    if not architecture:
        architecture = "RT-DETR (Ultralytics)" if loader == "rtdetr" else "YOLOv8 (Ultralytics)"

    raw_metrics = entry.get("metrics") or {}
    metrics: list[tuple[str, str]] = []
    if isinstance(raw_metrics, dict):
        for metric_key, metric_value in raw_metrics.items():
            if metric_value is None or str(metric_value).strip() == "":
                continue
            metrics.append((str(metric_key), str(metric_value)))

    return ModelSpec(
        key=key,
        display_name=name,
        filename=weights.replace("\\", "/"),
        url="",  # custom weights are placed by the user, never auto-fetched
        recommended_confidence=confidence,
        description=str(entry.get("description") or "").strip(),
        category=str(entry.get("category") or "custom").strip() or "custom",
        summary=str(entry.get("summary") or "").strip(),
        version=str(entry.get("version") or "").strip(),
        source=str(entry.get("source") or "User-provided").strip(),
        license=str(entry.get("license") or "").strip(),
        classes=classes,
        icon=str(entry.get("icon") or "🌊").strip() or "🌊",
        custom=True,
        homepage=str(entry.get("homepage") or "").strip(),
        architecture=architecture,
        loader=loader,
        auto=bool(entry.get("auto", True)),
        habitat=str(entry.get("habitat") or "").strip(),
        readiness=readiness,
        status_note=str(entry.get("status_note") or "").strip(),
        metrics=tuple(metrics),
    )


def iter_specs(manager: Any) -> list[ModelSpec]:
    """Every model spec visible through ``manager``.

    Uses :meth:`ModelManager.registry` when available (built-in + custom
    models) and falls back to the built-in registry for manager doubles
    that only implement the older interface.
    """
    registry = getattr(manager, "registry", None)
    if callable(registry):
        return list(registry().values())
    return list(MODEL_REGISTRY.values())


class ModelManager:
    """Loads, caches and downloads YOLOv8 model weights.

    Models are loaded **once** and kept in memory for the lifetime of the
    manager, so video frames never trigger repeated disk loads.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._cache: dict[str, object] = {}
        self._device: str | None = None
        self._custom_cache: tuple[Path, dict[str, ModelSpec]] | None = None

    # ------------------------------------------------------------------ #
    # Registry / availability
    # ------------------------------------------------------------------ #
    @property
    def models_dir(self) -> Path:
        return self.settings.models_dir

    def registry(self) -> dict[str, ModelSpec]:
        """Built-in models plus any user-registered custom model.

        The custom part is re-read whenever ``models/custom/registry.json``
        changes (cheap: a few KB of JSON).
        """
        custom = self._custom_specs()
        return {**MODEL_REGISTRY, **custom}

    def _custom_specs(self) -> dict[str, ModelSpec]:
        registry_file = _custom_registry_file(self.models_dir)
        try:
            stamp = (registry_file.stat().st_mtime_ns, registry_file.stat().st_size)
        except OSError:
            stamp = None
        cached = self._custom_cache
        if cached is not None and cached[0] == stamp:
            return cached[1]
        specs = load_custom_specs(self.models_dir)
        self._custom_cache = (stamp, specs)
        return specs

    def get_spec(self, key: str) -> ModelSpec:
        try:
            return self.registry()[key]
        except KeyError as exc:
            raise ModelNotAvailableError(
                f"Unknown model '{key}'. Available models: "
                f"{', '.join(self.registry())}."
            ) from exc

    def resolve_path(self, key: str) -> Path:
        return self.models_dir / self.get_spec(key).filename

    def is_available(self, key: str) -> bool:
        """True when the weight file for ``key`` exists on disk."""
        try:
            return self.resolve_path(key).is_file()
        except ModelNotAvailableError:
            return False

    def available_models(self) -> list[ModelSpec]:
        """Specs of every model whose weights are present locally."""
        return [spec for spec in self.registry().values() if self.is_available(spec.key)]

    def missing_models(self) -> list[ModelSpec]:
        return [spec for spec in self.registry().values() if not self.is_available(spec.key)]

    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    @property
    def device(self) -> str:
        """Compute device (``cpu`` / ``cuda:0``), resolved once and cached."""
        if self._device is None:
            self._device = self.settings.resolve_device()
            logger.info("Inference device: %s", self._device)
        return self._device

    # ------------------------------------------------------------------ #
    # Loading (cached)
    # ------------------------------------------------------------------ #
    def load(self, key: str):
        """Load (and cache) the YOLO model for ``key``.

        Raises :class:`ModelNotAvailableError` when the weight file is
        missing, with a message that tells the user how to obtain it.
        """
        if key in self._cache:
            return self._cache[key]

        spec = self.get_spec(key)
        path = self.resolve_path(key)
        if not path.is_file():
            raise ModelNotAvailableError(
                f"Model weights for '{spec.display_name}' not found at {path}. "
                "Run 'python scripts/download_models.py' (or use the download "
                "button in the web UI) to fetch them from the official "
                "marine-detect release."
            )

        try:
            import ultralytics
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ModelError(
                "The 'ultralytics' package is not installed. "
                "Install dependencies with 'pip install -r requirements.txt'."
            ) from exc

        loader_name = LOADERS.get(spec.loader, "YOLO")
        model_cls = getattr(ultralytics, loader_name, None)
        if model_cls is None:  # pragma: no cover - dependency guard
            raise ModelError(
                f"This Ultralytics build has no '{loader_name}' loader, which "
                f"model '{spec.display_name}' needs. Update ultralytics."
            )

        started = time.perf_counter()
        try:
            model = model_cls(str(path))
        except Exception as exc:  # noqa: BLE001 - surfaced as a friendly error
            raise ModelError(
                f"Failed to load model '{spec.display_name}' from {path}: {exc}"
            ) from exc
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("Loaded model '%s' in %.0f ms", key, elapsed)
        self._cache[key] = model
        return model

    def get_class_names(self, key: str) -> dict[int, str]:
        """Class names of a loaded model (``{index: name}``)."""
        model = self.load(key)
        names = getattr(model, "names", {}) or {}
        return {int(idx): str(name) for idx, name in names.items()}

    def auto_models(self) -> list[ModelSpec]:
        """Available models that take part in the ``auto`` selection."""
        return [spec for spec in self.available_models() if spec.auto]

    def verify(self, key: str) -> dict[str, Any]:
        """Real load + inference smoke test for one optional model.

        Returns ``{"status", "ok", "error", "classes"}`` where ``status`` is
        one of ``ready`` / ``not_installed`` / ``needs_training`` /
        ``incompatible``. It never raises: a broken optional model must not
        break the rest of the application.
        """
        import numpy as np

        spec = self.get_spec(key)
        if spec.readiness:
            return {"status": spec.readiness, "ok": False, "error": None, "classes": []}
        if not self.is_available(key):
            return {"status": "not_installed", "ok": False, "error": None, "classes": []}
        try:
            model = self.load(key)
            probe = np.zeros((64, 64, 3), dtype=np.uint8)
            model.predict(probe, conf=0.99, device=self.device, verbose=False)
        except Exception as exc:  # noqa: BLE001 - status must never crash
            logger.warning("Inference verification failed for %s: %s", key, exc)
            return {"status": "incompatible", "ok": False, "error": str(exc), "classes": []}
        names = getattr(model, "names", {}) or {}
        classes = [str(names[idx]) for idx in sorted(names)]
        return {"status": "ready", "ok": True, "error": None, "classes": classes}

    def clear_cache(self) -> None:
        """Drop all loaded models (frees memory)."""
        self._cache.clear()

    # ------------------------------------------------------------------ #
    # Download
    # ------------------------------------------------------------------ #
    def download(self, key: str, progress: ProgressCallback | None = None) -> Path:
        """Download the weights for ``key`` if they are not present.

        The transfer goes to a temporary ``.part`` file which is renamed
        only after it passed a sanity size check, so an interrupted
        download can never masquerade as a valid model.
        """
        spec = self.get_spec(key)
        destination = self.resolve_path(key)
        if self.is_available(key) and destination.stat().st_size >= _MIN_WEIGHT_BYTES:
            logger.info("Model '%s' already present at %s", key, destination)
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        logger.info("Downloading model '%s' -> %s", key, destination)

        def _hook(block_count: int, block_size: int, total_size: int) -> None:
            if progress is None:
                return
            downloaded = block_count * block_size
            total = total_size if total_size > 0 else None
            if total is not None:
                downloaded = min(downloaded, total)
            progress(downloaded, total)

        try:
            urllib.request.urlretrieve(spec.url, str(partial), reporthook=_hook)  # noqa: S310 - official HTTPS URLs
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            partial.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"Could not download '{spec.display_name}' weights: {exc}. "
                "Check your internet connection and try again."
            ) from exc

        if not partial.exists() or partial.stat().st_size < _MIN_WEIGHT_BYTES:
            size = partial.stat().st_size if partial.exists() else 0
            partial.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"Downloaded file for '{spec.display_name}' is incomplete ({size} bytes). "
                "Please retry."
            )

        partial.replace(destination)
        logger.info(
            "Downloaded model '%s' (%.1f MB)", key, destination.stat().st_size / (1024 * 1024)
        )
        return destination

    def download_all(self, progress: ProgressCallback | None = None) -> list[Path]:
        """Download every missing built-in model; returns the local paths.

        User-registered custom models have no official URL and are therefore
        never downloaded - they are placed on disk by the user.
        """
        return [
            self.download(spec.key, progress)
            for spec in self.registry().values()
            if spec.is_downloadable
        ]


def download_progress_printer(key: str) -> ProgressCallback:
    """Build a CLI progress printer for a single model download."""

    def _print(downloaded: int, total: int | None) -> None:
        if total:
            percent = downloaded * 100 / total
            print(f"  {key}: {percent:5.1f}% ({downloaded / 1e6:.1f} / {total / 1e6:.1f} MB)",
                  end="\r", flush=True)
        else:
            print(f"  {key}: {downloaded / 1e6:.1f} MB", end="\r", flush=True)
        if total and downloaded >= total:
            print()

    return _print
