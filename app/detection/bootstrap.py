"""Automatic, one-time weight bootstrap for ephemeral deployments.

Container platforms without a persistent disk (Streamlit Community Cloud)
wipe ``models/`` on every restart, so a fresh deployment starts with **no**
weights and every visitor would have to click *Download* before the first
detection could run. This module removes that manual step, honestly:

* when the app process starts, a **background daemon thread** downloads the
  core model (``fish_inv`` by default) if its weights are missing - silent,
  non-blocking, and never able to crash the app;
* the Detect page falls back to a *visible*, progress-reporting synchronous
  download when a user reaches it before the background transfer finished.

Only official, registry-listed weights are fetched, through the same code
path (and the same size / SHA-256 verification) as the Models-page buttons.
Nothing is ever fabricated, and the whole behaviour can be disabled with
``BLUEEYE_AUTO_DOWNLOAD_WEIGHTS=off``.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Mapping

from app.config import Settings
from app.detection.model_manager import ModelManager, ModelSpec, ProgressCallback

logger = logging.getLogger(__name__)

#: Core model auto-installed on a fresh deployment. It is the broad
#: workhorse model and, on its own, fits the 1 GB Community Cloud budget.
BOOTSTRAP_KEY = "fish_inv"

#: Guards both the once-per-process flag and every download (two sessions -
#: or a session and the background thread - must never write the same
#: ``.part`` file at the same time).
_LOCK = threading.Lock()

#: Set once the background bootstrap has been considered for this process.
_BACKGROUND_STARTED = False


def bootstrap_keys(settings: Settings, registry: Mapping[str, ModelSpec]) -> list[str]:
    """Registry keys worth fetching automatically on this deployment.

    Empty when auto-download is disabled. With the default ``auto`` model
    selection this is the core model only (never the nine-model set: that
    would blow the free tier's memory/disk budget); an explicit
    ``BLUEEYE_DEFAULT_MODEL`` names the model to prepare instead.
    """
    if not settings.auto_download_weights:
        return []
    selection = str(settings.default_model_selection or "auto").strip().lower()
    if selection in {"auto", "all", ""}:
        candidates = [BOOTSTRAP_KEY]
    else:
        candidates = [part for part in (p.strip() for p in selection.split(",")) if part]
    return [key for key in candidates if (spec := registry.get(key)) and spec.is_downloadable]


def ensure_weights(
    manager: ModelManager, key: str, progress: ProgressCallback | None = None
) -> Path:
    """Download ``key`` when missing (idempotent, process-wide lock).

    Returns the weight path whether the file was already present or just
    downloaded. Raises the manager's :class:`ModelError` subclasses on
    failure - callers decide how to surface that.
    """
    with _LOCK:
        if manager.is_available(key):
            return manager.resolve_path(key)
        return manager.download(key, progress=progress)


def start_background_bootstrap(manager: ModelManager, keys: list[str]) -> bool:
    """Download the missing ``keys`` in a daemon thread; once per process.

    Returns ``True`` when a worker thread was started. The worker swallows
    and logs every error: a failed background download must never take the
    app down - the Detect page and the Models page remain the visible
    fallbacks.
    """
    global _BACKGROUND_STARTED
    with _LOCK:
        if _BACKGROUND_STARTED:
            return False
        _BACKGROUND_STARTED = True
        missing = [key for key in keys if not manager.is_available(key)]
    if not missing:
        return False
    logger.info("Weight bootstrap: fetching %s in the background", ", ".join(missing))
    threading.Thread(
        target=_bootstrap_worker,
        args=(manager, missing),
        name="blueeye-weight-bootstrap",
        daemon=True,
    ).start()
    return True


def reset_bootstrap_state() -> None:
    """Forget that the background bootstrap ran (tests only)."""
    global _BACKGROUND_STARTED
    with _LOCK:
        _BACKGROUND_STARTED = False


def _bootstrap_worker(manager: ModelManager, keys: list[str]) -> None:
    for key in keys:
        try:
            ensure_weights(manager, key)
            logger.info("Weight bootstrap: model '%s' is installed", key)
        except Exception as exc:  # noqa: BLE001 - background threads must never kill the app
            logger.warning("Weight bootstrap: could not install '%s': %s", key, exc)
