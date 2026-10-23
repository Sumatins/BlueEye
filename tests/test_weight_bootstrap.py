"""Automatic weight bootstrap (app.detection.bootstrap).

These tests never touch the network: the manager is always a fake that
records download calls. They cover the fresh-deployment flow - no weights
on disk, background thread installs the core model, and every failure path
degrades to the honest manual-install UI instead of crashing.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest

from app.config import Settings, get_settings, reset_settings_cache
from app.detection import bootstrap
from app.detection.bootstrap import (
    bootstrap_keys,
    ensure_weights,
    start_background_bootstrap,
)
from app.detection.model_manager import MODEL_REGISTRY, ModelDownloadError


class FakeManager:
    """Minimal ModelManager double: records downloads, never opens a socket."""

    def __init__(self, available: bool = False, fail: bool = False) -> None:
        self.available = available
        self.fail = fail
        self.downloads: list[str] = []

    def is_available(self, key: str) -> bool:
        return self.available

    def resolve_path(self, key: str) -> Path:
        return Path("models") / f"{key}.pt"

    def download(self, key: str, progress=None) -> Path:
        self.downloads.append(key)
        if self.fail:
            raise ModelDownloadError("network down")
        self.available = True
        return self.resolve_path(key)


@pytest.fixture(autouse=True)
def _reset_bootstrap_state():
    bootstrap.reset_bootstrap_state()
    yield
    bootstrap.reset_bootstrap_state()


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = threading.Event()
    for _ in range(int(timeout / 0.05)):
        if predicate():
            return True
        deadline.wait(0.05)
    return predicate()


# --------------------------------------------------------------------------- #
# Which models are prepared automatically
# --------------------------------------------------------------------------- #
def test_core_model_selected_by_default():
    assert bootstrap_keys(Settings(), MODEL_REGISTRY) == ["fish_inv"]


def test_explicit_default_model_is_honoured():
    keys = bootstrap_keys(Settings(default_model_selection="megafauna"), MODEL_REGISTRY)
    assert keys == ["megafauna"]


def test_disabled_flag_prepares_nothing():
    assert bootstrap_keys(Settings(auto_download_weights=False), MODEL_REGISTRY) == []


def test_unknown_and_non_downloadable_models_are_skipped():
    assert bootstrap_keys(Settings(default_model_selection="ghost"), MODEL_REGISTRY) == []
    no_url = replace(MODEL_REGISTRY["fish_inv"], url="")
    keys = bootstrap_keys(Settings(default_model_selection="fish_inv"), {"fish_inv": no_url})
    assert keys == []


# --------------------------------------------------------------------------- #
# ensure_weights (idempotent, lock-guarded)
# --------------------------------------------------------------------------- #
def test_ensure_weights_skips_present_files():
    manager = FakeManager(available=True)
    path = ensure_weights(manager, "fish_inv")
    assert path.name == "fish_inv.pt"
    assert manager.downloads == []


def test_ensure_weights_downloads_missing_files():
    manager = FakeManager(available=False)
    ensure_weights(manager, "fish_inv")
    assert manager.downloads == ["fish_inv"]
    assert manager.available is True


def test_concurrent_ensure_downloads_once():
    """Two sessions (or a session and the background thread) racing on the
    same missing weights must trigger exactly one transfer - concurrent
    writers would corrupt the shared .part file."""
    manager = FakeManager(available=False)
    original = manager.download

    def slow_download(key: str, progress=None) -> Path:
        threading.Event().wait(0.15)  # hold the lock long enough to race
        return original(key, progress)

    manager.download = slow_download  # type: ignore[method-assign]
    threads = [
        threading.Thread(target=ensure_weights, args=(manager, "fish_inv")) for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert manager.downloads == ["fish_inv"]


# --------------------------------------------------------------------------- #
# Background worker
# --------------------------------------------------------------------------- #
def test_background_bootstrap_downloads_missing_weights_once():
    manager = FakeManager(available=False)
    assert start_background_bootstrap(manager, ["fish_inv"]) is True
    assert start_background_bootstrap(manager, ["fish_inv"]) is False  # once per process
    assert _wait_for(lambda: manager.downloads)
    assert manager.downloads == ["fish_inv"]


def test_background_bootstrap_is_noop_when_weights_present():
    manager = FakeManager(available=True)
    assert start_background_bootstrap(manager, ["fish_inv"]) is False
    assert manager.downloads == []


def test_background_bootstrap_swallows_failures():
    """A failed background download is logged, never raised - the app keeps
    working and the visible download paths stay available."""
    manager = FakeManager(available=False, fail=True)
    assert start_background_bootstrap(manager, ["fish_inv"]) is True
    assert _wait_for(lambda: manager.downloads)
    assert manager.downloads == ["fish_inv"]
    # The manual path still functions (and reports the honest error).
    with pytest.raises(ModelDownloadError):
        ensure_weights(manager, "fish_inv")


# --------------------------------------------------------------------------- #
# Settings plumbing
# --------------------------------------------------------------------------- #
def test_settings_enable_auto_download_by_default():
    assert Settings().auto_download_weights is True


def test_env_disables_auto_download(monkeypatch):
    monkeypatch.setenv("BLUEEYE_AUTO_DOWNLOAD_WEIGHTS", "off")
    reset_settings_cache()
    try:
        assert get_settings().auto_download_weights is False
    finally:
        reset_settings_cache()
