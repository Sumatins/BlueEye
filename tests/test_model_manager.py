"""Unit tests for the model registry, availability checks and downloads.

No network access and no GPU are required: ``urlretrieve`` is monkeypatched
and real weights are never loaded in these tests.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pytest

from app.detection.model_manager import (
    MODEL_REGISTRY,
    ModelDownloadError,
    ModelManager,
    ModelNotAvailableError,
)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_contains_both_reference_models() -> None:
    assert set(MODEL_REGISTRY) == {"fish_inv", "megafauna"}
    fish = MODEL_REGISTRY["fish_inv"]
    mega = MODEL_REGISTRY["megafauna"]
    # Thresholds published in the upstream marine-detect README.
    assert fish.recommended_confidence == pytest.approx(0.523)
    assert mega.recommended_confidence == pytest.approx(0.546)
    assert fish.url.startswith("https://")
    assert mega.url.startswith("https://")
    assert fish.filename.endswith(".pt") and mega.filename.endswith(".pt")


# --------------------------------------------------------------------------- #
# Availability (fresh directory -> nothing available)
# --------------------------------------------------------------------------- #
def test_models_not_available_without_weights(settings) -> None:
    manager = ModelManager(settings)
    assert not manager.is_available("fish_inv")
    assert manager.available_models() == []
    assert len(manager.missing_models()) == 2


def test_model_becomes_available_when_weight_file_exists(settings) -> None:
    weight = settings.models_dir / "fish_inv" / "FishInv.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"0" * 100)
    manager = ModelManager(settings)
    assert manager.is_available("fish_inv")
    assert [spec.key for spec in manager.available_models()] == ["fish_inv"]


def test_resolve_path_is_inside_models_dir(settings) -> None:
    manager = ModelManager(settings)
    for key in MODEL_REGISTRY:
        assert settings.models_dir in manager.resolve_path(key).parents


def test_load_without_weights_raises_friendly_error(settings) -> None:
    manager = ModelManager(settings)
    with pytest.raises(ModelNotAvailableError, match="download_models"):
        manager.load("fish_inv")


def test_unknown_model_key_raises(settings) -> None:
    manager = ModelManager(settings)
    with pytest.raises(ModelNotAvailableError, match="Unknown model"):
        manager.get_spec("dolphin")
    assert not manager.is_available("dolphin")


# --------------------------------------------------------------------------- #
# Download (network fully mocked)
# --------------------------------------------------------------------------- #
def _fake_successful_download(payload_size: int = 2_000_000):
    def fake_urlretrieve(url, filename, reporthook=None, **kwargs):
        Path(filename).write_bytes(b"0" * payload_size)
        if reporthook:
            reporthook(1, payload_size, payload_size)
        return str(filename), {}

    return fake_urlretrieve


def test_download_stores_weight_and_reports_progress(settings, monkeypatch) -> None:
    events: list[tuple[int, int | None]] = []
    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_successful_download())

    manager = ModelManager(settings)
    path = manager.download("fish_inv", progress=lambda d, t: events.append((d, t)))
    assert path == settings.models_dir / "fish_inv" / "FishInv.pt"
    assert path.exists() and path.stat().st_size >= 1_000_000
    assert events and events[-1][0] == events[-1][1]
    assert manager.is_available("fish_inv")


def test_download_skipped_when_weight_already_present(settings, monkeypatch) -> None:
    def fail_urlretrieve(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("download should not be triggered")

    monkeypatch.setattr(urllib.request, "urlretrieve", fail_urlretrieve)
    weight = settings.models_dir / "fish_inv" / "FishInv.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"0" * 2_000_000)

    manager = ModelManager(settings)
    assert manager.download("fish_inv") == weight


def test_download_network_failure_raises_model_error(settings, monkeypatch) -> None:
    def failing_urlretrieve(*args, **kwargs):
        raise urllib.error.URLError("no internet")

    monkeypatch.setattr(urllib.request, "urlretrieve", failing_urlretrieve)
    manager = ModelManager(settings)
    with pytest.raises(ModelDownloadError, match="Could not download"):
        manager.download("megafauna")
    # No partial files left behind.
    assert not list((settings.models_dir / "megafauna").glob("*.part"))


def test_download_incomplete_file_rejected(settings, monkeypatch) -> None:
    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_successful_download(payload_size=10))
    manager = ModelManager(settings)
    with pytest.raises(ModelDownloadError, match="incomplete"):
        manager.download("fish_inv")
    assert not manager.is_available("fish_inv")
    assert not list((settings.models_dir / "fish_inv").glob("*.part"))


def test_download_all_fetches_every_model(settings, monkeypatch) -> None:
    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_successful_download())
    manager = ModelManager(settings)
    paths = manager.download_all()
    assert len(paths) == 2
    assert all(path.exists() for path in paths)
    assert len(manager.available_models()) == 2


def test_clear_cache_is_safe_when_empty(settings) -> None:
    ModelManager(settings).clear_cache()
