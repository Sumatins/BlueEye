"""Tests for the Models page grouping and for download integrity.

The page is grouped into three honest buckets (ready / downloadable /
unavailable); :func:`_categorise` is a pure function so it can be tested
without Streamlit. The download path is exercised with ``urlretrieve``
monkeypatched, so no network access is required.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings
from app.detection.detector import MarineDetector
from app.detection.model_manager import (
    ModelDownloadError,
    ModelManager,
    ModelNotAvailableError,
    ModelSpec,
)
from app.ui.pages import models as models_page
from app.ui.pages.models import RUNTIME_LABEL, STATUS_LABEL, _categorise

_MODELS_SOURCE = (PROJECT_ROOT / "app" / "ui" / "pages" / "models.py").read_text(
    encoding="utf-8"
)


def _spec(key: str, **overrides) -> ModelSpec:
    base = dict(
        key=key,
        display_name=key,
        filename=f"{key}.pt",
        url="",
        recommended_confidence=0.5,
    )
    base.update(overrides)
    return ModelSpec(**base)


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
def test_categorise_splits_into_three_buckets() -> None:
    specs = [
        _spec("ready_model"),
        _spec("downloadable", download_url="https://example.com/best.pt"),
        _spec("needs", readiness="needs_training"),
        _spec("bad", readiness="incompatible"),
    ]
    results = {
        "ready_model": {"status": "ready"},
        "downloadable": {"status": "not_installed"},
        "needs": {"status": "needs_training"},
        "bad": {"status": "incompatible"},
    }
    ready, downloadable, unavailable = _categorise(specs, results)
    assert [s.key for s in ready] == ["ready_model"]
    assert [s.key for s in downloadable] == ["downloadable"]
    assert [s.key for s in unavailable] == ["needs", "bad"]


def test_a_model_with_weights_but_failed_inference_is_not_ready() -> None:
    """A file on disk is never enough - a failed test moves it out of Ready."""
    specs = [_spec("present")]
    results = {"present": {"status": "incompatible", "error": "boom"}}
    ready, _downloadable, unavailable = _categorise(specs, results)
    assert ready == []
    assert [s.key for s in unavailable] == ["present"]


def test_missing_custom_model_without_url_is_unavailable_not_downloadable() -> None:
    specs = [_spec("local_only")]
    results = {"local_only": {"status": "not_installed"}}
    _ready, downloadable, unavailable = _categorise(specs, results)
    assert downloadable == []
    assert [s.key for s in unavailable] == ["local_only"]


# --------------------------------------------------------------------------- #
# Page contract
# --------------------------------------------------------------------------- #
def test_required_buckets_and_fields_are_rendered() -> None:
    for marker in (
        "Ready and verified",
        "Downloadable but not yet verified",
        "Unavailable or incompatible",
        "Architecture",
        "Runtime",
        "Habitat",
        "Recommended threshold",
        "Weights",
        "Source",
        "License",
        "Image inference",
        "Video inference",
        "Last inference test",
        "Limitations",
    ):
        assert marker in _MODELS_SOURCE, f"missing Models-page field: {marker}"


def test_status_and_runtime_labels_cover_the_statuses() -> None:
    for status in ("ready", "not_installed", "needs_training", "incompatible"):
        assert status in STATUS_LABEL
    assert RUNTIME_LABEL["yolo"].startswith("Ultralytics")
    assert RUNTIME_LABEL["rtdetr"].startswith("Ultralytics")


def test_models_page_imports_render() -> None:
    assert callable(models_page.render)


# --------------------------------------------------------------------------- #
# Download integrity
# --------------------------------------------------------------------------- #
def _write_registry(settings: Settings, payload) -> Path:
    registry_dir = settings.models_dir / "custom"
    registry_dir.mkdir(parents=True, exist_ok=True)
    path = registry_dir / "registry.json"
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fake_download(payload: bytes):
    def fake_urlretrieve(url, filename, reporthook=None, **kwargs):
        Path(filename).write_bytes(payload)
        if reporthook:
            reporthook(1, len(payload), len(payload))
        return str(filename), {}

    return fake_urlretrieve


def test_custom_download_url_is_fetched_and_hashed(settings, monkeypatch) -> None:
    payload = b"0" * 2_000_000
    digest = hashlib.sha256(payload).hexdigest()
    _write_registry(
        settings,
        {
            "models": [
                {
                    "id": "dl_model",
                    "name": "Downloadable",
                    "weights": "additional/dl/D.pt",
                    "classes": ["fish"],
                    "download_url": "https://example.com/best.pt",
                    "sha256": digest,
                }
            ]
        },
    )
    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_download(payload))

    manager = ModelManager(settings)
    assert manager.get_spec("dl_model").is_downloadable
    path = manager.download("dl_model")
    assert path.is_file() and path.stat().st_size == len(payload)


def test_custom_download_sha256_mismatch_is_rejected(settings, monkeypatch) -> None:
    payload = b"0" * 2_000_000
    _write_registry(
        settings,
        {
            "models": [
                {
                    "id": "dl_bad",
                    "name": "Bad hash",
                    "weights": "additional/dl/Bad.pt",
                    "classes": ["fish"],
                    "download_url": "https://example.com/bad.pt",
                    "sha256": "0" * 64,
                }
            ]
        },
    )
    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_download(payload))

    manager = ModelManager(settings)
    with pytest.raises(ModelDownloadError, match="integrity"):
        manager.download("dl_bad")
    assert not manager.is_available("dl_bad")
    assert not list((settings.models_dir / "additional" / "dl").glob("*.part"))


def test_invalid_download_url_is_rejected(settings) -> None:
    from app.detection.model_manager import load_custom_specs

    _write_registry(
        settings,
        {
            "models": [
                {
                    "id": "bad_url",
                    "name": "Bad URL",
                    "weights": "additional/x/X.pt",
                    "download_url": "ftp://example.com/x.pt",
                }
            ]
        },
    )
    assert load_custom_specs(settings.models_dir) == {}


def test_invalid_sha256_is_rejected(settings) -> None:
    from app.detection.model_manager import load_custom_specs

    _write_registry(
        settings,
        {
            "models": [
                {
                    "id": "bad_hash",
                    "name": "Bad hash",
                    "weights": "additional/x/X.pt",
                    "download_url": "https://example.com/x.pt",
                    "sha256": "not-a-hash",
                }
            ]
        },
    )
    assert load_custom_specs(settings.models_dir) == {}


def test_archive_member_download_extracts_and_hashes(settings, monkeypatch) -> None:
    import io
    import zipfile

    payload = b"0" * 2_000_000
    digest = hashlib.sha256(payload).hexdigest()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model.pt", payload)
    archive_bytes = buffer.getvalue()

    _write_registry(
        settings,
        {
            "models": [
                {
                    "id": "zip_model",
                    "name": "Zip packaged",
                    "weights": "additional/zip/Z.pt",
                    "classes": ["Fish"],
                    "download_url": "https://example.com/model.zip",
                    "sha256": digest,
                    "archive_member": "model.pt",
                }
            ]
        },
    )
    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_download(archive_bytes))

    manager = ModelManager(settings)
    spec = manager.get_spec("zip_model")
    assert spec.archive_member == "model.pt"
    path = manager.download("zip_model")
    assert path.read_bytes() == payload
    # No leftover .zip / .part files.
    assert not list(path.parent.glob("*.part"))
    assert not list(path.parent.glob("*.zip"))


def test_archive_member_without_url_is_rejected(settings) -> None:
    from app.detection.model_manager import load_custom_specs

    _write_registry(
        settings,
        {
            "models": [
                {
                    "id": "orphan_member",
                    "name": "No URL",
                    "weights": "additional/x/X.pt",
                    "archive_member": "model.pt",
                }
            ]
        },
    )
    assert load_custom_specs(settings.models_dir) == {}


# --------------------------------------------------------------------------- #
# Runs without any optional / custom model installed
# --------------------------------------------------------------------------- #
def test_app_backend_runs_without_optional_models(settings) -> None:
    """With an empty models dir, only the two built-ins are registered.

    The app must still start and report the gap clearly instead of crashing.
    """
    detector = MarineDetector(settings=settings)
    registry = detector.model_manager.registry()
    assert set(registry) == {"fish_inv", "megafauna"}
    assert detector.model_manager.available_models() == []
    # "auto" needs the core weights; the failure is a clear domain error.
    with pytest.raises(ModelNotAvailableError):
        detector.resolve_keys("auto")


def test_backend_reports_not_runnable_without_crashing(settings) -> None:
    detector = MarineDetector(settings=settings)
    result = detector.verify_model("fish_inv")
    assert result["status"] in {"not_installed", "incompatible"}
    assert not detector.model_manager.is_available("fish_inv")
