"""Unit tests for the extensible model registry (custom / regional models).

These tests define the contract behind "a new model can be added without
rewriting the application": a JSON file plus a ``.pt`` file is all that is
needed, and a broken side-loaded file must never break the built-ins.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.detection.detector import MarineDetector
from app.detection.model_manager import (
    MODEL_REGISTRY,
    ModelManager,
    iter_specs,
    load_custom_specs,
)

CUSTOM_DIR = "custom"


def _write_registry(settings: Settings, payload) -> Path:
    registry_dir = settings.models_dir / CUSTOM_DIR
    registry_dir.mkdir(parents=True, exist_ok=True)
    path = registry_dir / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _add_weights(settings: Settings, relative: str, size: int = 1_500_000) -> Path:
    weight = settings.models_dir / relative
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"0" * size)
    return weight


_VALID = {
    "models": [
        {
            "id": "regional_karnataka",
            "name": "Karnataka Coastal Species",
            "weights": "regional/karnataka/BlueEyeRegional.pt",
            "description": "Species recorded along the Karnataka coast.",
            "classes": ["shark", "tuna", "mackerel"],
            "version": "0.1",
            "type": "yolov8",
            "recommended_confidence": 0.4,
        }
    ]
}


# --------------------------------------------------------------------------- #
# Discovery and availability
# --------------------------------------------------------------------------- #
def test_registry_without_custom_file_matches_builtins_only(settings) -> None:
    manager = ModelManager(settings)
    assert set(manager.registry()) == set(MODEL_REGISTRY)


def test_custom_model_is_discovered_but_not_available(settings) -> None:
    _write_registry(settings, _VALID)
    manager = ModelManager(settings)
    spec = manager.registry()["regional_karnataka"]

    assert spec.custom is True
    assert spec.display_name == "Karnataka Coastal Species"
    assert spec.classes == ("shark", "tuna", "mackerel")
    assert spec.recommended_confidence == pytest.approx(0.4)
    assert spec.url == ""  # never auto-downloaded
    assert not spec.is_downloadable
    assert not manager.is_available("regional_karnataka")
    assert [s.key for s in manager.missing_models()] == [
        "fish_inv",
        "megafauna",
        "regional_karnataka",
    ]


def test_custom_model_becomes_available_when_weights_exist(settings) -> None:
    _write_registry(settings, _VALID)
    _add_weights(settings, "regional/karnataka/BlueEyeRegional.pt")
    manager = ModelManager(settings)

    assert manager.is_available("regional_karnataka")
    assert "regional_karnataka" in [spec.key for spec in manager.available_models()]
    resolved = manager.resolve_path("regional_karnataka")
    assert resolved == settings.models_dir / "regional/karnataka/BlueEyeRegional.pt"
    assert settings.models_dir in resolved.parents


def test_auto_mode_includes_registered_custom_model(settings) -> None:
    _write_registry(settings, _VALID)
    _add_weights(settings, "regional/karnataka/BlueEyeRegional.pt")
    _add_weights(settings, "fish_inv/FishInv.pt")
    _add_weights(settings, "megafauna/MegaFauna.pt")

    keys = MarineDetector(settings=settings).resolve_keys("auto")
    assert set(keys) == {"fish_inv", "megafauna", "regional_karnataka"}


def test_auto_mode_uses_only_present_weights(settings) -> None:
    """``auto`` runs whatever is actually installed (unchanged semantics)."""
    _write_registry(settings, _VALID)
    _add_weights(settings, "regional/karnataka/BlueEyeRegional.pt")

    keys = MarineDetector(settings=settings).resolve_keys("auto")
    assert keys == ["regional_karnataka"]


def test_download_all_skips_custom_models(settings, monkeypatch) -> None:
    """Custom models have no official URL, so they are never fetched.

    With every model already present on disk ``download_all()`` must touch
    the network zero times - if a custom model were wrongly treated as
    downloadable, ``urlretrieve`` would be called with an empty URL and the
    test would fail loudly.
    """

    def fail(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("no download should be triggered")

    monkeypatch.setattr("urllib.request.urlretrieve", fail)
    _write_registry(settings, _VALID)
    _add_weights(settings, "regional/karnataka/BlueEyeRegional.pt")
    _add_weights(settings, "fish_inv/FishInv.pt")
    _add_weights(settings, "megafauna/MegaFauna.pt")

    manager = ModelManager(settings)
    assert len(manager.available_models()) == 3
    paths = manager.download_all()  # returns the (already present) built-in paths
    assert [p.name for p in paths] == ["FishInv.pt", "MegaFauna.pt"]
    assert all("regional" not in str(p) for p in paths)  # custom model skipped


def test_model_status_reports_custom_model(settings) -> None:
    _write_registry(settings, _VALID)
    status = MarineDetector(settings=settings).model_status()
    assert [entry["key"] for entry in status] == ["fish_inv", "megafauna", "regional_karnataka"]
    regional = status[-1]
    assert regional["available"] is False
    assert regional["name"] == "Karnataka Coastal Species"


# --------------------------------------------------------------------------- #
# Robustness: a broken side-loaded file must not break the built-ins
# --------------------------------------------------------------------------- #
def test_builtins_still_usable_with_invalid_entries(settings) -> None:
    _write_registry(
        settings,
        {
            "models": [
                {"id": "fish_inv", "name": "Override", "weights": "x.pt"},  # collision
                {"id": "../evil", "name": "Traversal", "weights": "../evil.pt"},
                {"id": "abs", "name": "Absolute", "weights": "C:/Windows/evil.pt"},
                {"id": "badtype", "name": "Odd", "weights": "odd.pt", "type": "onnx"},
                {"id": "noconf", "name": "Bad conf", "weights": "c.pt", "recommended_confidence": 5},
                {"id": "noname", "weights": "n.pt"},
                {"id": 123, "name": "Id not a string", "weights": "i.pt"},
                "not-an-object",
            ]
        },
    )
    specs = load_custom_specs(settings.models_dir)
    assert specs == {}  # every entry rejected
    manager = ModelManager(settings)
    assert set(manager.registry()) == set(MODEL_REGISTRY)
    assert manager.get_spec("fish_inv").display_name == "Fish & Invertebrates"


def test_malformed_json_is_ignored(settings) -> None:
    registry_dir = settings.models_dir / CUSTOM_DIR
    registry_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "registry.json").write_text("{not json", encoding="utf-8")

    assert load_custom_specs(settings.models_dir) == {}
    assert set(ModelManager(settings).registry()) == set(MODEL_REGISTRY)


def test_registry_refreshes_when_file_changes(settings) -> None:
    manager = ModelManager(settings)
    assert set(manager.registry()) == set(MODEL_REGISTRY)

    _write_registry(settings, _VALID)
    assert "regional_karnataka" in manager.registry()

    (settings.models_dir / CUSTOM_DIR / "registry.json").unlink()
    assert set(manager.registry()) == set(MODEL_REGISTRY)


# --------------------------------------------------------------------------- #
# Backwards compatibility of iter_specs
# --------------------------------------------------------------------------- #
def test_iter_specs_falls_back_for_managers_without_registry() -> None:
    class LegacyManager:
        pass

    assert [spec.key for spec in iter_specs(LegacyManager())] == list(MODEL_REGISTRY)
