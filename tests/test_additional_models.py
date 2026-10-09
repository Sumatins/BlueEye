"""Tests for the additional (verified) aquatic-species models.

Covers the behaviour added for the Indian-biodiversity expansion:

* curated models are **opt-in** - ``auto`` keeps running only the two core
  models while ``all`` runs every installed model;
* the honest status taxonomy (ready / not installed / needs training /
  incompatible) and the RT-DETR loader type;
* cross-model duplicate suppression;
* the shipped ``models/custom/registry.json`` declares the verified models
  as ready and the Indian targets as needing training.

No test touches the network; inference is mocked or not exercised.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings
from app.detection.detector import MarineDetector
from app.detection.model_manager import (
    ModelManager,
    ModelSpec,
    load_custom_specs,
)
from app.detection.results import Detection

CUSTOM_DIR = "custom"
REGISTRY_FILE = PROJECT_ROOT / "models" / "custom" / "registry.json"


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


_OPT_IN = {
    "models": [
        {
            "id": "optin_model",
            "name": "Opt-in Aquatic Model",
            "weights": "additional/optin/OptIn.pt",
            "type": "yolov8",
            "classes": ["fish"],
            "recommended_confidence": 0.3,
            "auto": False,
            "category": "additional",
        }
    ]
}


# --------------------------------------------------------------------------- #
# Opt-in selection
# --------------------------------------------------------------------------- #
def test_auto_excludes_opt_in_models(settings) -> None:
    _write_registry(settings, _OPT_IN)
    _add_weights(settings, "additional/optin/OptIn.pt")
    _add_weights(settings, "fish_inv/FishInv.pt")
    _add_weights(settings, "megafauna/MegaFauna.pt")

    detector = MarineDetector(settings=settings)
    assert detector.resolve_keys("auto") == ["fish_inv", "megafauna"]
    assert detector.resolve_keys("both") == ["fish_inv", "megafauna"]


def test_all_selection_includes_opt_in_models(settings) -> None:
    _write_registry(settings, _OPT_IN)
    _add_weights(settings, "additional/optin/OptIn.pt")
    _add_weights(settings, "fish_inv/FishInv.pt")
    _add_weights(settings, "megafauna/MegaFauna.pt")

    detector = MarineDetector(settings=settings)
    assert detector.resolve_keys("all") == ["fish_inv", "megafauna", "optin_model"]
    # Convenience aliases resolve to the same combined selection.
    for alias in ("everything", "combined", "full"):
        assert detector.resolve_keys(alias) == [
            "fish_inv",
            "megafauna",
            "optin_model",
        ]


def test_opt_in_model_is_selectable_by_id(settings) -> None:
    _write_registry(settings, _OPT_IN)
    _add_weights(settings, "additional/optin/OptIn.pt")
    assert MarineDetector(settings=settings).resolve_keys("optin_model") == ["optin_model"]


# --------------------------------------------------------------------------- #
# Status taxonomy
# --------------------------------------------------------------------------- #
def test_readiness_status_override_wins_over_availability() -> None:
    trained = ModelSpec(key="f", display_name="F", filename="f.pt", url="", recommended_confidence=0.5)
    assert trained.readiness_status(available=True) == "ready"
    assert trained.readiness_status(available=False) == "not_installed"

    needs = ModelSpec(
        key="d", display_name="D", filename="d.pt", url="", recommended_confidence=0.5,
        readiness="needs_training",
    )
    assert needs.readiness_status(available=False) == "needs_training"
    # A readiness override is never reported as ready, even if a file exists.
    assert needs.readiness_status(available=True) == "needs_training"

    bad = ModelSpec(
        key="x", display_name="X", filename="x.pt", url="", recommended_confidence=0.5,
        readiness="incompatible",
    )
    assert bad.readiness_status(available=False) == "incompatible"


def test_manager_verify_reports_status_without_loading(settings) -> None:
    _write_registry(settings, _OPT_IN)
    manager = ModelManager(settings)

    missing = manager.verify("optin_model")
    assert missing["status"] == "not_installed"
    assert missing["ok"] is False

    # A needs-training override answers immediately, even with no weights.
    _write_registry(
        settings,
        {
            "models": [
                {
                    "id": "needs_it",
                    "name": "Needs training",
                    "weights": "additional/needs_it/N.pt",
                    "type": "yolov8",
                    "readiness": "needs_training",
                }
            ]
        },
    )
    manager = ModelManager(settings)
    assert manager.verify("needs_it")["status"] == "needs_training"


def test_model_status_includes_readiness_fields(settings) -> None:
    _write_registry(settings, _OPT_IN)
    entry = {e["key"]: e for e in MarineDetector(settings=settings).model_status()}["optin_model"]
    assert entry["status"] == "not_installed"
    assert entry["architecture"]
    assert entry["auto"] is False


# --------------------------------------------------------------------------- #
# Loader types
# --------------------------------------------------------------------------- #
def test_rtdetr_entry_is_accepted(settings) -> None:
    _write_registry(
        settings,
        {
            "models": [
                {
                    "id": "detr_model",
                    "name": "DETR model",
                    "weights": "additional/detr/D.pt",
                    "type": "rtdetr",
                    "classes": ["shark"],
                }
            ]
        },
    )
    spec = load_custom_specs(settings.models_dir)["detr_model"]
    assert spec.loader == "rtdetr"
    assert "RT-DETR" in spec.architecture


def test_unknown_loader_type_is_rejected(settings) -> None:
    _write_registry(
        settings,
        {"models": [{"id": "bad", "name": "Bad", "weights": "bad.pt", "type": "onnx"}]},
    )
    assert load_custom_specs(settings.models_dir) == {}


# --------------------------------------------------------------------------- #
# Cross-model duplicate suppression
# --------------------------------------------------------------------------- #
def test_suppress_duplicates_keeps_highest_confidence() -> None:
    detections = [
        Detection(class_name="fish", confidence=0.9, bbox=(0, 0, 100, 100), model="m1"),
        Detection(class_name="fish", confidence=0.8, bbox=(4, 4, 96, 96), model="m2"),
        Detection(class_name="fish", confidence=0.7, bbox=(400, 400, 500, 500), model="m2"),
        Detection(class_name="shark", confidence=0.6, bbox=(0, 0, 100, 100), model="m2"),
    ]
    kept = MarineDetector._suppress_duplicates(detections)

    # The overlapping same-class duplicate is removed; distinct class and
    # non-overlapping boxes are preserved.
    assert len(kept) == 3
    fish = [d for d in kept if d.class_name == "fish"]
    assert {d.model for d in fish} == {"m1", "m2"}  # one fish from m1, one from m2
    assert max(d.confidence for d in fish) == 0.9
    assert any(d.class_name == "shark" for d in kept)


def test_suppress_duplicates_ignores_same_model_boxes() -> None:
    detections = [
        Detection(class_name="fish", confidence=0.9, bbox=(0, 0, 100, 100), model="m1"),
        Detection(class_name="fish", confidence=0.8, bbox=(4, 4, 96, 96), model="m1"),
    ]
    assert len(MarineDetector._suppress_duplicates(detections)) == 2


# --------------------------------------------------------------------------- #
# Shipped registry file
# --------------------------------------------------------------------------- #
def _registry_entries() -> dict[str, dict]:
    payload = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in payload["models"]}


def test_shipped_registry_lists_verified_models() -> None:
    entries = _registry_entries()
    assert {"aquatic_brackish", "underwater_fish", "aquarium_marine"} <= set(entries)
    for key in ("aquatic_brackish", "underwater_fish", "aquarium_marine"):
        assert entries[key]["category"] == "additional"
        assert entries[key]["auto"] is False
        assert not entries[key].get("readiness")  # derived -> ready when present


def test_shipped_registry_lists_indian_targets_as_needing_training() -> None:
    entries = _registry_entries()
    for key in (
        "freshwater_fish",
        "gangetic_dolphin",
        "freshwater_turtle",
        "gharial",
        "indian_marine",
    ):
        assert entries[key]["category"] == "indian"
        assert entries[key]["readiness"] == "needs_training"


def test_shipped_registry_documents_incompatible_model() -> None:
    entries = _registry_entries()
    assert entries["aquarium_axera"]["readiness"] == "incompatible"


def test_shipped_registry_parses_through_the_loader() -> None:
    """The real registry.json must be accepted by the custom-model parser."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        models_dir = Path(tmp) / "models"
        target = models_dir / CUSTOM_DIR
        target.mkdir(parents=True)
        (target / "registry.json").write_text(
            REGISTRY_FILE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        specs = load_custom_specs(models_dir)

    expected = {
        "aquatic_brackish",
        "underwater_fish",
        "aquarium_marine",
        "freshwater_fish",
        "gangetic_dolphin",
        "freshwater_turtle",
        "gharial",
        "indian_marine",
        "aquarium_axera",
    }
    assert expected <= set(specs)
    assert specs["aquarium_marine"].loader == "rtdetr"
    assert specs["aquarium_axera"].readiness_status(False) == "incompatible"
