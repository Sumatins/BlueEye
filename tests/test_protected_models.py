"""Protection tests for the five working (protected) BlueEye models.

The brief treats these as frozen: their keys, class mappings, thresholds,
inference contract and weights must not change. The registry assertions run
always; the SHA-256 checks run only when the weights are present on this
machine (weights are gitignored), so the suite still works on a fresh clone.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT
from app.detection.detector import MODEL_SELECTIONS, MarineDetector
from app.detection.model_manager import MODEL_REGISTRY, ModelManager

#: key -> (filename relative to models/, sha256 of the protected weights).
PROTECTED = {
    "fish_inv": (
        "fish_inv/FishInv.pt",
        "d0edcad3daa2130a35d7f960301e96f242458490c0210afa4009a0db11687b28",
    ),
    "megafauna": (
        "megafauna/MegaFauna.pt",
        "50fa92fc8be73d578415f163fe56f0a0e6effa3ab4c7df51ff243a9a50b3ee3b",
    ),
    "aquatic_brackish": (
        "additional/aquatic_brackish/BlueEyeAquaticBrackish.pt",
        "2a96029dbbe0477e3cf6dc9f0427d0400037c884ff4d4b4573e4dfe0de86090b",
    ),
    "underwater_fish": (
        "additional/underwater_fish/BlueEyeUnderwaterFish.pt",
        "2e3d70a04c5a5fc7aa1dce0d0e38d10169d32fe23c9d5314d7f0309be0eb36f3",
    ),
    "aquarium_marine": (
        "additional/aquarium_rtdetr/BlueEyeAquariumMarine.pt",
        "02c14f958e4d7a5bfb90fc082201d13d813d6d597f5fd65fa75738b95334b72d",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registry() -> dict:
    return ModelManager().registry()


def test_all_protected_models_are_registered() -> None:
    registry = _registry()
    for key in PROTECTED:
        assert key in registry, f"protected model {key} is missing from the registry"


def test_core_protected_models_keep_their_contract() -> None:
    fish = MODEL_REGISTRY["fish_inv"]
    mega = MODEL_REGISTRY["megafauna"]
    assert fish.recommended_confidence == pytest.approx(0.523)
    assert mega.recommended_confidence == pytest.approx(0.546)
    assert len(fish.classes) == 15
    assert mega.classes == ("shark", "ray", "turtle")
    # Both core models stay in the "auto" selection.
    assert fish.auto and mega.auto
    assert set(MODEL_SELECTIONS) == {"auto", "all", "fish_inv", "megafauna"}


def test_additional_protected_models_keep_their_classes() -> None:
    registry = _registry()
    assert registry["aquatic_brackish"].classes == (
        "crab",
        "fish",
        "jellyfish",
        "shrimp",
        "small_fish",
        "starfish",
    )
    assert registry["underwater_fish"].classes == ("fish",)
    assert registry["aquarium_marine"].classes == (
        "fish",
        "jellyfish",
        "penguin",
        "puffin",
        "shark",
        "starfish",
        "stingray",
    )
    assert registry["aquarium_marine"].loader == "rtdetr"
    # Additional models stay opt-in.
    for key in ("aquatic_brackish", "underwater_fish", "aquarium_marine"):
        assert registry[key].auto is False


@pytest.mark.parametrize("key", sorted(PROTECTED))
def test_protected_weights_are_unchanged(key: str) -> None:
    filename, expected = PROTECTED[key]
    path = PROJECT_ROOT / "models" / filename
    if not path.is_file():
        pytest.skip(f"protected weights not present: {path.name}")
    assert _sha256(path) == expected, f"protected weights changed: {filename}"


@pytest.mark.parametrize("key", sorted(PROTECTED))
def test_protected_models_are_selectable(key: str) -> None:
    detector = MarineDetector()
    if not detector.model_manager.is_available(key):
        pytest.skip(f"weights not present for {key}")
    assert detector.resolve_keys(key) == [key]


#: Newly integrated additional models (not part of the original five). Their
#: weights must not silently change either.
NEW_MODELS = {
    "obsea_mediterranean": (
        "additional/obsea_mediterranean/BlueEyeObseaMediterranean.pt",
        "12793170b79108bfbbaf730192a4f2d2e404414b00427f4bb148a46dc92262ab",
    ),
    "community_fish": (
        "additional/community_fish/BlueEyeCommunityFish.pt",
        "0b259afb3dca1d1d7f9d842ebf136884b2e3706b151172b266bb24e07f2b8e98",
    ),
    "fishial_detector": (
        "additional/fishial_detector/BlueEyeFishialDetector.pt",
        "5b786b334355fdb0c3faa9d375d70de24f3535ee6f11e9c3e26ec2e90810c03f",
    ),
}


@pytest.mark.parametrize("key", sorted(NEW_MODELS))
def test_integrated_weights_are_unchanged(key: str) -> None:
    filename, expected = NEW_MODELS[key]
    path = PROJECT_ROOT / "models" / filename
    if not path.is_file():
        pytest.skip(f"weights not present: {path.name}")
    assert _sha256(path) == expected, f"integrated weights changed: {filename}"


def test_integrated_models_are_registered_and_opt_in() -> None:
    registry = _registry()
    for key in NEW_MODELS:
        assert key in registry
        assert registry[key].auto is False
