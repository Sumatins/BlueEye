"""Tests for the Indian-biodiversity extension.

Two things are covered:

1. **Generic model selection** - any model that is present in the registry
   (built-in *or* custom / Indian) can be selected explicitly from the
   detector and the CLI, while ``auto`` and the legacy aliases keep working.
2. **Dataset/training configs** - the YOLO ``data.yaml`` templates under
   ``training/datasets/`` are valid and describe the intended classes.

These tests never touch the network and never require real weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from app.config import PROJECT_ROOT, Settings
from app.detection.detector import MarineDetector
from app.detection.model_manager import ModelNotAvailableError

CUSTOM_DIR = "custom"
DATASETS_DIR = PROJECT_ROOT / "training" / "datasets"


# --------------------------------------------------------------------------- #
# Helpers (mirror the pattern used by tests/test_custom_models.py)
# --------------------------------------------------------------------------- #
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


_INDIAN = {
    "models": [
        {
            "id": "freshwater_fish",
            "name": "Freshwater Fish (India)",
            "weights": "regional/freshwater_fish/BlueEyeFreshwaterFish.pt",
            "description": "Freshwater pond fish.",
            "classes": ["fish"],
            "version": "0.1",
            "type": "yolov8",
            "recommended_confidence": 0.5,
            "category": "freshwater",
        },
        {
            "id": "indian_marine",
            "name": "Indian Coastal Species",
            "weights": "regional/indian_marine/BlueEyeIndianMarine.pt",
            "description": "Coastal species.",
            "classes": ["whale_shark", "octopus", "ray", "olive_ridley_turtle"],
            "version": "0.1",
            "type": "yolov8",
            "recommended_confidence": 0.5,
            "category": "marine",
        },
    ]
}


# --------------------------------------------------------------------------- #
# Generic model selection
# --------------------------------------------------------------------------- #
def test_registered_custom_model_can_be_selected_by_id(settings) -> None:
    _write_registry(settings, _INDIAN)
    _add_weights(settings, "regional/freshwater_fish/BlueEyeFreshwaterFish.pt")

    keys = MarineDetector(settings=settings).resolve_keys("freshwater_fish")
    assert keys == ["freshwater_fish"]


def test_custom_selection_is_case_insensitive(settings) -> None:
    _write_registry(settings, _INDIAN)
    _add_weights(settings, "regional/indian_marine/BlueEyeIndianMarine.pt")

    keys = MarineDetector(settings=settings).resolve_keys("Indian_Marine")
    assert keys == ["indian_marine"]


def test_custom_selection_without_weights_raises_clear_error(settings) -> None:
    _write_registry(settings, _INDIAN)  # registered, but no .pt on disk

    with pytest.raises(ModelNotAvailableError) as excinfo:
        MarineDetector(settings=settings).resolve_keys("freshwater_fish")

    message = str(excinfo.value)
    assert "Freshwater Fish (India)" in message
    assert "not available" in message.lower()


def test_unknown_selection_lists_registered_models(settings) -> None:
    _write_registry(settings, _INDIAN)

    with pytest.raises(ModelNotAvailableError) as excinfo:
        MarineDetector(settings=settings).resolve_keys("dolphin")

    message = str(excinfo.value)
    assert "Unknown model selection 'dolphin'" in message
    # The error must tell the user what *is* available.
    assert "fish_inv" in message
    assert "freshwater_fish" in message


def test_builtin_aliases_and_auto_still_work(settings) -> None:
    _add_weights(settings, "fish_inv/FishInv.pt")
    _add_weights(settings, "megafauna/MegaFauna.pt")
    detector = MarineDetector(settings=settings)

    assert detector.resolve_keys("fish") == ["fish_inv"]
    assert detector.resolve_keys("mega") == ["megafauna"]
    assert detector.resolve_keys("auto") == ["fish_inv", "megafauna"]


def test_cli_accepts_any_registered_model_id() -> None:
    """The CLI must not hard-code the model choices any more."""
    from app.main import build_parser

    args = build_parser().parse_args(
        ["--image", "photo.jpg", "--model", "freshwater_fish"]
    )
    assert args.model == "freshwater_fish"


# --------------------------------------------------------------------------- #
# Dataset / training configuration templates
# --------------------------------------------------------------------------- #
_EXPECTED_CONFIGS = {
    "freshwater_fish.yaml": ["fish"],
    "gangetic_dolphin.yaml": ["gangetic_river_dolphin"],
    "freshwater_turtle.yaml": ["freshwater_turtle"],
    "gharial.yaml": ["gharial"],
    "indian_marine.yaml": ["whale_shark", "octopus", "ray", "olive_ridley_turtle"],
}


def test_training_configs_exist_for_every_target() -> None:
    for name in _EXPECTED_CONFIGS:
        assert (DATASETS_DIR / name).is_file(), f"missing {name}"


@pytest.mark.parametrize("name,expected", list(_EXPECTED_CONFIGS.items()))
def test_training_config_is_valid_yolo_yaml(name: str, expected: list[str]) -> None:
    data = yaml.safe_load((DATASETS_DIR / name).read_text(encoding="utf-8"))

    assert data["path"]  # dataset root
    assert data["train"] and data["val"]
    assert data["test"]
    names = data["names"]
    assert [names[i] for i in sorted(names)] == expected


def test_indian_marine_config_includes_octopus_and_ray() -> None:
    """Octopus is NOT covered by the built-in models, so it must be listed."""
    data = yaml.safe_load((DATASETS_DIR / "indian_marine.yaml").read_text(encoding="utf-8"))
    classes = set(data["names"].values())
    assert {"octopus", "ray", "whale_shark", "olive_ridley_turtle"} <= classes


def test_registry_example_is_a_generic_template() -> None:
    example = PROJECT_ROOT / "models" / "custom" / "registry.json.example"
    payload = json.loads(example.read_text(encoding="utf-8"))
    ids = {entry["id"] for entry in payload["models"]}
    # The shipped example is now a generic template (not the Indian targets).
    assert {"blueeye_custom", "regional_example"} <= ids

    inactive = json.loads(
        (PROJECT_ROOT / "models" / "custom" / "inactive_models.json").read_text(
            encoding="utf-8"
        )
    )
    inactive_ids = {entry["id"] for entry in inactive["models"]}
    # The Indian research targets are preserved in the inactive registry.
    assert {
        "freshwater_fish",
        "gangetic_dolphin",
        "freshwater_turtle",
        "gharial",
        "indian_marine",
    } <= inactive_ids
