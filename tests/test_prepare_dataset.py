"""Tests for ``scripts/prepare_dataset.py``.

The helper is the honest bridge between a downloaded, licensed dataset and
the training configs. These tests use a tiny synthetic dataset on disk: no
network, no real imagery, no model.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_dataset", SCRIPTS_DIR / "prepare_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare = _load_module()
DatasetError = prepare.DatasetError


def _make_source(root: Path, missing_label: bool = False) -> Path:
    """Create a minimal Roboflow-style export with two classes."""
    for split, image_name in (("train", "a"), ("valid", "c"), ("test", "d")):
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / f"{image_name}.jpg").write_bytes(b"\xff\xd8\xff")
        (labels / f"{image_name}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n", encoding="utf-8"
        )
    if not missing_label:
        (root / "train" / "images" / "b.jpg").write_bytes(b"\xff\xd8\xff")
        (root / "train" / "labels" / "b.txt").write_text(
            "0 0.4 0.4 0.2 0.2\n", encoding="utf-8"
        )
    else:
        # An image intentionally left without a label file.
        (root / "train" / "images" / "b.jpg").write_bytes(b"\xff\xd8\xff")
    (root / "data.yaml").write_text(
        "names:\n  0: fish\n  1: crab\n", encoding="utf-8"
    )
    return root


def test_discover_splits_maps_valid_to_val(tmp_path: Path) -> None:
    source = _make_source(tmp_path / "src")
    splits = prepare.discover_splits(source)
    assert set(splits) == {"train", "val", "test"}
    images_dir, labels_dir = splits["val"]
    assert images_dir.name == "images" and labels_dir.name == "labels"


def test_audit_counts_objects_and_missing_labels(tmp_path: Path) -> None:
    source = _make_source(tmp_path / "src", missing_label=True)
    splits = prepare.discover_splits(source)
    audit = prepare.audit_split(*splits["train"])
    assert audit["images"] == 2  # a.jpg + b.jpg (b has no label)
    assert audit["matched"] == 1
    assert audit["missing_labels"] == 1
    assert audit["objects"] == 2  # two objects in a.txt
    assert audit["per_class"] == {0: 1, 1: 1}


def test_prepare_dataset_writes_canonical_layout(tmp_path: Path) -> None:
    source = _make_source(tmp_path / "src")
    out = tmp_path / "out"

    info = prepare.prepare_dataset(source, out, name="demo")

    assert info["num_classes"] == 2
    assert info["classes"] == ["fish", "crab"]
    assert info["totals"]["matched"] == 4  # a, b, c, d
    assert info["totals"]["objects"] == 7  # 2+1+2+2

    for split in ("train", "val", "test"):
        assert (out / "images" / split).is_dir()
        assert (out / "labels" / split).is_dir()

    data_yaml = (out / "data.yaml").read_text(encoding="utf-8")
    assert "0: fish" in data_yaml and "1: crab" in data_yaml
    assert "train: images/train" in data_yaml

    audit = json.loads((out / "dataset_info.json").read_text(encoding="utf-8"))
    assert audit["name"] == "demo"
    assert audit["totals"]["objects"] == 7


def test_prepare_dataset_refuses_images_without_labels(tmp_path: Path) -> None:
    source = tmp_path / "src"
    for split in ("train", "valid", "test"):
        (source / split / "images").mkdir(parents=True)
        (source / split / "images" / "x.jpg").write_bytes(b"\xff\xd8\xff")
        (source / split / "labels").mkdir(parents=True)

    with pytest.raises(DatasetError):
        prepare.prepare_dataset(source, tmp_path / "out", name="broken")


def test_prepare_dataset_requires_real_class_names(tmp_path: Path) -> None:
    source = _make_source(tmp_path / "src")
    (source / "data.yaml").unlink()

    with pytest.raises(DatasetError):
        prepare.prepare_dataset(source, tmp_path / "out", name="demo")

    # Explicit class names make it usable again.
    info = prepare.prepare_dataset(
        source, tmp_path / "out", name="demo", classes=["fish", "crab"]
    )
    assert info["classes"] == ["fish", "crab"]


def test_too_few_class_names_is_rejected(tmp_path: Path) -> None:
    source = _make_source(tmp_path / "src")
    with pytest.raises(DatasetError):
        prepare.prepare_dataset(source, tmp_path / "out", name="demo", classes=["only_one"])


def test_audit_only_copies_nothing(tmp_path: Path) -> None:
    source = _make_source(tmp_path / "src")
    out = tmp_path / "out"
    info = prepare.prepare_dataset(source, out, name="demo", audit_only=True)
    assert info["copied"] is False
    assert not out.exists()


def test_cli_audit_only_returns_zero(tmp_path: Path, capsys) -> None:
    source = _make_source(tmp_path / "src")
    code = prepare.main(["--source", str(source), "--audit-only"])
    assert code == 0
    assert "fish" in capsys.readouterr().out
