"""Tests for labelled-dataset discovery used by the Research Evaluation Center.

Discovery only *finds* datasets the user already prepared on disk. These tests
build tiny throwaway dataset trees and verify that discovery is honest: a
folder is a dataset only when it really contains a parseable config, nested
folders are not double-counted, skipped directories are ignored and overlapping
search roots are de-duplicated.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from pathlib import Path

from app.evaluation.discovery import (
    available_splits,
    discover_datasets,
    find_data_yaml,
)

CONFIG = """\
name: {name}
path: {path}
train: images/train
val: images/val
test: images/test
names:
  0: fish
  1: scaridae
  2: shark
"""


def _make_dataset(root: Path, name: str = "my_ds") -> Path:
    dataset = root / name
    (dataset / "images" / "test").mkdir(parents=True)
    (dataset / "labels" / "test").mkdir(parents=True)
    (dataset / "images" / "test" / "a.jpg").write_bytes(b"not-a-real-image")
    (dataset / "labels" / "test" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (dataset / "data.yaml").write_text(
        CONFIG.format(name=name, path=dataset), encoding="utf-8"
    )
    return dataset


def test_discovers_a_prepared_dataset(tmp_path: Path) -> None:
    _make_dataset(tmp_path, "reef_survey")
    found = discover_datasets([tmp_path])
    assert len(found) == 1
    info = found[0]
    assert info.name == "reef_survey"
    assert info.class_names == ("fish", "scaridae", "shark")
    assert info.num_classes == 3
    assert info.data_yaml.name == "data.yaml"
    assert info.directory == info.data_yaml.parent


def test_reports_declared_splits(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    info = discover_datasets([tmp_path])[0]
    assert available_splits(info.data_yaml) == ("train", "val", "test")


def test_directory_without_config_is_not_a_dataset(tmp_path: Path) -> None:
    (tmp_path / "not_a_dataset" / "images").mkdir(parents=True)
    assert discover_datasets([tmp_path]) == []


def test_nested_folder_under_a_dataset_is_not_double_counted(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    info = discover_datasets([tmp_path])[0]
    # A stray config inside the dataset's own tree must not create a second hit.
    nested = info.directory / "images" / "test" / "nested"
    nested.mkdir(parents=True)
    (nested / "data.yaml").write_text(CONFIG.format(name="nested", path=nested), encoding="utf-8")
    assert len(discover_datasets([tmp_path])) == 1


def test_overlapping_roots_are_deduplicated(tmp_path: Path) -> None:
    _make_dataset(tmp_path / "a")
    _make_dataset(tmp_path, "outer")
    single = discover_datasets([tmp_path])
    double = discover_datasets([tmp_path, tmp_path])
    assert len(single) == 2
    assert [info.data_yaml for info in double] == [info.data_yaml for info in single]


def test_skipped_directories_are_ignored(tmp_path: Path) -> None:
    hidden = tmp_path / ".git" / "objects"
    (hidden).mkdir(parents=True)
    (hidden / "data.yaml").write_text(CONFIG.format(name="ghost", path=hidden), encoding="utf-8")
    assert discover_datasets([tmp_path]) == []


def test_invalid_config_is_ignored(tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "data.yaml").write_text("{ this is not: valid: yaml", encoding="utf-8")
    assert discover_datasets([tmp_path]) == []


def test_find_data_yaml_prefers_known_names(tmp_path: Path) -> None:
    (tmp_path / "dataset.yaml").write_text("names: [fish]", encoding="utf-8")
    found = find_data_yaml(tmp_path)
    assert found is not None and found.name == "dataset.yaml"
    assert find_data_yaml(tmp_path / "empty") is None
