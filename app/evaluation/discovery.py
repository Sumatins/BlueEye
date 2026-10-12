"""Discovery of locally prepared labelled datasets.

The Research Evaluation Center needs to find a dataset the user has already
prepared (with ``scripts/prepare_dataset.py``) or obtained themselves. This
module only *looks* at directories that already exist: it never downloads
anything, and a directory only counts as a dataset when it really contains a
readable ``data.yaml``/``dataset.yaml``. A directory with no config yields
nothing (so the UI can honestly report that no dataset is configured).

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from app.config import PROJECT_ROOT, Settings, get_settings
from app.evaluation.datasets import DatasetError, class_names_from_yaml, read_yaml

#: File names that mark a directory as a dataset root.
YAML_NAMES = ("data.yaml", "dataset.yaml", "data.yml")

#: Directories that are never worth scanning for datasets.
SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache"})


@dataclass(frozen=True)
class DatasetInfo:
    """One discovered dataset root."""

    name: str
    data_yaml: Path
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def directory(self) -> Path:
        return self.data_yaml.parent


def default_search_roots(settings: Settings | None = None) -> list[Path]:
    """Directories that are scanned for datasets, in priority order.

    These are the locations BlueEye itself writes to (``prepare_dataset.py``
    defaults to ``data/raw/<name>``) plus a conventional top-level
    ``datasets/`` folder.
    """
    settings = settings or get_settings()
    return [
        settings.data_dir / "raw",
        settings.data_dir / "datasets",
        settings.data_dir / "input",
        PROJECT_ROOT / "datasets",
    ]


def find_data_yaml(directory: Path) -> Path | None:
    """Return the dataset config inside ``directory``, if there is one."""
    for name in YAML_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _load_info(yaml_path: Path) -> DatasetInfo | None:
    try:
        data = read_yaml(yaml_path)
    except DatasetError:
        return None
    name = str(data.get("name") or yaml_path.parent.name)
    classes = class_names_from_yaml(data)
    return DatasetInfo(name=name, data_yaml=yaml_path.resolve(), class_names=classes)


def _walk(root: Path, max_depth: int) -> Iterable[Path]:
    """Yield ``data.yaml`` paths under ``root`` (breadth-first, bounded)."""
    root = root.resolve()
    queue: list[tuple[Path, int]] = [(root, 0)]
    while queue:
        directory, depth = queue.pop(0)
        candidate = find_data_yaml(directory)
        if candidate is not None:
            yield candidate
            # A dataset root owns its images/labels subdirectories - do not
            # treat a nested folder as a separate dataset.
            continue
        if depth >= max_depth:
            continue
        try:
            children = [
                child
                for child in directory.iterdir()
                if child.is_dir() and child.name not in SKIP_DIRS
            ]
        except OSError:
            continue
        queue.extend((child, depth + 1) for child in children)


def discover_datasets(
    roots: Sequence[Path] | None = None,
    *,
    max_depth: int = 4,
) -> list[DatasetInfo]:
    """Find every dataset under ``roots`` (deduplicated, sorted by name)."""
    roots = list(roots if roots is not None else default_search_roots())
    found: dict[Path, DatasetInfo] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for yaml_path in _walk(root, max_depth):
            if yaml_path in found:
                continue
            info = _load_info(yaml_path)
            if info is not None:
                found[yaml_path] = info
    return sorted(found.values(), key=lambda item: (item.name.casefold(), str(item.data_yaml)))


def available_splits(data_yaml: str | Path) -> tuple[str, ...]:
    """Split names declared in a ``data.yaml`` (``train`` / ``val`` / ``test``)."""
    try:
        data = read_yaml(Path(data_yaml))
    except DatasetError:
        return ()
    splits: list[str] = []
    for key in ("train", "val", "valid", "test"):
        if key in data and key not in splits:
            splits.append(key)
    return tuple(splits)


__all__ = [
    "DatasetInfo",
    "available_splits",
    "default_search_roots",
    "discover_datasets",
    "find_data_yaml",
]
