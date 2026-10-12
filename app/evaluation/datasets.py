"""YOLO dataset discovery, ``data.yaml`` parsing and ground-truth loading.

BlueEye ships **no** datasets. This module understands the canonical
Ultralytics layout produced by ``scripts/prepare_dataset.py``::

    <root>/
    |-- images/{train,val,test}/*.jpg
    |-- labels/{train,val,test}/*.txt   # YOLO: cls cx cy w h (normalised)
    └── data.yaml

Ground-truth boxes are read from the ``.txt`` labels and converted to pixel
coordinates using the real image size (Pillow). Nothing is invented: a
missing label file yields an empty (but valid) ground truth and that fact is
recorded as a note; an unreadable image is skipped and reported.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from app.evaluation.metrics import DetectionBox

#: Only these image extensions are considered part of a split.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be read at all."""


@dataclass
class DatasetSplit:
    """One split (``train`` / ``val`` / ``test``) of a YOLO dataset."""

    name: str
    data_yaml: Path
    split: str
    class_names: tuple[str, ...]
    images: tuple[Path, ...]
    ground_truth: dict[str, list[DetectionBox]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_objects(self) -> int:
        return sum(len(boxes) for boxes in self.ground_truth.values())

    def class_counts(self) -> dict[str, int]:
        """Ground-truth instances per class name."""
        counts: dict[int, int] = {}
        for boxes in self.ground_truth.values():
            for box in boxes:
                counts[box.class_id] = counts.get(box.class_id, 0) + 1
        named = {
            self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid): count
            for cid, count in counts.items()
        }
        return dict(sorted(named.items(), key=lambda item: (-item[1], item[0])))


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a ``data.yaml`` file into a mapping."""
    if not path.is_file():
        raise DatasetError(f"Dataset config not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear dataset error
        raise DatasetError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DatasetError(f"{path} is not a YAML mapping.")
    return data


def load_data_yaml(path: str | Path) -> dict[str, Any]:
    """Public alias for :func:`read_yaml`."""
    return read_yaml(Path(path))


def class_names_from_yaml(data: dict[str, Any]) -> tuple[str, ...]:
    """Read ``names`` from a ``data.yaml`` (list or ``{index: name}`` map)."""
    names = data.get("names")
    if isinstance(names, dict):
        try:
            ordered = sorted(names.items(), key=lambda item: int(item[0]))
        except (TypeError, ValueError):
            ordered = list(names.items())
        return tuple(str(value) for _, value in ordered)
    if isinstance(names, (list, tuple)):
        return tuple(str(value) for value in names)
    return ()


def _root_dir(data_yaml: Path, data: dict[str, Any]) -> Path:
    root = data.get("path")
    if root:
        candidate = Path(str(root)).expanduser()
        if not candidate.is_absolute():
            candidate = (data_yaml.parent / candidate).resolve()
        return candidate
    return data_yaml.parent


def _resolve_entry(root: Path, entry: Any) -> list[Path]:
    """Resolve one split entry to directories (a dir, or a ``.txt`` list)."""
    if not isinstance(entry, str) or not entry.strip():
        return []
    candidate = Path(entry).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if candidate.is_dir():
        return [candidate]
    if candidate.is_file() and candidate.suffix.lower() == ".txt":
        resolved: list[Path] = []
        for line in candidate.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            item = Path(text)
            if not item.is_absolute():
                item = root / item
            resolved.append(item.resolve())
        return resolved
    return []


def _image_dirs(root: Path, spec: Any) -> list[Path]:
    """All image directories referenced by one split specification."""
    if isinstance(spec, dict):
        value = spec.get("images") or spec.get("path")
    else:
        value = spec
    if isinstance(value, (list, tuple)):
        entries = list(value)
    elif value is None:
        entries = []
    else:
        entries = [value]

    directories: list[Path] = []
    for entry in entries:
        directories.extend(_resolve_entry(root, entry))
    return directories


def _find_split(data: dict[str, Any], split: str) -> tuple[str, Any]:
    names = [split]
    if split in ("val", "valid"):
        names = ["val", "valid"]
    for key in names:
        if key in data:
            return key, data[key]
    available = ", ".join(k for k in ("train", "val", "valid", "test") if k in data)
    raise DatasetError(
        f"No '{split}' split in the dataset config (available: {available or 'none'})."
    )


def discover_images(images_dir: Path) -> list[Path]:
    """Every image file under ``images_dir`` (recursively), sorted."""
    if not images_dir.is_dir():
        return []
    return [
        path
        for path in sorted(images_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def labels_dir_for(images_dir: Path) -> Path:
    """Map an ``images`` directory to its sibling ``labels`` directory."""
    parts = list(images_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels"


def parse_label_file(label_path: Path, width: int, height: int) -> list[DetectionBox]:
    """Convert a YOLO label file to pixel-space ground-truth boxes."""
    if not label_path.is_file():
        return []
    boxes: list[DetectionBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, box_w, box_h = (float(value) for value in parts[1:5])
        except ValueError:
            continue
        x1 = max(0.0, (cx - box_w / 2.0) * width)
        y1 = max(0.0, (cy - box_h / 2.0) * height)
        x2 = min(float(width), (cx + box_w / 2.0) * width)
        y2 = min(float(height), (cy + box_h / 2.0) * height)
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        boxes.append(
            DetectionBox(class_id=class_id, x1=x1, y1=y1, x2=x2, y2=y2, score=1.0)
        )
    return boxes


def load_split(data_yaml: str | Path, split: str = "test") -> DatasetSplit:
    """Load one split of a YOLO dataset with pixel-space ground truth.

    The image size needed to convert the normalised labels is read with
    Pillow. A split with no images is a hard error (it cannot be scored);
    missing labels for individual images are tolerated and noted.
    """
    data_path = Path(data_yaml).resolve()
    data = read_yaml(data_path)
    class_names = class_names_from_yaml(data)
    name = str(data.get("name") or data_path.parent.name)
    root = _root_dir(data_path, data)

    split_key, spec = _find_split(data, split)
    image_dirs: list[Path] = []
    for directory in _image_dirs(root, spec):
        image_dirs.append(directory)

    images: list[Path] = []
    for directory in image_dirs:
        images.extend(discover_images(directory))
    images = list(dict.fromkeys(images))
    if not images:
        raise DatasetError(
            f"The '{split_key}' split of {data_path} contains no images."
        )

    from PIL import Image

    ground_truth: dict[str, list[DetectionBox]] = {}
    missing_labels = 0
    unreadable = 0
    for image_path in images:
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:  # noqa: BLE001 - skip unreadable images, report them
            unreadable += 1
            continue
        label_path = labels_dir_for(image_path.parent) / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels += 1
        ground_truth[image_path.name] = parse_label_file(label_path, width, height)

    notes: list[str] = []
    if missing_labels:
        notes.append(f"{missing_labels} image(s) had no label file (empty ground truth).")
    if unreadable:
        notes.append(f"{unreadable} image(s) could not be opened and were skipped.")

    usable_images = tuple(path for path in images if path.name in ground_truth)

    return DatasetSplit(
        name=name,
        data_yaml=data_path,
        split=split_key,
        class_names=class_names,
        images=usable_images,
        ground_truth=ground_truth,
        notes=tuple(notes),
    )


def limit_split(dataset: DatasetSplit, limit: int) -> DatasetSplit:
    """Return ``dataset`` restricted to its first ``limit`` images."""
    if limit <= 0 or limit >= len(dataset.images):
        return dataset
    images = dataset.images[:limit]
    ground_truth = {
        path.name: dataset.ground_truth.get(path.name, []) for path in images
    }
    return replace(
        dataset,
        images=images,
        ground_truth=ground_truth,
        notes=dataset.notes + (f"Limited to the first {limit} images.",),
    )
