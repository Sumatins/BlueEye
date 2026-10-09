#!/usr/bin/env python
"""Prepare a downloaded, licensed dataset for BlueEye / YOLO training.

BlueEye deliberately ships **no** datasets. This helper takes a dataset you
obtained legally (for example a Roboflow YOLOv8 export, or any folder that
already has ``train`` / ``valid`` / ``test`` splits with ``images`` and
``labels``) and:

* discovers the splits (``train`` / ``valid|val`` / ``test``),
* validates that every image has a matching YOLO ``.txt`` label file,
* counts images, label files, objects and per-class instances,
* copies the data into the canonical Ultralytics layout expected by the
  configs in ``training/datasets/``::

      data/raw/<name>/
      ├── images/{train,val,test}/*.jpg
      ├── labels/{train,val,test}/*.txt
      └── data.yaml

* writes ``data/raw/<name>/dataset_info.json`` with the full audit.

It never invents annotations: a dataset with no label files is reported as
unusable for supervised training (exit code 2), never silently accepted.
Class names are taken from the dataset's own ``data.yaml`` (or ``--classes``);
BlueEye will not guess species names.

Examples
--------
Audit a dataset without copying anything::

    python scripts/prepare_dataset.py --source "D:/data/NR Underwater" --audit-only

Prepare it for training as ``indian_marine``::

    python scripts/prepare_dataset.py \
        --source "D:/data/NR Underwater" \
        --name indian_marine \
        --classes whale_shark octopus ray olive_ridley_turtle

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

#: Source split directory name -> canonical BlueEye split name.
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "val",
    "val": "val",
    "validation": "val",
    "test": "test",
}


class DatasetError(Exception):
    """A dataset cannot be prepared for training (missing/broken labels)."""


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_splits(source: Path) -> dict[str, tuple[Path, Path]]:
    """Find ``{split: (images_dir, labels_dir)}`` in a dataset directory.

    Supports the two most common exports:

    * Roboflow: ``source/train/images`` + ``source/train/labels``
    * flat:     ``source/images/train`` + ``source/labels/train``
    """
    if not source.is_dir():
        raise DatasetError(f"Source directory does not exist: {source}")

    found: dict[str, tuple[Path, Path]] = {}

    for candidate in sorted(source.iterdir()):
        if not candidate.is_dir():
            continue
        split = SPLIT_ALIASES.get(candidate.name.lower())
        if split is None:
            continue
        images = candidate / "images"
        if images.is_dir():
            found[split] = (images, candidate / "labels")

    if found:
        return found

    flat_images = source / "images"
    if flat_images.is_dir():
        for candidate in sorted(flat_images.iterdir()):
            split = SPLIT_ALIASES.get(candidate.name.lower())
            if split is not None and candidate.is_dir():
                found[split] = (candidate, source / "labels" / candidate.name)

    return found


def read_class_names(source: Path) -> list[str]:
    """Read ``names`` from the dataset's own ``data.yaml`` (never invented)."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - PyYAML is a project dependency
        return []

    for filename in ("data.yaml", "data.yml"):
        path = source / filename
        if not path.is_file():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 - a broken file is simply ignored
            return []
        names = payload.get("names") if isinstance(payload, dict) else None
        if isinstance(names, dict):
            try:
                return [str(names[key]) for key in sorted(names, key=lambda k: int(k))]
            except (TypeError, ValueError):
                return [str(names[key]) for key in sorted(names, key=str)]
        if isinstance(names, list):
            return [str(name) for name in names]
    return []


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def audit_split(images_dir: Path, labels_dir: Path) -> dict[str, Any]:
    """Count images, labels, objects and per-class instances for one split."""
    images = (
        sorted(p for p in images_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if images_dir.is_dir()
        else []
    )
    labels = (
        {p.stem: p for p in labels_dir.glob("*.txt")} if labels_dir.is_dir() else {}
    )

    matched = 0
    missing_labels = 0
    objects = 0
    per_class: dict[int, int] = {}
    empty_labels = 0

    for image in images:
        label = labels.get(image.stem)
        if label is None:
            missing_labels += 1
            continue
        matched += 1
        lines = [
            line.strip()
            for line in label.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        if not lines:
            empty_labels += 1
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_index = int(float(parts[0]))
            except ValueError:
                continue
            objects += 1
            per_class[class_index] = per_class.get(class_index, 0) + 1

    return {
        "images": len(images),
        "label_files": len(labels),
        "matched": matched,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "objects": objects,
        "per_class": dict(sorted(per_class.items())),
    }


def _resolve_names(
    source: Path, audits: dict[str, dict[str, Any]], classes: list[str] | None
) -> list[str]:
    if classes:
        names = [str(name) for name in classes]
    else:
        names = read_class_names(source)

    if not names:
        raise DatasetError(
            "No class names found. Pass the dataset's real class names with "
            "--classes (BlueEye will not guess species names), or provide a "
            "data.yaml that lists them under 'names'."
        )

    max_index = max(
        (max(audit["per_class"]) for audit in audits.values() if audit["per_class"]),
        default=-1,
    )
    if max_index >= len(names):
        raise DatasetError(
            f"Label files reference class index {max_index}, but only "
            f"{len(names)} class name(s) were given ({names}). Pass the full, "
            "ordered class list with --classes."
        )
    return names


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def prepare_dataset(
    source: Path,
    out: Path,
    *,
    name: str | None = None,
    classes: list[str] | None = None,
    copy: bool = True,
    overwrite: bool = False,
    audit_only: bool = False,
) -> dict[str, Any]:
    """Validate (and optionally copy) a dataset into the canonical layout."""
    source = Path(source)
    out = Path(out)

    splits = discover_splits(source)
    if not splits:
        raise DatasetError(
            f"No train/valid/test splits found under {source}. Expected "
            "train|valid|test folders each containing images/ (and labels/)."
        )

    audits = {
        split: audit_split(images_dir, labels_dir)
        for split, (images_dir, labels_dir) in sorted(splits.items())
    }
    if sum(audit["matched"] for audit in audits.values()) == 0:
        raise DatasetError(
            "No image/label pairs were found. The dataset has images but no "
            "YOLO .txt annotations, so it cannot be used for supervised "
            "training as-is - obtain the annotations first."
        )

    names = _resolve_names(source, audits, classes)
    total_objects = sum(audit["objects"] for audit in audits.values())
    dataset_name = name or source.name

    info: dict[str, Any] = {
        "name": dataset_name,
        "source": str(source.resolve()),
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "classes": names,
        "num_classes": len(names),
        "splits": audits,
        "totals": {
            "images": sum(a["images"] for a in audits.values()),
            "matched": sum(a["matched"] for a in audits.values()),
            "missing_labels": sum(a["missing_labels"] for a in audits.values()),
            "objects": total_objects,
        },
        "copied": bool(copy and not audit_only),
    }

    if audit_only:
        return info

    if out.exists() and any(out.iterdir()) and not overwrite:
        raise DatasetError(
            f"Output directory {out} is not empty. Re-run with --overwrite to "
            "replace it, or choose another --out directory."
        )

    for split, (images_dir, labels_dir) in splits.items():
        dest_images = out / "images" / split
        dest_labels = out / "labels" / split
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)

        for image in sorted(
            p for p in images_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES
        ):
            label = labels_dir / f"{image.stem}.txt"
            if not label.is_file():
                continue  # image without a label: skipped, counted in the audit
            if copy:
                shutil.copy2(image, dest_images / image.name)
                shutil.copy2(label, dest_labels / label.name)

    names_block = "\n".join(f"  {index}: {value}" for index, value in enumerate(names))
    (out / "data.yaml").write_text(
        "# Generated by scripts/prepare_dataset.py - do not edit by hand.\n"
        f"# Source: {source.resolve()}\n"
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        f"{names_block}\n",
        encoding="utf-8",
    )
    (out / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return info


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and lay out a downloaded dataset for YOLO training."
    )
    parser.add_argument("--source", required=True, help="Dataset directory to prepare.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: data/raw/<name>).",
    )
    parser.add_argument("--name", default=None, help="Dataset name (default: source folder name).")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Ordered class names, when the source has no data.yaml.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate and report without copying any files.",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Write data.yaml + audit but do not copy the image/label files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output dir.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source)
    name = args.name or source.name
    out = Path(args.out) if args.out else Path("data") / "raw" / name

    try:
        info = prepare_dataset(
            source,
            out,
            name=name,
            classes=args.classes,
            copy=not args.no_copy,
            overwrite=args.overwrite,
            audit_only=args.audit_only,
        )
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    totals = info["totals"]
    print(f"Dataset: {info['name']}")
    print(f"Classes ({info['num_classes']}): {', '.join(info['classes'])}")
    for split, audit in info["splits"].items():
        print(
            f"  {split:5s} images={audit['matched']:5d} "
            f"objects={audit['objects']:6d} missing_labels={audit['missing_labels']}"
        )
    print(
        f"Totals: {totals['matched']} labelled images, {totals['objects']} objects "
        f"({totals['missing_labels']} image(s) without labels)"
    )
    if not args.audit_only:
        print(f"Prepared at: {out.resolve()}")
        print(f"Train with:  python scripts/train.py --data {out.as_posix()}/data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
