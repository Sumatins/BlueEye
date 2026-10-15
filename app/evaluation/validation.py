"""Real, read-only validation of a YOLO dataset before it is evaluated.

The Research Evaluation Center must never accept a dataset it cannot honestly
score. This module inspects a ``data.yaml`` and the files it references and
reports two clearly separated classes of finding:

* **errors** - the dataset cannot be scored as-is (missing/empty requested
  split, unreadable config, malformed labels, out-of-range boxes, class ids
  with no name, an inconsistent ``nc``, ...);
* **warnings** - the dataset is still usable but the user should know
  (missing label file treated as a valid negative image, an image that could
  not be opened, a declared split whose directory is absent, possible
  train/test leakage, duplicate class names, ...).

Nothing is ever rewritten or repaired here: the source dataset is opened
read-only. A dataset is only *ready* when it has no errors and the requested
split actually contains images.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app.evaluation.datasets import (
    IMAGE_EXTENSIONS,
    DatasetError,
    _image_dirs,
    _root_dir,
    class_names_from_yaml,
    labels_dir_for,
    read_yaml,
)

#: Splits BlueEye understands, in preference order.
KNOWN_SPLITS: tuple[str, ...] = ("train", "val", "valid", "test")

#: Images larger than this are not content-hashed for leakage detection.
_LEAK_HASH_LIMIT = 25 * 1024 * 1024


class DatasetValidationError(RuntimeError):
    """Raised only when the config itself cannot be read at all."""


@dataclass(frozen=True)
class DatasetIssue:
    """One validation finding, optionally tied to a file or class."""

    severity: str  # "error" | "warning"
    code: str
    message: str
    path: str | None = None
    cls: str | None = None

    def as_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "class": self.cls,
        }


@dataclass
class SplitReport:
    """What was actually found for one declared split."""

    name: str
    declared: bool = False
    images_dirs: tuple[str, ...] = ()
    labels_dirs: tuple[str, ...] = ()
    num_images: int = 0
    num_labels: int = 0
    num_objects: int = 0
    empty_labels: int = 0
    missing_labels: int = 0
    orphan_labels: int = 0
    malformed_rows: int = 0
    out_of_range: int = 0
    bad_class_ids: int = 0
    unreadable_images: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.num_images > 0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "declared": self.declared,
            "images_dirs": list(self.images_dirs),
            "labels_dirs": list(self.labels_dirs),
            "images": self.num_images,
            "labels": self.num_labels,
            "objects": self.num_objects,
            "empty_labels": self.empty_labels,
            "missing_labels": self.missing_labels,
            "orphan_labels": self.orphan_labels,
            "malformed_rows": self.malformed_rows,
            "out_of_range": self.out_of_range,
            "bad_class_ids": self.bad_class_ids,
            "unreadable_images": self.unreadable_images,
            "class_counts": dict(self.class_counts),
        }


@dataclass
class DatasetValidation:
    """The full validation outcome for one ``data.yaml``."""

    data_yaml: Path
    name: str
    root: str
    class_names: tuple[str, ...]
    reports: tuple[SplitReport, ...]
    issues: tuple[DatasetIssue, ...]
    fingerprint: str
    requested_split: str | None = None

    @property
    def errors(self) -> tuple[DatasetIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[DatasetIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def splits_with_images(self) -> tuple[str, ...]:
        return tuple(report.name for report in self.reports if report.usable)

    @property
    def ready(self) -> bool:
        """``True`` only when there are no errors and the split has images.

        ``nc``/config errors block readiness; so does a requested split that is
        missing or empty.
        """
        if self.errors:
            return False
        if self.requested_split is None:
            return any(report.usable for report in self.reports)
        report = self.split(self.requested_split)
        if report is None and self.requested_split in ("val", "valid"):
            report = self.split("val") or self.split("valid")
        return bool(report and report.usable)

    def split(self, name: str) -> SplitReport | None:
        for report in self.reports:
            if report.name == name:
                return report
        return None

    @property
    def is_synthetic(self) -> bool:
        """True when the dataset name, path, or root indicates synthetic or test fixture data."""
        name_l = self.name.lower()
        path_l = str(self.data_yaml).lower()
        root_l = str(self.root).lower()
        return any(
            token in name_l or token in path_l or token in root_l
            for token in ("synthetic", "toy", "mock", "dummy")
        )

    def as_dict(self) -> dict:
        return {
            "data_yaml": str(self.data_yaml),
            "name": self.name,
            "root": str(self.root),
            "class_names": list(self.class_names),
            "splits": [report.as_dict() for report in self.reports],
            "splits_with_images": list(self.splits_with_images),
            "errors": [issue.as_dict() for issue in self.errors],
            "warnings": [issue.as_dict() for issue in self.warnings],
            "fingerprint": self.fingerprint,
            "ready": self.ready,
            "is_synthetic": self.is_synthetic,
        }


# --------------------------------------------------------------------------- #
# Per-split scanning
# --------------------------------------------------------------------------- #
def _scan_split(
    name: str,
    images_dirs: Sequence[Path],
    class_names: Sequence[str],
    issues: list[DatasetIssue],
) -> tuple[SplitReport, list[Path]]:
    """Validate one split; append findings to ``issues``."""
    report = SplitReport(
        name=name,
        declared=True,
        images_dirs=tuple(str(path) for path in images_dirs),
    )
    class_count = len(class_names)
    images: list[Path] = []
    seen: set[Path] = set()
    label_stems: set[str] = set()

    for images_dir in images_dirs:
        labels_dir = labels_dir_for(images_dir)
        if labels_dir.is_dir():
            report.labels_dirs = report.labels_dirs + (str(labels_dir),)
            for label in labels_dir.rglob("*.txt"):
                if label.is_file():
                    label_stems.add(label.stem)
        else:
            issues.append(
                DatasetIssue(
                    "warning",
                    "labels_dir_missing",
                    f"Split '{name}': no 'labels' directory beside "
                    f"'{images_dir}'.",
                    path=str(labels_dir),
                )
            )

        found = [
            path
            for path in (sorted(images_dir.rglob("*")) if images_dir.is_dir() else [])
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        for image in found:
            if image in seen:
                continue
            seen.add(image)
            images.append(image)
            report.num_images += 1
            label_path = labels_dir_for(image.parent) / f"{image.stem}.txt"
            if label_path.is_file():
                report.num_labels += 1
                _scan_label(label_path, class_names, report, issues, class_count)
            else:
                report.missing_labels += 1
                issues.append(
                    DatasetIssue(
                        "warning",
                        "missing_label",
                        "No label file; treated as a negative (background) image.",
                        path=str(image),
                    )
                )
            _check_image(image, report, issues)

    # Orphan label files (a label with no matching image) often indicate a
    # half-copied dataset.
    image_stems = {image.stem for image in images}
    for stem in sorted(label_stems - image_stems):
        report.orphan_labels += 1
        issues.append(
            DatasetIssue(
                "warning",
                "orphan_label",
                "Label file has no matching image.",
                path=stem,
            )
        )
    return report, images


def _scan_label(
    label_path: Path,
    class_names: Sequence[str],
    report: SplitReport,
    issues: list[DatasetIssue],
    class_count: int,
) -> None:
    try:
        text = label_path.read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(
            DatasetIssue(
                "error",
                "label_unreadable",
                f"Could not read label file: {exc}",
                path=str(label_path),
            )
        )
        return

    rows = [line for line in text.splitlines() if line.strip()]
    if not rows:
        report.empty_labels += 1
        return

    for line_number, line in enumerate(rows, start=1):
        parts = line.split()
        if len(parts) != 5:
            report.malformed_rows += 1
            issues.append(
                DatasetIssue(
                    "error",
                    "label_malformed",
                    f"Line {line_number}: expected 5 fields "
                    f"(class cx cy w h), found {len(parts)}.",
                    path=str(label_path),
                )
            )
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, box_w, box_h = (float(value) for value in parts[1:5])
        except ValueError:
            report.malformed_rows += 1
            issues.append(
                DatasetIssue(
                    "error",
                    "label_malformed",
                    f"Line {line_number}: values are not numeric.",
                    path=str(label_path),
                )
            )
            continue

        if class_count and not (0 <= class_id < class_count):
            report.bad_class_ids += 1
            issues.append(
                DatasetIssue(
                    "error",
                    "label_bad_class",
                    f"Line {line_number}: class id {class_id} has no name "
                    f"(known ids 0..{class_count - 1}).",
                    path=str(label_path),
                )
            )
            continue

        problems: list[str] = []
        for value, label in ((cx, "cx"), (cy, "cy")):
            if not 0.0 <= value <= 1.0:
                problems.append(f"{label}={value:g}")
        for value, label in ((box_w, "w"), (box_h, "h")):
            if not 0.0 < value <= 1.0:
                problems.append(f"{label}={value:g}")
        if not problems:
            if not (0.0 <= cx - box_w / 2.0 and cx + box_w / 2.0 <= 1.0 + 1e-6):
                problems.append("x extends outside the image")
            if not (0.0 <= cy - box_h / 2.0 and cy + box_h / 2.0 <= 1.0 + 1e-6):
                problems.append("y extends outside the image")
        if problems:
            report.out_of_range += 1
            issues.append(
                DatasetIssue(
                    "error",
                    "label_out_of_range",
                    f"Line {line_number}: invalid normalised box "
                    f"({'; '.join(problems)}).",
                    path=str(label_path),
                )
            )
            continue

        report.num_objects += 1
        name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        report.class_counts[name] = report.class_counts.get(name, 0) + 1


def _check_image(image: Path, report: SplitReport, issues: list[DatasetIssue]) -> None:
    try:
        from PIL import Image

        with Image.open(image) as handle:
            handle.size  # touching .size proves the header parsed
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        report.unreadable_images += 1
        issues.append(
            DatasetIssue(
                "warning",
                "image_unreadable",
                f"Could not open image: {exc}",
                path=str(image),
            )
        )


# --------------------------------------------------------------------------- #
# Config-level checks and fingerprint
# --------------------------------------------------------------------------- #
def _class_issues(data: dict[str, Any], class_names: Sequence[str]) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    if not class_names:
        issues.append(
            DatasetIssue(
                "error",
                "names_missing",
                "The config has no 'names' mapping, so labels cannot be named.",
            )
        )
        return issues

    normalised = [name.strip().casefold() for name in class_names if name.strip()]
    seen: set[str] = set()
    for name in normalised:
        if name in seen:
            issues.append(
                DatasetIssue(
                    "warning",
                    "class_name_duplicate",
                    f"Duplicate class name '{name}' - per-class metrics will be "
                    f"ambiguous.",
                    cls=name,
                )
            )
        seen.add(name)

    nc = data.get("nc")
    if nc is not None:
        try:
            declared = int(nc)
        except (TypeError, ValueError):
            declared = -1
        if declared != len(class_names):
            issues.append(
                DatasetIssue(
                    "error",
                    "nc_mismatch",
                    f"'nc' is {nc} but {len(class_names)} class names are "
                    f"defined.",
                )
            )
    return issues


def _image_digest(path: Path) -> str | None:
    try:
        if path.stat().st_size > _LEAK_HASH_LIMIT:
            return None
    except OSError:
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _leakage_issues(
    split_images: dict[str, list[Path]],
) -> list[DatasetIssue]:
    """Warn when the same image content appears in two different splits."""
    digests: dict[str, dict[str, str]] = {}
    for split, images in split_images.items():
        table: dict[str, str] = {}
        for image in images:
            value = _image_digest(image)
            if value:
                table.setdefault(value, str(image))
        digests[split] = table

    issues: list[DatasetIssue] = []
    names = [name for name in ("train", "val", "valid", "test") if name in digests]
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = set(digests[left]) & set(digests[right])
            if not shared:
                continue
            example = digests[left][next(iter(shared))]
            issues.append(
                DatasetIssue(
                    "warning",
                    "split_leakage",
                    f"{len(shared)} identical image(s) appear in both "
                    f"'{left}' and '{right}' (train/test leakage).",
                    path=example,
                )
            )
    return issues


def _fingerprint(
    data_yaml: Path,
    class_names: Sequence[str],
    split_images: dict[str, list[Path]],
) -> str:
    """Deterministic fingerprint of the config, classes and image inventory."""
    digest = hashlib.sha256()
    digest.update(b"blueeye-dataset-v1|")
    digest.update("|".join(class_names).encode("utf-8"))
    for split in sorted(split_images):
        for image in sorted(split_images[split]):
            try:
                size = image.stat().st_size
            except OSError:
                size = -1
            digest.update(f"\n{split}:{image.name}:{size}".encode("utf-8"))
    digest.update(b"\n")
    try:
        digest.update(Path(data_yaml).read_bytes())
    except OSError:
        digest.update(str(data_yaml).encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def validate_dataset(
    data_yaml: str | Path,
    split: str | None = None,
) -> DatasetValidation:
    """Validate a YOLO dataset config and return findings + statistics.

    Raises :class:`DatasetValidationError` only when the config file itself is
    missing or unparseable; every other problem is reported as an issue.
    """
    path = Path(data_yaml)
    try:
        data = read_yaml(path)
    except DatasetError as exc:
        raise DatasetValidationError(str(exc)) from exc

    class_names = class_names_from_yaml(data)
    name = str(data.get("name") or path.parent.name)
    root = _root_dir(path, data)

    issues: list[DatasetIssue] = list(_class_issues(data, class_names))

    declared = [key for key in ("train", "val", "valid", "test") if key in data]
    reports: list[SplitReport] = []
    split_images: dict[str, list[Path]] = {}

    for key in declared:
        images_dirs = _image_dirs(root, data[key])
        if not images_dirs:
            issues.append(
                DatasetIssue(
                    "warning",
                    "split_unresolved",
                    f"Split '{key}' is declared but its directory could not be "
                    f"found under {root}.",
                    path=str(data.get(key)),
                )
            )
            reports.append(SplitReport(name=key, declared=True))
            split_images[key] = []
            continue
        report, images = _scan_split(key, images_dirs, class_names, issues)
        reports.append(report)
        split_images[key] = images

    if not declared:
        issues.append(
            DatasetIssue(
                "error",
                "splits_missing",
                "The config declares no 'train'/'val'/'test' split.",
            )
        )

    # A requested split must exist and contain images to be scorable.
    if split:
        aliases = {split}
        if split in ("val", "valid"):
            aliases = {"val", "valid"}
        report = next((item for item in reports if item.name in aliases), None)
        if report is None:
            issues.append(
                DatasetIssue(
                    "error",
                    "requested_split_missing",
                    f"No '{split}' split in the dataset config "
                    f"(available: {', '.join(declared) or 'none'}).",
                )
            )
        elif not report.usable:
            issues.append(
                DatasetIssue(
                    "error",
                    "requested_split_empty",
                    f"The '{split}' split contains no images.",
                )
            )

    issues.extend(_leakage_issues(split_images))

    fingerprint = _fingerprint(path, class_names, split_images)
    return DatasetValidation(
        data_yaml=path,
        name=name,
        root=str(root),
        class_names=class_names,
        reports=tuple(reports),
        issues=tuple(issues),
        fingerprint=fingerprint,
        requested_split=split,
    )
