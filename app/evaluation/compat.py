"""Dataset <-> model class compatibility.

BlueEye never silently remaps a class label. Before any metric is computed
this module compares the class names a *dataset* declares with the names a
*model* actually emits, and reports:

* **supported** dataset classes - an exact (case-insensitive) name match;
* **unsupported** dataset classes - the model cannot emit them, so they can
  only ever become false negatives; they are excluded from the per-class
  comparison and listed, never remapped to a different label;
* **extra** model classes - the model can emit them but the dataset has no
  such class; they are ignored for scoring.

When nothing overlaps the comparison is refused so that an incompatible
model/dataset pair is never presented as a valid result.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def _normalise(name: str) -> str:
    """Case- and whitespace-insensitive form of a class name."""
    return " ".join(str(name).split()).casefold()


@dataclass(frozen=True)
class ClassCompat:
    """Result of matching dataset class names against model class names."""

    supported: tuple[str, ...]
    unsupported: tuple[str, ...]
    extra_model: tuple[str, ...]
    mapping: tuple[tuple[str, str], ...]

    @property
    def comparable(self) -> bool:
        """True when at least one class name is shared."""
        return bool(self.supported)

    @property
    def summary(self) -> str:
        if not self.comparable:
            return "No shared class names: the model and this dataset are not comparable."
        return (
            f"{len(self.supported)} comparable class(es); "
            f"{len(self.unsupported)} dataset class(es) the model cannot detect; "
            f"{len(self.extra_model)} model class(es) absent from the dataset."
        )


def compare_classes(
    dataset_classes: Sequence[str],
    model_classes: Iterable[str],
) -> ClassCompat:
    """Match dataset class names to model class names by exact name.

    A match is case-insensitive and whitespace-insensitive (``"Fish"`` and
    ``"fish"`` are the same class). Anything else is reported as unsupported
    (dataset side) or extra (model side) - never guessed.
    """
    model_lookup: dict[str, str] = {}
    for name in model_classes:
        text = str(name).strip()
        if text:
            model_lookup.setdefault(_normalise(text), text)

    supported: list[str] = []
    unsupported: list[str] = []
    mapping: list[tuple[str, str]] = []
    matched_model_keys: set[str] = set()

    for name in dataset_classes:
        text = str(name).strip()
        if not text:
            continue
        key = _normalise(text)
        model_name = model_lookup.get(key)
        if model_name is None:
            unsupported.append(text)
        else:
            supported.append(text)
            mapping.append((text, model_name))
            matched_model_keys.add(key)

    extra_model = [
        model_name
        for key, model_name in model_lookup.items()
        if key not in matched_model_keys
    ]
    return ClassCompat(
        supported=tuple(supported),
        unsupported=tuple(unsupported),
        extra_model=tuple(extra_model),
        mapping=tuple(mapping),
    )
