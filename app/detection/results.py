"""Structured detection results, statistics and report serialisation.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Iterator, Sequence

#: Presentation-only mapping from raw model class ids to friendly names.
#: Sources: species scope of the upstream *marine-detect* README
#: (https://github.com/Orange-OpenSource/marine-detect). Raw class ids are
#: always preserved in JSON reports; this mapping never invents classes.
CLASS_DISPLAY_NAMES = {
    # Fish & invertebrates model
    "fish": "Fish",
    "serranidae": "Grouper (Serranidae)",
    "scaridae": "Parrotfish (Scaridae)",
    "chaetodontidae": "Butterfly fish (Chaetodontidae)",
    "lutjanidae": "Snapper (Lutjanidae)",
    "muraenidae": "Moray eel (Muraenidae)",
    "haemulidae": "Sweetlips (Haemulidae)",
    "cromileptes_altivelis": "Barramundi cod",
    "cheilinus_undulatus": "Humphead wrasse",
    "bolbometopon_muricatum": "Bumphead parrotfish",
    "giant_clam": "Giant clam",
    "urchin": "Sea urchin",
    "sea_cucumber": "Sea cucumber",
    "crown_of_thorns": "Crown-of-thorns",
    "lobster": "Lobster",
    # MegaFauna model
    "shark": "Shark",
    "turtle": "Sea turtle",
    "ray": "Ray",
}


def display_class_name(class_name: str) -> str:
    """Return a human-friendly name for a raw model class id."""
    key = str(class_name).strip().lower()
    if key in CLASS_DISPLAY_NAMES:
        return CLASS_DISPLAY_NAMES[key]
    return key.replace("_", " ").capitalize()


def compute_statistics(
    confidences: Sequence[float], class_counts: dict[str, int] | Counter
) -> dict[str, Any]:
    """Compute the standard BlueEye detection statistics.

    Used for single images (all detections of one frame) and aggregated for
    videos (every detection of every processed frame).
    """
    total = sum(class_counts.values())
    unique_classes = len(class_counts)
    if confidences:
        average = sum(confidences) / len(confidences)
        maximum = max(confidences)
    else:
        average = 0.0
        maximum = 0.0
    return {
        "total_detections": total,
        "unique_classes": unique_classes,
        "detections_per_class": dict(
            sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "average_confidence": round(average, 4),
        "max_confidence": round(maximum, 4),
    }


@dataclass(frozen=True)
class Detection:
    """A single detected object.

    ``bbox`` is ``(x1, y1, x2, y2)`` in pixels of the analysed frame.
    """

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    model: str = ""

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @classmethod
    def from_xyxy(
        cls,
        xyxy: Iterable[float],
        class_name: str,
        confidence: float,
        model: str = "",
    ) -> "Detection":
        """Build a detection from raw corner coordinates (floats allowed)."""
        x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
        return cls(class_name=str(class_name), confidence=float(confidence), bbox=(x1, y1, x2, y2), model=model)

    @property
    def display_name(self) -> str:
        return display_class_name(self.class_name)

    @property
    def confidence_percent(self) -> str:
        return f"{self.confidence * 100:.1f}%"

    def to_dict(self) -> dict[str, Any]:
        """Structured representation (schema from the BlueEye report)."""
        x1, y1, x2, y2 = self.bbox
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "model": self.model,
        }

    def to_report_dict(self) -> dict[str, Any]:
        """Compact representation used inside JSON report files."""
        x1, y1, x2, y2 = self.bbox
        return {
            "class": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [x1, y1, x2, y2],
            "model": self.model,
        }


@dataclass
class DetectionResult:
    """All detections produced for a single image (or video frame)."""

    source: str
    detections: list[Detection] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    #: User-requested threshold, or ``None`` when per-model recommended
    #: thresholds were used.
    confidence_threshold: float | None = None
    #: Threshold actually applied per model key.
    thresholds: dict[str, float] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    enhanced: bool = False

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)

    @property
    def is_empty(self) -> bool:
        return not self.detections

    def stats(self) -> dict[str, Any]:
        """Detection statistics for this image/frame."""
        counts: Counter = Counter(det.class_name for det in self.detections)
        return compute_statistics([det.confidence for det in self.detections], counts)

    def sorted_detections(self) -> list[Detection]:
        """Detections sorted by descending confidence (for display)."""
        return sorted(self.detections, key=lambda det: det.confidence, reverse=True)

    def to_report(self) -> dict[str, Any]:
        """Machine-readable report payload (written to ``outputs/reports``)."""
        return {
            "input": self.source,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "models": list(self.models_used),
            "confidence_threshold": self.confidence_threshold,
            "thresholds_applied": dict(self.thresholds),
            "enhanced": self.enhanced,
            "image_size": {"width": self.image_width, "height": self.image_height},
            "inference_time_ms": round(self.inference_time_ms, 2),
            "detections": [det.to_report_dict() for det in self.sorted_detections()],
            "statistics": self.stats(),
        }
