"""Image pipeline: validate -> read -> enhance? -> detect -> annotate -> save.

Adapted for BlueEye from the image inference flow of *marine-detect* by
Orange Business Services SA (AGPL-3.0-only).

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config import Settings, get_settings
from app.detection.results import DetectionResult
from app.processing.drawing import draw_detections
from app.processing.enhancement import EnhancementConfig, enhance
from app.utils.file_utils import (
    read_image_bgr,
    save_json as write_json,
    unique_output_path,
    write_image_bgr,
)
from app.utils.validation import validate_image_file

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.detection.detector import MarineDetector

logger = logging.getLogger(__name__)


class ProcessingError(RuntimeError):
    """User-facing processing failure (bad codec, unwritable output, ...)."""


@dataclass
class ImageProcessResult:
    """Outcome of one processed image."""

    input_path: Path
    output_path: Path
    report_path: Path | None
    detection: DetectionResult
    elapsed_seconds: float
    enhanced: bool

    @property
    def stats(self) -> dict[str, Any]:
        return self.detection.stats()

    @property
    def is_empty(self) -> bool:
        return self.detection.is_empty


class ImageProcessor:
    """Processes a single image end to end."""

    def __init__(self, detector: "MarineDetector", settings: Settings | None = None) -> None:
        self.detector = detector
        self.settings = settings or get_settings()

    def process(
        self,
        input_path: str | Path,
        model_selection: str | None = None,
        confidence: float | None = None,
        enhancement: EnhancementConfig | None = None,
        output_dir: str | Path | None = None,
        save_json: bool = True,
        annotate: bool = True,
    ) -> ImageProcessResult:
        """Run the full image pipeline and return the stored artefacts.

        Raises
        ------
        app.utils.validation.ValidationError
            When the input file is missing, empty, oversized or corrupt.
        app.detection.model_manager.ModelNotAvailableError
            When the selected model weights are not present.
        ProcessingError
            When the image cannot be read or the result cannot be written.
        """
        settings = self.settings
        file_path = validate_image_file(input_path)

        try:
            image = read_image_bgr(file_path)
        except Exception as exc:  # noqa: BLE001 - friendly message
            raise ProcessingError(f"Could not read image '{file_path.name}': {exc}") from exc
        if image is None or image.size == 0:
            raise ProcessingError(f"Image '{file_path.name}' is empty or unreadable.")

        # Optional enhancement (kept separate from detection so it can be
        # toggled without touching the model code).
        enhanced_flag = bool(enhancement is not None and enhancement.enabled)
        detection_image = enhance(image, enhancement) if enhanced_flag else image
        if enhanced_flag:
            logger.info(
                "Enhancement enabled (color_correction=%s, contrast=%s, denoise=%s)",
                enhancement.color_correction,
                enhancement.contrast,
                enhancement.denoise,
            )

        started = time.perf_counter()
        detection = self.detector.predict_image(
            detection_image,
            model_selection=model_selection,
            confidence=confidence,
            source_name=file_path.name,
            enhanced=enhanced_flag,
        )

        # Boxes were computed on the enhanced copy (same dimensions), so
        # they can be drawn on the original image for a faithful preview.
        annotated = draw_detections(image, detection.detections) if annotate else image

        out_dir = Path(output_dir) if output_dir else settings.images_output_dir
        suffix = file_path.suffix.lower() if file_path.suffix.lower() else ".jpg"
        try:
            output_path = unique_output_path(out_dir, "detection", suffix)
            write_image_bgr(output_path, annotated)
        except OSError as exc:
            raise ProcessingError(
                f"Could not save the annotated image to '{out_dir}': {exc}"
            ) from exc

        report_path: Path | None = None
        if save_json:
            report = detection.to_report()
            report["output_image"] = output_path.name
            report["processing_time_seconds"] = round(time.perf_counter() - started, 3)
            report_path = settings.reports_output_dir / f"{output_path.stem}.json"
            try:
                write_json(report_path, report)
            except OSError as exc:
                raise ProcessingError(
                    f"Could not save the JSON report to '{settings.reports_output_dir}': {exc}"
                ) from exc

        elapsed = time.perf_counter() - started
        logger.info(
            "Image '%s': %d detection(s), %.2f s, output %s",
            file_path.name,
            len(detection),
            elapsed,
            output_path,
        )
        return ImageProcessResult(
            input_path=file_path,
            output_path=output_path,
            report_path=report_path,
            detection=detection,
            elapsed_seconds=elapsed,
            enhanced=enhanced_flag,
        )
