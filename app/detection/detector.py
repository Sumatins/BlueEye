"""BlueEye aquatic-life detector: the single inference entry point.

``MarineDetector`` hides model loading, device selection, per-model
confidence thresholds and result construction behind a small interface:

* :meth:`MarineDetector.load_model`        - preload weights
* :meth:`MarineDetector.predict_image`     - detect on an image array
* :meth:`MarineDetector.predict_frame`     - detect on one video frame
* :meth:`MarineDetector.predict_video`     - full video pipeline
* :meth:`MarineDetector.get_class_names`   - class names of a model

The UI and the CLI never talk to Ultralytics directly.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from app.config import Settings, get_settings
from app.detection.model_manager import (
    ModelManager,
    ModelNotAvailableError,
)
from app.detection.results import Detection, DetectionResult

logger = logging.getLogger(__name__)

#: Built-in shortcuts for the "model selection" parameter. Any key that is
#: present in the registry (built-in *or* user-registered custom / regional
#: model) is also accepted directly - see :meth:`MarineDetector.resolve_keys`.
MODEL_SELECTIONS = ("auto", "all", "fish_inv", "megafauna")

#: Pseudo-selection that runs **every** installed model (including the
#: opt-in additional models). ``auto`` keeps running only the core models.
ALL_SELECTION = "all"

#: Aliases accepted for convenience.
_SELECTION_ALIASES = {
    "auto": "auto",
    "both": "auto",
    "all": ALL_SELECTION,
    "everything": ALL_SELECTION,
    "combined": ALL_SELECTION,
    "full": ALL_SELECTION,
    "fish": "fish_inv",
    "fish_inv": "fish_inv",
    "fishinv": "fish_inv",
    "fish_&_invertebrates": "fish_inv",
    "invertebrates": "fish_inv",
    "mega": "megafauna",
    "megafauna": "megafauna",
}

#: Two detections from *different* models are treated as duplicates when they
#: have the same class name and overlap by more than this IoU. This keeps the
#: combined ("all") mode from double-reporting the same animal.
_DUPLICATE_IOU = 0.7


def _iou(box_a, box_b) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


class MarineDetector:
    """YOLOv8 / RT-DETR-based aquatic-life detector.

    Parameters
    ----------
    model_manager:
        Optional injected :class:`ModelManager` (tests substitute a fake).
    settings:
        Optional injected :class:`Settings`.
    """

    def __init__(
        self,
        model_manager: ModelManager | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_manager = model_manager or ModelManager(self.settings)

    # ------------------------------------------------------------------ #
    # Model handling
    # ------------------------------------------------------------------ #
    @property
    def device(self) -> str:
        return self.model_manager.device

    def _registered_keys(self) -> list[str]:
        """Every model key visible to the manager (built-in + custom)."""
        registry = getattr(self.model_manager, "registry", None)
        if callable(registry):
            return list(registry().keys())
        # Fallback for manager doubles that only implement the older interface.
        from app.detection.model_manager import iter_specs

        return [spec.key for spec in iter_specs(self.model_manager)]

    def resolve_keys(self, model_selection: str | None = None) -> list[str]:
        """Turn a selection into the list of model keys that will run.

        ``auto`` runs **every model whose weights are available** (combined
        mode). In addition to the built-in shortcuts and their aliases
        (``fish``/``mega``), **any key present in the registry** can be
        selected explicitly - this is what lets a user-registered custom /
        regional model (for example an Indian-biodiversity model) be run
        directly from the UI, the CLI and the API.

        Raises :class:`ModelNotAvailableError` when nothing usable is
        selected or present.
        """
        selection = (model_selection or self.settings.default_model_selection or "auto")
        raw = str(selection).strip()

        normalized = _SELECTION_ALIASES.get(raw.lower())
        if normalized is None:
            # A registered model key is a valid explicit selection. Match
            # case-insensitively so ``--model MegaFauna`` also works.
            registered = {key.lower(): key for key in self._registered_keys()}
            normalized = registered.get(raw.lower())

        if normalized is None:
            known = ", ".join(self._registered_keys()) or "none"
            raise ModelNotAvailableError(
                f"Unknown model selection '{selection}'. "
                f"Choose 'auto' or one of the registered models: {known}."
            )

        if normalized in ("auto", ALL_SELECTION):
            specs = list(self.model_manager.available_models())
            if normalized == "auto":
                # ``auto`` keeps the original behaviour: the core models
                # only. Additional (opt-in) models have ``auto=False``.
                specs = [spec for spec in specs if getattr(spec, "auto", True)]
            available = [spec.key for spec in specs]
            if not available:
                raise ModelNotAvailableError(
                    "No model weights found. Download them with "
                    "'python scripts/download_models.py' or via the download "
                    "button in the web UI, then try again."
                )
            logger.info("Model selection '%s' -> models: %s", normalized, ", ".join(available))
            return available

        if not self.model_manager.is_available(normalized):
            spec = self.model_manager.get_spec(normalized)
            raise ModelNotAvailableError(
                f"Model weights for '{spec.display_name}' are not available "
                f"locally. Run 'python scripts/download_models.py' to download them "
                f"(custom / regional weights must be placed under 'models/' - see "
                f"models/custom/README.md)."
            )
        return [normalized]

    def load_model(self, model_selection: str | None = None) -> list[str]:
        """Preload the selected model(s) into memory; returns their keys."""
        keys = self.resolve_keys(model_selection)
        for key in keys:
            self.model_manager.load(key)
        return keys

    def get_class_names(self, model_selection: str | None = None) -> dict[str, dict[int, str]]:
        """Class names per model key for the selected model(s)."""
        keys = self.resolve_keys(model_selection)
        return {key: self.model_manager.get_class_names(key) for key in keys}

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict_image(
        self,
        image: np.ndarray,
        model_selection: str | None = None,
        confidence: float | None = None,
        source_name: str = "image",
        enhanced: bool = False,
    ) -> DetectionResult:
        """Run detection on a BGR image array and return structured results.

        Parameters
        ----------
        image:
            ``H x W x 3`` BGR ``uint8`` array (OpenCV convention).
        model_selection:
            ``auto`` (all available models), ``fish_inv`` or ``megafauna``.
        confidence:
            Minimum confidence in ``[0, 1]``. ``None`` uses each model's
            recommended threshold (FishInv 0.523 / MegaFauna 0.546, as
            published upstream).
        """
        if image is None or not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("predict_image expects an HxWx3 BGR numpy array.")
        if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")

        keys = self.resolve_keys(model_selection)
        height, width = image.shape[:2]

        detections: list[Detection] = []
        thresholds: dict[str, float] = {}
        started = time.perf_counter()

        for key in keys:
            spec = self.model_manager.get_spec(key)
            applied = spec.recommended_confidence if confidence is None else float(confidence)
            thresholds[key] = applied
            model = self.model_manager.load(key)

            try:
                raw_results = model.predict(
                    image,
                    conf=applied,
                    device=self.device,
                    verbose=False,
                )
            except Exception as exc:  # noqa: BLE001 - friendly surfaced error
                raise RuntimeError(
                    f"Inference failed on model '{spec.display_name}': {exc}"
                ) from exc

            result = raw_results[0] if raw_results else None
            if result is None or getattr(result, "boxes", None) is None:
                continue
            names = getattr(result, "names", {}) or {}
            for box in result.boxes:
                # Explicit post-filter (the threshold is also passed to the
                # model, this guarantees the documented behaviour).
                box_conf = float(box.conf[0] if hasattr(box.conf, "__len__") else box.conf)
                if box_conf < applied:
                    continue
                cls_idx = int(box.cls[0] if hasattr(box.cls, "__len__") else box.cls)
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                # Clamp boxes to the frame and drop degenerate results.
                x1 = max(0.0, min(x1, width - 1))
                x2 = max(0.0, min(x2, width - 1))
                y1 = max(0.0, min(y1, height - 1))
                y2 = max(0.0, min(y2, height - 1))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                detections.append(
                    Detection.from_xyxy(
                        (x1, y1, x2, y2),
                        class_name=str(names.get(cls_idx, cls_idx)),
                        confidence=box_conf,
                        model=key,
                    )
                )

        if len(keys) > 1 and detections:
            detections = self._suppress_duplicates(detections)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Detected %d object(s) in %s using [%s] in %.0f ms",
            len(detections),
            source_name,
            ", ".join(keys),
            elapsed_ms,
        )
        return DetectionResult(
            source=str(source_name),
            detections=detections,
            models_used=list(keys),
            confidence_threshold=None if confidence is None else float(confidence),
            thresholds=thresholds,
            inference_time_ms=elapsed_ms,
            image_width=width,
            image_height=height,
            enhanced=enhanced,
        )

    @staticmethod
    def _suppress_duplicates(detections: list[Detection]) -> list[Detection]:
        """Drop same-class boxes that two models both reported for one object.

        Only detections coming from *different* models are compared, and only
        when the class names match, so genuinely distinct species are never
        merged. The highest-confidence box is kept.
        """
        kept: list[Detection] = []
        for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
            duplicate = any(
                other.model != det.model
                and other.class_name == det.class_name
                and _iou(other.bbox, det.bbox) >= _DUPLICATE_IOU
                for other in kept
            )
            if not duplicate:
                kept.append(det)
        return kept

    def predict_frame(
        self,
        frame: np.ndarray,
        model_selection: str | None = None,
        confidence: float | None = None,
        frame_index: int | None = None,
        enhanced: bool = False,
    ) -> DetectionResult:
        """Detect on a single video frame (alias of :meth:`predict_image`)."""
        name = f"frame {frame_index}" if frame_index is not None else "frame"
        return self.predict_image(
            frame,
            model_selection=model_selection,
            confidence=confidence,
            source_name=name,
            enhanced=enhanced,
        )

    def predict_video(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        model_selection: str | None = None,
        confidence: float | None = None,
        enhancement: Any = None,
        progress_callback: Any = None,
        save_json: bool = True,
        max_frames: int | None = None,
    ):
        """Run the full frame-by-frame video pipeline.

        Delegates to :class:`app.processing.video_processor.VideoProcessor`
        (imported lazily to avoid a package cycle).
        """
        from app.processing.video_processor import VideoProcessor

        processor = VideoProcessor(self, settings=self.settings)
        return processor.process(
            input_path=input_path,
            output_path=output_path,
            model_selection=model_selection,
            confidence=confidence,
            enhancement=enhancement,
            progress_callback=progress_callback,
            save_json=save_json,
            max_frames=max_frames,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def verify_model(self, key: str) -> dict[str, Any]:
        """Run a real inference smoke test for ``key`` (never raises).

        Returns ``{"status", "ok", "error", "classes"}``; see
        :meth:`app.detection.model_manager.ModelManager.verify`.
        """
        verify = getattr(self.model_manager, "verify", None)
        if callable(verify):
            return verify(key)
        available = self.model_manager.is_available(key)
        return {"status": "ready" if available else "not_installed", "ok": available,
                "error": None, "classes": []}

    def model_status(self) -> list[dict[str, Any]]:
        """Per-model status used by the CLI and the UI.

        Includes user-registered custom models (see
        :meth:`app.detection.model_manager.ModelManager.registry`).
        """
        from app.detection.model_manager import iter_specs

        status = []
        for spec in iter_specs(self.model_manager):
            available = self.model_manager.is_available(spec.key)
            entry: dict[str, Any] = {
                "key": spec.key,
                "name": spec.display_name,
                "description": spec.description,
                "available": available,
                "status": spec.readiness_status(available),
                "recommended_confidence": spec.recommended_confidence,
                "path": str(self.model_manager.resolve_path(spec.key)),
                "architecture": spec.architecture,
                "loader": spec.loader,
                "habitat": spec.habitat,
                "auto": spec.auto,
                "custom": spec.custom,
                "license": spec.license,
                "source": spec.source,
                "classes": [],
            }
            if available:
                try:
                    names = self.model_manager.get_class_names(spec.key)
                    entry["classes"] = [names[idx] for idx in sorted(names)]
                except Exception:  # noqa: BLE001 - status must never crash
                    entry["classes"] = []
            status.append(entry)
        return status
