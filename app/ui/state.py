"""Shared BlueEye UI state: session defaults, navigation, cached resources.

This module is the seam between the web UI and the detection backend. It
holds the *only* code the UI uses to reach the pipeline (cached detector,
upload intake, history recording) - page modules never construct models or
open files themselves.

Navigation works through ``st.session_state["nav_sel"]``; a page requests a
jump with :func:`navigate`, which defers the change to the next run so that
Streamlit's widget-state rules are respected.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import streamlit as st

from app.config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, get_settings
from app.detection.detector import MarineDetector
from app.detection.model_manager import ModelSpec
from app.processing.enhancement import EnhancementConfig
from app.utils.file_utils import unique_output_path
from app.utils.history import HistoryEntry, add_entry
from app.utils.validation import (
    ValidationError,
    ensure_extension,
    ensure_safe_path,
    ensure_size,
    sanitize_filename,
    validate_image_file,
    validate_video_file,
)

logger = logging.getLogger(__name__)

#: Friendly label for the "run every installed model" pseudo-selection.
AUTO_LABEL = "Auto - core models (Fish & Invertebrates + MegaFauna)"

#: Friendly label for the pseudo-selection that also runs the additional models.
ALL_LABEL = "Every installed model (core + additional)"

#: Registry categories that belong to the "additional aquatic species" set.
ADDITIONAL_CATEGORIES = ("additional", "indian")

#: Media kinds accepted by :func:`save_upload`.
KINDS = ("image", "video")


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_detector() -> MarineDetector:
    """One detector (and one set of loaded models) per Streamlit server."""
    return MarineDetector()


@st.cache_resource(show_spinner=False)
def get_settings_cached():
    settings = get_settings()
    settings.ensure_dirs()
    return settings


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_session(settings: Any) -> None:
    """Seed session defaults. Safe to call on every run (uses setdefault)."""
    st.session_state.setdefault("nav_sel", "dashboard")
    st.session_state.setdefault("detect_media", "Image")
    st.session_state.setdefault("model_selection", "auto")
    st.session_state.setdefault("confidence", float(settings.default_confidence))
    st.session_state.setdefault("use_model_default", False)
    st.session_state.setdefault(
        "enhancement_config",
        EnhancementConfig(
            enabled=False,
            color_correction=False,
            contrast=False,
            denoise=False,
        ),
    )


def navigate(page: str) -> None:
    """Ask the shell to show ``page`` on the next run.

    The pending value is stored separately because ``st.session_state`` keys
    bound to live widgets cannot be modified during the same run.
    """
    st.session_state["nav_pending"] = page
    st.rerun()


def apply_pending_navigation() -> None:
    """Consume a queued navigation request (called before the nav widget)."""
    pending = st.session_state.pop("nav_pending", None)
    if pending:
        st.session_state["nav_sel"] = pending


# --------------------------------------------------------------------------- #
# Upload intake (identical validation to the CLI's input handling)
# --------------------------------------------------------------------------- #
def save_upload(uploaded: Any, settings: Any, kind: str) -> Path:
    """Validate a Streamlit upload and store it under ``data/input/``.

    Raises :class:`ValidationError` with a user-facing message when the file
    is of an unsupported type, too large, empty or not decodable.
    """
    if kind not in KINDS:
        raise ValidationError(f"Unsupported input kind: {kind}")

    name = sanitize_filename(uploaded.name, fallback=f"upload.{kind}")
    allowed = IMAGE_EXTENSIONS if kind == "image" else VIDEO_EXTENSIONS
    ensure_extension(name, allowed, kind)
    size = uploaded.size if hasattr(uploaded, "size") else len(uploaded.getbuffer())
    limit = (settings.max_image_mb if kind == "image" else settings.max_video_mb) * 1024 * 1024
    ensure_size(size, limit, kind)

    data = uploaded.getvalue()
    if not data:
        raise ValidationError("The uploaded file is empty.")

    target = ensure_safe_path(settings.input_dir, settings.input_dir / name)
    if target.exists():
        target = ensure_safe_path(
            settings.input_dir,
            unique_output_path(settings.input_dir, Path(name).stem, Path(name).suffix).with_suffix(
                Path(name).suffix
            ),
        )
    target.write_bytes(data)
    if kind == "image":
        validate_image_file(target)
    else:
        validate_video_file(target)
    return target


# --------------------------------------------------------------------------- #
# Model selection helpers
# --------------------------------------------------------------------------- #
def registry(detector: MarineDetector) -> dict[str, ModelSpec]:
    """Every registered model (built-in plus user-registered custom ones)."""
    return detector.model_manager.registry()


def model_label(key: str, spec: ModelSpec | None, available: bool) -> str:
    """Human-friendly label used by selectors and cards."""
    if key == "auto":
        return AUTO_LABEL
    if key == "all":
        return ALL_LABEL
    if spec is None:
        return key
    label = spec.display_name
    if not available:
        label += " - not installed"
    return label


def model_description(key: str, spec: ModelSpec | None) -> str:
    """Short explanation shown under the model selector."""
    if key == "auto":
        return (
            "Runs the two core models and merges the detections. The class "
            "sets are disjoint, so results contain no duplicates."
        )
    if key == "all":
        return (
            "Runs every installed model, including the additional aquatic "
            "models. Overlapping detections of the same class are merged."
        )
    if spec is None:
        return ""
    return spec.summary or spec.description


def additional_specs(registry: dict[str, ModelSpec]) -> list[ModelSpec]:
    """Specs that belong to the additional / Indian aquatic set."""
    return [
        spec for spec in registry.values() if spec.category in ADDITIONAL_CATEGORIES
    ]


def available_keys(detector: MarineDetector) -> list[str]:
    """Keys whose weights are present locally."""
    return [spec.key for spec in detector.model_manager.available_models()]


def any_model_available(detector: MarineDetector) -> bool:
    return bool(available_keys(detector))


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
def record_history(
    kind: str,
    report: dict[str, Any] | None,
    output: Path | str | None,
    report_path: Path | str | None,
) -> None:
    """Append one finished detection to the persistent history.

    History is a convenience feature: any failure here is logged and never
    interrupts the detection flow that triggered it.
    """
    if not report:
        return
    try:
        entry = HistoryEntry.from_report(
            kind,
            report,
            output=str(output or ""),
            report_path=str(report_path) if report_path else None,
        )
        add_entry(entry)
    except Exception:  # noqa: BLE001 - history must never break detection
        logger.exception("Could not record detection history")
