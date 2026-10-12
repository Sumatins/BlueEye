"""Model Comparison: run several installed models on the same image.

The page reuses the unmodified detector (one call per model) and the shared
annotation renderer, so what is shown is exactly what the application would
produce for each model.

Honesty rules
-------------
* A model can only emit classes in its own list. The page shows each model's
  class list and its unique classes so a missing species is never read as
  proof of absence.
* The numbers shown are real per-model detection counts, confidences and
  inference timing. No accuracy metric is implied - that requires the labelled
  dataset used by the Research Evaluation Center.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import cv2
import streamlit as st

from app.detection.comparison import class_coverage, run_comparison
from app.detection.detector import MarineDetector
from app.detection.results import display_class_name
from app.processing.drawing import draw_detections
from app.ui import state
from app.ui.components import (
    error_card,
    hero,
    metrics_row,
    notice,
)

logger = logging.getLogger(__name__)


def render(detector: MarineDetector, settings, user=None) -> None:
    """Render the comparison console for the signed-in account."""
    hero(
        st,
        "Explore · side by side",
        "Model Comparison",
        "Run several installed models on one image and compare what each "
        "actually detects.",
        icon="compare_arrows",
    )
    if user is None:
        st.caption("Sign in to compare models on your own images.")
        return

    notice(
        st,
        "info",
        "How to read this",
        "Each model runs on the exact same image with the same confidence. A "
        "model can only emit classes in its own list, so a species missing from "
        "one column is not evidence that it is absent - it may be outside that "
        "model's scope. This is a qualitative check, not a benchmark.",
        icon="info",
    )

    available = state.available_keys(detector)
    if not available:
        error_card(
            st,
            "No model weights installed",
            "Download the core weights from the AI Models page first.",
        )
        return

    registry = state.registry(detector)
    upload = st.file_uploader(
        "Image to analyse",
        type=["jpg", "jpeg", "png", "webp"],
        key="compare_upload",
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        selections = st.multiselect(
            "Models to compare",
            options=available,
            default=available[: min(3, len(available))],
            format_func=lambda key: state.model_label(key, registry.get(key), True),
            key="compare_models",
            help="Pick two or more to see them side by side.",
        )
    with c2:
        confidence = st.slider(
            "Confidence",
            min_value=0.01,
            max_value=0.99,
            value=float(settings.default_confidence),
            step=0.01,
            key="compare_confidence",
        )

    collect = st.button(
        "Run comparison",
        type="primary",
        icon=":material/play_arrow:",
        key="compare_run",
    )
    if collect:
        if upload is None:
            error_card(st, "Upload an image first", "Choose a JPG, PNG or WebP file to analyse.")
        elif len(selections) < 2:
            error_card(st, "Select at least two models", "A comparison needs two or more models.")
        else:
            _run(detector, settings, user.id, upload, list(selections), float(confidence))

    result = st.session_state.get("compare_result")
    if result:
        _render_result(result)
    elif upload is None:
        st.caption(
            "Upload an image to begin. Files are processed on this machine "
            "only and are scoped to your account."
        )


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def _run(detector, settings, user_id, upload, selections, confidence) -> None:
    try:
        stored = state.save_upload(upload, settings, "image", user_id=user_id)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user below
        error_card(st, "That image could not be accepted", str(exc))
        return

    image = cv2.imread(str(stored))
    if image is None:
        error_card(st, "That image could not be read", "The file may be corrupt.")
        return

    with st.spinner("Running models on the same image..."):
        runs = run_comparison(detector, image, selections, confidence=confidence)

    st.session_state["compare_result"] = _build_result(image, runs, upload.name, confidence)


def _encode(image) -> bytes:
    """JPEG bytes for an image array (empty bytes if encoding fails)."""
    ok, buffer = cv2.imencode(".jpg", image)
    return buffer.tobytes() if ok else b""


def _build_result(image, runs, input_name: str, confidence: float) -> dict:
    """Serialise a comparison into a session-safe payload (bytes + plain data)."""
    return {
        "input": input_name,
        "confidence": confidence,
        "created": datetime.now().isoformat(timespec="seconds"),
        "original": _encode(image),
        "annotated": {run.key: _encode(draw_detections(image, run.detections)) for run in runs},
        "coverage": class_coverage(runs),
        "runs": [
            {
                "key": run.key,
                "display_name": run.display_name,
                "error": run.error,
                "total_detections": run.total_detections,
                "detected_classes": run.detected_classes,
                "average_confidence": round(run.average_confidence, 4),
                "max_confidence": round(run.max_confidence, 4),
                "inference_ms": round(run.inference_ms, 2),
                "class_names": list(run.class_names),
            }
            for run in runs
        ],
    }


def comparison_json(result: dict) -> str:
    """JSON text of a comparison, without the (large) inline image bytes."""
    payload = {
        key: value for key, value in result.items() if key not in {"original", "annotated"}
    }
    return json.dumps(payload, indent=2)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _render_result(result: dict) -> None:
    st.markdown("### Comparison")
    st.caption(
        f"{result['input']} · confidence {result['confidence']:.2f} · "
        f"{result['created']}"
    )

    coverage = result.get("coverage", {})
    if coverage and not coverage.get("all_equal", True):
        notice(
            st,
            "warning",
            "The selected models have different class lists",
            "Shared classes: "
            + (", ".join(coverage.get("shared", [])) or "none")
            + ". Each model's unique classes are listed under its column, so "
            "results are not directly interchangeable.",
            icon="warning",
        )

    st.image(result.get("original"), caption="Original image", width="stretch")

    runs = result.get("runs", [])
    if not runs:
        return
    columns = st.columns(min(len(runs), 3))
    for index, run in enumerate(runs):
        with columns[index % len(columns)]:
            _render_run_column(result, run)

    st.write("")
    st.download_button(
        "Download comparison (JSON)",
        data=comparison_json(result),
        file_name="blueeye_model_comparison.json",
        mime="application/json",
        icon=":material/download:",
        key="compare_download",
    )
    st.caption(
        "The JSON records the real per-model detection counts, confidences and "
        "timing. No accuracy metric is implied."
    )


def _render_run_column(result: dict, run: dict) -> None:
    st.markdown(f"#### {run['display_name']}")
    annotated = result.get("annotated", {}).get(run["key"])
    if annotated:
        st.image(annotated, width="stretch")
    if run.get("error"):
        error_card(st, "This model failed", str(run["error"]), log=False)
        return
    metrics_row(
        st,
        [
            ("Objects", run["total_detections"], "above threshold"),
            ("Classes", len(run["detected_classes"]), "distinct"),
            ("Avg conf", f"{run['average_confidence'] * 100:.1f}%", "across detections"),
            ("Inference", f"{run['inference_ms'] / 1000:.2f} s", "this image"),
        ],
    )
    if run["detected_classes"]:
        st.dataframe(
            [
                {"Species": display_class_name(name), "Class": name, "Count": count}
                for name, count in run["detected_classes"].items()
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("No detections above the confidence threshold.")
    unique = result.get("coverage", {}).get("unique", {}).get(run["key"], [])
    if unique:
        st.caption("Classes unique to this model: " + ", ".join(unique))


__all__ = ["render", "comparison_json"]
