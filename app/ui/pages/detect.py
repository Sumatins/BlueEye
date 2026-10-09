"""Detect page: upload -> model -> confidence -> run -> results.

This page is a pure presentation layer over the existing pipeline. The
inference calls are byte-for-byte the ones the original UI used:

* ``ImageProcessor(...).process(input_path=..., model_selection=...,
  confidence=..., enhancement=..., save_json=True)``
* ``VideoProcessor(...).process(..., progress_callback=..., save_json=True)``

Nothing here changes model behaviour, thresholds or output files.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
import streamlit as st

from app.config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.detection.detector import MarineDetector
from app.detection.model_manager import ModelError
from app.processing.enhancement import EnhancementConfig
from app.processing.image_processor import ImageProcessor, ProcessingError
from app.processing.video_processor import VideoProcessor, probe_video
from app.ui import state
from app.ui.components import (
    card,
    chips,
    detection_table,
    download_button,
    empty_state,
    error_card,
    hero,
    kv,
    metrics_row,
    model_status_badge,
    notice,
    species_chart,
    status_badge,
    step,
)
from app.ui.icons import span
from app.ui.theme import esc
from app.utils.file_utils import human_size, load_json, read_file_bytes
from app.utils.validation import ValidationError

logger = logging.getLogger(__name__)

#: Media modes offered by the page.
MEDIA_MODES = ("Image", "Video")


def render(detector: MarineDetector, settings) -> None:
    """Render the detection workflow."""
    hero(
        st,
        "Detection",
        "Aquatic Detection",
        "Identify aquatic animals in underwater images and supported videos.",
        icon="center_focus_strong",
    )

    media = st.radio(
        "Media type",
        options=list(MEDIA_MODES),
        horizontal=True,
        key="detect_media",
        help="Choose what you want to analyze.",
    )

    if not state.any_model_available(detector):
        _render_no_weights(detector)
        return

    if media == "Image":
        _image_workflow(detector, settings)
    else:
        _video_workflow(detector, settings)


# --------------------------------------------------------------------------- #
# Model / weights guards
# --------------------------------------------------------------------------- #
def _render_no_weights(detector: MarineDetector) -> None:
    """Friendly 'model unavailable' state instead of a crash (see README)."""
    error_card(
        st,
        "Model unavailable",
        "No model weights are installed, so detection cannot run.",
        hint=(
            "BlueEye never ships fabricated weights. Install the official "
            "marine-detect models below, or run: "
            "python scripts/download_models.py"
        ),
    )
    if st.button(
        "Download model weights",
        type="primary",
        icon=":material/download:",
        width="stretch",
    ):
        _download_all(detector)


def _download_all(detector: MarineDetector) -> None:
    """Download every missing built-in model, then refresh the UI."""
    try:
        with st.spinner("Downloading model weights (~87 MB per model)..."):
            detector.model_manager.download_all()
        st.cache_resource.clear()
        st.rerun()
    except ModelError as exc:
        error_card(st, "Download failed", str(exc))
        logger.warning("Model download failed: %s", exc)


def _selection_guard(detector: MarineDetector) -> str | None:
    """Return a user-facing error message when the selection cannot run."""
    selection = st.session_state.get("model_selection", "auto")
    if selection in ("auto", "all"):
        if not state.any_model_available(detector):
            return "No model weights are installed."
        return None
    if not detector.model_manager.is_available(selection):
        spec = state.registry(detector).get(selection)
        name = spec.display_name if spec else selection
        return f"The weights for '{name}' are not installed."
    return None


# --------------------------------------------------------------------------- #
# Shared controls (steps 2-4)
# --------------------------------------------------------------------------- #
def _model_step(detector: MarineDetector) -> None:
    """Step 2 - friendly model selection driven by the registry."""
    registry = state.registry(detector)
    options = ["auto", "all", *registry.keys()]
    # The key is seeded in init_session and may be set by the Models page's
    # "Use this model" action, so validate it instead of forcing an index
    # (passing index alongside a session-state value triggers a warning).
    if st.session_state.get("model_selection", "auto") not in options:
        st.session_state["model_selection"] = "auto"

    step(st, 2, "Choose a model")
    st.selectbox(
        "Model",
        options=options,
        key="model_selection",
        format_func=lambda key: state.model_label(
            key,
            registry.get(key),
            True
            if key in ("auto", "all")
            else detector.model_manager.is_available(key),
        ),
        help=(
            "'Auto' runs the two core models. 'Every installed model' also "
            "runs the additional aquatic models; overlapping detections of "
            "the same class are merged, so results contain no duplicates."
        ),
        width="stretch",
    )
    _model_info_card(detector, registry)


def _model_info_card(detector: MarineDetector, registry: dict) -> None:
    """Readable explanation of the current selection (registry data only)."""
    key = st.session_state.get("model_selection", "auto")
    description = state.model_description(key, registry.get(key))

    if key in ("auto", "all"):
        if key == "all":
            ready_keys = state.available_keys(detector)
        else:
            auto_models = getattr(detector.model_manager, "auto_models", None)
            ready_keys = (
                [spec.key for spec in auto_models()]
                if callable(auto_models)
                else state.available_keys(detector)
            )
        names = [registry[k].display_name for k in ready_keys if k in registry]
        status = status_badge(bool(names))
        meta = chips(names) if names else ""
        meta += f'<span class="be-chip">{len(names)} ready</span>'
        title = (
            "Auto - core models"
            if key == "auto"
            else "Every installed model (core + additional)"
        )
        icon = "done_all"
    else:
        spec = registry.get(key)
        if spec is None:  # defensive: selection was validated above
            st.caption(description)
            return
        available = detector.model_manager.is_available(key)
        status = model_status_badge(spec.readiness_status(available))
        parts = [f"{len(spec.classes)} classes"]
        if spec.recommended_confidence:
            parts.append(f"threshold {spec.recommended_confidence:.3f}")
        parts.append(spec.key)
        meta = chips(parts)
        title, icon = spec.display_name, "psychology"

    st.markdown(
        f'<div class="be-card be-card--flush"><div class="be-iblock">'
        f'<div class="tile">{span(icon)}</div><div class="body">'
        f'<p class="t">{esc(title)}{status}</p>'
        f'<p class="s">{esc(description)}</p>'
        f'<div class="row">{meta}</div>'
        f"</div></div></div>",
        unsafe_allow_html=True,
    )


def _confidence_step() -> None:
    """Step 3 - confidence threshold (defaults preserved: 0.50)."""
    settings = state.get_settings_cached()
    step(st, 3, "Set the confidence threshold")

    use_default = st.toggle(
        "Use each model's recommended threshold",
        key="use_model_default",
        help=(
            "Recommended values published upstream: 0.523 (Fish & "
            "Invertebrates) and 0.546 (MegaFauna)."
        ),
    )
    slider_value = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(st.session_state.get("confidence") or settings.default_confidence),
        step=0.01,
        disabled=use_default,
        key="confidence_slider",
        help="Detections below this confidence are ignored.",
        width="stretch",
    )
    st.session_state["confidence"] = None if use_default else float(slider_value)
    if use_default:
        st.caption("Using the recommended per-model threshold for every model.")


def _enhancement_step(settings) -> None:
    """Optional preprocessing - off by default, never claimed to help."""
    enhance_on = st.toggle(
        "Underwater enhancement (optional preprocessing)",
        value=settings.enhancement_enabled,
        key="enh_toggle",
        help=(
            "Colour correction → contrast → denoising. Disabled by default: "
            "it is not guaranteed to improve detection accuracy - compare "
            "both on your data."
        ),
    )
    if enhance_on:
        with st.expander("Enhancement options", expanded=True):
            color_on = st.checkbox("Colour correction", value=True, key="enh_color")
            contrast_on = st.checkbox("Contrast enhancement (CLAHE)", value=True, key="enh_contrast")
            denoise_on = st.checkbox("Denoising", value=False, key="enh_denoise")
    else:
        color_on, contrast_on, denoise_on = True, True, False

    st.session_state["enhancement_config"] = EnhancementConfig(
        enabled=enhance_on,
        color_correction=color_on if enhance_on else False,
        contrast=contrast_on if enhance_on else False,
        denoise=denoise_on if enhance_on else False,
    )


def _run_settings() -> dict:
    """Arguments shared by the image and video pipeline calls."""
    return {
        "model_selection": st.session_state.get("model_selection", "auto"),
        "confidence": st.session_state.get("confidence"),
        "enhancement": st.session_state.get("enhancement_config"),
    }


# --------------------------------------------------------------------------- #
# Image workflow
# --------------------------------------------------------------------------- #
def _image_workflow(detector: MarineDetector, settings) -> None:
    step(st, 1, "Upload an underwater image")
    uploaded = st.file_uploader(
        "Upload an underwater image",
        type=[ext.lstrip(".") for ext in sorted(IMAGE_EXTENSIONS)],
        key="image_uploader",
        help=(
            f"Formats: {', '.join(sorted(IMAGE_EXTENSIONS))} · "
            f"max {settings.max_image_mb} MB"
        ),
        label_visibility="collapsed",
        width="stretch",
    )
    if uploaded is None:
        empty_state(
            st,
            "image",
            "No media selected",
            "Upload an underwater image to begin aquatic-life detection. "
            "Example images are available in assets/images/input_folder/.",
        )
        _model_step(detector)
        _confidence_step()
        _enhancement_step(settings)
        return

    try:
        input_path = state.save_upload(uploaded, settings, "image")
    except ValidationError as exc:
        error_card(
            st,
            "Unable to process this image",
            str(exc),
            hint="Check the file format and size, then try another image.",
        )
        return

    preview_left, preview_right = st.columns(2, gap="medium")
    with preview_left:
        st.markdown("**Uploaded image**")
        st.image(read_file_bytes(input_path), width="stretch")
    with preview_right:
        card(st, title="File details", body_html=_image_meta(input_path))

    _model_step(detector)
    _confidence_step()
    _enhancement_step(settings)

    guard = _selection_guard(detector)
    if guard:
        error_card(st, "Model unavailable", guard, hint="See the Models page for setup instructions.")
        return

    st.write("")
    if st.button(
        "Detect Aquatic Life",
        type="primary",
        icon=":material/rocket_launch:",
        width="stretch",
        key="run_image",
    ):
        _run_image(detector, settings, input_path)

    _render_image_results()


def _image_meta(input_path: Path) -> str:
    """Key/value HTML with the upload's real on-disk properties."""
    size_bytes = input_path.stat().st_size
    try:
        with Image.open(input_path) as image:
            width, height = image.size
            fmt = (image.format or "").upper()
    except OSError:
        width = height = 0
        fmt = ""
    return kv(
        (
            ("File", input_path.name),
            ("Format", fmt or input_path.suffix.lstrip(".").upper()),
            ("Size", human_size(size_bytes)),
            ("Dimensions", f"{width} × {height} px" if width and height else "—"),
        )
    )


def _run_image(detector: MarineDetector, settings, input_path: Path) -> None:
    """Run the image pipeline exactly as the original UI did."""
    try:
        with st.spinner("Analyzing image with YOLOv8..."):
            processor = ImageProcessor(detector, settings=settings)
            result = processor.process(
                input_path=input_path,
                **_run_settings(),
                save_json=True,
            )
    except (ValidationError, ModelError, ProcessingError, ValueError) as exc:
        error_card(
            st,
            "Unable to process this image",
            str(exc),
            hint="The uploaded file may be invalid or unsupported.",
            log=False,
        )
        logger.warning("Image detection rejected: %s", exc)
        return
    except Exception as exc:  # noqa: BLE001 - surface, but log the traceback
        logger.exception("Image detection failed")
        error_card(
            st,
            "Unable to process this image",
            f"Unexpected error during detection: {exc}",
            hint="The technical details were written to the application log.",
            log=False,
        )
        return

    st.session_state["image_result"] = str(result.output_path)
    st.session_state["image_report"] = str(result.report_path) if result.report_path else None
    st.session_state["image_input"] = str(result.input_path)

    report = load_json(result.report_path) if result.report_path else None
    state.record_history("image", report, result.output_path, result.report_path)


def _render_image_results() -> None:
    output_path = st.session_state.get("image_result")
    if not output_path or not Path(output_path).exists():
        return

    report_path = st.session_state.get("image_report")
    report = load_json(report_path) if report_path and Path(report_path).exists() else None
    if report is None:
        return

    detections = report.get("detections", [])
    stats = report.get("statistics", {})

    st.markdown("### Detection result")
    notice(
        st,
        "info",
        "Detection completed.",
        f"{len(detections)} object(s) found.",
        icon="check_circle",
    )

    col_original, col_result = st.columns(2)
    with col_original:
        st.markdown("**Original image**")
        st.image(read_file_bytes(st.session_state["image_input"]), width="stretch")
    with col_result:
        st.markdown("**Detection result**")
        st.image(read_file_bytes(output_path), width="stretch")

    st.markdown("### Detection summary")
    _summary_metrics(stats)

    if not detections:
        notice(
            st,
            "warning",
            "No organisms detected",
            "Nothing was found above the confidence threshold. Try lowering it "
            "or enabling a different model.",
        )
    else:
        st.markdown("### Detected aquatic life")
        detection_table(st, detections)
        with st.expander("Species breakdown", expanded=True):
            species_chart(st, stats.get("detections_per_class", {}) or {})

    threshold = report.get("confidence_threshold")
    st.caption(
        f"Model(s): {', '.join(report.get('models', []))} · "
        f"inference {report.get('inference_time_ms', 0):.0f} ms · "
        f"enhanced: {'yes' if report.get('enhanced') else 'no'} · "
        f"threshold: {'model default' if threshold is None else threshold}"
    )

    download_left, download_right = st.columns(2)
    with download_left:
        download_button(st, "Download annotated image", output_path, "image/jpeg")
    with download_right:
        if report_path:
            download_button(st, "Download JSON report", report_path, "application/json")


def _summary_metrics(stats: dict) -> None:
    total = stats.get("total_detections", 0)
    unique = stats.get("unique_classes", 0)
    average = float(stats.get("average_confidence", 0.0)) * 100
    maximum = float(stats.get("max_confidence", 0.0)) * 100
    metrics_row(
        st,
        [
            ("Total objects", total, "above threshold"),
            ("Species detected", unique, "distinct classes"),
            ("Average confidence", f"{average:.1f}%", "across detections"),
            ("Max confidence", f"{maximum:.1f}%", "best match"),
        ],
    )


# --------------------------------------------------------------------------- #
# Video workflow
# --------------------------------------------------------------------------- #
def _video_workflow(detector: MarineDetector, settings) -> None:
    step(st, 1, "Upload an underwater video")
    uploaded = st.file_uploader(
        "Upload an underwater video",
        type=[ext.lstrip(".") for ext in sorted(VIDEO_EXTENSIONS)],
        key="video_uploader",
        help=(
            f"Formats: {', '.join(sorted(VIDEO_EXTENSIONS))} · "
            f"max {settings.max_video_mb} MB"
        ),
        label_visibility="collapsed",
        width="stretch",
    )
    if uploaded is None:
        empty_state(
            st,
            "videocam",
            "No media selected",
            "Upload an underwater video to begin frame-by-frame aquatic-life "
            "detection. Supported formats are shown in the upload area.",
        )
        _model_step(detector)
        _confidence_step()
        _enhancement_step(settings)
        return

    try:
        input_path = state.save_upload(uploaded, settings, "video")
        info = probe_video(input_path)
    except ValidationError as exc:
        error_card(
            st,
            "Unable to process this video",
            str(exc),
            hint="Check the file format and size, then try another video.",
        )
        return

    with st.expander("Video information", expanded=True):
        metrics_row(
            st,
            [
                ("Resolution", f"{info.width}×{info.height}", ""),
                ("FPS", f"{info.fps:.2f}", ""),
                ("Frames", info.frame_count or "unknown", ""),
                ("Duration", f"{info.duration_seconds:.1f} s", human_size(info.file_size_bytes)),
            ],
            accent_first=False,
        )
    st.video(read_file_bytes(input_path))

    _model_step(detector)
    _confidence_step()
    _enhancement_step(settings)

    guard = _selection_guard(detector)
    if guard:
        error_card(st, "Model unavailable", guard, hint="See the Models page for setup instructions.")
        return

    st.write("")
    if st.button(
        "Detect Aquatic Life",
        type="primary",
        icon=":material/rocket_launch:",
        width="stretch",
        key="run_video",
    ):
        _run_video(detector, settings, input_path, info.frame_count or 0)

    _render_video_results()


def _run_video(detector: MarineDetector, settings, input_path: Path, total_hint: int) -> None:
    """Run the video pipeline with a real frame-by-frame progress bar."""
    progress = st.progress(0.0, text="Starting video processing...")

    def _on_progress(done: int, total: int | None) -> None:
        if total:
            progress.progress(
                min(done / total, 1.0),
                text=f"Processing video — frame {done:,} / {total:,}",
            )
        else:
            progress.progress(0.0, text=f"Processing video — frame {done:,}")

    try:
        with st.spinner("Processing video frame by frame..."):
            processor = VideoProcessor(detector, settings=settings)
            result = processor.process(
                input_path=input_path,
                progress_callback=_on_progress,
                save_json=True,
                **_run_settings(),
            )
        progress.progress(1.0, text="Detection completed.")
    except (ValidationError, ModelError, ProcessingError, ValueError) as exc:
        progress.empty()
        error_card(
            st,
            "Unable to process this video",
            str(exc),
            hint="The uploaded file may be invalid or unsupported.",
            log=False,
        )
        logger.warning("Video detection rejected: %s", exc)
        return
    except Exception as exc:  # noqa: BLE001 - surface, but log the traceback
        progress.empty()
        logger.exception("Video detection failed")
        error_card(
            st,
            "Unable to process this video",
            f"Unexpected error during video processing: {exc}",
            hint="The technical details were written to the application log.",
            log=False,
        )
        return

    st.session_state["video_output"] = str(result.output_path)
    st.session_state["video_report"] = str(result.report_path) if result.report_path else None
    st.session_state["video_stats"] = result.summary_stats()
    st.session_state["video_total_hint"] = total_hint

    report = load_json(result.report_path) if result.report_path else None
    state.record_history("video", report, result.output_path, result.report_path)


def _render_video_results() -> None:
    output_path = st.session_state.get("video_output")
    if not output_path or not Path(output_path).exists():
        return

    report_path = st.session_state.get("video_report")
    report = load_json(report_path) if report_path and Path(report_path).exists() else {}
    stats = report.get("statistics", {})
    processing = report.get("processing", {})

    st.markdown("### Detection result")
    notice(
        st,
        "info",
        "Detection completed.",
        f"{processing.get('frames_processed', 0)} frame(s) analyzed.",
        icon="check_circle",
    )

    st.markdown("### Detection summary")
    _summary_metrics(stats)

    metrics_row(
        st,
        [
            ("Frames processed", processing.get("frames_processed", "?"), "of the video"),
            (
                "Frames with detections",
                processing.get("frames_with_detections", 0),
                "organic frames",
            ),
            ("Processing speed", f"{processing.get('processing_fps', 0):.1f} fps", "end to end"),
            ("Elapsed", f"{processing.get('elapsed_seconds', 0):.1f} s", "wall clock"),
        ],
        accent_first=False,
    )

    per_class = stats.get("detections_per_class", {}) or {}
    if per_class:
        with st.expander("Species breakdown", expanded=True):
            species_chart(st, per_class)

    st.markdown("### Processed video")
    st.video(read_file_bytes(output_path))

    threshold = report.get("confidence_threshold")
    st.caption(
        f"Model(s): {', '.join(report.get('models', []))} · "
        f"codec: {report.get('output_codec', 'unknown')} · "
        f"enhanced: {'yes' if report.get('enhanced') else 'no'} · "
        f"threshold: {'model default' if threshold is None else threshold}"
    )

    download_left, download_right = st.columns(2)
    with download_left:
        download_button(st, "Download annotated video", output_path, "video/mp4")
    with download_right:
        if report_path:
            download_button(st, "Download JSON report", report_path, "application/json")
