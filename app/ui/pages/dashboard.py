"""Dashboard: BlueEye's aquatic-biodiversity landing page.

Every number shown here is computed from the persisted detection history or
from the live model registry - nothing is simulated. Rendering the dashboard
never loads a model: model availability is a cheap file-existence check, so
opening the home page stays fast even when weights are not installed.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app import __version__
from app.config import IMAGE_EXTENSIONS, PROJECT_ROOT
from app.detection.detector import MarineDetector
from app.detection.results import display_class_name
from app.ui import state
from app.ui.components import empty_state, metrics_row, notice
from app.ui.icons import span
from app.ui.theme import esc
from app.utils.history import history_path, load_entries, summarize

#: Hero illustration shipped with the project (upstream marine-detect assets,
#: AGPL-3.0-only). Purely decorative - no inference depends on it.
HERO_IMAGE = PROJECT_ROOT / "assets" / "images" / "input_folder" / "regq.jpg"


def render(detector: MarineDetector, settings) -> None:
    """Render the landing page (A-F of the dashboard brief)."""
    _hero(detector)                 # A. hero + two real actions
    _how_it_works()                 # B. four-step explanation
    _observation_overview(settings)  # C. real observation metrics
    _recent_detections(settings)     # D. genuine recent records
    _capabilities()                  # E. feature cards that navigate
    _responsible_use()               # F. honest scope and limits


# --------------------------------------------------------------------------- #
# A. Hero
# --------------------------------------------------------------------------- #
def _hero(detector: MarineDetector) -> None:
    text_column, art_column = st.columns([1.35, 1], gap="large")

    with text_column:
        st.markdown(
            '<div class="be-hero">'
            f'<div class="be-tag">{span("visibility")}See beneath the surface</div>'
            "<h1>Understand aquatic life</h1>"
            "<p>BlueEye is an AI-assisted aquatic animal detection and biodiversity "
            "monitoring application. It analyses uploaded underwater images and "
            "supported videos, identifies the animals its selected models recognise, "
            "and summarises every observation.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        image_action, video_action = st.columns(2)
        with image_action:
            if st.button(
                "Detect an Image",
                type="primary",
                icon=":material/image:",
                width="stretch",
                key="hero_detect_image",
            ):
                st.session_state["detect_media"] = "Image"
                state.navigate("detect_image")
        with video_action:
            if st.button(
                "Analyse a Video",
                icon=":material/videocam:",
                width="stretch",
                key="hero_detect_video",
            ):
                st.session_state["detect_media"] = "Video"
                state.navigate("detect_video")
        st.caption(_status_line(detector))

    with art_column:
        if HERO_IMAGE.is_file():
            st.image(
                str(HERO_IMAGE),
                width="stretch",
                caption="Sample aquatic image - marine-detect project assets.",
            )
        else:
            st.markdown(
                '<div class="be-card be-card--muted"><div class="be-empty">'
                f'<div class="tile">{span("waves")}</div>'
                '<p class="be-card-sub" style="margin:0">Aquatic imagery</p>'
                "</div></div>",
                unsafe_allow_html=True,
            )


def _status_line(detector: MarineDetector) -> str:
    """Compact, honest readiness line (no per-model list, no fake numbers)."""
    ready = state.available_keys(detector)
    total = len(state.registry(detector))
    if ready:
        return (
            f"Ready on this machine: {len(ready)} of {total} registered models · "
            f"inference device {str(detector.device).upper()}. "
            f"BlueEye v{__version__}."
        )
    return "No model weights are installed yet - open Models to download them."


# --------------------------------------------------------------------------- #
# B. How BlueEye works
# --------------------------------------------------------------------------- #
def _how_it_works() -> None:
    st.markdown("### How BlueEye works")
    steps = (
        ("cloud_upload", "1. Upload", "Add an aquatic image or a supported video."),
        ("psychology", "2. Choose a model", "Pick a compatible model, or Auto for the core set."),
        ("bolt", "3. Run inference", "The detection pipeline finds the animals it recognises."),
        ("fact_check", "4. Review results", "Inspect the annotated output and saved observations."),
    )
    columns = st.columns(len(steps))
    for column, (icon, title, body) in zip(columns, steps):
        with column:
            st.markdown(
                '<div class="be-card be-card--flush">'
                f'<div class="be-card-title">{span(icon)}{esc(title)}</div>'
                f'<p class="be-card-sub">{esc(body)}</p></div>',
                unsafe_allow_html=True,
            )
    st.caption(
        "Results depend on the selected model, the image or video quality and the "
        "species each model supports - not every animal in a scene is detectable."
    )


# --------------------------------------------------------------------------- #
# C. Observation overview
# --------------------------------------------------------------------------- #
def _observation_overview(settings) -> None:
    stats = summarize(load_entries(history_path(settings)))

    st.markdown("### Observation overview")
    if not stats["runs"]:
        empty_state(
            st,
            "insights",
            "No observations yet",
            "Once you finish a detection, this application's own observations "
            "appear here. Nothing is estimated or invented.",
        )
        return

    metrics_row(
        st,
        [
            ("Detection runs", stats["runs"], f"{stats['images']} image · {stats['videos']} video"),
            ("Observed detections", stats["total_objects"], "objects across every run"),
            ("Distinct classes", stats["unique_species"], "classes observed so far"),
            ("Recent activity", stats["last_run"] or "—", "latest finished run"),
        ],
    )
    st.caption(
        "These are counts of this application's own observations, not estimates "
        "of animal populations."
    )


# --------------------------------------------------------------------------- #
# D. Recent detections
# --------------------------------------------------------------------------- #
def _recent_detections(settings) -> None:
    entries = load_entries(history_path(settings))

    st.markdown("### Recent detections")
    if not entries:
        empty_state(
            st,
            "photo_library",
            "Nothing detected yet",
            "Your finished detections will be listed here with their real inputs, "
            "classes and outputs.",
        )
        if st.button(
            "Start an analysis",
            type="primary",
            icon=":material/image:",
            key="recent_start",
        ):
            st.session_state["detect_media"] = "Image"
            state.navigate("detect_image")
        return

    for index, entry in enumerate(entries[:4]):
        _recent_row(entry, index)

    st.write("")
    if st.button(
        "Open full history",
        icon=":material/history:",
        width="stretch",
        key="recent_open_history",
    ):
        state.navigate("history")


def _recent_row(entry, index: int) -> None:
    thumbnail_column, info_column = st.columns([1, 2.8], gap="medium")

    with thumbnail_column:
        output = Path(entry.output) if entry.output else None
        if (
            entry.kind == "image"
            and output is not None
            and output.is_file()
            and output.suffix.lower() in IMAGE_EXTENSIONS
        ):
            st.image(str(output), width="stretch")
        else:
            st.markdown(
                '<div class="be-card be-card--muted"><div class="be-empty">'
                f'<div class="tile">{span("videocam" if entry.kind == "video" else "image")}</div>'
                '<p class="be-card-sub" style="margin:0">No thumbnail</p>'
                "</div></div>",
                unsafe_allow_html=True,
            )

    with info_column:
        st.markdown(f"**{esc(Path(entry.filename).name)}**")
        classes = ", ".join(
            display_class_name(name) for name in entry.detections_per_class
        ) or "No detections recorded"
        st.caption(
            f"{entry.display_time} · {entry.kind} · {entry.total_detections} object(s) · "
            f"{entry.average_confidence * 100:.1f}% avg confidence"
        )
        st.markdown(f"<p class='be-card-sub'>{esc(classes)}</p>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# E. Capabilities
# --------------------------------------------------------------------------- #
def _capabilities() -> None:
    st.markdown("### Explore BlueEye")
    capabilities = (
        ("image", "Image Detection", "Detect animals in a single underwater image.", "detect_image"),
        ("videocam", "Video Detection", "Analyse supported videos frame by frame.", "detect_video"),
        ("psychology", "Aquatic Species Models", "Browse models, species and honest status.", "models"),
        ("insights", "Biodiversity Analytics", "Review statistics from your own detections.", "analytics"),
    )
    columns = st.columns(len(capabilities))
    for column, (icon, title, body, target) in zip(columns, capabilities):
        with column:
            st.markdown(
                '<div class="be-card be-card--flush">'
                f'<div class="be-card-title">{span(icon)}{esc(title)}</div>'
                f'<p class="be-card-sub">{esc(body)}</p></div>',
                unsafe_allow_html=True,
            )
            if st.button("Open", key=f"cap_{target}", width="stretch"):
                state.navigate(target)


# --------------------------------------------------------------------------- #
# F. Responsible use
# --------------------------------------------------------------------------- #
def _responsible_use() -> None:
    st.markdown("### Capabilities and limits")
    can_column, cannot_column = st.columns(2)
    with can_column:
        st.markdown(
            '<div class="be-card"><div class="be-card-title">'
            f'{span("check_circle")}What BlueEye helps with</div>'
            "<ul class='be-card-sub' style='margin:.2rem 0 0 1.1rem'>"
            "<li>Inspecting uploaded aquatic imagery and supported video.</li>"
            "<li>Identifying animals that the selected models recognise.</li>"
            "<li>Summarising detections and keeping an observation history.</li>"
            "<li>Organising image and video analysis for research and education.</li>"
            "<li>Assisting conservation monitoring once predictions are validated.</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
    with cannot_column:
        st.markdown(
            '<div class="be-card"><div class="be-card-title">'
            f'{span("block")}What BlueEye does not do</div>'
            "<ul class='be-card-sub' style='margin:.2rem 0 0 1.1rem'>"
            "<li>It does not measure water quality or pollution.</li>"
            "<li>It does not estimate population size or species abundance.</li>"
            "<li>It does not detect species outside each model's trained classes.</li>"
            "<li>It is not a substitute for expert field surveys.</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    notice(
        st,
        "info",
        "Responsible use",
        "Detections are AI-generated observations. Species-level reliability depends "
        "on each model's training data and on image quality. Independently validate "
        "predictions before using them for conservation, regulatory or research "
        "decisions.",
    )
