"""Dashboard: BlueEye's welcoming landing page.

The page is intentionally short and task-focused. It uses the real navigation
actions (the same session-state routing the sidebar uses) and never loads a
model or shows invented statistics. The full model registry lives on the
Models page and the observation history lives on the History / Analytics
pages, so the dashboard does not duplicate them.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import streamlit as st

from app import __version__
from app.config import PROJECT_ROOT
from app.detection.detector import MarineDetector
from app.ui import state
from app.ui.components import notice
from app.ui.icons import span
from app.ui.theme import esc

#: Hero illustration shipped with the project (upstream marine-detect assets,
#: AGPL-3.0-only). Local file only - nothing is fetched over the network.
HERO_IMAGE = PROJECT_ROOT / "assets" / "images" / "input_folder" / "regq.jpg"


def render(detector: MarineDetector, settings) -> None:
    """Render the landing page (hero -> explore -> how -> capabilities -> use)."""
    _hero(detector)
    _explore()
    _how_it_works()
    _capabilities()
    _responsible_use()


# --------------------------------------------------------------------------- #
# 1. Hero
# --------------------------------------------------------------------------- #
def _hero(detector: MarineDetector) -> None:
    text_column, art_column = st.columns([1.35, 1], gap="large")

    with text_column:
        st.markdown(
            '<div class="be-hero">'
            f'<div class="be-tag">{span("visibility")}See beneath the surface</div>'
            "<h1>Understand aquatic life</h1>"
            "<p>BlueEye uses AI to detect aquatic animals in uploaded underwater "
            "images and supported videos. It identifies what its selected models "
            "recognise and records every observation for review.</p>"
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
    if ready:
        return (
            f"{len(ready)} model(s) ready on this machine · "
            f"inference device {str(detector.device).upper()} · BlueEye v{__version__}."
        )
    return "No model weights are installed yet - open Models to download them."


# --------------------------------------------------------------------------- #
# 2. Explore BlueEye
# --------------------------------------------------------------------------- #
def _explore() -> None:
    st.markdown("### Explore BlueEye")
    cards = (
        ("image", "Image Detection", "Detect animals in a single underwater image.", "detect_image"),
        ("videocam", "Video Detection", "Analyse a supported video frame by frame.", "detect_video"),
        ("menu_book", "Aquatic Species", "Browse the educational species reference library.", "species"),
        ("psychology", "Models", "See every model, its classes, licence and honest status.", "models"),
        ("insights", "Biodiversity Analytics", "Review statistics from your own detections.", "analytics"),
    )
    columns = st.columns(len(cards))
    for column, (icon, title, body, target) in zip(columns, cards):
        with column:
            st.markdown(
                '<div class="be-card be-card--flush" style="min-height:150px">'
                f'<div class="be-card-title">{span(icon)}{esc(title)}</div>'
                f'<p class="be-card-sub">{esc(body)}</p></div>',
                unsafe_allow_html=True,
            )
            if st.button("Open", key=f"explore_{target}", width="stretch"):
                if target in ("detect_image", "detect_video"):
                    st.session_state["detect_media"] = (
                        "Image" if target == "detect_image" else "Video"
                    )
                state.navigate(target)


# --------------------------------------------------------------------------- #
# 3. How BlueEye works
# --------------------------------------------------------------------------- #
def _how_it_works() -> None:
    st.markdown("### How BlueEye works")
    steps = (
        ("cloud_upload", "1. Upload", "Add an image or a supported video."),
        ("psychology", "2. Choose a model", "Pick a model and configure inference."),
        ("fact_check", "3. Review results", "Inspect boxes, labels and confidence scores."),
        ("history", "4. Save and review", "Find every observation in History and Analytics."),
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
        "Results depend on the selected model, the input quality and the species "
        "each model supports."
    )


# --------------------------------------------------------------------------- #
# 4. Capabilities
# --------------------------------------------------------------------------- #
def _capabilities() -> None:
    st.markdown("### Capabilities")
    st.markdown(
        "BlueEye runs object detection on **underwater images and supported "
        "videos**, using the model you choose to recognise the aquatic animals it "
        "was trained on. Every run produces an annotated image or video plus a JSON "
        "report, and each finished observation is organised locally so you can "
        "review it over time in **History** and summarise it in **Analytics**. "
        "BlueEye **never uploads your media**.",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# 5. Responsible use
# --------------------------------------------------------------------------- #
def _responsible_use() -> None:
    notice(
        st,
        "info",
        "Responsible use",
        "Predictions depend on the selected model and on input quality, and are "
        "not guaranteed to generalise to every water body or camera. Independently "
        "validate detections before using them for scientific, conservation or "
        "regulatory decisions.",
    )
