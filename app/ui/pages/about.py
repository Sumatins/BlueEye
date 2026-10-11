"""About page: what BlueEye is, why it exists and how it works.

BlueEye is presented as *our* application: an AI-assisted aquatic animal
detection and biodiversity-exploration tool. The page describes only features
that exist in the running system and is not a repository showcase. Licence
and attribution notices remain in the source files, the README and the model
metadata; they are intentionally not repeated here.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import streamlit as st

from app.detection.detector import MarineDetector
from app.ui.components import hero, notice
from app.ui.icons import span
from app.ui.theme import esc


def render(detector: MarineDetector, settings, user=None) -> None:
    """Render the BlueEye project description."""
    hero(
        st,
        "About BlueEye",
        "About BlueEye",
        "An AI-assisted aquatic animal detection and biodiversity-exploration "
        "application for underwater imagery and supported video.",
        icon="info",
    )

    _about()
    _purpose()
    _features()
    _how_it_works()
    _technology()
    _responsible_use()


# --------------------------------------------------------------------------- #
# Layout helper
# --------------------------------------------------------------------------- #
def _tile(icon: str, title: str, body: str) -> str:
    """One labelled icon tile for the content grids."""
    return (
        '<div class="be-tile">'
        f'<div class="ic">{span(icon)}</div>'
        f"<h4>{esc(title)}</h4><p>{esc(body)}</p></div>"
    )


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _about() -> None:
    st.markdown("### About BlueEye")
    st.markdown(
        "BlueEye is an application we developed to make aquatic animal "
        "observation easier to carry out and easier to review. It runs object "
        "detection on supported underwater **images and video**, presents the "
        "predicted labels, confidence scores and bounding boxes, and keeps each "
        "finished observation organised locally so it can be revisited later. "
        "Alongside detection, it offers an educational **Species Explorer** and "
        "**Biodiversity Analytics** built from your own runs. Predictions are "
        "model output, not laboratory identification: BlueEye reports what its "
        "selected models were trained to recognise."
    )


def _purpose() -> None:
    st.markdown("### Our purpose")
    st.markdown(
        "Observing aquatic life by hand is slow, expensive and hard to repeat, "
        "especially underwater where visibility and field time are limited. "
        "BlueEye's goal is to make aquatic animal observation, biodiversity "
        "education and organised detection review more accessible: give a "
        "single clear workflow for running a detection model over imagery, "
        "understanding what it found, and building a reviewable record of those "
        "findings over time."
    )
    st.markdown(
        "We focus on being honest about scope - the interface states which "
        "classes a model supports and where it cannot confirm a species - "
        "rather than overstating what the system knows."
    )


def _features() -> None:
    st.markdown("### Key features")
    st.markdown(
        '<div class="be-grid">'
        + _tile(
            "image",
            "Image detection",
            "Detect aquatic animals in a single underwater image and download the annotated result.",
        )
        + _tile(
            "videocam",
            "Video analysis",
            "Process a supported video frame by frame and re-encode a browser-playable annotated video.",
        )
        + _tile(
            "psychology",
            "Multiple models",
            "Choose from the registered detection models, each with its documented class support.",
        )
        + _tile(
            "menu_book",
            "Species Explorer",
            "Browse educational species profiles with an honest note on detection compatibility.",
        )
        + _tile(
            "insights",
            "History and Analytics",
            "Every run is stored locally and summarised - nothing is uploaded.",
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def _how_it_works() -> None:
    st.markdown("### How it works")
    st.markdown(
        "You upload an image or a supported video. After validation, the file "
        "is passed to the detection model you select (optionally after "
        "underwater enhancement, which is off by default). The model returns "
        "boxes with labels and confidence scores; BlueEye draws them on the "
        "output, writes a JSON report, and records the observation so you can "
        "review it in History and summarise it in Analytics."
    )


def _technology() -> None:
    st.markdown("### Technology")
    st.markdown(
        '<div class="be-grid">'
        + _tile(
            "code",
            "Python",
            "The application, CLI and detection pipeline are written in Python.",
        )
        + _tile(
            "web",
            "Streamlit",
            "The interactive web interface is built with Streamlit.",
        )
        + _tile(
            "center_focus_strong",
            "Ultralytics",
            "Object-detection inference (YOLO / RT-DETR) through the Ultralytics library.",
        )
        + _tile(
            "psychology",
            "PyTorch",
            "The deep-learning backend that runs the models on CPU or GPU.",
        )
        + _tile(
            "movie",
            "OpenCV + imageio-ffmpeg",
            "Image and video decoding, annotation and H.264 output encoding.",
        )
        + _tile(
            "calculate",
            "NumPy + Pillow",
            "Array operations and image handling behind detection and reports.",
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "BlueEye runs on ordinary hardware - a CPU is enough to run inference; "
        "an NVIDIA GPU speeds it up when available. The technology above is the "
        "stack BlueEye is built on, not models or datasets we created."
    )


def _responsible_use() -> None:
    notice(
        st,
        "info",
        "Responsible use",
        "Detections are model predictions that depend on the selected model, "
        "the input quality and the classes it was trained on. They should be "
        "validated before being used as scientific evidence or for "
        "conservation decisions.",
    )


__all__ = ["render"]
