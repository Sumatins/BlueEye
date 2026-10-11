"""Dashboard: BlueEye's product landing page.

A single, information-rich overview assembled only from real sources:

* detection-history statistics (:mod:`app.utils.history`) for the summary cards
  and the recent-activity feed;
* the live model registry and resolved inference device for the model-status
  panel.

No number, status or model claim is fabricated: when there is no history the
cards read zero and the activity section shows a real empty state, and model
status always comes from the registry on disk. Nothing here loads a model, so
the page stays fast.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app import __version__
from app.content.species import get_species
from app.detection.detector import MarineDetector
from app.detection.model_manager import ModelSpec
from app.detection.results import display_class_name
from app.ui import state
from app.ui.components import badge, chips, empty_state, model_status_badge
from app.ui.icons import span
from app.ui.theme import esc
from app.utils.history import history_path, load_entries, summarize

#: Primary action panels: (icon, title, description, media, button label, target).
ACTIONS: tuple[tuple[str, str, str, str, str, str], ...] = (
    (
        "image",
        "Detect an Image",
        "Upload a single underwater photo and detect aquatic animals with the "
        "model you choose. Download the annotated image and JSON report.",
        "Image",
        "Detect an Image",
        "detect_image",
    ),
    (
        "videocam",
        "Analyse a Video",
        "Process a supported video frame by frame with real progress, then "
        "review the annotated clip and the aggregated report.",
        "Video",
        "Analyse a Video",
        "detect_video",
    ),
)

#: Four-step workflow: (icon, title, description).
WORKFLOW_STEPS: tuple[tuple[str, str, str], ...] = (
    ("cloud_upload", "Upload", "Add an underwater image or a supported video."),
    ("psychology", "Select a model", "Choose one of the available detection models."),
    ("fact_check", "Review predictions", "Inspect labels, confidence scores and bounding boxes."),
    ("insights", "Explore observations", "Organise and review results in History and Analytics."),
)

#: Species shown in the biodiversity preview (diverse groups, Indian first).
PREVIEW_SPECIES: tuple[str, ...] = (
    "ganges_river_dolphin",
    "gharial",
    "indian_softshell_turtle",
    "golden_mahseer",
    "whale_shark",
    "humphead_wrasse",
)

#: How many recent runs the activity feed shows.
RECENT_LIMIT = 5


def render(detector: MarineDetector, settings) -> None:
    """Render the dashboard in the requested section order."""
    entries = load_entries(history_path(settings))
    stats = summarize(entries)

    _header(detector)
    _summary_cards(detector, stats)
    _primary_actions()

    activity_column, models_column = st.columns([1.55, 1], gap="large")
    with activity_column:
        _recent_activity(entries)
    with models_column:
        _model_status(detector)

    _how_it_works()
    _species_preview()
    _responsible_observation()


# --------------------------------------------------------------------------- #
# A. Header
# --------------------------------------------------------------------------- #
def _header(detector: MarineDetector) -> None:
    """Welcoming header with a real detection-readiness indicator."""
    ready = state.available_keys(detector)
    if ready:
        status = badge(
            f"Detection ready · {len(ready)} model(s) available", "ok", dot=True
        )
    else:
        status = badge("Detection unavailable · no weights installed", "warn", dot=True)

    st.markdown(
        '<div class="be-hero">'
        f'<div class="be-tag">{span("dashboard")}AI-powered aquatic detection</div>'
        "<h1>BlueEye Dashboard</h1>"
        "<p>Detect and explore aquatic animals in underwater images and "
        "supported videos using AI. BlueEye identifies what the selected models "
        "recognise, records every run locally, and helps you study the species "
        "behind each observation.</p>"
        f'<div class="be-statusrow">{status}'
        f'<span class="dev">Inference device {esc(str(detector.device).upper())} '
        f"· BlueEye v{esc(__version__)}</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# B. Summary cards (real history + registry data)
# --------------------------------------------------------------------------- #
def _summary_cards(detector: MarineDetector, stats: dict) -> None:
    registry = state.registry(detector)
    available = state.available_keys(detector)

    cards = (
        _stat("Detection runs", f"{stats['runs']:,}", "stored on this machine"),
        _stat("Images analysed", f"{stats['images']:,}", "image runs"),
        _stat("Videos analysed", f"{stats['videos']:,}", "video runs"),
        _stat(
            "Unique species",
            f"{stats['unique_species']:,}",
            f"{stats['total_objects']:,} object(s) detected",
        ),
        _stat(
            "AI models available",
            f"{len(available)}",
            f"of {len(registry)} registered",
        ),
    )
    st.markdown(f'<div class="be-statgrid">{"".join(cards)}</div>', unsafe_allow_html=True)
    if not stats["runs"]:
        st.caption(
            "No finished detections yet - the numbers above come from your own "
            "runs and will fill in as you use BlueEye."
        )


def _stat(label: str, value: str, hint: str) -> str:
    """One summary card (reuses the shared ``.be-metric`` styling)."""
    return (
        '<div class="be-metric">'
        f'<div class="k">{esc(label)}</div>'
        f'<div class="v">{esc(value)}</div>'
        f'<div class="h">{esc(hint)}</div></div>'
    )


# --------------------------------------------------------------------------- #
# C. Primary actions
# --------------------------------------------------------------------------- #
def _primary_actions() -> None:
    st.markdown("### Start a detection")
    columns = st.columns(2, gap="medium")
    for column, (icon, title, body, media, button_label, target) in zip(columns, ACTIONS):
        with column:
            st.markdown(
                '<div class="be-action">'
                f'<div class="ic">{span(icon)}</div>'
                f"<h4>{esc(title)}</h4><p>{esc(body)}</p>"
                '<div class="spacer"></div></div>',
                unsafe_allow_html=True,
            )
            if st.button(
                button_label,
                type="primary" if target == "detect_image" else "secondary",
                icon=f":material/{icon}:",
                width="stretch",
                key=f"action_{target}",
            ):
                st.session_state["detect_media"] = media
                state.navigate(target)


# --------------------------------------------------------------------------- #
# D. Recent activity
# --------------------------------------------------------------------------- #
def _recent_activity(entries: list) -> None:
    st.markdown("### Recent activity")
    if not entries:
        empty_state(
            st,
            "history",
            "No detections yet",
            "Run your first detection and it will appear here with its input, "
            "type, species and result.",
        )
        first, second = st.columns(2)
        with first:
            if st.button(
                "Detect an Image",
                type="primary",
                icon=":material/image:",
                width="stretch",
                key="empty_image",
            ):
                st.session_state["detect_media"] = "Image"
                state.navigate("detect_image")
        with second:
            if st.button(
                "Analyse a Video",
                icon=":material/videocam:",
                width="stretch",
                key="empty_video",
            ):
                st.session_state["detect_media"] = "Video"
                state.navigate("detect_video")
        return

    st.markdown(
        f'<div class="be-feed">{"".join(_feed_item(e) for e in entries[:RECENT_LIMIT])}</div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "Open Detection History",
        icon=":material/history:",
        width="stretch",
        key="open_history",
    ):
        state.navigate("history")


def _feed_item(entry) -> str:
    """One recent-run row: input, type, species and completed status."""
    icon = "videocam" if entry.kind == "video" else "image"
    species = list(entry.detections_per_class)
    species_html = (
        f'<div class="species">{chips(display_class_name(name) for name in species[:5])}</div>'
        if species
        else '<div class="meta">No species above threshold</div>'
    )
    meta = (
        f"{entry.kind.capitalize()} · {esc(entry.display_time)} · "
        f"{entry.total_detections} object(s) · {entry.unique_classes} species"
    )
    return (
        '<div class="item">'
        f'<div class="tile">{span(icon)}</div>'
        '<div class="body">'
        f'<div class="fn">{esc(Path(entry.filename).name)}</div>'
        f'<div class="meta">{meta}</div>{species_html}'
        "</div>"
        f'<div class="st">{badge("Completed", "ok", dot=True)}</div>'
        "</div>"
    )


# --------------------------------------------------------------------------- #
# E. Model status
# --------------------------------------------------------------------------- #
def _model_status(detector: MarineDetector) -> None:
    st.markdown("### Model status")
    registry = state.registry(detector)
    available = set(state.available_keys(detector))

    ordered = sorted(
        registry.values(),
        key=lambda spec: (spec.key not in available, spec.display_name.lower()),
    )
    rows = "".join(_model_row(spec, spec.key in available) for spec in ordered)
    st.markdown(rows, unsafe_allow_html=True)
    st.caption(
        f"Inference device {str(detector.device).upper()}. A model's verified "
        "load-and-inference status is confirmed on the AI Models page."
    )
    if st.button(
        "Open AI Models",
        icon=":material/psychology:",
        width="stretch",
        key="open_models",
    ):
        state.navigate("models")


def _model_row(spec: ModelSpec, available: bool) -> str:
    classes = len(spec.classes)
    detail = spec.habitat or "habitat not recorded"
    if classes:
        detail = f"{classes} classes · {detail}"
    sub = f'<span class="sub">{esc(detail)}</span>'
    return (
        '<div class="be-modelrow">'
        f'<span class="nm">{esc(spec.display_name)}{sub}</span>'
        f'<span class="st">{model_status_badge(spec.readiness_status(available))}</span>'
        "</div>"
    )


# --------------------------------------------------------------------------- #
# Workflow
# --------------------------------------------------------------------------- #
def _how_it_works() -> None:
    st.markdown("### How BlueEye works")
    tiles = "".join(
        '<div class="be-tile">'
        f'<div class="ic">{span(icon)}</div>'
        f"<h4>{index} · {esc(title)}</h4><p>{esc(body)}</p></div>"
        for index, (icon, title, body) in enumerate(WORKFLOW_STEPS, start=1)
    )
    st.markdown(f'<div class="be-grid">{tiles}</div>', unsafe_allow_html=True)
    st.caption(
        "Results depend on the selected model, the input quality and the species "
        "each model supports - BlueEye never claims a species outside a model's "
        "trained classes."
    )


# --------------------------------------------------------------------------- #
# Species preview
# --------------------------------------------------------------------------- #
def _species_preview() -> None:
    st.markdown("### Explore aquatic biodiversity")
    st.markdown(
        '<p class="be-dash-lead">A preview from the Species Explorer reference '
        "library - Indian and South Asian species first.</p>",
        unsafe_allow_html=True,
    )
    selected = [get_species(species_id) for species_id in PREVIEW_SPECIES]
    species = [item for item in selected if item is not None]

    for start in range(0, len(species), 3):
        row = species[start : start + 3]
        columns = st.columns(3, gap="medium")
        for column, item in zip(columns, row):
            with column:
                st.markdown(
                    '<div class="be-card be-species-card">'
                    f'<div class="be-card-title">{esc(item.common_name)}</div>'
                    f"<p class='be-card-sub'><em>{esc(item.scientific_name)}</em></p>"
                    f"<div>{chips((item.habitat, item.group))}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    "View species",
                    key=f"home_species_{item.id}",
                    icon=":material/arrow_forward:",
                    width="stretch",
                ):
                    st.session_state["species_selected"] = item.id
                    state.navigate("species")

    if st.button(
        "Explore all species",
        key="home_all_species",
        icon=":material/menu_book:",
        width="stretch",
    ):
        st.session_state.pop("species_selected", None)
        state.navigate("species")


# --------------------------------------------------------------------------- #
# Responsible observation
# --------------------------------------------------------------------------- #
def _responsible_observation() -> None:
    st.markdown(
        '<div class="be-responsible">'
        f'{span("verified_user")}'
        "<div><b>Responsible observation</b> - every detection is a model "
        "prediction that depends on the input and on the classes the selected "
        "model was trained on. Validate predictions before using them for "
        "scientific, conservation or regulatory decisions.</div></div>",
        unsafe_allow_html=True,
    )


__all__ = ["render"]
