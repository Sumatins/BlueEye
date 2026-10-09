"""Dashboard: landing page with quick actions, system status and real stats.

Every number shown here is computed from the persisted detection history or
from the live model registry - nothing is simulated.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app import __version__
from app.detection.detector import MarineDetector
from app.ui import state
from app.ui.components import card, empty_state, hero, metrics_row, status_badge
from app.ui.icons import span
from app.ui.theme import esc
from app.utils.history import history_path, load_entries, summarize


def render(detector: MarineDetector, settings) -> None:
    """Render the landing page."""
    hero(
        st,
        "See beneath the surface",
        "BlueEye",
        "Analyze marine life with AI-powered underwater detection.",
        icon="visibility",
    )

    _quick_actions()
    _system_status(detector)
    _statistics(settings)
    _recent_detection(settings)
    _model_preview(detector)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _quick_actions() -> None:
    st.markdown("### Quick actions")
    image_column, video_column = st.columns(2)
    with image_column:
        if st.button(
            "Detect image",
            type="primary",
            icon=":material/image:",
            width="stretch",
        ):
            st.session_state["detect_media"] = "Image"
            state.navigate("detect")
    with video_column:
        if st.button(
            "Detect video",
            icon=":material/videocam:",
            width="stretch",
        ):
            st.session_state["detect_media"] = "Video"
            state.navigate("detect")
    st.write("")


def _system_status(detector: MarineDetector) -> None:
    registry = state.registry(detector)
    ready = state.available_keys(detector)
    models_label = f"{len(ready)} / {len(registry)}"
    status = "Ready" if ready else "Model weights required"

    st.markdown("### System status")
    metrics_row(
        st,
        [
            ("Device", str(detector.device).upper(), "CPU / CUDA is detected automatically"),
            ("Models", models_label, "installed and ready"),
            ("Status", status, "inference service"),
            ("Version", __version__, "BlueEye release"),
        ],
    )


def _statistics(settings) -> None:
    entries = load_entries(history_path(settings))
    stats = summarize(entries)

    st.markdown("### Detection statistics")
    if not stats["runs"]:
        empty_state(
            st,
            "bar_chart",
            "No detections yet",
            "Statistics from your finished detections will appear here. "
            "Run a detection on the Detect page to get started.",
        )
        return

    metrics_row(
        st,
        [
            ("Detections run", stats["runs"], "stored on this machine"),
            ("Objects detected", stats["total_objects"], "across every run"),
            ("Species detected", stats["unique_species"], "distinct classes"),
            (
                "Average confidence",
                f"{stats['average_confidence'] * 100:.1f}%",
                "weighted by detections",
            ),
        ],
    )
    st.caption(
        f"Computed from {stats['runs']} stored detection(s) · "
        f"{stats['frames_processed']:,} video frame(s) processed · "
        f"latest run {stats['last_run']}"
    )


def _recent_detection(settings) -> None:
    entries = load_entries(history_path(settings))
    if not entries:
        return
    latest = entries[0]
    st.markdown("### Recent detection")
    body = (
        '<dl class="be-kv">'
        f"<dt>When</dt><dd>{esc(latest.display_time)}</dd>"
        f"<dt>Input</dt><dd>{esc(Path(latest.filename).name)}</dd>"
        f"<dt>Type</dt><dd>{esc(latest.kind.capitalize())}</dd>"
        f"<dt>Model(s)</dt><dd>{esc(', '.join(latest.models) or '—')}</dd>"
        f"<dt>Objects</dt><dd>{latest.total_detections} "
        f"({latest.average_confidence * 100:.1f}% avg confidence)</dd>"
        "</dl>"
    )
    card(
        st,
        title=Path(latest.output).name if latest.output else "Latest run",
        body_html=body,
        subtitle=f"Finished {latest.display_time}",
    )
    if st.button("Open history", icon=":material/history:", width="stretch"):
        state.navigate("history")


def _model_preview(detector: MarineDetector) -> None:
    registry = state.registry(detector)
    ready = set(state.available_keys(detector))

    st.markdown("### Available models")
    rows = []
    for key, spec in registry.items():
        rows.append(
            "<div class='be-card be-card--flush' style='margin-bottom:.4rem'>"
            f"<div class='be-card-title'>{span('psychology')}{esc(spec.display_name)} "
            f"{status_badge(key in ready)}</div>"
            f"<p class='be-card-sub'>{esc(spec.summary or spec.description)}</p>"
            f"<div>{esc(key)}</div></div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)
    if st.button("Manage models", icon=":material/psychology:", width="stretch"):
        state.navigate("models")
    st.caption(
        f"{len(ready)} of {len(registry)} models installed · weights are "
        "downloaded on demand and never fabricated."
    )
