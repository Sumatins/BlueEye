"""Analytics page: real statistics computed from the stored detection history.

Charts are built exclusively from persisted entries
(:mod:`app.utils.history`). If there is no data, the page says so instead of
showing invented numbers.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import streamlit as st

from app.detection.detector import MarineDetector
from app.detection.results import display_class_name
from app.ui import state
from app.ui.components import badge, empty_state, hero, metrics_row, notice
from app.utils.history import (
    confidence_histogram,
    history_path,
    load_entries,
    summarize,
)


def render(detector: MarineDetector, settings) -> None:
    """Render session/cumulative analytics."""
    entries = load_entries(history_path(settings))
    stats = summarize(entries)

    hero(
        st,
        "Measured, not estimated",
        "Analytics",
        "Everything on this page is computed from your finished detections.",
        icon="insights",
    )

    if not stats["runs"]:
        empty_state(
            st,
            "bar_chart",
            "Not enough data yet",
            "Run a detection on the Detect page - finished runs are stored "
            "locally and summarized here.",
        )
        notice(
            st,
            "info",
            "Where the data comes from",
            "Each completed detection appends an entry to "
            f"{history_path(settings).name} in the outputs folder. Nothing is "
            "uploaded anywhere.",
        )
        return

    metrics_row(
        st,
        [
            ("Objects detected", f"{stats['total_objects']:,}", "across all stored runs"),
            ("Species detected", stats["unique_species"], "distinct classes"),
            ("Detections run", stats["runs"], f"{stats['images']} image(s), {stats['videos']} video(s)"),
            (
                "Average confidence",
                f"{stats['average_confidence'] * 100:.1f}%",
                "weighted by detections",
            ),
        ],
    )
    st.caption(
        f"{stats['frames_processed']:,} video frame(s) processed · "
        f"first run {stats['first_run']} · last run {stats['last_run']}"
    )
    st.write("")

    left, right = st.columns(2)

    with left:
        st.markdown("### Most detected species")
        if stats["top_species"]:
            _species_chart(stats["species_counts"])
            for rank, (name, count) in enumerate(stats["top_species"], start=1):
                st.write(f"{rank}. **{display_class_name(name)}** — {count}")
        else:
            empty_state(st, "pets", "No species recorded", "Stored runs had no class labels.")

    with right:
        st.markdown("### Confidence distribution")
        _confidence_chart(entries)
        st.caption(
            "Buckets the *average* confidence of each finished run "
            "(runs are the unit we store)."
        )

    st.markdown("### Models used")
    if stats["models_used"]:
        labels = {
            key: spec.display_name
            for key, spec in state.registry(detector).items()
        }
        st.markdown(
            "".join(
                badge(labels.get(key, display_class_name(key)), "info")
                for key in stats["models_used"]
            ),
            unsafe_allow_html=True,
        )
    else:
        st.caption("No model information recorded.")

    notice(
        st,
        "info",
        "Stored locally",
        f"{stats['runs']} detection(s) are kept in "
        f"{history_path(settings)}. You can clear them on the History page.",
    )


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _species_chart(counts: dict[str, int]) -> None:
    """Bar chart of detections per species (top 12, real counts)."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - streamlit depends on pandas
        return
    top = list(counts.items())[:12]
    frame = pd.DataFrame(
        {"Species": [display_class_name(name) for name, _ in top], "Objects": [count for _, count in top]}
    )
    st.bar_chart(frame.set_index("Species"), height=260)


def _confidence_chart(entries) -> None:
    """Bar chart of run-average confidence buckets."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - streamlit depends on pandas
        return
    buckets = confidence_histogram(entries, bins=10)
    frame = pd.DataFrame(
        {"Confidence": [label for label, _ in buckets], "Runs": [count for _, count in buckets]}
    )
    st.bar_chart(frame.set_index("Confidence"), height=260)


__all__ = ["render"]
