"""History page: the persisted list of finished detections.

Entries are read from :mod:`app.utils.history` (``outputs/history/
detections.json``) - they are real records of runs performed on this
machine, never seeded or faked.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.detection.detector import MarineDetector
from app.detection.results import display_class_name
from app.ui.components import badge, download_button, empty_state, hero, metrics_row, notice
from app.utils.history import clear_history, history_path, load_entries


def render(detector: MarineDetector, settings) -> None:
    """Render the detection history."""
    store = history_path(settings)
    entries = load_entries(store)

    hero(
        st,
        "Stored on this machine",
        "History",
        "Every finished detection, with its real inputs and outputs.",
        icon="history",
    )

    if not entries:
        empty_state(
            st,
            "history",
            "No detections recorded yet",
            "Completed runs are saved here automatically. Start on the Detect page.",
        )
        notice(
            st,
            "info",
            "Session history",
            f"History is persisted to {store} and survives restarts. "
            "It is never sent anywhere.",
        )
        return

    metrics_row(
        st,
        [
            ("Stored runs", len(entries), "newest first"),
            ("Objects", sum(entry.total_detections for entry in entries), "all time"),
            ("Images", sum(1 for entry in entries if entry.kind == "image"), "runs"),
            ("Videos", sum(1 for entry in entries if entry.kind == "video"), "runs"),
        ],
    )
    st.write("")

    _history_table(entries)
    st.write("")
    _actions(store)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _history_table(entries) -> None:
    """Table of stored runs."""
    rows = []
    for index, entry in enumerate(entries, start=1):
        rows.append(
            {
                "#": index,
                "Date": entry.display_time,
                "Input": Path(entry.filename).name,
                "Type": entry.kind.capitalize(),
                "Model(s)": ", ".join(entry.models) or "—",
                "Objects": entry.total_detections,
                "Species": entry.unique_classes,
                "Avg conf": f"{entry.average_confidence * 100:.1f}%",
                "Threshold": (
                    "model default"
                    if entry.confidence_threshold is None
                    else f"{entry.confidence_threshold:.2f}"
                ),
                "Enhanced": "yes" if entry.enhanced else "no",
                "Output": Path(entry.output).name if entry.output else "—",
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")

    with st.expander("Show the newest run in detail", expanded=False):
        _entry_detail(entries[0])


def _entry_detail(entry) -> None:
    """Detail block for one entry (class breakdown + report download)."""
    st.markdown(f"**{Path(entry.filename).name}** · {entry.display_time}")
    body = (
        "<dl class='be-kv'>"
        f"<dt>Type</dt><dd>{entry.kind.capitalize()}</dd>"
        f"<dt>Model(s)</dt><dd>{', '.join(entry.models) or '—'}</dd>"
        f"<dt>Objects</dt><dd>{entry.total_detections}</dd>"
        f"<dt>Species</dt><dd>{entry.unique_classes}</dd>"
        f"<dt>Average confidence</dt><dd>{entry.average_confidence * 100:.1f}%</dd>"
        f"<dt>Max confidence</dt><dd>{entry.max_confidence * 100:.1f}%</dd>"
        f"<dt>Frames processed</dt><dd>{entry.frames_processed or '—'}</dd>"
        f"<dt>Elapsed</dt><dd>{f'{entry.elapsed_seconds:.1f} s' if entry.elapsed_seconds else '—'}</dd>"
        f"<dt>Output</dt><dd>{Path(entry.output).name if entry.output else '—'}</dd>"
        "</dl>"
    )
    st.markdown(body, unsafe_allow_html=True)

    if entry.detections_per_class:
        st.markdown(
            "  ".join(
                badge(f"{display_class_name(name)} · {count}", "info")
                for name, count in entry.detections_per_class.items()
            ),
            unsafe_allow_html=True,
        )

    if entry.report and Path(entry.report).exists():
        download_button(st, "Download JSON report", entry.report, "application/json")
    if entry.output and Path(entry.output).exists():
        suffix = Path(entry.output).suffix.lower()
        mime = "video/mp4" if suffix in {".mp4", ".mov", ".mkv", ".avi"} else "image/jpeg"
        download_button(st, "Download output file", entry.output, mime)


def _actions(store: Path) -> None:
    """Export and clear actions."""
    left, right = st.columns(2)
    with left:
        download_button(st, "Export history (JSON)", store, "application/json")
    with right:
        if st.button(
            "Clear history",
            icon=":material/delete:",
            width="stretch",
        ):
            clear_history(store)
            st.rerun()

    st.caption(f"Stored at {store}")


__all__ = ["render"]
