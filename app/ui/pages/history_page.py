"""History page: the persisted list of finished detections.

Entries are read from :mod:`app.utils.history` (one store per account under
``outputs/history/users/<user_id>/``) - they are real records of runs
performed on this machine, never seeded or faked.

Every stored run can be re-opened: its original input and annotated result
are shown inline (image) or as a playable processed video, alongside the
real detection summary and downloads. Files that were moved or deleted are
reported honestly instead of breaking the page, and internal storage paths
are never rendered.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.detection.detector import MarineDetector
from app.detection.results import display_class_name
from app.ui import state
from app.ui.components import (
    badge,
    download_button,
    empty_state,
    hero,
    kv,
    metrics_row,
    notice,
)
from app.utils.history import clear_history, history_path, load_entries


def render(detector: MarineDetector, settings, user=None) -> None:
    """Render the detection history for the signed-in account."""
    user_id = user.id if user else None
    store = history_path(settings, user_id=user_id)
    entries = load_entries(store)

    hero(
        st,
        "Saved locally under your account",
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
            "Your history",
            "Finished detections are stored privately under your account and "
            "survive restarts. They are never sent anywhere.",
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

    _history_table(entries, settings, user_id)
    st.write("")
    _actions(store)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _history_table(entries, settings=None, user_id=None) -> None:
    """Table of stored runs plus an inline viewer for any selected run."""
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
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")

    st.markdown("### Open a stored run")
    labels = [
        f"#{index} · {entry.display_time} · {Path(entry.filename).name}"
        for index, entry in enumerate(entries, start=1)
    ]
    selected = st.selectbox(
        "Stored run",
        options=list(range(len(entries))),
        format_func=lambda index: labels[index],
        key="history_select",
        label_visibility="collapsed",
    )
    st.write("")
    _entry_detail(entries[selected], settings, user_id)


def _entry_detail(entry, settings=None, user_id=None) -> None:
    """Full detail for one stored run, with inline media when available.

    Downloads and inline media are only offered for files that belong to the
    signed-in account's own scoped directories; anything else is reported as
    unavailable without revealing where it actually lives.
    """
    st.markdown(f"#### {Path(entry.filename).name}")
    st.caption(f"{entry.kind.capitalize()} · {entry.display_time}")

    output = Path(entry.output) if entry.output else None
    output_ok = (
        output is not None
        and state.is_owned_path(settings, user_id, output)
        and output.is_file()
    )
    original = state.resolve_owned_input(settings, user_id, entry.filename)

    if entry.kind == "video":
        _video_media(output if output_ok else None, original)
    else:
        _image_media(output if output_ok else None, original)

    metrics_row(
        st,
        [
            ("Total objects", entry.total_detections, "above threshold"),
            ("Species", entry.unique_classes, "distinct classes"),
            (
                "Average confidence",
                f"{entry.average_confidence * 100:.1f}%",
                "across detections",
            ),
            (
                "Max confidence",
                f"{entry.max_confidence * 100:.1f}%",
                "best match",
            ),
        ],
    )

    if entry.detections_per_class:
        st.markdown("**Detected classes**")
        st.markdown(
            "  ".join(
                badge(f"{display_class_name(name)} · {count}", "info")
                for name, count in entry.detections_per_class.items()
            ),
            unsafe_allow_html=True,
        )

    rows = [
        ("Type", entry.kind.capitalize()),
        ("Model(s)", ", ".join(entry.models) or "—"),
        (
            "Confidence threshold",
            "model default"
            if entry.confidence_threshold is None
            else f"{entry.confidence_threshold:.2f}",
        ),
        ("Enhanced", "yes" if entry.enhanced else "no"),
        (
            "Frames processed",
            entry.frames_processed if entry.frames_processed is not None else "",
        ),
        (
            "Elapsed",
            f"{entry.elapsed_seconds:.1f} s" if entry.elapsed_seconds else "",
        ),
    ]
    details = kv(rows)
    if details:
        st.markdown(details, unsafe_allow_html=True)

    st.write("")
    download_left, download_right = st.columns(2)
    with download_left:
        if output_ok:
            suffix = output.suffix.lower()
            mime = "video/mp4" if suffix in {".mp4", ".mov", ".mkv", ".avi"} else "image/jpeg"
            download_button(st, "Download output file", output, mime)
        elif entry.output:
            st.caption("The output file is no longer available on this device.")
    with download_right:
        report_ok = (
            bool(entry.report)
            and state.is_owned_path(settings, user_id, entry.report)
            and Path(entry.report).is_file()
        )
        if report_ok:
            download_button(st, "Download JSON report", entry.report, "application/json")
        elif entry.report:
            st.caption("The JSON report is no longer available on this device.")


def _image_media(output: Path | None, original: Path | None) -> None:
    """Original input beside the annotated result (either side may be missing)."""
    left, right = st.columns(2)
    with left:
        st.markdown("**Original image**")
        if original is not None:
            _show_image(original)
        else:
            st.caption("The original image is no longer available on this device.")
    with right:
        st.markdown("**Detection result**")
        if output is not None:
            _show_image(output)
        else:
            st.caption("The annotated result is no longer available on this device.")


def _video_media(output: Path | None, original: Path | None) -> None:
    """Processed video (and the original, if still present)."""
    left, right = st.columns(2)
    with left:
        st.markdown("**Original video**")
        if original is not None:
            _show_video(original)
        else:
            st.caption("The original video is no longer available on this device.")
    with right:
        st.markdown("**Processed video**")
        if output is not None:
            _show_video(output)
        else:
            st.caption("The processed video is no longer available on this device.")


def _show_image(path: Path) -> None:
    """Render an image, tolerating a file that vanished between checks."""
    try:
        st.image(path.read_bytes(), width="stretch")
    except OSError:
        st.caption("That image could not be read from disk.")


def _show_video(path: Path) -> None:
    """Render a video, tolerating a file that vanished between checks."""
    try:
        st.video(path.read_bytes())
    except OSError:
        st.caption("That video could not be read from disk.")


def _actions(store: Path) -> None:
    """Export and clear actions."""
    entry_count = len(load_entries(store))
    left, right = st.columns(2)
    with left:
        if entry_count:
            download_button(st, "Export history (JSON)", store, "application/json")
    with right:
        if st.button(
            "Clear history",
            icon=":material/delete:",
            width="stretch",
        ):
            clear_history(store)
            st.rerun()

    st.caption(
        "History is kept privately under your account; it is never uploaded "
        "or shared."
    )


__all__ = ["render"]
