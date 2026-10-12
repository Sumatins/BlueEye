"""History page: the persisted list of finished detections.

Entries are read from :mod:`app.utils.history` (one store per account under
``outputs/history/users/<user_id>/``) - they are real records of runs
performed on this machine, never seeded or faked.

Every stored run can be searched, filtered, re-opened and deleted. Its original
input and annotated result are shown inline (image) or as a playable processed
video, alongside the real detection summary and downloads. Files that were
moved or deleted are reported honestly instead of breaking the page, and
internal storage paths are never rendered.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import json
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
from app.utils.history import (
    clear_history,
    entries_csv,
    filter_entries,
    history_path,
    load_entries,
    remove_entry,
    research_summary,
)


def render(detector: MarineDetector, settings, user=None) -> None:
    """Render the stored-run browser for the signed-in account."""
    user_id = user.id if user else None
    store = history_path(settings, user_id=user_id)
    entries = load_entries(store)

    hero(
        st,
        "Your stored runs",
        "Detection History",
        "Search, re-open and export every detection recorded on this machine.",
        icon="history",
    )

    if not entries:
        empty_state(
            st,
            "history",
            "No detections recorded yet",
            "Run a detection on the AI Detection page and it will appear here.",
        )
        return

    metrics_row(
        st,
        [
            ("Runs", len(entries), "newest first"),
            ("Objects", sum(entry.total_detections for entry in entries), "all time"),
            ("Images", sum(1 for entry in entries if entry.kind == "image"), "runs"),
            ("Videos", sum(1 for entry in entries if entry.kind == "video"), "runs"),
        ],
    )
    st.write("")

    filtered = _filters(entries)
    st.caption(f"Showing {len(filtered)} of {len(entries)} run(s).")

    if filtered:
        _history_table(filtered, settings, user_id, store)
    else:
        empty_state(
            st,
            "filter_alt_off",
            "No runs match the current filters",
            "Clear the search box or reset the type/model filters to see more.",
        )

    st.write("")
    _actions(store, filtered or entries)


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def _filters(entries):
    """Free-text + type + model filters, all applied to the real stored fields."""
    models = sorted({model for entry in entries for model in entry.models})
    left, mid, right = st.columns([2, 1, 1])
    with left:
        query = st.text_input(
            "Search by file or species",
            key="history_query",
            placeholder="e.g. reef, shark, turtle",
        )
    with mid:
        kind = st.selectbox(
            "Type", options=["All", "image", "video"], key="history_kind"
        )
    with right:
        model = st.selectbox(
            "Model", options=["All", *models] if models else ["All"], key="history_model"
        )
    return filter_entries(
        entries,
        query=query,
        kind=None if kind == "All" else kind,
        model=None if model == "All" else model,
    )


# --------------------------------------------------------------------------- #
# Table + detail
# --------------------------------------------------------------------------- #
def _history_table(entries, settings, user_id, store) -> None:
    """Table of stored runs plus an inline viewer for the selected run."""
    rows = []
    for index, entry in enumerate(entries, start=1):
        rows.append(
            {
                "#": index,
                "Run": entry.run_id,
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
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")

    st.markdown("### Open a stored run")
    labels = [
        f"{entry.run_id} · {entry.display_time} · {Path(entry.filename).name}"
        for entry in entries
    ]
    selected = st.selectbox(
        "Stored run",
        options=list(range(len(entries))),
        format_func=lambda index: labels[index],
        key="history_select",
        label_visibility="collapsed",
    )
    st.write("")
    _entry_detail(entries[selected], settings, user_id, store)


def _entry_detail(entry, settings, user_id, store) -> None:
    """Full detail for one stored run, with inline media and downloads.

    Downloads and inline media are only offered for files that belong to the
    signed-in account's own scoped directories; anything else is reported as
    unavailable without revealing where it actually lives.
    """
    st.markdown(f"#### {Path(entry.filename).name}")
    st.caption(f"{entry.kind.capitalize()} · {entry.display_time} · {entry.run_id}")

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
        ("Run id", entry.run_id),
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
        ("Elapsed", f"{entry.elapsed_seconds:.1f} s" if entry.elapsed_seconds else ""),
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

    st.download_button(
        "Download this run (research summary JSON)",
        data=json.dumps(research_summary([entry]), indent=2),
        file_name=f"{entry.run_id}_research_summary.json",
        mime="application/json",
        icon=":material/download:",
        key="history_run_summary",
    )

    _delete_section(entry, settings, user_id, store)


def _delete_section(entry, settings, user_id, store) -> None:
    """Delete one stored run behind an explicit confirmation."""
    with st.expander("Delete this run"):
        st.caption(
            "Removes this record from your history. This cannot be undone; "
            "other runs are unaffected."
        )
        also_files = st.checkbox(
            "Also delete this run's stored output and report files",
            key="history_delete_files",
        )
        confirm = st.checkbox(
            "I understand this permanently removes the selected run",
            key="history_delete_confirm",
        )
        if st.button(
            "Delete run",
            icon=":material/delete:",
            disabled=not confirm,
            key="history_delete",
        ):
            _delete_entry(store, entry, settings, user_id, also_files)
            st.success("The stored run was deleted.")
            st.rerun()


def _delete_entry(store, entry, settings, user_id, also_files: bool) -> None:
    """Remove ``entry`` from the store (and optionally its owned files)."""
    store_entries = load_entries(store)
    index = next((i for i, stored in enumerate(store_entries) if _matches(stored, entry)), None)
    if index is not None:
        remove_entry(store, index)
    if also_files:
        for candidate in (entry.output, entry.report):
            if candidate and state.is_owned_path(settings, user_id, candidate):
                try:
                    Path(candidate).unlink(missing_ok=True)
                except OSError:  # pragma: no cover - a locked/missing file is fine
                    continue


def _matches(a, b) -> bool:
    """Identify the same stored run across a reload by its stable fields."""
    return a.run_id == b.run_id and (
        a.timestamp,
        a.kind,
        Path(a.filename).name,
        a.total_detections,
        a.unique_classes,
    ) == (
        b.timestamp,
        b.kind,
        Path(b.filename).name,
        b.total_detections,
        b.unique_classes,
    )


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Export / clear
# --------------------------------------------------------------------------- #
def _actions(store: Path, entries) -> None:
    """Export the (filtered) runs and clear the whole history."""
    st.markdown("### Export & maintenance")
    count = len(entries)
    left, mid, right = st.columns(3)
    with left:
        st.download_button(
            "Export runs (CSV)",
            data=entries_csv(entries),
            file_name="blueeye_runs.csv",
            mime="text/csv",
            icon=":material/download:",
            disabled=count == 0,
            key="history_csv",
        )
    with mid:
        st.download_button(
            "Research summary (JSON)",
            data=json.dumps(research_summary(entries), indent=2),
            file_name="blueeye_research_summary.json",
            mime="application/json",
            icon=":material/download:",
            disabled=count == 0,
            key="history_summary",
        )
    with right:
        if store.is_file():
            download_button(st, "Raw store (JSON)", store, "application/json")

    with st.expander("Clear all history"):
        st.caption(
            "Permanently removes every stored run for this account. Output and "
            "report files already on disk are left untouched."
        )
        confirm = st.checkbox(
            "I understand this clears my entire detection history",
            key="history_clear_confirm",
        )
        if st.button(
            "Clear history",
            icon=":material/delete_sweep:",
            disabled=not confirm,
            key="history_clear",
        ):
            clear_history(store)
            st.rerun()

    notice(
        st,
        "info",
        "Your data stays here",
        "History is stored locally under your account and is exported only when "
        "you ask. The figures are ordinary model predictions, not validated "
        "accuracy.",
        icon="lock",
    )


__all__ = ["render"]
