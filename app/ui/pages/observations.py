"""My Observations - the account's private library of saved detections.

This page is the long-term home of every saved analysis. It is a deliberate
extension of the detection History: History lists *runs*, whereas My
Observations lists the records a user chose to keep, with search, filters,
survey collections, side-by-side comparison and structured exports.

Opening a record never re-runs the detector: the detail view
(:mod:`app.ui.pages.observation_detail`) reads the saved database row and the
private media it references. Everything is scoped to the signed-in account and
no internal file path is ever printed.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import streamlit as st

from app.detection.results import display_class_name
from app.observations import compare as obs_compare
from app.observations import export as obs_export
from app.observations.models import REVIEW_STATUSES, REVIEW_STATUS_LABELS, ObservationError
from app.ui import state
from app.ui.components import (
    empty_state,
    error_card,
    hero,
    metrics_row,
    notice,
)
from app.ui.pages import observation_detail
from app.ui.theme import esc


def _status_label(status: str) -> str:
    return REVIEW_STATUS_LABELS.get(status, (status,))[0]


def render(detector, settings, user=None) -> None:
    """Render the private observation library for the signed-in account."""
    if user is None:
        error_card(
            st,
            "Sign in required",
            "My Observations is private. Sign in to open your saved records.",
        )
        return

    open_id = st.session_state.get("observation_open")
    if open_id:
        observation_detail.render(settings, user, open_id)
        return

    hero(
        st,
        "Your library",
        "My Observations",
        "Saved detections only you can see: search, review, organise into surveys "
        "and export as research data.",
        icon="bookmarks",
    )

    store = state.observation_store(settings)
    _metrics(store.stats(user.id))

    tab_browse, tab_collections, tab_compare = st.tabs(
        ["Observations", "Collections & surveys", "Compare & export"]
    )
    with tab_browse:
        _browse(store, settings, user)
    with tab_collections:
        _collections(store, settings, user)
    with tab_compare:
        _compare_export(store, settings, user)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metrics(stats: dict) -> None:
    metrics_row(
        st,
        [
            ("Saved observations", stats["observations"], "individual detections"),
            ("Analyses", stats["analyses"], "saved runs"),
            ("Awaiting review", stats["awaiting_review"], "predicted or flagged"),
            ("Collections", stats["collections"], "surveys / projects"),
        ],
    )


# --------------------------------------------------------------------------- #
# Browse
# --------------------------------------------------------------------------- #
def _summary_row(analysis) -> dict:
    return {
        "Source": analysis.source_filename,
        "Detected": analysis.detected_at,
        "Type": analysis.kind,
        "Objects": analysis.total_detections,
        "Classes": analysis.unique_classes,
        "Avg confidence": f"{analysis.average_confidence * 100:.1f}%",
        "Project": analysis.project or "—",
        "Review": _status_label(analysis.review_status),
    }


def _browse(store, settings, user) -> None:
    if store.count_analyses(user.id) == 0:
        empty_state(
            st,
            "bookmarks",
            "No observations saved yet",
            "Run a detection on the AI Detection page and choose 'Save observation' "
            "to build your private library.",
        )
        return

    filters = _filters(store, user)
    total = store.count_analyses(user.id, **filters)
    if total == 0:
        notice(
            st,
            "info",
            "No matching observations",
            "No saved observation matches the current search or filters.",
            icon="search_off",
        )
        return

    page_size = st.selectbox("Per page", options=(10, 25, 50), key="observations_page_size")
    offset = int(st.session_state.get("observations_offset", 0) or 0)
    offset = max(0, min(offset, max(0, total - 1)))
    analyses = store.list_analyses(user.id, **filters, limit=page_size, offset=offset)

    st.caption(f"Showing {len(analyses)} of {total} saved analysis(es).")
    for analysis in analyses:
        _observation_row(store, settings, user, analysis)

    columns = st.columns([1, 1, 3])
    with columns[0]:
        if st.button("Previous", key="observations_prev", disabled=offset <= 0):
            st.session_state["observations_offset"] = max(0, offset - page_size)
            st.rerun()
    with columns[1]:
        if st.button(
            "Next", key="observations_next", disabled=offset + page_size >= total
        ):
            st.session_state["observations_offset"] = offset + page_size
            st.rerun()

    _export_view(store, user, analyses, total)


def _observation_row(store, settings, user, analysis) -> None:
    columns = st.columns([5, 1])
    with columns[0]:
        st.markdown(
            f"**{esc(analysis.source_filename)}** · {esc(analysis.detected_at)} · "
            f"{analysis.kind} · {analysis.total_detections} object(s) · "
            f"{esc(_status_label(analysis.review_status))}",
            unsafe_allow_html=True,
        )
        if analysis.project:
            st.caption(f"Project: {analysis.project}")
    with columns[1]:
        if st.button("Open", key=f"obs_open_{analysis.id}", width="stretch"):
            st.session_state["observation_open"] = analysis.id
            st.rerun()


def _filters(store, user) -> dict:
    st.markdown("**Search and filter**")
    columns = st.columns([2, 1, 1, 1, 1])
    with columns[0]:
        st.text_input(
            "Search (label, filename, project, notes)",
            key="observations_query",
            placeholder="e.g. parrotfish or reef_2024.jpg",
        )
    with columns[1]:
        st.selectbox("Media", ("All", "Image", "Video"), key="observations_kind")
    with columns[2]:
        st.selectbox(
            "Review",
            ("All", *REVIEW_STATUSES),
            key="observations_status",
            format_func=lambda key: "All" if key == "All" else _status_label(key),
        )
    with columns[3]:
        projects = ["All", *store.distinct_projects(user.id)]
        if st.session_state.get("observations_project") not in projects:
            st.session_state["observations_project"] = "All"
        st.selectbox("Project", projects, key="observations_project")
    with columns[4]:
        models = ["All", *store.distinct_models(user.id)]
        if st.session_state.get("observations_model") not in models:
            st.session_state["observations_model"] = "All"
        st.selectbox("Model", models, key="observations_model")

    kind = st.session_state.get("observations_kind", "All")
    status = st.session_state.get("observations_status", "All")
    project = st.session_state.get("observations_project", "All")
    model = st.session_state.get("observations_model", "All")
    query = st.session_state.get("observations_query", "")
    signature = (query, kind, status, project, model)
    if st.session_state.get("observations_filter_sig") != signature:
        st.session_state["observations_offset"] = 0
        st.session_state["observations_filter_sig"] = signature
    return {
        "query": query,
        "kind": None if kind == "All" else kind.lower(),
        "review_status": None if status == "All" else status,
        "project": None if project == "All" else project,
        "model": None if model == "All" else model,
    }


def _export_view(store, user, analyses, total: int) -> None:
    st.markdown("**Export the current view**")
    full = [analysis for analysis in (store.get_analysis(a.id, user.id) for a in analyses)
            if analysis is not None]
    st.caption(f"Exports the {len(full)} analysis(es) currently shown (of {total}).")
    columns = st.columns(2)
    with columns[0]:
        st.download_button(
            "Download observations CSV",
            data=obs_export.observations_csv(full),
            file_name="blueeye_observations.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
    with columns[1]:
        st.download_button(
            "Download observations JSON",
            data=obs_export.json_dumps(obs_export.observations_to_dict(full)),
            file_name="blueeye_observations.json",
            mime="application/json",
            icon=":material/download:",
            width="stretch",
        )


# --------------------------------------------------------------------------- #
# Collections
# --------------------------------------------------------------------------- #
def _collections(store, settings, user) -> None:
    collections = store.list_collections(user.id)
    st.markdown("### Your surveys and collections")
    if not collections:
        empty_state(
            st,
            "folder_open",
            "No collections yet",
            "Create a survey or project below. Add observations from an "
            "observation's detail page.",
        )
    else:
        for collection in collections:
            with st.container(border=True):
                st.markdown(f"**{collection.name}**")
                if collection.description:
                    st.caption(collection.description)
                st.caption(
                    f"{collection.item_count} observation(s) · updated "
                    f"{collection.updated_at}"
                )

    with st.form(key="observations_new_collection"):
        name = st.text_input("New collection name", key="observations_col_name")
        description = st.text_input("Description (optional)", key="observations_col_desc")
        submitted = st.form_submit_button("Create collection", icon=":material/create_new_folder:")
    if submitted:
        if not name.strip():
            st.warning("Enter a collection name first.")
        else:
            collection = store.create_collection(user.id, name.strip(), description.strip())
            st.success(f"Created '{collection.name}'.")
            st.rerun()

    if not collections:
        return

    names = [collection.name for collection in collections]
    if st.session_state.get("observations_collection_pick") not in names:
        st.session_state["observations_collection_pick"] = names[0]
    chosen = st.selectbox("Open a collection", names, key="observations_collection_pick")
    collection = next(item for item in collections if item.name == chosen)
    _collection_detail(store, user, collection)


def _collection_detail(store, user, collection) -> None:
    analyses = store.list_collection_analyses(collection.id, user.id)
    if not analyses:
        st.caption("This collection has no saved observations yet.")
    else:
        st.dataframe(
            [_summary_row(item) for item in analyses], hide_index=True, width="stretch"
        )
        options = {
            f"{item.source_filename} · {item.detected_at} · {item.id[:6]}": item.id
            for item in analyses
        }
        pick = st.selectbox(
            "Remove an observation from this collection",
            list(options.keys()),
            key=f"col_remove_pick_{collection.id}",
        )
        if st.button("Remove from collection", key=f"col_remove_{collection.id}"):
            store.remove_from_collection(collection.id, options[pick], user.id)
            st.success("Observation removed from this collection.")
            st.rerun()

        full = [item for item in (store.get_analysis(a.id, user.id) for a in analyses)
                if item is not None]
        columns = st.columns(3)
        with columns[0]:
            st.download_button(
                "Survey JSON",
                data=obs_export.json_dumps(obs_export.collection_to_dict(collection, full)),
                file_name=f"collection_{collection.id[:8]}.json",
                mime="application/json",
                icon=":material/download:",
                width="stretch",
            )
        with columns[1]:
            st.download_button(
                "Survey CSV",
                data=obs_export.collection_csv(collection, full),
                file_name=f"collection_{collection.id[:8]}.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )
        with columns[2]:
            st.download_button(
                "Survey HTML report",
                data=obs_export.collection_report_html(collection, full),
                file_name=f"collection_{collection.id[:8]}.html",
                mime="text/html",
                icon=":material/download:",
                width="stretch",
            )

    rename = st.text_input("Rename collection", value=collection.name, key=f"col_rename_{collection.id}")
    rename_columns = st.columns(2)
    with rename_columns[0]:
        if st.button("Save name", key=f"col_rename_save_{collection.id}", width="stretch"):
            try:
                store.update_collection(collection.id, user.id, name=rename)
            except ObservationError as exc:
                st.error(str(exc))
            else:
                st.success("Collection renamed.")
                st.rerun()
    with rename_columns[1]:
        if st.button(
            "Delete collection",
            key=f"col_delete_{collection.id}",
            icon=":material/delete:",
            width="stretch",
        ):
            store.delete_collection(collection.id, user.id)
            st.success("Collection deleted. The observations themselves were kept.")
            st.rerun()


# --------------------------------------------------------------------------- #
# Compare & export
# --------------------------------------------------------------------------- #
def _compare_export(store, settings, user) -> None:
    analyses = store.list_analyses(user.id)
    if not analyses:
        empty_state(
            st,
            "compare",
            "Nothing to compare yet",
            "Save at least two observations to compare model predictions side by side.",
        )
        return

    labels = {
        f"{item.source_filename} · {item.detected_at} · {item.id[:6]}": item.id
        for item in analyses
    }
    if "observations_compare_pick" in st.session_state:
        st.multiselect(
            "Choose saved observations to compare",
            list(labels.keys()),
            key="observations_compare_pick",
        )
    else:
        st.multiselect(
            "Choose saved observations to compare",
            list(labels.keys()),
            default=list(labels.keys())[: min(2, len(labels))],
            key="observations_compare_pick",
        )
    chosen = st.session_state.get("observations_compare_pick", [])
    selected_ids = [labels[label] for label in chosen if label in labels]
    full = [item for item in (store.get_analysis(i, user.id) for i in selected_ids) if item]

    if len(full) < 2:
        st.caption("Select at least two observations to compare.")
        return

    comparison = obs_compare.compare_analyses(full)
    notice(st, "info", "About this comparison", comparison["note"], icon="balance")
    st.dataframe(obs_compare.comparison_rows(full), hide_index=True, width="stretch")
    if comparison["labels_in_every_observation"]:
        st.markdown(
            "**Labels present in every selected observation:** "
            + ", ".join(
                display_class_name(label)
                for label in comparison["labels_in_every_observation"]
            )
        )
    st.download_button(
        "Download comparison CSV",
        data=obs_compare.comparison_csv(full),
        file_name="blueeye_comparison.csv",
        mime="text/csv",
        icon=":material/download:",
        width="stretch",
    )
