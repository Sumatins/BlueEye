"""Saved observation detail view (identification review and evidence).

This module opens an **already saved** record and renders it from the database
and the private media it references. It never runs inference: opening or
exporting a record simply re-reads what was stored when it was saved.

Everything is account-scoped. A record can only be loaded when its owning
account id matches the signed-in user, and every media file is re-checked with
:func:`app.ui.state.owned_media` before it is rendered or offered for download.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from app.content.species import species_for_class
from app.detection.results import display_class_name
from app.observations import export as obs_export
from app.observations.models import REVIEW_STATUS_LABELS, Analysis, ObservationError
from app.ui import state
from app.ui.components import (
    badge,
    download_button,
    error_card,
    hero,
    kv,
    metrics_row,
    notice,
)
from app.ui.theme import esc
from app.utils.file_utils import read_file_bytes

logger = logging.getLogger(__name__)

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".json": "application/json",
}


def render(settings, user, analysis_id: str) -> None:
    """Render one saved observation (returns with an error card when denied)."""
    if user is None:
        error_card(st, "Sign in required", "Sign in to open saved observations.")
        return
    store = state.observation_store(settings)
    analysis = store.get_analysis(analysis_id, user.id)
    if analysis is None:
        error_card(
            st,
            "Observation not found",
            "This observation does not exist, was deleted, or belongs to another account.",
            hint="Saved observations are private to the account that created them.",
            log=False,
        )
        _back_button()
        return

    _flash()
    _header(analysis)
    _evidence(analysis, settings, user)
    _detections_section(analysis)
    _identification(analysis)
    _review_controls(settings, analysis, user)
    _corrections_section(settings, analysis, user)
    _research_metadata(settings, analysis, user)
    _collections_section(settings, analysis, user)
    _downloads(analysis, settings, user)
    _exports(analysis)
    _delete_section(settings, analysis, user)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _back_button(key: str = "observation_back") -> None:
    if st.button("Back to My Observations", icon=":material/arrow_back:", key=key):
        st.session_state.pop("observation_open", None)
        st.rerun()


def _flash() -> None:
    message = st.session_state.pop("observation_flash", None)
    if message:
        st.success(message)


def _set_flash(message: str) -> None:
    st.session_state["observation_flash"] = message


def _status_badge(status: str) -> str:
    label, kind = REVIEW_STATUS_LABELS.get(status, (status, "info"))
    return badge(label, kind, dot=True)


def _human(value: object) -> str:
    return "unavailable" if value in (None, "") else str(value)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
def _header(analysis: Analysis) -> None:
    _back_button()
    hero(
        st,
        "Saved observation",
        analysis.source_filename,
        f"{analysis.detected_at or 'timestamp unavailable'}"
        f"{' · ' + analysis.project if analysis.project else ''}",
        icon="bookmarks",
    )
    st.markdown(_status_badge(analysis.review_status), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
def _evidence(analysis: Analysis, settings, user) -> None:
    st.markdown("### Evidence")
    original = state.owned_media(settings, user.id, analysis.original_path)
    annotated = state.owned_media(settings, user.id, analysis.annotated_path)

    columns = st.columns(2)
    with columns[0]:
        st.markdown("**Original media**")
        _render_media(original, analysis.kind)
    with columns[1]:
        st.markdown("**Annotated result (model output)**")
        _render_media(annotated, analysis.kind)

    if original is None or annotated is None:
        notice(
            st,
            "warning",
            "Some evidence is no longer on disk",
            "A referenced file is missing or is not readable by this account. "
            "The saved metadata and detections are still shown below.",
            icon="folder_off",
        )


def _render_media(path: Path | None, kind: str) -> None:
    if path is None:
        st.caption("Not available.")
        return
    try:
        data = read_file_bytes(path)
    except OSError:  # pragma: no cover - defensive
        st.caption("Not available.")
        return
    if kind == "video":
        st.video(data)
    else:
        st.image(data, width="stretch")


# --------------------------------------------------------------------------- #
# Detections
# --------------------------------------------------------------------------- #
def _detection_rows(analysis: Analysis) -> list[dict]:
    rows = []
    for index, det in enumerate(analysis.detections, start=1):
        latest = det.latest_correction
        rows.append(
            {
                "#": index,
                "AI prediction": display_class_name(det.class_name),
                "Confidence": f"{det.confidence * 100:.1f}%",
                "Corrected label": display_class_name(latest.corrected_label) if latest else "—",
                "Bounding box": "—" if det.bbox is None else ", ".join(str(v) for v in det.bbox),
                "Frame": "—" if det.frame_index is None else det.frame_index,
                "Model": det.model or "—",
            }
        )
    return rows


def _detections_section(analysis: Analysis) -> None:
    st.markdown("### Detected objects")
    if not analysis.detections:
        st.caption("No individual objects were recorded for this run.")
        return
    st.dataframe(_detection_rows(analysis), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# Identification
# --------------------------------------------------------------------------- #
def _species_note(class_id: str) -> str | None:
    profile = species_for_class(class_id)
    if profile is None or profile.detection != "direct":
        return None
    return f"{profile.common_name} ({profile.scientific_name})"


def _identification(analysis: Analysis) -> None:
    st.markdown("### Identification")
    metrics_row(
        st,
        [
            ("Total objects", analysis.total_detections, "above threshold"),
            ("Distinct classes", analysis.unique_classes, "predicted"),
            ("Average confidence", f"{analysis.average_confidence * 100:.1f}%", ""),
            ("Max confidence", f"{analysis.max_confidence * 100:.1f}%", ""),
        ],
    )

    distinct: list[str] = []
    for det in analysis.detections:
        if det.class_name not in distinct:
            distinct.append(det.class_name)

    species_lines = [name for class_id in distinct if (name := _species_note(class_id))]
    broad_lines = [display_class_name(c) for c in distinct if not _species_note(c)]

    if species_lines:
        st.markdown("**Species names supported by the model:** " + "; ".join(species_lines))
    if broad_lines:
        st.markdown(
            "**Broad classes (species not established):** " + ", ".join(broad_lines)
        )
    if not distinct:
        st.caption("No individual classes were recorded for this run.")

    st.markdown(
        kv(
            [
                ("Model(s)", ", ".join(analysis.models) or "unavailable"),
                ("Model version", analysis.model_version or "unavailable"),
                ("Detection timestamp", analysis.detected_at or "unavailable"),
                ("Saved", analysis.created_at or "unavailable"),
                ("Media type", analysis.kind),
                (
                    "Confidence threshold",
                    "model default"
                    if analysis.confidence_threshold is None
                    else f"{analysis.confidence_threshold:.3f}",
                ),
                ("Enhanced preprocessing", "yes" if analysis.enhanced else "no"),
                (
                    "Review status",
                    REVIEW_STATUS_LABELS.get(analysis.review_status, ("unknown",))[0],
                ),
            ]
        ),
        unsafe_allow_html=True,
    )

    notice(
        st,
        "info",
        "These are model predictions",
        "Labels here are AI detection outputs, not confirmed biological "
        "identifications. A model that only supports a broad class (for example "
        "'Fish') does not establish a species.",
        icon="info",
    )


# --------------------------------------------------------------------------- #
# Review status
# --------------------------------------------------------------------------- #
def _review_controls(settings, analysis: Analysis, user) -> None:
    st.markdown("### Review workflow")
    store = state.observation_store(settings)
    columns = st.columns(2)
    with columns[0]:
        if st.button(
            "Mark needs review",
            key=f"review_needs_{analysis.id}",
            icon=":material/flag:",
            width="stretch",
        ):
            store.update_analysis(analysis.id, user.id, review_status="needs_review")
            _set_flash("Marked as needing review.")
            st.rerun()
    with columns[1]:
        if st.button(
            "Reset to AI-predicted",
            key=f"review_reset_{analysis.id}",
            icon=":material/restart_alt:",
            width="stretch",
        ):
            if analysis.review_status == "verified" and not settings.is_reviewer(user.email):
                st.error("Only a configured reviewer can change a verified record.")
            else:
                store.update_analysis(analysis.id, user.id, review_status="predicted")
                _set_flash("Review status reset to AI-predicted.")
                st.rerun()
    if analysis.review_status == "corrected":
        st.caption("Contains a user correction. It is not a verified identification.")
    elif analysis.review_status == "verified":
        st.caption("This identification was verified by an authorised reviewer.")
    else:
        st.caption(
            "Only a configured reviewer can mark a record as a verified identification."
        )


# --------------------------------------------------------------------------- #
# Corrections
# --------------------------------------------------------------------------- #
def _corrections_section(settings, analysis: Analysis, user) -> None:
    st.markdown("### Review and correction")
    existing = [correction for det in analysis.detections for correction in det.corrections]
    if existing:
        st.markdown("**Correction history** (original predictions are preserved):")
        for correction in existing:
            note = f" — {esc(correction.note)}" if correction.note else ""
            st.markdown(
                f"- {esc(display_class_name(correction.original_label))} → "
                f"**{esc(display_class_name(correction.corrected_label))}** "
                f"({esc(correction.created_at)}){note}",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No user corrections yet; the model predictions are unchanged.")

    if not analysis.detections:
        st.caption("Nothing to correct: no individual objects were recorded.")
        return

    options = {
        f"#{i} · {display_class_name(det.class_name)} · {det.confidence * 100:.0f}%": det.id
        for i, det in enumerate(analysis.detections, start=1)
    }
    choice = st.selectbox(
        "Object to correct", list(options.keys()), key=f"correct_pick_{analysis.id}"
    )
    corrected = st.text_input(
        "Corrected label (class id or free text)",
        key=f"correct_label_{analysis.id}",
        placeholder="e.g. serranidae",
    )
    note = st.text_input("Correction note (optional)", key=f"correct_note_{analysis.id}")
    verify = False
    if settings.is_reviewer(user.email):
        verify = st.checkbox(
            "Mark this correction as a verified identification",
            key=f"correct_verify_{analysis.id}",
            help="Only configured reviewers can mark a record as verified.",
        )
    if st.button("Save correction", key=f"correct_save_{analysis.id}", icon=":material/edit:"):
        if not corrected.strip():
            st.warning("Enter the corrected label first.")
            return
        try:
            state.observation_store(settings).add_correction(
                options[choice], user.id, corrected.strip(), note.strip(), verified=verify
            )
        except ObservationError as exc:
            st.error(str(exc))
            return
        _set_flash("Correction saved. The original AI prediction was kept unchanged.")
        st.rerun()


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def _research_metadata(settings, analysis: Analysis, user) -> None:
    st.markdown("### Research notes and metadata")
    with st.form(key=f"metadata_form_{analysis.id}"):
        project = st.text_input(
            "Survey / project", value=analysis.project, key=f"meta_proj_{analysis.id}"
        )
        location = st.text_input(
            "Collection location (optional)",
            value=analysis.location,
            key=f"meta_loc_{analysis.id}",
            help="Private to your account.",
        )
        habitat = st.text_input(
            "Habitat (optional)", value=analysis.habitat, key=f"meta_hab_{analysis.id}"
        )
        notes = st.text_area(
            "Notes", value=analysis.notes, key=f"meta_notes_{analysis.id}", height=120
        )
        submitted = st.form_submit_button("Save metadata", icon=":material/save:")
    if submitted:
        state.observation_store(settings).update_analysis(
            analysis.id,
            user.id,
            project=project,
            location=location,
            habitat=habitat,
            notes=notes,
        )
        _set_flash("Observation metadata saved.")
        st.rerun()


# --------------------------------------------------------------------------- #
# Collections
# --------------------------------------------------------------------------- #
def _collections_section(settings, analysis: Analysis, user) -> None:
    st.markdown("### Collections and surveys")
    store = state.observation_store(settings)
    collections = store.list_collections(user.id)
    # Current membership requires the item list per collection.
    current_names = []
    for collection in collections:
        analyses = store.list_collection_analyses(collection.id, user.id)
        if analysis.id in {item.id for item in analyses}:
            current_names.append(collection.name)

    if collections:
        selected = st.multiselect(
            "Assign to collections",
            options=[collection.name for collection in collections],
            default=current_names,
            key=f"assign_{analysis.id}",
        )
        if st.button(
            "Update collections",
            key=f"assign_save_{analysis.id}",
            icon=":material/bookmark:",
        ):
            by_name = {collection.name: collection for collection in collections}
            chosen = set(selected)
            for name in chosen - set(current_names):
                store.add_to_collection(by_name[name].id, analysis.id, user.id)
            for name in set(current_names) - chosen:
                store.remove_from_collection(by_name[name].id, analysis.id, user.id)
            _set_flash("Collection membership updated.")
            st.rerun()
    else:
        st.caption("You have no collections yet. Create one below to group observations.")

    with st.form(key=f"new_collection_{analysis.id}"):
        name = st.text_input("New collection name", key=f"new_col_name_{analysis.id}")
        description = st.text_input(
            "Description (optional)", key=f"new_col_desc_{analysis.id}"
        )
        created = st.form_submit_button(
            "Create and add", icon=":material/create_new_folder:"
        )
    if created:
        if not name.strip():
            st.warning("Enter a collection name first.")
        else:
            collection = store.create_collection(user.id, name.strip(), description.strip())
            store.add_to_collection(collection.id, analysis.id, user.id)
            _set_flash(f"Created '{collection.name}' and added this observation.")
            st.rerun()


# --------------------------------------------------------------------------- #
# Downloads / exports
# --------------------------------------------------------------------------- #
def _downloads(analysis: Analysis, settings, user) -> None:
    st.markdown("### Download evidence")
    original = state.owned_media(settings, user.id, analysis.original_path)
    annotated = state.owned_media(settings, user.id, analysis.annotated_path)
    report = state.owned_media(settings, user.id, analysis.report_path)
    columns = st.columns(3)
    with columns[0]:
        if original is not None:
            download_button(st, "Original media", original, _mime_for(original))
        else:
            st.caption("Original media unavailable.")
    with columns[1]:
        if annotated is not None:
            download_button(st, "Annotated result", annotated, _mime_for(annotated))
        else:
            st.caption("Annotated result unavailable.")
    with columns[2]:
        if report is not None:
            download_button(st, "Detection JSON", report, "application/json")
        else:
            st.caption("Report unavailable.")


def _mime_for(path: Path) -> str:
    return _MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")


def _exports(analysis: Analysis) -> None:
    st.markdown("### Export this observation")
    payload = obs_export.analysis_to_dict(analysis)
    columns = st.columns(3)
    with columns[0]:
        st.download_button(
            "Download JSON",
            data=obs_export.json_dumps(payload),
            file_name=f"observation_{analysis.id[:8]}.json",
            mime="application/json",
            icon=":material/download:",
            width="stretch",
        )
    with columns[1]:
        st.download_button(
            "Download CSV",
            data=obs_export.analysis_csv(analysis),
            file_name=f"observation_{analysis.id[:8]}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
    with columns[2]:
        st.download_button(
            "Download HTML report",
            data=obs_export.analysis_report_html(analysis),
            file_name=f"observation_{analysis.id[:8]}.html",
            mime="text/html",
            icon=":material/download:",
            width="stretch",
        )
    st.caption(
        "The HTML report is self-contained and can be printed to PDF from any browser."
    )


# --------------------------------------------------------------------------- #
# Delete
# --------------------------------------------------------------------------- #
def _delete_section(settings, analysis: Analysis, user) -> None:
    st.markdown("### Delete this observation")
    confirm = st.checkbox(
        "I understand this permanently removes the saved record",
        key=f"delete_confirm_{analysis.id}",
    )
    purge = st.checkbox(
        "Also delete the stored media and report files",
        key=f"delete_files_{analysis.id}",
    )
    if st.button(
        "Delete observation",
        key=f"delete_{analysis.id}",
        icon=":material/delete:",
        disabled=not confirm,
    ):
        store = state.observation_store(settings)
        removed = store.delete_analysis(analysis.id, user.id)
        if not removed:
            st.error("That record could not be found for your account.")
            return
        if purge:
            _purge_files(settings, user.id, analysis)
        st.session_state.pop("observation_open", None)
        _set_flash(
            "Observation deleted." + (" Media files were also removed." if purge else "")
        )
        st.rerun()


def _purge_files(settings, user_id: str, analysis: Analysis) -> None:
    for path in (analysis.annotated_path, analysis.report_path, analysis.original_path):
        owned = state.owned_media(settings, user_id, path)
        if owned is not None:
            try:
                Path(owned).unlink(missing_ok=True)
            except OSError:  # pragma: no cover - defensive
                logger.warning("Could not remove observation media %s", owned)
