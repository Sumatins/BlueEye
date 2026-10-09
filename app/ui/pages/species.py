"""Species Explorer: an educational aquatic-species reference library.

This page is **separate from model inference**. It presents verified reference
profiles (see :mod:`app.content.species`) together with an honest statement of
whether any active BlueEye model can actually detect each species. It never
claims that a generic ``fish`` / ``turtle`` / ``shark`` class identifies a
species.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import streamlit as st

from app.content.species import (
    DETECTION_CATEGORIES,
    DETECTION_ORDER,
    SPECIES,
    Species,
    all_groups,
    all_habitats,
    filter_species,
    get_species,
)
from app.detection.detector import MarineDetector
from app.ui.components import badge, chips, hero, notice
from app.ui.theme import esc

#: Badge colour per detection capability.
_DETECTION_KIND = {
    "direct": "ok",
    "broad": "info",
    "none": "off",
    "unverified": "warn",
}

_SELECTED_KEY = "species_selected"


def render(detector: MarineDetector, settings) -> None:
    """Render the explorer, or the detail view when a species is selected."""
    selected_id = st.session_state.get(_SELECTED_KEY)
    if selected_id:
        species = get_species(selected_id)
        if species is not None:
            _detail(species)
            return
        st.session_state.pop(_SELECTED_KEY, None)

    hero(
        st,
        "Reference library",
        "Species Explorer",
        "An educational guide to aquatic life, with an honest note on whether "
        "BlueEye's models can actually detect each species.",
        icon="menu_book",
    )
    _legend()
    _filters_and_grid()


# --------------------------------------------------------------------------- #
# Legend / scope
# --------------------------------------------------------------------------- #
def _legend() -> None:
    notice(
        st,
        "info",
        "This is a reference library, not a detection claim",
        "BlueEye can only detect what its installed models were trained to "
        "recognise. A generic 'fish', 'turtle' or 'shark' model does not "
        "establish a species identity.",
    )
    legend = " ".join(
        f"{_detection_badge(key)} {esc(DETECTION_CATEGORIES[key])}"
        for key in DETECTION_ORDER
    )
    st.markdown(f"<div class='be-card be-card--muted'>{legend}</div>", unsafe_allow_html=True)


def _detection_badge(key: str, dot: bool = True) -> str:
    label = DETECTION_CATEGORIES.get(key, "Unverified")
    return badge(label, _DETECTION_KIND.get(key, "info"), dot=dot)


# --------------------------------------------------------------------------- #
# Filters + grid
# --------------------------------------------------------------------------- #
def _filters_and_grid() -> None:
    filter_columns = st.columns([1.4, 1, 1])
    with filter_columns[0]:
        query = st.text_input(
            "Search by common or scientific name",
            key="species_query",
            placeholder="e.g. mahseer, Platanista, turtle",
        )
    with filter_columns[1]:
        habitats = st.multiselect(
            "Habitat", options=all_habitats(), default=[], key="species_habitat"
        )
    with filter_columns[2]:
        groups = st.multiselect(
            "Animal group", options=all_groups(), default=[], key="species_group"
        )

    matches = filter_species(query, habitats, groups)
    st.caption(f"{len(matches)} of {len(SPECIES)} species shown")

    if not matches:
        notice(
            st,
            "warning",
            "No species match your search",
            "Try a different name, or clear the habitat and group filters.",
        )
        return

    columns_per_row = 3
    for start in range(0, len(matches), columns_per_row):
        row = matches[start : start + columns_per_row]
        columns = st.columns(columns_per_row)
        for column, species in zip(columns, row):
            with column:
                _card(species)


def _card(species: Species) -> None:
    st.markdown(
        '<div class="be-card">'
        f'<div class="be-card-title">{esc(species.common_name)}</div>'
        f"<p class='be-card-sub'><em>{esc(species.scientific_name)}</em></p>"
        f"<div style='margin:.2rem 0 .45rem'>{_detection_badge(species.detection)}</div>"
        f"<div>{chips((species.group, species.habitat))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    if st.button(
        "View profile",
        key=f"species_open_{species.id}",
        icon=":material/arrow_forward:",
        width="stretch",
    ):
        st.session_state[_SELECTED_KEY] = species.id
        st.rerun()


# --------------------------------------------------------------------------- #
# Detail view
# --------------------------------------------------------------------------- #
def _detail(species: Species) -> None:
    if st.button(
        "Back to species list",
        icon=":material/arrow_back:",
        key="species_back",
    ):
        st.session_state.pop(_SELECTED_KEY, None)
        st.rerun()

    hero(
        st,
        species.group,
        species.common_name,
        species.scientific_name,
        icon="menu_book",
    )

    st.markdown(
        f"<div>{chips((species.habitat, species.group))}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p>{esc(species.appearance)}</p>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        _field("Scientific name", species.scientific_name)
        _field("Group", species.group)
        _field("Habitat", species.habitat)
        _field("Distribution (with Indian relevance)", species.distribution)
        _field("Diet", species.diet)
    with right:
        _field("Ecological role", species.ecological_role)
        _field("Conservation status", species.conservation_status)
        _field("Status source", species.conservation_source)
        _field("Interesting fact", species.interesting_fact)

    st.markdown("### Can BlueEye detect this species?")
    st.markdown(
        f"<div class='be-card'><div class='be-card-title'>"
        f"{_detection_badge(species.detection)} "
        f"{esc(species.detection_label)}</div>"
        f"<p class='be-card-sub'>{esc(species.detection_note)}</p></div>",
        unsafe_allow_html=True,
    )
    if species.image:
        st.image(
            species.image,
            caption=species.image_credit or None,
            width="stretch",
        )

    if st.button(
        "Back to species list",
        icon=":material/arrow_back:",
        key="species_back_bottom",
        width="stretch",
    ):
        st.session_state.pop(_SELECTED_KEY, None)
        st.rerun()


def _field(label: str, value: str) -> None:
    st.markdown(
        '<div class="be-card be-card--flush">'
        f"<div class='be-card-sub' style='margin:0 0 .15rem'>{esc(label)}</div>"
        f"<div>{esc(value)}</div></div>",
        unsafe_allow_html=True,
    )


__all__ = ["render"]
