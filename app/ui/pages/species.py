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
from app.ui.icons import span
from app.ui.theme import esc

#: Badge colour per detection capability.
_DETECTION_KIND = {
    "direct": "ok",
    "broad": "info",
    "none": "off",
    "unverified": "warn",
}

_SELECTED_KEY = "species_selected"

#: Material icon used as the photo fallback tile, per animal group.
_GROUP_ICONS: dict[str, str] = {
    "Aquatic reptile": "pets",
    "Coastal & marine fish": "set_meal",
    "Freshwater fish": "set_meal",
    "Freshwater shark": "set_meal",
    "Freshwater turtle": "pets",
    "Invertebrate": "bug_report",
    "Reef fish": "set_meal",
    "River dolphin & cetacean": "pets",
    "Sea turtle": "pets",
    "Shark & ray": "waves",
}


def _photo(species: Species, modifier: str = "") -> str:
    """Species photograph with a graceful icon fallback.

    The image lazy-loads and, if it ever fails (offline / moved file),
    ``onerror`` flips the container to its fallback class so the browser's
    broken-image placeholder is replaced by the group icon tile. A species
    with no verified image simply shows the icon tile - never a fake photo.
    """
    icon = span(_GROUP_ICONS.get(species.group, "photo_camera"))
    fallback = f'<div class="be-photo-fallback">{icon}</div>'
    classes = f"be-photo {modifier}".strip()
    if not species.image:
        return f'<div class="{classes} be-photo--missing">{fallback}</div>'
    return (
        f'<div class="{classes}">'
        f'<img src="{esc(species.image)}" alt="Photograph of {esc(species.common_name)}" '
        'loading="lazy" referrerpolicy="no-referrer" '
        f'title="Photo: {esc(species.image_credit or "Wikimedia Commons")}" '
        "onerror=\"this.closest('.be-photo').classList.add('be-photo--missing')\">"
        f"{fallback}</div>"
    )


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
        '<div class="be-card be-species-card">'
        f"{_photo(species)}"
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
    _back_button(key="species_back")

    st.markdown(
        '<div class="be-hero">'
        f'<div class="be-tag">{span("menu_book")}{esc(species.group)}</div>'
        f"<h1>{esc(species.common_name)}</h1>"
        f'<p class="sci"><em>{esc(species.scientific_name)}</em></p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div>{chips((species.habitat, species.group))} "
        f"{_detection_badge(species.detection)}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(_photo(species, "be-photo--wide"), unsafe_allow_html=True)
    if species.image and species.image_credit:
        st.caption(f"Photo: {species.image_credit} (via Wikimedia Commons)")

    _heading("Overview")
    st.markdown(f"<p>{esc(species.overview)}</p>", unsafe_allow_html=True)
    _two_columns(
        [("Identification", species.identification)],
        [("Habitat", species.habitat_notes)],
    )

    _heading("Habitat and distribution")
    st.markdown(f"<p>{esc(species.distribution)}</p>", unsafe_allow_html=True)

    _heading("Diet and behaviour")
    _two_columns(
        [("Diet", species.diet)],
        [("Behaviour", species.behaviour)],
    )

    _heading("Ecological importance")
    st.markdown(f"<p>{esc(species.ecological_role)}</p>", unsafe_allow_html=True)

    _heading("Conservation")
    _two_columns(
        [("Threats", species.threats)],
        [
            ("Conservation status", species.conservation_status),
            ("Status source", species.conservation_source),
        ],
    )
    _field("Conservation actions", species.conservation_actions)

    _heading("Interesting facts")
    facts = "".join(f"<li>{esc(fact)}</li>" for fact in species.facts)
    st.markdown(f"<ul class='be-facts'>{facts}</ul>", unsafe_allow_html=True)

    _heading("Can BlueEye detect this species?")
    st.markdown(
        f"<div class='be-card'><div class='be-card-title'>"
        f"{_detection_badge(species.detection)} "
        f"{esc(species.detection_label)}</div>"
        f"<p class='be-card-sub'>{esc(species.detection_note)}</p></div>",
        unsafe_allow_html=True,
    )

    _heading("References")
    _references(species.references)

    _back_button(key="species_back_bottom", stretch=True)


def _heading(text: str) -> None:
    st.markdown(f"### {esc(text)}")


def _back_button(key: str, stretch: bool = False) -> None:
    if st.button(
        "Back to species list",
        icon=":material/arrow_back:",
        key=key,
        width="stretch" if stretch else "content",
    ):
        st.session_state.pop(_SELECTED_KEY, None)
        st.rerun()


def _two_columns(left: list[tuple[str, str]], right: list[tuple[str, str]]) -> None:
    left_column, right_column = st.columns(2)
    with left_column:
        for label, value in left:
            _field(label, value)
    with right_column:
        for label, value in right:
            _field(label, value)


def _field(label: str, value: str) -> None:
    st.markdown(
        '<div class="be-card be-card--flush">'
        f"<div class='be-card-sub' style='margin:0 0 .15rem'>{esc(label)}</div>"
        f"<div>{esc(value)}</div></div>",
        unsafe_allow_html=True,
    )


def _references(references: tuple[tuple[str, str], ...]) -> None:
    """Render the verified source links for a species profile."""
    if not references:
        st.caption("No external references recorded for this species.")
        return
    items = "".join(
        f'<li><a href="{esc(url)}" target="_blank" rel="noopener noreferrer">'
        f"{esc(label)}</a></li>"
        for label, url in references
    )
    st.markdown(f"<ul class='be-reflist'>{items}</ul>", unsafe_allow_html=True)


__all__ = ["render"]
