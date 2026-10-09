"""Tests for the Species Explorer reference library and its page wiring.

These are data/structural tests: they never load a model or start Streamlit.
They enforce the honesty rule — every species must carry a real detection
capability and a note explaining it — and they check that search and the
habitat / group filters behave.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from app.config import PROJECT_ROOT
from app.content.species import (
    DETECTION_CATEGORIES,
    DETECTION_ORDER,
    SPECIES,
    all_groups,
    all_habitats,
    filter_species,
    get_species,
    search_species,
)
from app.ui import theme
from app.ui.streamlit_app import PAGE_MODULES

_PAGE_SOURCE = (PROJECT_ROOT / "app" / "ui" / "pages" / "species.py").read_text(
    encoding="utf-8"
)


# --------------------------------------------------------------------------- #
# Library content
# --------------------------------------------------------------------------- #
def test_library_has_at_least_30_species() -> None:
    assert len(SPECIES) >= 30


def test_ids_are_unique_and_well_formed() -> None:
    ids = [species.id for species in SPECIES]
    assert len(ids) == len(set(ids))
    assert all(species.id and species.id.isascii() for species in SPECIES)


def test_every_species_has_its_required_facts() -> None:
    for species in SPECIES:
        assert species.common_name.strip()
        assert species.scientific_name.strip()
        assert species.group.strip()
        assert species.habitat.strip()
        assert species.distribution.strip()
        assert species.appearance.strip()
        assert species.diet.strip()
        assert species.ecological_role.strip()
        assert species.conservation_status.strip()
        assert species.conservation_source.strip()
        assert species.interesting_fact.strip()


def test_detection_capability_is_honest() -> None:
    for species in SPECIES:
        assert species.detection in DETECTION_CATEGORIES
        # A capability claim must always be explained.
        assert species.detection_note.strip()
        # A generic detector must never be presented as species-level support.
        if species.detection == "broad":
            assert "generic" in species.detection_note.lower() or "broad" in (
                species.detection_note.lower()
            ) or "cannot" in species.detection_note.lower()


def test_detection_categories_have_labels_and_order() -> None:
    assert set(DETECTION_ORDER) == set(DETECTION_CATEGORIES)
    assert all(DETECTION_CATEGORIES[key].strip() for key in DETECTION_ORDER)


def test_indian_priority_species_are_present() -> None:
    names = " | ".join(species.common_name.lower() for species in SPECIES)
    for required in (
        "ganges river dolphin",
        "gharial",
        "dugong",
        "whale shark",
        "olive ridley sea turtle",
    ):
        assert required in names, f"missing priority species: {required}"


def test_only_verified_classes_are_claimed_as_direct() -> None:
    """'direct' must map to a class that really exists in an active model."""
    from app.detection.model_manager import MODEL_REGISTRY, load_custom_specs
    from app.config import get_settings

    known_classes: set[str] = set()
    for spec in MODEL_REGISTRY.values():
        known_classes.update(spec.classes)
    for spec in load_custom_specs(get_settings().models_dir).values():
        known_classes.update(spec.classes)

    # Species names that map to an explicit class in a shipped registry.
    assert get_species("crown_of_thorns").detection == "direct"
    assert "crown_of_thorns" in known_classes
    assert get_species("humphead_wrasse").detection == "direct"
    assert "cheilinus_undulatus" in known_classes


# --------------------------------------------------------------------------- #
# Search + filters
# --------------------------------------------------------------------------- #
def test_search_matches_common_and_scientific_names() -> None:
    by_common = search_species("mahseer")
    assert by_common and all("mahseer" in s.common_name.lower() for s in by_common)

    by_scientific = search_species("Platanista")
    assert [s.id for s in by_scientific] == ["ganges_river_dolphin"]

    # Case-insensitive.
    assert search_species("GHARIAL") == search_species("gharial")


def test_empty_query_returns_everything() -> None:
    assert search_species("") == list(SPECIES)
    assert search_species("   ") == list(SPECIES)


def test_filter_by_habitat_and_group() -> None:
    freshwater = filter_species(habitats=["Freshwater"])
    assert freshwater
    assert all(s.habitat == "Freshwater" for s in freshwater)
    assert len(freshwater) < len(SPECIES)

    turtles = filter_species(groups=["Sea turtle"])
    assert turtles
    assert all(s.group == "Sea turtle" for s in turtles)


def test_filters_combine_with_search() -> None:
    result = filter_species("turtle", groups=["Sea turtle"])
    assert result
    assert all(s.group == "Sea turtle" for s in result)


def test_filter_with_no_match_returns_empty() -> None:
    assert filter_species("nonexistent-species-xyz") == []


def test_lookup_and_option_lists() -> None:
    assert get_species("rohu").common_name == "Rohu"
    assert get_species("does-not-exist") is None
    assert all_habitats() == sorted(set(all_habitats()))
    assert all_groups() == sorted(set(all_groups()))


# --------------------------------------------------------------------------- #
# Page wiring
# --------------------------------------------------------------------------- #
def test_species_page_is_registered_in_navigation() -> None:
    ids = [page_id for page_id, _, _ in theme.PAGES]
    assert "species" in ids
    assert PAGE_MODULES["species"] is not None


def test_species_page_target_is_under_explore() -> None:
    grouped = {title: ids for title, ids in theme.NAV_GROUPS}
    assert "species" in grouped["Explore"]


def test_species_page_has_search_filters_detail_and_back() -> None:
    for marker in (
        "Search by common or scientific name",
        "Habitat",
        "Animal group",
        "View profile",
        "Back to species list",
        "Can BlueEye detect this species?",
    ):
        assert marker in _PAGE_SOURCE, f"missing UI element: {marker}"
