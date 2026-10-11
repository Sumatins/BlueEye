"""Tests for the Marine Areas Map page and its geographic data layer.

Structural and data-honesty tests, in the style of the other page suites:
no Streamlit server and no models are started. They assert that

* the Map is wired into the navigation exactly once, under "Explore";
* the region catalogue is well-formed (no invented or nonsensical bounds);
* species are attributed to regions strictly from the catalogue's own
  ``distribution`` text - never invented - including two regions that
  intentionally have no records yet (honest empty states);
* the optional OBIS client builds a valid bounding-box query, parses
  occurrence payloads without fabricating counts, and links records to
  catalogue profiles only on an exact scientific-name match;
* the page itself renders the required honesty markers and rejects
  fabricated demo data.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from app.config import PROJECT_ROOT
from app.content.regions import (
    REGIONS,
    get_region,
    match_species_by_scientific_name,
    normalize_scientific_name,
    obis_query_url,
    obis_web_search_url,
    parse_obis_occurrences,
    region_species,
    regions_with_species,
    search_regions,
)
from app.ui import theme
from app.ui.streamlit_app import PAGE_MODULES

_PAGE_SOURCE = (PROJECT_ROOT / "app" / "ui" / "pages" / "map_page.py").read_text(
    encoding="utf-8"
)


# --------------------------------------------------------------------------- #
# Navigation wiring
# --------------------------------------------------------------------------- #
def test_map_is_a_single_explore_navigation_entry() -> None:
    ids = [page_id for page_id, _, _ in theme.PAGES]
    assert ids.count("map") == 1
    # Map sits alongside the other exploration tools, after Species Explorer.
    assert ids.index("species") < ids.index("map")
    labels = [label for _, label, _ in theme.PAGES]
    assert "Map" in labels


def test_map_is_under_explore_group() -> None:
    grouped = {title: ids for title, ids in theme.NAV_GROUPS}
    assert "map" in grouped["Explore"]


def test_map_page_has_render_module() -> None:
    assert PAGE_MODULES["map"] is not None


# --------------------------------------------------------------------------- #
# Region catalogue
# --------------------------------------------------------------------------- #
def test_regions_are_well_formed() -> None:
    ids = [region.id for region in REGIONS]
    assert len(ids) == len(set(ids))
    for region in REGIONS:
        assert region.name and region.kind and region.note
        assert region.search_terms, region.id
        assert region.area_labels, region.id
        lat, lon = region.center
        assert -90 <= lat <= 90, region.id
        assert -180 <= lon <= 180, region.id
        assert region.min_lat < region.max_lat, region.id
        assert region.min_lon < region.max_lon, region.id
        assert -90 <= region.min_lat and region.max_lat <= 90, region.id
        assert -180 <= region.min_lon and region.max_lon <= 180, region.id


def test_species_attribution_comes_only_from_catalogue_text() -> None:
    """Every attribution must be traceable to the species' distribution text."""
    assert REGIONS, "expected at least one marine region"
    for region in REGIONS:
        for species, labels in region_species(region):
            text = species.distribution.lower()
            for label in labels:
                assert (
                    label.lower() in text
                ), f"{species.id} attributed to {region.id} via {label!r}, absent from its distribution text"


def test_known_regional_records_are_present() -> None:
    def ids_of(region_id: str) -> set[str]:
        region = get_region(region_id)
        assert region is not None
        return {species.id for species, _ in region_species(region)}

    assert {"green_sea_turtle", "humphead_wrasse", "reef_manta_ray"} <= ids_of(
        "lakshadweep"
    )
    assert {"dugong", "leatherback_turtle"} <= ids_of("andaman_nicobar")
    assert "sandfish_sea_cucumber" in ids_of("gulf_of_mannar")
    assert "whale_shark" in ids_of("gulf_of_kutch")
    assert {"mud_crab", "hilsa"} <= ids_of("sundarbans_mangroves")
    assert {"olive_ridley_turtle", "irrawaddy_dolphin"} <= ids_of("odisha_coast")
    assert {"indian_oil_sardine", "scalloped_hammerhead"} <= ids_of("arabian_sea")
    assert {"scalloped_hammerhead", "irrawaddy_dolphin"} <= ids_of("bay_of_bengal")


def test_regions_without_catalogue_records_are_honestly_empty() -> None:
    """Some areas have no catalogue records yet - that must stay an empty state."""
    for region_id in ("kerala_coast", "tamil_nadu_coast"):
        assert region_species(get_region(region_id)) == []
    assert all(len(region_species(region)) >= 1 for region in regions_with_species())
    assert regions_with_species()


def test_search_regions_matches_by_name_and_alias() -> None:
    assert {region.id for region in search_regions("kutch")} == {"gulf_of_kutch"}
    assert "gulf_of_mannar" in {region.id for region in search_regions("mannar")}
    assert "gulf_of_mannar" in {region.id for region in search_regions("palk")}
    assert "odisha_coast" in {region.id for region in search_regions("gahirmatha")}
    assert "sundarbans_mangroves" in {region.id for region in search_regions("mangrove")}
    assert search_regions("atlantis-does-not-exist") == []
    assert search_regions("") == []


# --------------------------------------------------------------------------- #
# OBIS live observations (georeferenced records)
# --------------------------------------------------------------------------- #
def test_obis_urls_use_the_regions_bounding_box() -> None:
    region = get_region("gulf_of_mannar")
    assert region is not None
    api_url = obis_query_url(region)
    assert api_url.startswith("https://api.obis.org/v3/occurrence?")
    assert "geometry=POLYGON%28%28" in api_url
    assert "78.0+8.0" in api_url  # min_lon min_lat corner
    assert "size=500" in api_url
    web_url = obis_web_search_url(region)
    assert web_url.startswith("https://obis.org/occurrence/search?geometry=")


def test_obis_occurrence_parsing_is_aggregated_honestly() -> None:
    payload = {
        "total": 5,
        "results": [
            {
                "scientificName": "Chelonia mydas",
                "decimalLatitude": 9.1,
                "decimalLongitude": 79.2,
                "date_year": 2018,
            },
            {
                "scientificName": "Chelonia mydas (Linnaeus, 1758)",
                "decimalLatitude": 9.2,
                "decimalLongitude": 79.3,
                "date_year": 2019,
            },
            {"scientificName": "Halophila ovalis", "date_year": 2005},
            {"scientificName": ""},
            {
                "scientificName": "Halophila ovalis",
                "decimalLatitude": 9.0,
                "decimalLongitude": 79.0,
                "date_year": "2002",
            },
        ],
    }
    summary = parse_obis_occurrences(payload, limit=5)
    assert summary["total"] == 5
    assert summary["downloaded"] == 5
    assert len(summary["points"]) == 3  # only the three records with coordinates
    by_name = {row["name"]: row for row in summary["top"]}

    assert by_name["Chelonia mydas"]["count"] == 2
    assert by_name["Chelonia mydas"]["recent_year"] == 2019
    assert by_name["Chelonia mydas"]["species"].id == "green_sea_turtle"

    assert by_name["Halophila ovalis"]["count"] == 2
    assert by_name["Halophila ovalis"]["recent_year"] == 2005
    assert by_name["Halophila ovalis"]["species"] is None  # no catalogue profile

    assert by_name["Unidentified"]["count"] == 1
    assert summary["top"][0]["name"] == "Chelonia mydas"  # sorted by count


def test_obis_parsing_handles_empty_and_bad_payloads() -> None:
    assert parse_obis_occurrences(None)["top"] == []
    assert parse_obis_occurrences({})["total"] == 0
    empty = parse_obis_occurrences({"results": [], "total": 0})
    assert empty["downloaded"] == 0 and empty["points"] == []


def test_scientific_name_matching_is_exact() -> None:
    assert normalize_scientific_name("Chelonia mydas (Linnaeus, 1758)") == "Chelonia mydas"
    assert match_species_by_scientific_name("Chelonia mydas").id == "green_sea_turtle"
    assert (
        match_species_by_scientific_name("Chelonia mydas (Linnaeus, 1758)").id
        == "green_sea_turtle"
    )
    assert match_species_by_scientific_name("Ehrenbergina carinata") is None
    assert match_species_by_scientific_name("") is None


# --------------------------------------------------------------------------- #
# Page wiring / honesty
# --------------------------------------------------------------------------- #
def test_map_page_source_keeps_data_honest() -> None:
    for marker in (
        "OBIS",
        "CC0",
        "never geotagged",
        "BlueEye educational catalogue",
        "st.pydeck_chart",
        "on_select",
        "region_species",
        "search_regions",
        "?page=species",
        "not verified by BlueEye",
    ):
        assert marker in _PAGE_SOURCE, f"missing honesty marker: {marker}"
    for fabricated in ("demo_count", "sample_count", "Lorem ipsum", "fake"):
        assert fabricated not in _PAGE_SOURCE, f"page still contains: {fabricated}"


def test_region_data_module_has_no_streamlit_dependency() -> None:
    """The data layer stays importable outside Streamlit (pure data + logic)."""
    source = (PROJECT_ROOT / "app" / "content" / "regions.py").read_text(encoding="utf-8")
    assert "import streamlit" not in source