"""Tests for the Map's worldwide search layer (geocoding, coordinates, boxes).

These cover the pure data logic added for global (non-India) use of the
Marine Areas Map:

* :func:`app.content.regions.parse_lat_lon` and
  :func:`app.content.regions.region_from_coordinates` - exact-coordinate areas
  anywhere on Earth, with strict validation and no silent clamping to a
  country;
* :func:`app.content.regions.parse_geocode_response`,
  :func:`app.content.regions.pick_best_place` and
  :func:`app.content.regions.region_from_place` - geocoded places become the
  same exploration model as curated areas (and point-sized matches are
  expanded, never invented);
* :func:`app.content.regions.bbox_windows` and
  :func:`app.content.regions.obis_query_urls` - boxes crossing the
  international date line are split into valid query windows.

No network calls happen in these tests (the geocoder HTTP layer itself is
cached-out and offline-tolerant; it fails to ``None`` - the tests pin the
pure parsing/geometry logic).

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from app.content.regions import (
    GeoPlace,
    MarineRegion,
    bbox_windows,
    obis_query_url,
    obis_query_urls,
    parse_geocode_response,
    parse_lat_lon,
    pick_best_place,
    region_from_coordinates,
    region_from_place,
)


def _place(
    *,
    name: str,
    lat: float,
    lon: float,
    box: tuple[float, float, float, float] | None = None,
    precision: str = "natural · reef",
) -> GeoPlace:
    """Build a ``GeoPlace`` (like one Nominatim would return)."""
    west, south, east, north = box if box else (lon, lat, lon, lat)
    return GeoPlace(
        name=name,
        display_name=f"{name}, somewhere",
        center=(lat, lon),
        west=west,
        south=south,
        east=east,
        north=north,
        source="OpenStreetMap (Nominatim)",
        precision=precision,
        key="osm-way-123",
    )


# --------------------------------------------------------------------------- #
# Exact-coordinate explorer
# --------------------------------------------------------------------------- #
def test_parse_lat_lon_accepts_sensible_formats() -> None:
    assert parse_lat_lon("12.5, 77.2") == (12.5, 77.2)
    assert parse_lat_lon("12.5 77.2") == (12.5, 77.2)
    assert parse_lat_lon("12.5;77.2") == (12.5, 77.2)
    assert parse_lat_lon("12.5N 77.2E") == (12.5, 77.2)
    assert parse_lat_lon("12.5S 77.2W") == (-12.5, -77.2)
    assert parse_lat_lon("12.5°S 45.2°W") == (-12.5, -45.2)
    assert parse_lat_lon("  -3.2, -179.9 ") == (-3.2, -179.9)


def test_parse_lat_lon_rejects_invalid_input() -> None:
    assert parse_lat_lon("") is None
    assert parse_lat_lon("not coordinates") is None
    assert parse_lat_lon("12.5") is None  # needs both values
    assert parse_lat_lon("120, 5") is None  # lat out of range
    assert parse_lat_lon("12, 200") is None  # lon out of range
    assert parse_lat_lon("12.5N 77.2E extra 9.9N") is None  # more than two numbers


def test_region_from_coordinates_is_a_valid_exploration_box() -> None:
    region = region_from_coordinates(23.8, 66.1)
    assert region.kind == "Exact coordinates"
    assert region.id.startswith("coords_")
    assert -90 <= region.min_lat < region.max_lat <= 90
    assert -180 <= region.min_lon < region.max_lon <= 180
    # The entered point is the centre of the box.
    assert abs(region.max_lat - region.min_lat - 2.0) < 1e-9  # default span
    assert 22.8 <= region.min_lat < 23.8 < region.max_lat <= 24.8


def test_region_from_coordinates_clamps_at_the_globe_edge() -> None:
    north = region_from_coordinates(89.9, 45.0)
    assert north.max_lat <= 90.0
    assert north.min_lat < north.max_lat
    south = region_from_coordinates(-89.9, -45.0)
    assert south.min_lat >= -90.0


def test_region_from_coordinates_rejects_bad_input() -> None:
    for lat, lon in ((91, 0), (-91, 0), (0, 181), (0, -181)):
        try:
            region_from_coordinates(lat, lon)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for lat={lat}, lon={lon}")


# --------------------------------------------------------------------------- #
# Geocoding (Nominatim response parsing - pure logic, no network)
# --------------------------------------------------------------------------- #
def test_parse_geocode_response_skips_invalid_rows() -> None:
    payload = [
        {"name": "Reef", "lat": "-18.2", "lon": "147.7", "display_name": "Reef, AU",
         "boundingbox": ["-18.5", "-18.0", "147.5", "148.0"],
         "category": "natural", "type": "reef", "osm_type": "way", "osm_id": 1},
        {"name": "Bad lat", "lat": "not-a-number", "lon": "10.0"},
        {"name": "Out of range", "lat": "95", "lon": "10.0"},
        "not a dict",
    ]
    places = parse_geocode_response(payload)
    assert len(places) == 1
    assert places[0].name == "Reef"
    assert places[0].west == 147.5 and places[0].east == 148.0
    assert places[0].south == -18.5 and places[0].north == -18.0
    assert places[0].precision == "natural · reef"
    assert places[0].source == "OpenStreetMap (Nominatim)"


def test_parse_geocode_response_handles_garbage() -> None:
    assert parse_geocode_response(None) == []
    assert parse_geocode_response({}) == []
    assert parse_geocode_response([{"name": "missing coordinates"}]) == []


def test_pick_best_place_prefers_a_real_area_box() -> None:
    point = _place(name="Reef point", lat=-18.2, lon=147.7)
    country = _place(
        name="Brazil",
        lat=-10.0,
        lon=-52.0,
        box=(-73.98, -33.87, -28.63, 5.27),
        precision="boundary · administrative",
    )
    assert pick_best_place([point, country]) is country
    assert pick_best_place([point]) is point
    assert pick_best_place([]) is None


def test_region_from_place_uses_the_geocoded_box() -> None:
    place = _place(
        name="Brazil",
        lat=-10.0,
        lon=-52.0,
        box=(-73.98, -33.87, -28.63, 5.27),
        precision="boundary · administrative",
    )
    region = region_from_place(place)
    assert region.id == "geo_brazil"
    assert region.area_labels == ("Brazil",)
    assert (region.min_lon, region.min_lat) == (-73.98, -33.87)
    assert (region.max_lon, region.max_lat) == (-28.63, 5.27)
    assert region.center == (-10.0, -52.0)


def test_region_from_place_expands_point_sized_matches() -> None:
    point = _place(name="Great Barrier Reef", lat=-18.2, lon=147.7)
    region = region_from_place(point)
    assert region.min_lon < region.max_lon
    assert region.min_lat < region.max_lat
    span_lat = region.max_lat - region.min_lat
    span_lon = region.max_lon - region.min_lon
    assert span_lat >= 2.0 and span_lon >= 2.0
    assert region.area_labels == ("Great Barrier Reef",)


# --------------------------------------------------------------------------- #
# Boxes crossing the international date line
# --------------------------------------------------------------------------- #
def _crossing_region() -> MarineRegion:
    return MarineRegion(
        id="geo_fiji",
        name="Fiji",
        kind="boundary · administrative",
        note="Crosses the date line.",
        search_terms=(),
        area_labels=("Fiji",),
        center=(0.0, 178.0),
        min_lat=-19.0,
        max_lat=20.0,
        min_lon=175.0,  # west bound east of the east bound => crosses 180°
        max_lon=-179.0,
    )


def test_bbox_windows_splits_antimeridian_boxes() -> None:
    windows = bbox_windows(_crossing_region())
    assert windows == [(175.0, 180.0), (-180.0, -179.0)]
    plain = MarineRegion(
        id="x", name="x", kind="k", note="", search_terms=(), area_labels=(),
        center=(10.0, 70.0), min_lat=5.0, max_lat=15.0, min_lon=65.0, max_lon=75.0,
    )
    assert bbox_windows(plain) == [(65.0, 75.0)]


def test_obis_query_urls_covers_each_window() -> None:
    urls = obis_query_urls(_crossing_region(), size=500)
    assert len(urls) == 2
    for url in urls:
        assert url.startswith("https://api.obis.org/v3/occurrence?")
        assert "size=500" in url
    assert "POLYGON%28%28175.0" in urls[0]
    assert "POLYGON%28%28-180.0" in urls[1]


def test_obis_query_url_uses_first_window_for_simple_boxes() -> None:
    plain = MarineRegion(
        id="x", name="x", kind="k", note="", search_terms=(), area_labels=(),
        center=(10.0, 70.0), min_lat=5.0, max_lat=15.0, min_lon=65.0, max_lon=75.0,
    )
    assert obis_query_urls(plain) == [obis_query_url(plain)]
    assert "65.0+5.0" in obis_query_url(plain)  # min_lon min_lat corner


def test_merge_payloads_sums_totals_and_concatenates_results() -> None:
    from app.content.regions import _merge_payloads

    merged = _merge_payloads(
        [
            {"total": 3, "results": [{"scientificName": "A"}]},
            {"total": 4, "results": [{"scientificName": "B"}, {"scientificName": "C"}]},
            "garbage",
            {"total": "x"},
        ]
    )
    assert merged["total"] == 7
    assert len(merged["results"]) == 3