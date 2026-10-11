"""Geographic marine-area reference data for the BlueEye Marine Areas Map.

This module is deliberately separated from inference. It holds:

* a small curated catalogue of well-known Indian Ocean marine areas
  (archipelagos, gulfs, seas, mangrove deltas and coastlines), modelled as
  :class:`MarineRegion` records with approximate centres and bounding boxes
  for map exploration;
* **worldwide** place lookup: the map is not restricted to India. Any place
  on Earth can be explored through :func:`geocode_area` (OpenStreetMap
  Nominatim, no key required), through exact latitude/longitude input
  (:func:`parse_lat_lon` + :func:`region_from_coordinates`), or by picking a
  curated area. Geocoded places become the same :class:`MarineRegion` model
  with an **approximate** exploration box, so every downstream behaviour
  (catalogue linking, OBIS queries, map drawing) works identically;
* an honest link between areas and the educational species catalogue in
  :mod:`app.content.species`: a species is **only ever** attributed to an
  area when its ``distribution`` text names one of the area's location
  labels (:func:`region_species`). Nothing is invented and the page shows
  exactly which label(s) matched. The catalogue is India-first, so most
  worldwide areas honestly return an empty catalogue list;
* a thin client for the Ocean Biodiversity Information System (OBIS)
  occurrence API (:data:`OBIS_API_URL`) used for **georeferenced
  observations only**. OBIS aggregates openly licensed (CC0) marine
  occurrence records, needs no API key and accepts bounding-box queries, so
  it fits this project without new infrastructure. Boxes that cross the
  international date line are split into two valid windows
  (:func:`bbox_windows`, :func:`obis_query_urls`).

Honesty rules enforced here:

* areas are approximate boxes for exploration, never claimed as precise
  political or ecological boundaries;
* geocoding never fabricates a location: a place that cannot be resolved
  returns ``None`` and the UI says so;
* OBIS coordinates come from published records, shown as reported by third
  parties with source and licence attribution;
* BlueEye media detections are never geotagged, so this module never
  attaches a detection to a location.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from app.content.species import SPECIES, Species

logger = logging.getLogger(__name__)

#: OBIS occurrence REST API (no key required; data CC0 / permissive).
OBIS_API_URL = "https://api.obis.org/v3/occurrence"
#: Human-friendly OBIS occurrence search UI (same geometry parameter).
OBIS_SITE_URL = "https://obis.org/occurrence/search"

#: Cap for scatter-points plotted on the map (keeps the deck layer responsive).
_MAX_PREVIEW_POINTS = 1200

#: OpenStreetMap geocoder (Nominatim) - free, no key, global coverage.
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

#: User-Agent used for public APIs (identifies the app politely; Nominatim's
#: usage policy requires a descriptive agent).
API_USER_AGENT = "BlueEye/1.0 Marine Areas Map (educational dashboard)"

#: Minimum exploration-box span (degrees) for geocoded places that resolve to
#: a point or a tiny feature, so exploration still has a usable area.
_MIN_AREA_SPAN = 2.0


def _valid_lat(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value) and -90.0 <= value <= 90.0


def _valid_lon(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value) and -180.0 <= value <= 180.0


@dataclass(frozen=True)
class GeoPlace:
    """A geocoded worldwide place with an approximate exploration box.

    ``source`` names the geocoder so the UI can attribute the result, and
    ``precision`` is a short description of the matched feature type (e.g.
    ``"natural · reef"``) so the box is never mistaken for an exact boundary.
    Numbers are always validated against real geographic limits.
    """

    name: str
    display_name: str
    center: tuple[float, float]  # (lat, lon)
    west: float
    south: float
    east: float
    north: float
    source: str
    precision: str = ""
    key: str = ""


def _feature_type(row: dict) -> str:
    category = str(row.get("category") or "").strip()
    kind = str(row.get("type") or "").strip()
    label = f"{category} · {kind}" if category and kind else (category or kind)
    return label[:60]


def _feature_key(row: dict) -> str:
    osm_type = str(row.get("osm_type") or "")
    osm_id = row.get("osm_id")
    return f"osm-{osm_type}-{osm_id}" if osm_type and osm_id else ""


def parse_geocode_response(payload: Any) -> list[GeoPlace]:
    """Convert a Nominatim ``format=jsonv2`` response into :class:`GeoPlace`.

    Rows with missing or out-of-range coordinates are skipped - a location is
    never fabricated from a malformed response.
    """
    if not isinstance(payload, list):
        return []
    places: list[GeoPlace] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        try:
            lat = float(row.get("lat"))
            lon = float(row.get("lon"))
        except (TypeError, ValueError):
            continue
        if not (_valid_lat(lat) and _valid_lon(lon)):
            continue
        try:
            box = [float(value) for value in row.get("boundingbox") or []]
        except (TypeError, ValueError):
            box = []
        west = box[2] if len(box) >= 4 and _valid_lon(box[2]) else lon
        south = box[0] if len(box) >= 4 and _valid_lat(box[0]) else lat
        east = box[3] if len(box) >= 4 and _valid_lon(box[3]) else lon
        north = box[1] if len(box) >= 4 and _valid_lat(box[1]) else lat
        places.append(
            GeoPlace(
                name=str(row.get("name") or "").strip(),
                display_name=str(row.get("display_name") or row.get("name") or "").strip(),
                center=(lat, lon),
                west=west,
                south=south,
                east=east,
                north=north,
                source="OpenStreetMap (Nominatim)",
                precision=_feature_type(row),
                key=_feature_key(row),
            )
        )
    return places


def pick_best_place(places: list[GeoPlace]) -> GeoPlace | None:
    """Choose the most useful geocoding result.

    Prefers candidates with a real (non-point) bounding box, then the largest
    box - a country or sea is more useful for exploration than a single point.
    """
    if not places:
        return None

    def area(place: GeoPlace) -> float:
        return abs(place.east - place.west) * abs(place.north - place.south)

    def has_area(place: GeoPlace) -> int:
        return 1 if area(place) > 1e-9 else 0

    return max(places, key=lambda place: (has_area(place), area(place)))


def geocode_area(query: str, timeout: int = 12) -> GeoPlace | None:
    """Geocode an arbitrary worldwide place via OpenStreetMap Nominatim.

    Returns the best :class:`GeoPlace`, or ``None`` when the query matched
    nothing or the geocoder could not be reached, so callers can say so
    honestly and fall back to curated areas / exact coordinates.
    """
    text = (query or "").strip()
    if not text:
        return None
    try:
        import requests  # local import: live geocoding is an optional service

        response = requests.get(
            NOMINATIM_URL,
            params={
                "format": "jsonv2",
                "q": text,
                "limit": 5,
                "accept-language": "en",
            },
            timeout=timeout,
            headers={"User-Agent": API_USER_AGENT},
        )
        response.raise_for_status()
        return pick_best_place(parse_geocode_response(response.json()))
    except (OSError, ValueError) as exc:  # requests failures subclass OSError
        logger.info("Geocoding failed for %r: %s", text, exc)
        return None


def parse_lat_lon(text: str) -> tuple[float, float] | None:
    """Parse ``lat, lon`` from free text into ``(lat, lon)`` or ``None``.

    Accepts decimal degrees with optional direction letters and degree
    symbols: ``12.5, 77.2``, ``12.5N 77.2E``, ``12.5S 77.2W``, ``12.5 77.2``.
    Latitude must lie in [-90, 90] and longitude in [-180, 180]; anything
    else is rejected rather than clamped (callers decide how to show it).
    """
    raw = re.sub(r"[°º,;]", " ", (text or "").strip())
    if not raw:
        return None
    numbers = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", raw)]
    if len(numbers) < 2:
        return None
    lat, lon = numbers[0], numbers[1]
    letters = [part.upper() for part in re.findall(r"[NSEW]", raw)]
    for index, letter in enumerate(letters[:2]):
        if index == 0:
            if letter == "S":
                lat = -abs(lat)
            elif letter == "N":
                lat = abs(lat)
        else:
            if letter == "W":
                lon = -abs(lon)
            elif letter == "E":
                lon = abs(lon)
    if not (_valid_lat(lat) and _valid_lon(lon)) or len(letters) > 2:
        return None
    return lat, lon


def region_from_coordinates(
    lat: float, lon: float, span: float = _MIN_AREA_SPAN, label: str = "Exact coordinates"
) -> MarineRegion:
    """Build an exploration :class:`MarineRegion` around a latitude/longitude.

    The box is centred on the point with a span of ``span`` degrees (clamped
    to the real globe), so the result is always a valid, drawable area.
    """
    if not (_valid_lat(lat) and _valid_lon(lon)):
        raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")
    span = float(span)
    if not (0 < span <= 180):
        raise ValueError(f"Invalid span: {span}")
    half = span / 2.0
    min_lat = max(-90.0, lat - half)
    max_lat = min(90.0, lat + half)
    min_lon = max(-180.0, lon - half)
    max_lon = min(180.0, lon + half)
    if min_lat >= max_lat or min_lon >= max_lon:  # degenerate pole/edge box
        raise ValueError(
            f"Coordinates at the globe's edge collapse to no area: lat={lat}, lon={lon}"
        )
    return MarineRegion(
        id=f"coords_{lat:.3f}_{lon:.3f}",
        name=f"{lat:.3f}°/ {lon:.3f}°",
        kind="Exact coordinates",
        note="An approximate exploration box centred on the coordinates you entered.",
        search_terms=(),
        area_labels=(),
        center=(lat, lon),
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
    )


def region_from_place(place: GeoPlace, label: str | None = None) -> MarineRegion:
    """Turn a geocoded :class:`GeoPlace` into an explorable :class:`MarineRegion`.

    Point-sized matches are expanded to a minimum exploration span
    (:data:`_MIN_AREA_SPAN`) so the user still gets a usable area. The
    returned region behaves exactly like a curated one everywhere downstream.
    """
    if place is None:
        raise ValueError("Cannot build a region from an empty geocoding result")
    lat, lon = place.center
    if place.east - place.west > 1e-9 and place.north - place.south > 1e-9:
        min_lat, max_lat = place.south, place.north
        min_lon, max_lon = place.west, place.east
    else:  # point-sized match: expand to the minimum exploration span
        min_lat = max(-90.0, lat - _MIN_AREA_SPAN / 2.0)
        max_lat = min(90.0, lat + _MIN_AREA_SPAN / 2.0)
        min_lon = max(-180.0, lon - _MIN_AREA_SPAN / 2.0)
        max_lon = min(180.0, lon + _MIN_AREA_SPAN / 2.0)
    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError(f"Geocoded place collapses to no area: {place.display_name!r}")
    slug = re.sub(r"[^a-z0-9]+", "-", (label or place.name).lower()).strip("-")[:48]
    name = (
        label
        or place.name
        or place.display_name.split(",")[0]
        or "Geocoded area"
    )
    return MarineRegion(
        id=f"geo_{slug or 'place'}",
        name=name,
        kind=place.precision or f"Geocoded · {place.source}",
        note=(
            "Geocoded online - this is an approximate exploration box derived "
            f"from {place.source}, not an exact boundary."
        ),
        search_terms=(),
        area_labels=(name,),
        center=(lat, lon),
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
    )


@dataclass(frozen=True)
class MarineRegion:
    """A well-known marine area users can select on the Marine Areas Map.

    ``center`` and the ``*_lat``/``*_lon`` bounds describe an **approximate**
    exploration box around the area - deliberately not exact boundaries.
    ``area_labels`` are the geographic names checked against the species
    catalogue's ``distribution`` text; a species is attributed to the region
    only when its catalogue record names at least one of these labels, and
    the UI displays exactly which label(s) matched.
    """

    id: str
    name: str
    kind: str
    note: str
    search_terms: tuple[str, ...]
    area_labels: tuple[str, ...]
    center: tuple[float, float]  # (lat, lon)
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


#: The marine areas offered by the Marine Areas Map. Centres/bounds are
#: approximate (values chosen from standard atlas geography, not inferred
#: from detections). ``area_labels`` are matched verbatim (case-insensitive)
#: against the catalogue's ``distribution`` text - see :func:`region_species`.
REGIONS: tuple[MarineRegion, ...] = (
    MarineRegion(
        id="lakshadweep",
        name="Lakshadweep",
        kind="Coral archipelago",
        note="A chain of coral atolls off India's south-west coast, ringed by "
        "reefs, lagoons and sea-turtle nesting beaches.",
        search_terms=("Laccadive", "Kavaratti", "Agatti", "Minicoy", "atoll"),
        area_labels=("Lakshadweep",),
        center=(10.3, 72.6),
        min_lat=7.9, min_lon=70.9, max_lat=12.8, max_lon=74.2,
    ),
    MarineRegion(
        id="andaman_nicobar",
        name="Andaman & Nicobar Islands",
        kind="Island archipelago",
        note="A tropical island chain in the Bay of Bengal with coral reefs, "
        "mangrove creeks and leatherback nesting beaches.",
        search_terms=("Andaman", "Nicobar", "Port Blair", "Havelock"),
        area_labels=("Andaman", "Nicobar"),
        center=(11.2, 92.9),
        min_lat=5.9, min_lon=91.7, max_lat=14.2, max_lon=94.4,
    ),
    MarineRegion(
        id="gulf_of_mannar",
        name="Gulf of Mannar",
        kind="Gulf · biosphere reserve",
        note="A protected gulf between India and Sri Lanka known for seagrass "
        "meadows, coral reefs and the last Indian dugong herds.",
        search_terms=("Mannar", "Palk", "Rameswaram", "Thoothukudi", "seagrass"),
        area_labels=("Gulf of Mannar", "Palk"),
        center=(9.2, 79.4),
        min_lat=8.0, min_lon=78.0, max_lat=10.6, max_lon=80.4,
    ),
    MarineRegion(
        id="gulf_of_kutch",
        name="Gulf of Kutch",
        kind="Gulf · marine national park",
        note="A macro-tidal gulf on Gujarat's coast hosting India's first "
        "marine national park and important dugong sites.",
        search_terms=("Kutch", "Kachchh", "Gujarat", "Dwarka", "Jamnagar"),
        area_labels=("Gulf of Kutch", "Gujarat"),
        center=(22.6, 69.3),
        min_lat=21.9, min_lon=68.0, max_lat=23.6, max_lon=70.4,
    ),
    MarineRegion(
        id="sundarbans_mangroves",
        name="Sundarbans Mangroves",
        kind="Mangrove delta",
        note="The world's largest mangrove delta, shared by India and "
        "Bangladesh, with tidal mudflats, creeks and brackish swamps.",
        search_terms=(
            "Sundarban",
            "Sundarbans",
            "Hooghly",
            "Padma",
            "Ganges delta",
            "mangrove",
            "delta",
        ),
        area_labels=("Sundarbans", "Hooghly", "Padma"),
        center=(21.9, 89.4),
        min_lat=21.0, min_lon=87.9, max_lat=22.6, max_lon=91.1,
    ),
    MarineRegion(
        id="odisha_coast",
        name="Odisha Coast",
        kind="Coastline · turtle rookeries",
        note="Long sandy beaches sheltering the world's largest olive ridley "
        "mass-nesting sites at Gahirmatha and Rushikulya.",
        search_terms=("Odisha", "Orissa", "Gahirmatha", "Rushikulya", "Puri", "Chilika"),
        area_labels=("Odisha", "Gahirmatha", "Rushikulya"),
        center=(20.3, 86.6),
        min_lat=19.0, min_lon=84.9, max_lat=21.6, max_lon=87.8,
    ),
    MarineRegion(
        id="arabian_sea",
        name="Arabian Sea (Indian shelf)",
        kind="Sea · open water",
        note="The north-east Indian Ocean basin along India's western coast, "
        "from Gujarat to Kerala and into the Lakshadweep atolls.",
        search_terms=("Arabian Sea", "Arabian", "west coast", "Gujarat", "Lakshadweep"),
        area_labels=("Arabian Sea", "Lakshadweep", "Gujarat", "Gulf of Kutch"),
        center=(16.8, 71.4),
        min_lat=5.5, min_lon=64.0, max_lat=23.8, max_lon=76.5,
    ),
    MarineRegion(
        id="bay_of_bengal",
        name="Bay of Bengal (Indian shelf)",
        kind="Sea · open water",
        note="A large tropical bay east of India, fed by the Ganga and "
        "Brahmaputra deltas, with dolphins, rays and nesting turtles.",
        search_terms=("Bay of Bengal", "Bengal", "east coast", "Odisha", "Andaman"),
        area_labels=(
            "Bay of Bengal",
            "Gulf of Mannar",
            "Palk",
            "Odisha",
            "Gahirmatha",
            "Rushikulya",
            "Sundarbans",
            "Hooghly",
            "Padma",
            "Andaman",
            "Nicobar",
        ),
        center=(13.2, 86.0),
        min_lat=5.5, min_lon=78.5, max_lat=22.6, max_lon=94.5,
    ),
    MarineRegion(
        id="kerala_coast",
        name="Kerala Coast",
        kind="Coastline",
        note="South-west coast of backwaters, estuaries and palm-lined "
        "beaches; offshore reefs lie beyond the shelf edge.",
        search_terms=("Kerala", "Kochi", "Cochin", "Malabar", "Kollam", "Vizhinjam"),
        area_labels=("Kerala",),
        center=(9.8, 76.1),
        min_lat=8.0, min_lon=74.6, max_lat=12.7, max_lon=77.2,
    ),
    MarineRegion(
        id="tamil_nadu_coast",
        name="Tamil Nadu Coast",
        kind="Coastline",
        note="South-east coast with long beaches, lagoons and offshore reefs; "
        "the mainland shore of the Gulf of Mannar.",
        search_terms=("Tamil Nadu", "Chennai", "Coromandel", "Puducherry", "Nagapattinam"),
        area_labels=("Tamil Nadu", "Chennai"),
        center=(10.4, 79.5),
        min_lat=8.1, min_lon=77.6, max_lat=13.4, max_lon=80.5,
    ),
)

_REGION_BY_ID: dict[str, MarineRegion] = {region.id: region for region in REGIONS}


def get_region(region_id: str) -> MarineRegion | None:
    """Look a region up by its stable id."""
    return _REGION_BY_ID.get(region_id)


def search_regions(query: str) -> list[MarineRegion]:
    """Substring-search the regions by name and search terms.

    Empty input returns an empty list (the page shows its empty state rather
    than an arbitrary default).
    """
    token = (query or "").strip().lower()
    if not token:
        return []
    matches: list[MarineRegion] = []
    for region in REGIONS:
        haystack = (region.name, *region.search_terms)
        if any(token in part.lower() for part in haystack):
            matches.append(region)
    return matches


def region_species(region: MarineRegion) -> list[tuple[Species, list[str]]]:
    """Catalogue species whose ``distribution`` text names the region.

    Attribution is strictly text-based: only labels that appear verbatim
    (case-insensitive) in a species' distribution record are reported, and
    the returned per-species labels are exactly those matches - so the UI can
    show *why* a species appears here and nothing is invented.
    """
    matches: list[tuple[Species, list[str]]] = []
    for species in SPECIES:
        text = (species.distribution or "").lower()
        hit = [label for label in region.area_labels if label.lower() in text]
        if hit:
            matches.append((species, hit))
    return matches


def regions_with_species() -> list[MarineRegion]:
    """Regions that currently have at least one catalogue species record."""
    return [region for region in REGIONS if region_species(region)]


# --------------------------------------------------------------------------- #
# OBIS live observations (georeferenced records, opt-in)
# --------------------------------------------------------------------------- #
def normalize_scientific_name(raw: str) -> str:
    """Strip author/rank suffixes such as ``Chelonia mydas (Linnaeus, 1758)``."""
    name = re.sub(r"\s*\([^)]*\)", "", raw or "").strip()
    return re.sub(r"\s+", " ", name)


def match_species_by_scientific_name(name: str) -> Species | None:
    """Return the catalogue species whose scientific name matches, if any."""
    cleaned = normalize_scientific_name(name).lower()
    if not cleaned:
        return None
    for species in SPECIES:
        if species.scientific_name.lower() == cleaned:
            return species
    return None


def bbox_windows(region: MarineRegion) -> list[tuple[float, float]]:
    """Longitude windows that cover the region's box within [-180, 180].

    A box whose west bound lies east of its east bound crosses the
    international date line and is split into two valid windows so both the
    OBIS query and the drawn map polygon remain valid.
    """
    west, east = region.min_lon, region.max_lon
    if west > east:
        return [(west, 180.0), (-180.0, east)]
    return [(west, east)]


def _polygon_payload(min_lon, max_lon, min_lat, max_lat) -> str:
    return (
        "POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
        "{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    ).format(min_lon=min_lon, max_lon=max_lon, min_lat=min_lat, max_lat=max_lat)


def obis_query_url(region: MarineRegion, size: int = 500) -> str:
    """The OBIS occurrence API URL for a region's bounding box.

    Boxes that do not cross the date line produce the same single URL they
    always did; for crossing boxes see :func:`obis_query_urls`.
    """
    west, east = bbox_windows(region)[0]
    polygon = _polygon_payload(west, east, region.min_lat, region.max_lat)
    return f"{OBIS_API_URL}?{urlencode({'geometry': polygon, 'size': size})}"


def obis_query_urls(region: MarineRegion, size: int = 500) -> list[str]:
    """OBIS URLs covering the region's box (one per longitude window)."""
    return [
        f"{OBIS_API_URL}?{urlencode({'geometry': _polygon_payload(west, east, region.min_lat, region.max_lat), 'size': size})}"
        for west, east in bbox_windows(region)
    ]


def obis_web_search_url(region: MarineRegion) -> str:
    """A human-readable OBIS search for the same area (for attribution links)."""
    west, east = bbox_windows(region)[0]
    polygon = _polygon_payload(west, east, region.min_lat, region.max_lat)
    return f"{OBIS_SITE_URL}?{urlencode({'geometry': polygon})}"


def parse_obis_occurrences(payload: Any, limit: int = 12) -> dict[str, Any]:
    """Aggregate an OBIS occurrence response into a small honest summary.

    Returns ``{"total": int, "downloaded": int, "top": [..], "points": [..]}``
    where each ``top`` row is ``{"name", "count", "recent_year", "species"}``
    (``species`` is the matching catalogue profile or ``None``) and ``points``
    are plotted on the map, capped for responsiveness.
    """
    if not isinstance(payload, dict):
        return {"total": 0, "downloaded": 0, "top": [], "points": []}
    results = payload.get("results") or []
    try:
        total = int(payload.get("total") or 0)
    except (TypeError, ValueError):
        total = 0

    counts: dict[str, int] = {}
    years: dict[str, int] = {}
    points: list[dict[str, Any]] = []
    for record in results:
        if not isinstance(record, dict):
            continue
        name = normalize_scientific_name(record.get("scientificName") or "")
        if not name:
            name = "Unidentified"
        counts[name] = counts.get(name, 0) + 1
        try:
            year = int(record.get("date_year") or 0)
        except (TypeError, ValueError):
            year = 0
        if year:
            years[name] = max(years.get(name, 0), year)
        lat, lon = record.get("decimalLatitude"), record.get("decimalLongitude")
        if lat is not None and lon is not None:
            points.append({"lon": float(lon), "lat": float(lat), "name": name})

    if len(points) > _MAX_PREVIEW_POINTS:
        step = math.ceil(len(points) / _MAX_PREVIEW_POINTS)
        points = points[::step]

    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0].lower()))
    top = [
        {
            "name": name,
            "count": count,
            "recent_year": years.get(name),
            "species": match_species_by_scientific_name(name),
        }
        for name, count in ranked[:limit]
    ]
    return {"total": total, "downloaded": len(results), "top": top, "points": points}


def _merge_payloads(payloads: list[Any]) -> dict[str, Any]:
    """Merge several OBIS payloads (used when a box crosses the date line)."""
    total = 0
    results: list[Any] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        try:
            total += int(payload.get("total") or 0)
        except (TypeError, ValueError):
            continue
        records = payload.get("results")
        if isinstance(records, list):  # noqa: SIM102 - nested for brevity
            results.extend(record for record in records if isinstance(record, dict))
    return {"total": total, "results": results}


def fetch_obis_observations(
    region: MarineRegion,
    size: int = 500,
    timeout: int = 20,
    limit: int = 12,
) -> dict[str, Any]:
    """Download and summarise published OBIS occurrences for a region's box.

    A box crossing the international date line is queried once per longitude
    window and the raw results are merged before summarising, so counts stay
    honest. The caller is expected to catch :class:`requests.RequestException`
    (including timeout) and show an informative offline/error state: live
    observations are an optional enhancement, never a requirement.
    """
    import requests  # local import: only needed for the optional live source

    headers = {"User-Agent": API_USER_AGENT}
    urls = obis_query_urls(region, size=size)
    if len(urls) == 1:
        response = requests.get(urls[0], timeout=timeout, headers=headers)
        response.raise_for_status()
        return parse_obis_occurrences(response.json(), limit=limit)

    payloads: list[Any] = []
    for url in urls:
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        payloads.append(response.json())
    return parse_obis_occurrences(_merge_payloads(payloads), limit=limit)