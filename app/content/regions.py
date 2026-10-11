"""Geographic marine-area reference data for the BlueEye Marine Areas Map.

This module is deliberately separated from inference. It holds:

* a small curated catalogue of well-known Indian Ocean marine areas
  (archipelagos, gulfs, seas, mangrove deltas and coastlines), modelled as
  :class:`MarineRegion` records with approximate centres and bounding boxes
  for map exploration;
* an honest link between those areas and the educational species catalogue
  in :mod:`app.content.species`: a species is **only ever** attributed to an
  area when its ``distribution`` text names one of the area's location
  labels (:func:`region_species`). Nothing is invented and the page shows
  exactly which label(s) matched;
* a thin client for the Ocean Biodiversity Information System (OBIS)
  occurrence API (:data:`OBIS_API_URL`) used for **georeferenced
  observations only**. OBIS aggregates openly licensed (CC0) marine
  occurrence records, needs no API key and accepts bounding-box queries, so
  it fits this project without new infrastructure.

Honesty rules enforced here:

* areas are approximate boxes for exploration, never claimed as precise
  political or ecological boundaries;
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


def obis_query_url(region: MarineRegion, size: int = 500) -> str:
    """The OBIS occurrence API URL for a region's bounding box."""
    polygon = (
        f"POLYGON(({region.min_lon} {region.min_lat}, "
        f"{region.max_lon} {region.min_lat}, {region.max_lon} {region.max_lat}, "
        f"{region.min_lon} {region.max_lat}, {region.min_lon} {region.min_lat}))"
    )
    return f"{OBIS_API_URL}?{urlencode({'geometry': polygon, 'size': size})}"


def obis_web_search_url(region: MarineRegion) -> str:
    """A human-readable OBIS search for the same area (for attribution links)."""
    polygon = (
        f"POLYGON(({region.min_lon} {region.min_lat}, "
        f"{region.max_lon} {region.min_lat}, {region.max_lon} {region.max_lat}, "
        f"{region.min_lon} {region.max_lat}, {region.min_lon} {region.min_lat}))"
    )
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


def fetch_obis_observations(
    region: MarineRegion, size: int = 500, timeout: int = 20
) -> dict[str, Any]:
    """Download and summarise published OBIS occurrences for a region's box.

    The caller is expected to catch :class:`requests.RequestException`
    (including timeout) and show an informative offline/error state: live
    observations are an optional enhancement, never a requirement.
    """
    import requests  # local import: only needed for the optional live source

    response = requests.get(
        obis_query_url(region, size=size),
        timeout=timeout,
        headers={"User-Agent": "BlueEye/1.0 Marine Areas Map (Streamlit demo)"},
    )
    response.raise_for_status()
    return parse_obis_occurrences(response.json())