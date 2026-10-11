"""Marine Areas Map: geographic marine-organism exploration.

The Map is a discovery surface, not a map of detection history. Users pick a
coastline, sea, island or protected area (by search, dropdown or a click on
the map itself) and are shown:

* **Reported** - species from the BlueEye educational catalogue whose
  ``distribution`` text explicitly names the selected area (see
  :func:`app.content.regions.region_species`). Each card lists exactly which
  location names matched, so nothing is invented.
* **Observed** - live georeferenced occurrence records for the area box,
  fetched (opt-in, online) from the Ocean Biodiversity Information System
  (OBIS). Records are shown as reported by third-party datasets, with source
  and licence attribution; OBIS data are predominantly CC0.
* **Detected** - BlueEye media detections are never geotagged, so this page
  never attaches a detection to a location and says so explicitly.

Rendering is scoped to this page; nothing here touches inference, uploads,
history or models.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
from typing import Any

import pydeck as pdk
import streamlit as st

from app.content.regions import (
    REGIONS,
    MarineRegion,
    fetch_obis_observations,
    get_region,
    obis_web_search_url,
    region_species,
    regions_with_species,
    search_regions,
)
from app.content.species import Species
from app.detection.detector import MarineDetector
from app.ui.components import badge, empty_state, hero, kv, notice
from app.ui.icons import span
from app.ui.theme import esc

logger = logging.getLogger(__name__)

#: Region preselected on first visit (has rich catalogue coverage).
_DEFAULT_REGION = "lakshadweep"

#: Sample size for the optional OBIS download (kept modest for responsiveness).
_OBS_SAMPLE_SIZE = 500

#: Session keys owned by this page.
_REGION_KEY = "map_region_id"
_SEARCH_KEY = "map_search"
_SELECT_KEY = "map_area_select"
_GROUP_KEY = "map_group_filter"
_OBS_KEY = "map_obis_toggle"
_LAST_REGION_KEY = "map_last_region"
_MAP_CHART_KEY = "map_chart"

#: deck.gl layer ids used for click selection.
_AREA_LAYER_ID = "region-areas"
_SELECTED_LAYER_ID = "region-selected"
_OBS_LAYER_ID = "region-observations"


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
def render(detector: MarineDetector, settings) -> None:
    """Render the Marine Areas Map page."""
    st.session_state.setdefault(_REGION_KEY, _DEFAULT_REGION)
    base = get_region(st.session_state.get(_REGION_KEY, _DEFAULT_REGION))
    if base is None:
        base = get_region(_DEFAULT_REGION)
    st.session_state[_REGION_KEY] = base.id

    hero(
        st,
        "Explorer · Geography",
        "Marine Areas Map",
        "Pick a coastline, sea, island or protected area, then discover the "
        "marine life reported for that stretch of ocean.",
        icon="map",
    )

    region = _controls(base)

    # Live observations are opt-in; the toggle lives in the side panel below,
    # so on a toggle change this fetch re-runs before the map is re-rendered.
    obs = _observations(region)

    map_col, side_col = st.columns([7, 3], gap="medium")
    with map_col:
        _map(region, obs)
    with side_col:
        _side_panel(region, obs)

    _catalogue_results(region)
    _observations_panel(region, obs)


# --------------------------------------------------------------------------- #
# Location controls (search at the top, guaranteed selection below)
# --------------------------------------------------------------------------- #
def _controls(region: MarineRegion) -> MarineRegion:
    """Search + pick controls; returns the effective (selected) region."""
    query = st.session_state.get(_SEARCH_KEY, "").strip()
    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.text_input(
            "Search a coastline, sea, island or protected area",
            key=_SEARCH_KEY,
            placeholder="e.g. Lakshadweep, Gulf of Kutch, Sundarbans…",
        )
        if query:
            matches = search_regions(query)
            if matches:
                st.caption(
                    f"{len(matches)} matching area{'s' if len(matches) != 1 else ''}:"
                )
                for col, match in zip(st.columns(min(len(matches), 4)), matches[:4]):
                    with col:
                        st.button(
                            f"Select · {match.name}",
                            key=f"map_sug_{match.id}",
                            on_click=_select_region,
                            args=(match.id,),
                        )
            else:
                st.caption(
                    f"No named marine area matches “{esc(query)}” in the BlueEye "
                    "region catalogue — try a sea, island or coast name."
                )
                _suggestion_chips("Try")

    index_by_id = {r.id: position for position, r in enumerate(REGIONS)}
    # Dynamic widget key: the dropdown re-mounts whenever the selected area
    # changes (however it was changed - dropdown, search suggestion or map
    # click), so its label always matches the currently selected region
    # instead of holding a stale value from an earlier selection.
    select_key = f"{_SELECT_KEY}::{region.id}"
    with right:
        st.selectbox(
            "or choose from all areas",
            options=[r.id for r in REGIONS],
            format_func=lambda rid: next(r.name for r in REGIONS if r.id == rid),
            key=select_key,
            index=index_by_id[region.id],
        )
        via_select = st.session_state.get(select_key)
        if via_select and get_region(via_select) is not None:
            region = get_region(via_select)
            st.session_state[_REGION_KEY] = region.id

    # The location changed: drop a stale group filter from a previous area.
    if st.session_state.get(_LAST_REGION_KEY) != region.id:
        st.session_state.pop(_GROUP_KEY, None)
        st.session_state[_LAST_REGION_KEY] = region.id
    return region


def _select_region(region_id: str) -> None:
    """Choose an area from a suggestion button.

    Only the page-level selection is written here; the "or choose from all
    areas" dropdown re-mounts with the current region (dynamic widget key in
    :func:`_controls`), so it always reflects the selected area no matter how
    the selection was made (dropdown, search suggestion, or map click).
    """
    if get_region(region_id) is not None:
        st.session_state[_REGION_KEY] = region_id


def _suggestion_chips(prefix: str) -> None:
    """Quick-select buttons for areas that already have catalogue records."""
    with_data = regions_with_species()
    if not with_data:
        return
    cols = st.columns(min(len(with_data), 4))
    for col, region in zip(cols, with_data[:4]):
        with col:
            st.button(
                f"{prefix} · {region.name}",
                key=f"map_chip_{prefix.lower()}_{region.id}",
                on_click=_select_region,
                args=(region.id,),
            )


# --------------------------------------------------------------------------- #
# Map (pydeck - the project's existing mapping library)
# --------------------------------------------------------------------------- #
def _map(region: MarineRegion, obs: dict[str, Any] | None) -> None:
    layers = [
        pdk.Layer(
            "PolygonLayer",
            id=_AREA_LAYER_ID,
            data=[_area_row(item) for item in REGIONS],
            get_polygon="coords",
            stroked=True,
            get_fill_color=[11, 58, 102, 150],
            get_line_color=[17, 84, 139, 230],
            line_width_min_pixels=1.4,
            pickable=True,
            auto_highlight=True,
        ),
        pdk.Layer(
            "PolygonLayer",
            id=_SELECTED_LAYER_ID,
            data=[_area_row(region)],
            get_polygon="coords",
            stroked=True,
            get_fill_color=[34, 211, 238, 110],
            get_line_color=[34, 211, 238, 255],
            line_width_min_pixels=2.4,
            pickable=True,
            auto_highlight=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            id="region-centres",
            data=[
                {"lon": item.center[1], "lat": item.center[0], "name": item.name}
                for item in REGIONS
            ],
            get_position="[lon, lat]",
            get_radius=25000,
            get_fill_color=[34, 211, 238, 90],
            pickable=False,
        ),
        pdk.Layer(
            "TextLayer",
            id="region-labels",
            data=[
                {"lon": item.center[1], "lat": item.center[0], "name": item.name}
                for item in REGIONS
            ],
            get_position="[lon, lat]",
            get_text="name",
            get_size=12,
            get_color=[214, 235, 247, 200],
        ),
    ]
    if obs and obs.get("points"):
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id=_OBS_LAYER_ID,
                data=obs["points"],
                get_position="[lon, lat]",
                get_radius=9000,
                get_fill_color=[252, 211, 77, 200],
                pickable=True,
                auto_highlight=True,
            )
        )

    view_state = pdk.ViewState(
        latitude=region.center[0],
        longitude=region.center[1],
        zoom=4.8 if region.kind.startswith("Sea") else 6.2,
        controller=True,
    )
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="dark",
        tooltip={"text": "{name}"},
    )
    st.pydeck_chart(
        deck,
        height=540,
        key=_MAP_CHART_KEY,
        on_select=_on_map_pick,
        selection_mode="single-object",
    )
    st.caption(
        "Pan and zoom with the mouse or trackpad; click any shaded area to "
        "select it. Shaded boxes are approximate exploration areas, not exact "
        "boundaries. Basemap tiles come from an external provider and need the "
        "internet (the layer shapes render regardless)."
    )


def _area_row(region: MarineRegion) -> dict[str, Any]:
    return {
        "id": region.id,
        "name": region.name,
        "coords": [
            [region.min_lon, region.min_lat],
            [region.max_lon, region.min_lat],
            [region.max_lon, region.max_lat],
            [region.min_lon, region.max_lat],
            [region.min_lon, region.min_lat],
        ],
    }


def _on_map_pick(event: Any = None) -> None:
    """Map click: select the shaded area the user clicked (best effort)."""
    region_id = _picked_region_id(event)
    if region_id and region_id != st.session_state.get(_REGION_KEY):
        st.session_state[_REGION_KEY] = region_id


def _picked_region_id(event: Any) -> str | None:
    """Extract a region id from a pydeck selection event, if any."""
    try:
        selection = getattr(event, "selection", None)
        if selection is None:
            return None
        objects = selection.get("objects") if hasattr(selection, "get") else getattr(selection, "objects", {})
        if not isinstance(objects, dict):
            return None
        for layer_id in (_AREA_LAYER_ID, _SELECTED_LAYER_ID):
            rows = objects.get(layer_id) or []
            if not rows:
                continue
            region_id = (rows[0] or {}).get("id")
            if region_id and get_region(region_id) is not None:
                return region_id
    except Exception as exc:  # noqa: BLE001 - never break the page on a UI quirk
        logger.warning("Map selection ignored: %s", exc)
    return None


# --------------------------------------------------------------------------- #
# Side panel: selected location, legend, OBIS toggle
# --------------------------------------------------------------------------- #
def _side_panel(region: MarineRegion, obs: dict[str, Any] | None) -> None:
    species_count = len(region_species(region))
    centre = (
        f"{_deg(region.center[0], 'N', 'S')} · {_deg(region.center[1], 'E', 'W')}"
    )
    area_text = (
        f"{region.min_lat:.1f}–{region.max_lat:.1f}°N, "
        f"{region.min_lon:.1f}–{region.max_lon:.1f}°E"
    )
    st.markdown(
        '<div class="be-card be-map-area">'
        f'<div class="be-card-title">{span("my_location")} Selected location</div>'
        f"<h4>{esc(region.name)}</h4>"
        '<div class="tags">'
        f"{badge(region.kind, 'info')}"
        "</div>"
        f'<p class="note">{esc(region.note)}</p>'
        f"{kv([('Centre', centre), ('Approximate area', area_text), ('Catalogue species', f'{species_count} reported')])}"
        '<div class="be-map-legend">'
        f'<div class="row">{badge("Reported", "info")} Catalogue record names this area</div>'
        f'<div class="row">{badge("Observed", "warn")} Live OBIS occurrence records (CC0)</div>'
        f'<div class="row">{badge("Detected", "off")} BlueEye detections are never geotagged</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.toggle(
        "Load georeferenced observations (OBIS)",
        value=bool(st.session_state.get(_OBS_KEY, False)),
        key=_OBS_KEY,
        help="Optional: queries the Ocean Biodiversity Information System for "
        "published occurrence records inside the selected area. Needs the internet.",
    )
    st.caption(
        "Media you analyse with BlueEye is stored without geographic "
        "coordinates, so detections are never mapped to a location."
    )


def _deg(value: float, positive: str, negative: str) -> str:
    return f"{value:.1f}°{positive}" if value >= 0 else f"{-value:.1f}°{negative}"


# --------------------------------------------------------------------------- #
# Catalogue results (the honest, source-driven species list)
# --------------------------------------------------------------------------- #
def _catalogue_results(region: MarineRegion) -> None:
    st.markdown("### Marine species reported here")
    st.markdown(
        '<p class="be-dash-lead">From the BlueEye educational catalogue — species '
        "whose distribution record explicitly names this area. No records are "
        "invented.</p>",
        unsafe_allow_html=True,
    )
    matches = region_species(region)
    if not matches:
        empty_state(
            st,
            "explore",
            "No catalogue records for this area yet",
            "The BlueEye reference catalogue does not name a species record for "
            f"{region.name}. Try a nearby area that already has records, or load "
            "live observations (OBIS) below for georeferenced records instead.",
        )
        _suggestion_chips("Try")
        return

    groups = sorted({species.group for species, _ in matches})
    selected_groups = st.multiselect(
        "Filter by animal group", groups, key=_GROUP_KEY
    )
    shown = [
        (species, labels)
        for species, labels in matches
        if not selected_groups or species.group in selected_groups
    ]
    if not shown:
        notice(
            st,
            "info",
            "No species match the selected groups",
            "Clear the group filter to see every species reported for this area.",
        )
        return

    names = " · ".join(
        dict.fromkeys(label for _, labels in matches for label in labels)
    )
    st.caption(
        f"{len(shown)} species reported here · location names: {esc(names)}"
    )
    grid = "".join(_species_card(species, labels) for species, labels in shown)
    st.markdown(f'<div class="be-mapgrid">{grid}</div>', unsafe_allow_html=True)


def _species_card(species: Species, labels: list[str]) -> str:
    """A species card like the dashboard's, with the matched area labels."""
    status = species.conservation_status or "Not assessed"
    status_badge = badge(status, _status_kind(status))
    places = " · ".join(dict.fromkeys(labels))
    href = f"?page=species&amp;species={esc(species.id)}"
    return (
        '<div class="be-map-species">'
        '<div class="head">'
        f'<div class="ic">{span(_SPECIES_ICONS.get(species.id, "water_drop"))}</div><div>'
        f"<h4>{esc(species.common_name)}</h4>"
        f'<p class="sci">{esc(species.scientific_name)}</p>'
        "</div></div>"
        '<div class="tags">'
        f"{badge(species.group, 'info')}{badge(species.habitat, 'off')}{status_badge}"
        "</div>"
        f'<p class="about">{esc(_clamp(species.overview or species.distribution or ""))}</p>'
        '<div class="foot">'
        f'<span class="loc">{span("location_on")} Reported in: {esc(places)}</span>'
        f'<a class="be-view" href="{href}">View →</a>'
        "</div></div>"
    )


def _clamp(text: str, limit: int = 170) -> str:
    """Shorten a description to ``limit`` characters at a word boundary."""
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    cut = compact[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(".,;:") + "…"


#: Conservation status -> badge colour (same display logic as the dashboard).
_CON_STATUS_KIND = {
    "Critically Endangered": "warn",
    "Endangered": "warn",
    "Vulnerable": "warn",
    "Near Threatened": "warn",
    "Least Concern": "ok",
    "Data Deficient": "off",
    "Not Evaluated": "off",
}


def _status_kind(status: str) -> str:
    """Map a catalogue conservation status to a badge colour."""
    kind = _CON_STATUS_KIND.get(status)
    if kind is not None:
        return kind
    lowered = status.lower()
    if "not" in lowered or "deficient" in lowered or "assess" in lowered:
        return "off"
    if (
        "endangered" in lowered
        or "threatened" in lowered
        or "vulnerable" in lowered
        or "critically" in lowered
    ):
        return "warn"
    if "least concern" in lowered:
        return "ok"
    return "info"


#: Decorative icon per species most likely to appear here (Material names).
_SPECIES_ICONS = {
    "green_sea_turtle": "turtle",
    "olive_ridley_turtle": "water_drop",
    "leatherback_turtle": "water_drop",
    "dugong": "ecosystem",
    "whale_shark": "waves",
    "reef_manta_ray": "air",
    "humphead_wrasse": "pets",
    "humphead_parrotfish": "pets",
    "camouflage_grouper": "pets",
    "spotted_seahorse": "pets",
    "indian_mackerel": "pets",
    "milkfish": "pets",
    "indian_oil_sardine": "pets",
    "sandfish_sea_cucumber": "ecosystem",
    "mud_crab": "pets",
    "crown_of_thorns": "auto_awesome",
    "scalloped_hammerhead": "waves",
    "indian_ocean_humpback_dolphin": "waves",
    "irrawaddy_dolphin": "waves",
    "hilsa": "pets",
}


# --------------------------------------------------------------------------- #
# Live observations (OBIS)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=1800)
def _obis_cached(region_id: str, size: int) -> dict[str, Any] | None:
    """Cached OBIS lookup per region (network failures cache as ``None``)."""
    region = get_region(region_id)
    if region is None:
        return None
    try:
        return fetch_obis_observations(region, size=size)
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly notice
        logger.info("OBIS lookup failed for %s: %s", region_id, exc)
        return None


def _observations(region: MarineRegion) -> dict[str, Any] | None:
    """Fetch OBIS observations for the map when the user asked for them."""
    if not st.session_state.get(_OBS_KEY, False):
        return None
    with st.spinner("Contacting the OBIS occurrence service…"):
        return _obis_cached(region.id, _OBS_SAMPLE_SIZE)


def _observations_panel(region: MarineRegion, obs: dict[str, Any] | None) -> None:
    st.markdown("### Live georeferenced observations")
    st.markdown(
        '<p class="be-dash-lead">Published occurrence records inside the selected '
        "area box, fetched from the Ocean Biodiversity Information System (OBIS).</p>",
        unsafe_allow_html=True,
    )
    if obs is None:
        if st.session_state.get(_OBS_KEY, False):
            notice(
                st,
                "warning",
                "OBIS is unreachable right now",
                "Live observations need an internet connection, and the request "
                "timed out or failed. Everything else on this page is unaffected.",
            )
        else:
            st.caption(
                "Use the toggle next to the map to load georeferenced records for "
                "this area (optional, needs the internet)."
            )
        return

    total = int(obs.get("total") or 0)
    downloaded = int(obs.get("downloaded") or 0)
    top = obs.get("top") or []
    st.caption(
        f"{total:,} published records occur within the {esc(region.name)} area box. "
        f"Showing the {len(top)} most common reported taxa from a "
        f"{downloaded}-record download."
    )
    if top:
        rows = [
            {
                "Reported scientific name": row["name"],
                "Records": str(row["count"]),
                "Most recent year": str(row["recent_year"] or "—"),
                "In BlueEye catalogue": (
                    f"Yes — {row['species'].common_name}"
                    if row["species"]
                    else "No"
                ),
            }
            for row in top
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Record counts come from the downloaded sample, not a full census.")
    else:
        st.caption("This area box returned no occurrence records in the download.")

    st.markdown(
        '<div class="be-obs-src">'
        f'<a class="be-link" href="{obis_web_search_url(region)}" target="_blank">'
        f"{span('open_in_new')} Open the same area on obis.org</a>"
        f'<p>OBIS aggregates marine occurrence data from many contributing '
        "datasets shared under permissive licences (predominantly CC0). "
        "Records are shown as their source datasets reported them and "
        "are not verified by BlueEye.</p></div>",
        unsafe_allow_html=True,
    )