"""Dashboard: a premium marine-intelligence overview and discovery page.

The dashboard is **display-only**. It never runs a detection and never
uploads a file - image and video analysis live exclusively on the unified
AI Detection page. Nothing here is fabricated: every number, chip, label,
comparison panel and chart is computed from the on-disk detection history
(:mod:`app.utils.history`), the installed model registry and the species
catalogue (:mod:`app.content.species`).

Sections
--------
1. Cinematic hero band (real underwater imagery from the project assets).
2. Marine statistics - real counts from stored runs + installed models.
3. Latest Detection - original input vs annotated result vs details.
4. Indian Marine Biodiversity - real profiles from the species catalogue.
5. Marine Biodiversity Overview - real charts over a selectable time range.
6. Recent Discoveries - the newest stored runs.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

from app.config import PROJECT_ROOT
from app.content.species import Species, get_species
from app.detection.detector import MarineDetector
from app.detection.results import display_class_name
from app.ui import state
from app.ui.components import badge, empty_state, kv
from app.ui.icons import span
from app.ui.theme import esc
from app.utils.file_utils import read_file_bytes
from app.utils.history import history_path, load_entries, summarize

#: How many recent runs the Recent Discoveries list shows.
RECENT_LIMIT = 4

#: Output suffixes the comparison panel can display as images.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}

#: Real underwater photography shipped with the project (used as the hero
#: background under a readability overlay - this same sample is what the
#: bundled detection demo runs on).
_HERO_IMAGE = PROJECT_ROOT / "assets" / "images" / "input_folder" / "regq.jpg"

#: Model classes that map one-to-one to a species profile in the catalogue
#: (only these carry a verified scientific name / profile link).
_CLASS_SPECIES: dict[str, str] = {
    "cheilinus_undulatus": "humphead_wrasse",
    "bolbometopon_muricatum": "humphead_parrotfish",
    "cromileptes_altivelis": "camouflage_grouper",
    "crown_of_thorns": "crown_of_thorns",
}

#: Indian-biodiversity picks (all verified present in Indian coastal waters
#: by their catalogue records - Lakshadweep, Andaman & Nicobar, Gujarat,
#: Gulf of Mannar, Odisha and Indian coastal backwaters).
_INDIAN_FEATURED = "green_sea_turtle"
_INDIAN_MINIS = (
    "olive_ridley_turtle",
    "dugong",
    "whale_shark",
    "spotted_seahorse",
)

#: Time-range options for the overview charts (real filtering of timestamps).
_RANGES = ("All time", "Last 30 days", "Last 7 days")

#: Conservation status -> badge kind (display only; text from the catalogue).
_CON_STATUS_KIND = {
    "Critically Endangered": "warn",
    "Endangered": "warn",
    "Vulnerable": "warn",
    "Near Threatened": "warn",
    "Least Concern": "ok",
    "Data Deficient": "off",
    "Not Evaluated": "off",
}

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def render(detector: MarineDetector, settings, user=None) -> None:
    """Render the marine-intelligence dashboard for the signed-in account."""
    user_id = user.id if user else None
    entries = load_entries(history_path(settings, user_id=user_id))
    stats = summarize(entries)
    registry = state.registry(detector)
    available = state.available_keys(detector)

    _hero_band()
    _marine_stats(registry, stats, available)
    _latest_detection(entries, registry, settings, user_id)
    _indian_biodiversity()
    _biodiversity_overview(entries)
    _recent_discoveries(entries, registry)


# --------------------------------------------------------------------------- #
# 1. Hero band
# --------------------------------------------------------------------------- #
def _hero_band() -> None:
    """Cinematic underwater banner with the project's own photography."""
    uri = _image_data_uri(str(_HERO_IMAGE))
    st.markdown(
        '<div class="be-mhero" role="banner" '
        "style=\"background-image: "
        "linear-gradient(180deg, rgba(2, 9, 20, .10) 0%, rgba(2, 9, 20, .35) 58%, rgba(2, 9, 20, .80) 100%), "
        "linear-gradient(100deg, "
        "rgba(2, 9, 20, .94) 0%, rgba(2, 9, 20, .74) 48%, rgba(2, 9, 20, .28) 82%, rgba(2, 9, 20, .18) 100%), "
        f"url({uri});\">"
        '<div class="eyebrow">' + span("waves") + " BLUEEYE · MARINE INTELLIGENCE</div>"
        "<h1>See Beneath the Surface</h1>"
        "<p>Discover marine life through AI-powered detection and intelligent "
        "biodiversity monitoring.</p>"
        '<a class="be-cta" href="?page=species">Explore Marine Species '
        '<span class="arr">→</span></a>'
        "</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# 2. Marine statistics (real counts only)
# --------------------------------------------------------------------------- #
def _marine_stats(registry: dict, stats: dict, available: list[str]) -> None:
    """Four statistic cards computed from stored runs and installed models."""
    sessions = stats["runs"]
    unique = f"{stats['unique_species']}" if sessions else "—"
    cards = [
        (
            "folder_open",
            "Total Detection Sessions",
            f"{sessions:,}",
            f"{stats['images']} image · {stats['videos']} video"
            if sessions
            else "begins on your first run",
        ),
        (
            "pets",
            "Unique Species Identified",
            unique,
            "distinct classes recorded",
        ),
        (
            "scan",
            "Total Organisms Detected",
            f"{stats['total_objects']:,}",
            "objects across all runs",
        ),
        (
            "memory",
            "Available AI Models",
            f"{len(available)}",
            f"{len(available)} installed weights · {len(registry)} registered"
            if available
            else "no weights installed yet",
        ),
    ]
    inner = "".join(
        '<div class="be-mstat"><div class="ic">' + span(icon) + "</div>"
        f'<div class="bd"><div class="v">{esc(value)}</div>'
        f'<div class="k">{esc(label)}</div><div class="h">{esc(hint)}</div></div></div>'
        for icon, label, value, hint in cards
    )
    st.markdown(f'<div class="be-mstatgrid">{inner}</div>', unsafe_allow_html=True)
    st.caption(
        "Figures are computed from your stored detection history and the "
        "installed model registry - no estimates."
    )


# --------------------------------------------------------------------------- #
# 3. Latest detection (original vs annotated vs details)
# --------------------------------------------------------------------------- #
def _latest_detection(entries: list, registry: dict, settings=None, user_id=None) -> None:
    """Show the newest real result with full media comparison, or an empty state."""
    st.markdown("### Latest Detection")
    st.markdown(
        '<p class="be-dash-lead">See how AI helps identify marine life.</p>',
        unsafe_allow_html=True,
    )

    # Only runs whose artefacts belong to the signed-in account are shown.
    latest = next(
        (
            e
            for e in entries
            if e.output
            and state.is_owned_path(settings, user_id, e.output)
            and Path(e.output).is_file()
        ),
        None,
    )
    if latest is None:
        empty_state(
            st,
            "search",
            "No detections yet",
            "Your latest detection - original input, annotated result and "
            "details - will appear here after your first run on the "
            "AI Detection page.",
        )
        if st.button(
            "Start a detection",
            type="primary",
            icon=":material/center_focus_strong:",
            key="dash_start_first",
        ):
            state.navigate("detect")
        return

    if latest.kind == "video" or Path(latest.output).suffix.lower() in VIDEO_SUFFIXES:
        _latest_video(latest, registry, settings, user_id)
    else:
        _latest_image(latest, registry, settings, user_id)


def _latest_image(entry, registry: dict, settings=None, user_id=None) -> None:
    """Original image | arrow | annotated image, plus a details panel."""
    out = Path(entry.output)
    out_uri = _image_data_uri(str(out))
    original = _resolve_input(entry, settings, user_id)

    if original is not None and original.suffix.lower() in IMAGE_SUFFIXES:
        panel_a = (
            f'<div class="media"><img src="{_image_data_uri(str(original))}" '
            'alt="Original input image"/></div>'
        )
    else:
        panel_a = (
            '<div class="media">'
            '<div class="ph">Original file is not available on this device.</div>'
            "</div>"
        )
    panel_b = (
        f'<div class="media"><img src="{out_uri}" '
        'alt="Annotated detection output"/></div>'
    )

    st.markdown(
        '<div class="be-cmp">'
        f'<div class="panel">{panel_a}'
        f'<div class="lbl">{span("image")} Original Image</div></div>'
        f'<div class="arrow">{span("arrow_forward")}</div>'
        f'<div class="panel">{panel_b}'
        f'<div class="lbl">{span("verified")} Detected Result</div></div>'
        f'<div class="panel panel--c">{_details_html(entry, registry)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Latest run · {Path(entry.filename).name} · {entry.display_time} · "
        f"{entry.total_detections} detection(s)"
    )


def _latest_video(entry, registry: dict, settings=None, user_id=None) -> None:
    """Annotated video (and original when its file is still on disk)."""
    out = Path(entry.output)
    original = _resolve_input(entry, settings, user_id)

    left, right = st.columns(2)
    with left:
        st.markdown("**Original Video**")
        if original is not None:
            st.video(read_file_bytes(str(original)))
        else:
            st.caption("Original file is not available on this device.")
    with right:
        st.markdown("**Detected Result**")
        st.video(read_file_bytes(str(out)))

    st.markdown(
        '<div class="be-cmp"><div class="panel panel--c panel--span">'
        f"{_details_html(entry, registry)}</div></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Latest run · {Path(entry.filename).name} · {entry.display_time} · "
        f"{entry.total_detections} detection(s)"
    )


def _details_html(entry, registry: dict) -> str:
    """Detection Details panel: real classes, confidence, models, time."""
    labels = {key: spec.display_name for key, spec in registry.items()}
    classes = list(entry.detections_per_class.items())

    chips_html = "".join(
        f'<span class="be-chip">{esc(f"{display_class_name(name)} ×{count}")}</span>'
        for name, count in classes[:6]
    )
    models = ", ".join(
        labels.get(m, display_class_name(m)) for m in entry.models
    ) or "—"
    top_class = display_class_name(classes[0][0]) if classes else "—"

    rows = [
        ("Media", entry.kind.capitalize()),
        ("Top category", top_class),
        (
            "Peak confidence",
            f"{entry.max_confidence * 100:.1f}%" if entry.max_confidence else "—",
        ),
        ("Detections", str(entry.total_detections)),
        ("Model(s)", models),
        ("Timestamp", entry.display_time),
    ]
    species = _species_for(entry)
    if species is not None:
        rows.append(("Scientific name", species.scientific_name))
    rows.append(("Enhanced", "Yes" if entry.enhanced else "No"))

    return (
        f'<div class="lbl">{span("info")} Detection Details</div>'
        f'<div class="detail">{chips_html}{kv(rows)}{_species_link(species)}</div>'
    )


def _species_for(entry) -> Species | None:
    """The catalogue species for a detected class, when a verified map exists."""
    for class_id in entry.detections_per_class:
        species = get_species(_CLASS_SPECIES.get(class_id, ""))
        if species is not None:
            return species
    return None


def _species_link(species: Species | None) -> str:
    """Deep link into the Species Explorer (only when a profile exists)."""
    if species is not None:
        href = f"?page=species&amp;species={esc(species.id)}"
        label = f"View {esc(species.common_name)} profile"
    else:
        href = "?page=species"
        label = "Browse species in the explorer"
    return (
        f'<a class="be-link" href="{href}">{span("menu_book")} {label}</a>'
    )


# --------------------------------------------------------------------------- #
# 4. Indian marine biodiversity (real catalogue profiles)
# --------------------------------------------------------------------------- #
def _indian_biodiversity() -> None:
    """Featured and mini species cards from the verified catalogue."""
    st.markdown("### Indian Marine Biodiversity")
    st.markdown(
        '<p class="be-dash-lead">Explore species found in Indian coastal waters.</p>',
        unsafe_allow_html=True,
    )

    featured = get_species(_INDIAN_FEATURED)
    if featured is not None:
        st.markdown(_featured_html(featured), unsafe_allow_html=True)

    minis = [spec for spec in (get_species(sid) for sid in _INDIAN_MINIS) if spec is not None]
    if minis:
        st.markdown(
            f'<div class="be-biogrid">{"".join(_mini_html(s) for s in minis)}</div>',
            unsafe_allow_html=True,
        )

    if st.button(
        "View More",
        icon=":material/menu_book:",
        key="dash_species_all",
    ):
        state.navigate("species")


def _featured_html(species: Species) -> str:
    """A large profile card for the featured Indian marine species."""
    status = species.conservation_status or "Not assessed"
    return (
        '<div class="be-bio-featured">'
        f'<div class="photo">{span(_MINI_ICONS.get(species.id, "water_drop"))}</div><div>'
        f"<h4>{esc(species.common_name)}</h4>"
        f'<p class="sci">{esc(species.scientific_name)}</p>'
        '<div class="badges">'
        f'{badge(species.group, "info")}{badge(species.habitat, "off")}'
        f"{badge(status, _status_kind(status))}</div>"
        f'<p class="meta">{esc(species.distribution)}</p>'
        f'<a class="be-link" href="?page=species&amp;species={esc(species.id)}">'
        f'{span("menu_book")} View profile</a>'
        "</div></div>"
    )


def _mini_html(species: Species) -> str:
    """A compact species card with conservation badge and profile link."""
    status = species.conservation_status or "Not assessed"
    icon = _MINI_ICONS.get(species.id, "water_drop")
    return (
        '<div class="be-bio-mini">'
        '<div class="row">'
        f'<div class="ic">{span(icon)}</div><div>'
        f"<h4>{esc(species.common_name)}</h4>"
        f'<p class="sci">{esc(species.scientific_name)}</p>'
        "</div></div>"
        '<div class="st">'
        f"{badge(status, _status_kind(status))} "
        f'<a class="be-view" href="?page=species&amp;species={esc(species.id)}">View →</a>'
        "</div></div>"
    )


#: Decorative icon per mini species (Material Symbols names).
_MINI_ICONS = {
    "green_sea_turtle": "water_drop",
    "olive_ridley_turtle": "water_drop",
    "dugong": "water",
    "whale_shark": "waves",
    "spotted_seahorse": "pets",
}


def _status_kind(status: str) -> str:
    """Map a catalogue conservation status to a badge colour."""
    kind = _CON_STATUS_KIND.get(status)
    if kind is not None:
        return kind
    lowered = status.lower()
    if "not" in lowered or "deficient" in lowered or "assess" in lowered:
        return "off"
    return "info"


# --------------------------------------------------------------------------- #
# 5. Marine biodiversity overview (real history, selectable range)
# --------------------------------------------------------------------------- #
def _biodiversity_overview(entries: list) -> None:
    """Detection-activity line chart + species bar chart from stored runs."""
    st.markdown("### Marine Biodiversity Overview")
    st.markdown(
        '<p class="be-dash-lead">Detection activity from your stored runs.</p>',
        unsafe_allow_html=True,
    )

    label = st.radio(
        "Time range",
        options=list(_RANGES),
        horizontal=True,
        key="dash_range",
        help="Window for the charts below, applied to the stored history timestamps.",
    )
    filtered = _filter_entries(entries, label)
    if not filtered:
        empty_state(
            st,
            "bar_chart",
            "Not enough data yet",
            "No stored runs fall in this range. Run a detection on the "
            "AI Detection page - finished runs are summarized here.",
        )
        return

    fstats = summarize(filtered)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Detection activity")
        _activity_chart(filtered)
        st.caption(
            f"{len(filtered)} run(s) in this range · newest {fstats['last_run']}"
        )
    with right:
        st.markdown("#### Detections by category")
        if fstats["species_counts"]:
            _species_chart(fstats["species_counts"])
        else:
            st.caption("No class labels were recorded for these runs.")

    if st.button(
        "View Analytics",
        icon=":material/bar_chart:",
        key="dash_open_analytics",
    ):
        state.navigate("analytics")


def _filter_entries(entries: list, label: str) -> list:
    """Keep only entries whose timestamp falls inside the selected window."""
    if label == "All time":
        return list(entries)
    days = 7 if "7" in label else 30
    cutoff = datetime.now() - timedelta(days=days)
    kept = []
    for entry in entries:
        try:
            timestamp = datetime.fromisoformat(entry.timestamp)
        except (ValueError, TypeError):
            continue
        if timestamp >= cutoff:
            kept.append(entry)
    return kept


def _activity_chart(entries: list) -> None:
    """Line chart of runs per calendar day (real timestamps)."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - streamlit depends on pandas
        return
    per_day = Counter()
    for entry in entries:
        try:
            per_day[datetime.fromisoformat(entry.timestamp).date().isoformat()] += 1
        except (ValueError, TypeError):
            continue
    if not per_day:
        st.caption("No usable timestamps in this range.")
        return
    days = sorted(per_day)
    frame = pd.DataFrame({"Runs": [per_day[day] for day in days]}, index=days)
    st.line_chart(frame, height=270)


def _species_chart(counts: dict) -> None:
    """Bar chart of detections per class (top 8, real counts)."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - streamlit depends on pandas
        return
    top = list(counts.items())[:8]
    frame = pd.DataFrame(
        {"Species": [display_class_name(name) for name, _ in top],
         "Objects": [count for _, count in top]}
    )
    st.bar_chart(frame.set_index("Species"), height=270)


# --------------------------------------------------------------------------- #
# 6. Recent discoveries (real stored runs)
# --------------------------------------------------------------------------- #
def _recent_discoveries(entries: list, registry: dict) -> None:
    """Compact list of the newest stored runs, with a link to History."""
    st.markdown("### Recent Discoveries")
    st.markdown(
        '<p class="be-dash-lead">The latest detections from your on-device history.</p>',
        unsafe_allow_html=True,
    )
    if not entries:
        empty_state(
            st,
            "pets",
            "No discoveries yet",
            "Finished detection runs appear here automatically with the "
            "species, confidence and model used.",
        )
        return

    rows = "".join(_discovery_row(entry, registry) for entry in entries[:RECENT_LIMIT])
    st.markdown(f'<div class="be-feed">{rows}</div>', unsafe_allow_html=True)
    if st.button(
        "View all detection history",
        icon=":material/history:",
        key="dash_history_all",
    ):
        state.navigate("history")


def _discovery_row(entry, registry: dict) -> str:
    """One discovery row: media, top species, confidence, model, date."""
    labels = {key: spec.display_name for key, spec in registry.items()}
    classes = list(entry.detections_per_class.items())
    top = display_class_name(classes[0][0]) if classes else "No class labels"
    confidence = (
        f" · {entry.max_confidence * 100:.1f}% confidence"
        if entry.max_confidence
        else ""
    )
    models = ", ".join(
        labels.get(m, display_class_name(m)) for m in entry.models[:2]
    ) or "—"
    icon = "videocam" if entry.kind == "video" else "image"
    return (
        '<div class="item">'
        f'<div class="tile">{span(icon)}</div>'
        '<div class="body">'
        f'<div class="fn">{esc(Path(entry.filename).name)}</div>'
        f'<div class="meta">{esc(f"{top}{confidence} · {models}")}</div>'
        "</div>"
        '<div class="st"><a class="be-view" href="?page=history">View →</a></div>'
        "</div>"
    )


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _resolve_input(entry, settings=None, user_id=None) -> Path | None:
    """Locate the original upload referenced by a history entry.

    Thin wrapper over :func:`app.ui.state.resolve_owned_input` so the
    dashboard and the History page locate the original media identically -
    and only ever the signed-in account's own files.
    """
    return state.resolve_owned_input(settings, user_id, entry.filename)


@st.cache_data(show_spinner=False)
def _image_data_uri(path: str) -> str:
    """A data URI for a local image so HTML panels can render it inline."""
    suffix = Path(path).suffix.lower()
    mime = _MIME_BY_SUFFIX.get(suffix, "image/jpeg")
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


__all__ = ["render"]