"""Reusable BlueEye UI components.

Every page is assembled from these primitives so that cards, metrics,
badges, notices and tables look and behave identically everywhere. They are
pure presentation: none of them runs inference or changes detection
behaviour.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import streamlit as st

from app.detection.results import display_class_name
from app.ui.icons import material, span
from app.ui.theme import esc

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #
def hero(container, tag: str, title: str, subtitle: str, icon: str | None = None) -> None:
    """Large branded header band (``icon`` is a Material Symbols name)."""
    tag_html = f"{span(icon)}{esc(tag)}" if icon else esc(tag)
    container.markdown(
        f'<div class="be-hero"><div class="be-tag">{tag_html}</div>'
        f"<h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def section_title(container, title: str, subtitle: str | None = None) -> None:
    """Section heading with an optional muted sub-line."""
    sub = f'<p class="be-card-sub">{esc(subtitle)}</p>' if subtitle else ""
    container.markdown(f'<div class="be-card-title">{esc(title)}</div>{sub}', unsafe_allow_html=True)


def step(container, number: int, title: str) -> None:
    """Numbered workflow step marker."""
    container.markdown(
        f'<div class="be-step"><span class="n">{number}</span>'
        f'<span class="t">{esc(title)}</span></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Data display
# --------------------------------------------------------------------------- #
def metric_card(
    container,
    label: str,
    value: Any,
    hint: str = "",
    accent: bool = False,
) -> None:
    """Single statistic card (label / value / optional hint)."""
    accent_class = " be-metric--accent" if accent else ""
    hint_html = f'<div class="h">{esc(hint)}</div>' if hint else ""
    container.markdown(
        f'<div class="be-metric{accent_class}">'
        f'<div class="k">{esc(label)}</div>'
        f'<div class="v">{esc(value)}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def metrics_row(container, items: Sequence[tuple[str, Any, str]], accent_first: bool = True) -> None:
    """Render a row of up to four metric cards."""
    if not items:
        return
    columns = container.columns(min(len(items), 4))
    for index, (label, value, hint) in enumerate(items[:4]):
        metric_card(columns[index], label, value, hint, accent=accent_first and index == 0)


def kv(pairs: Iterable[tuple[str, Any]]) -> str:
    """Key/value list as HTML (empty string when there are no pairs)."""
    rows = [(esc(key), esc(value)) for key, value in pairs if str(value).strip() != ""]
    if not rows:
        return ""
    inner = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    return f'<dl class="be-kv">{inner}</dl>'


def chips(items: Iterable[str]) -> str:
    """Render a list of small class-name chips as HTML."""
    return "".join(f'<span class="be-chip">{esc(item)}</span>' for item in items)


def badge(text: str, kind: str = "info", dot: bool = False) -> str:
    """Status badge (``ok`` / ``warn`` / ``off`` / ``info``) as HTML.

    With ``dot=True`` a small status dot is rendered before the text.
    """
    safe_kind = kind if kind in {"ok", "warn", "off", "info"} else "info"
    dot_html = '<span class="be-dot"></span>' if dot else ""
    return f'<span class="be-badge be-badge--{safe_kind}">{dot_html}{esc(text)}</span>'


def status_badge(available: bool, loading: bool = False) -> str:
    """Model availability badge: Ready / Loading / Not installed."""
    if loading:
        return badge("Loading", "warn", dot=True)
    if available:
        return badge("Ready", "ok", dot=True)
    return badge("Not installed", "off", dot=True)


#: Display label + badge kind for each model readiness status.
MODEL_STATUS_STYLE = {
    "ready": ("Ready", "ok"),
    "not_installed": ("Not installed", "off"),
    "needs_training": ("Needs training", "warn"),
    "incompatible": ("Incompatible", "warn"),
}


def model_status_badge(status: str, dot: bool = True) -> str:
    """Badge for a model readiness status.

    ``status`` is one of ``ready`` / ``not_installed`` / ``needs_training`` /
    ``incompatible`` (see :meth:`ModelSpec.readiness_status`).
    """
    label, kind = MODEL_STATUS_STYLE.get(
        status, (status.replace("_", " ").title() or "Unknown", "info")
    )
    return badge(label, kind, dot=dot)


# --------------------------------------------------------------------------- #
# Notices / states
# --------------------------------------------------------------------------- #
#: Default icon per notice kind (Material Symbols name).
NOTICE_ICONS = {"error": "error", "warning": "warning", "info": "info"}


def notice(container, kind: str, title: str, body: str = "", icon: str | None = None) -> None:
    """Coloured notice block (``error`` / ``warning`` / ``info``).

    ``icon`` overrides the kind's default Material Symbols icon.
    """
    safe_kind = kind if kind in NOTICE_ICONS else "info"
    icon_name = icon or NOTICE_ICONS[safe_kind]
    body_html = f"<p>{esc(body)}</p>" if body else ""
    container.markdown(
        f'<div class="be-notice be-notice--{safe_kind}">'
        f"<h3>{span(icon_name)}{esc(title)}</h3>{body_html}</div>",
        unsafe_allow_html=True,
    )


def error_card(
    container,
    title: str,
    message: str,
    hint: str | None = None,
    log: bool = True,
) -> None:
    """Professional error block: friendly text for the user, detail in logs.

    The technical message is still written to the log so developers can
    diagnose problems (nothing is hidden, nothing is a raw traceback).
    """
    if log:
        logger.warning("%s: %s", title, message)
    body = message if hint is None else f"{message}\n\n{hint}"
    notice(container, "error", title, body)


def empty_state(container, icon: str, title: str, body: str) -> None:
    """Placeholder shown when a page has no data yet.

    ``icon`` is a Material Symbols name (for example ``"image"``); it is
    rendered inside a tinted tile so empty states look intentional rather
    than blank.
    """
    container.markdown(
        f'<div class="be-card be-card--muted"><div class="be-empty">'
        f'<div class="tile">{span(icon)}</div>'
        f'<div class="be-card-title">{esc(title)}</div>'
        f'<p class="be-card-sub" style="margin:0">{esc(body)}</p></div></div>',
        unsafe_allow_html=True,
    )


def card(container, title: str, body_html: str = "", subtitle: str = "") -> None:
    """Generic white card with a title."""
    sub = f'<p class="be-card-sub">{esc(subtitle)}</p>' if subtitle else ""
    container.markdown(
        f'<div class="be-card"><div class="be-card-title">{esc(title)}</div>'
        f"{sub}{body_html}</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Detection-specific components
# --------------------------------------------------------------------------- #
def detection_rows(detections: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert report detections into table rows with friendly names."""
    rows = []
    for index, det in enumerate(detections, start=1):
        raw = str(det.get("class", det.get("class_name", "?")))
        bbox = det.get("bbox") or det.get("bounding_box")
        rows.append(
            {
                "#": index,
                "Species": display_class_name(raw),
                "Class": raw,
                "Confidence": f"{float(det.get('confidence', 0)) * 100:.1f}%",
                "Location": str(bbox),
                "Model": str(det.get("model", "")),
            }
        )
    return rows


def detection_table(container, detections: Sequence[dict[str, Any]]) -> None:
    """Table of detections (species / confidence / box / model)."""
    rows = detection_rows(detections)
    if rows:
        container.dataframe(rows, hide_index=True, width="stretch")


def species_chart(container, per_class: dict[str, int]) -> None:
    """Bar chart of detections per species (real counts only)."""
    if not per_class:
        return
    container.bar_chart({display_class_name(k): v for k, v in per_class.items()}, width="stretch")


def download_button(
    container,
    label: str,
    path: str | Path,
    mime: str,
    icon: str | None = None,
) -> bool:
    """Download button for a file on disk; silent no-op when missing.

    ``icon`` is a Material Symbols name (defaults to ``download``).
    """
    file_path = Path(path)
    if not file_path.exists():
        container.caption(f"File not available: {file_path.name}")
        return False
    return bool(
        container.download_button(
            label=label,
            data=file_path.read_bytes(),
            file_name=file_path.name,
            mime=mime,
            icon=icon or material("download"),
            width="stretch",
        )
    )
