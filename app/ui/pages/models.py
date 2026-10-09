"""Models page: registry cards, status and installation actions.

Cards are generated from :meth:`app.detection.model_manager.ModelManager.registry`,
so a model registered in ``models/custom/registry.json`` appears here with
no code changes. Custom / regional models are described honestly: if no
weights are installed the card says so, and BlueEye never claims a model
exists when it does not.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from app.config import PROJECT_ROOT
from app.detection.detector import MarineDetector
from app.detection.model_manager import CUSTOM_REGISTRY_PATH, ModelError, ModelSpec
from app.ui import state
from app.ui.components import (
    badge,
    chips,
    download_button,
    error_card,
    hero,
    metrics_row,
    model_status_badge,
    notice,
    status_badge,
)
from app.ui.icons import span
from app.ui.theme import esc

logger = logging.getLogger(__name__)

#: Path of the extension file, relative to the project root (documentation).
CUSTOM_REGISTRY_FILE = "/".join(CUSTOM_REGISTRY_PATH)


def render(detector: MarineDetector, settings) -> None:
    """Render the model catalogue."""
    registry = state.registry(detector)
    ready = set(state.available_keys(detector))

    hero(
        st,
        "Model catalogue",
        "Marine Models",
        "Friendly names, real status and honest provenance for every model "
        "BlueEye can run.",
        icon="psychology",
    )

    metrics_row(
        st,
        [
            ("Registered", len(registry), "models in the registry"),
            ("Ready", len(ready), "weights installed"),
            ("Missing", len(registry) - len(ready), "needs download or training"),
            ("Device", str(detector.device).upper(), "inference device"),
        ],
    )
    st.write("")

    missing = [spec for key, spec in registry.items() if key not in ready]
    downloadable = [spec for spec in missing if spec.is_downloadable]
    if downloadable:
        _download_panel(detector, downloadable)

    core = [
        spec
        for spec in registry.values()
        if spec.category not in state.ADDITIONAL_CATEGORIES
    ]
    _model_grid(detector, core, ready)
    _additional_section(detector, registry, ready)
    _custom_section(detector, registry)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _download_panel(detector: MarineDetector, downloadable: list[ModelSpec]) -> None:
    notice(
        st,
        "info",
        "Some weights are not installed",
        f"{len(downloadable)} official model(s) can be downloaded on demand "
        "(~87 MB each). BlueEye never ships or fabricates weights.",
    )
    if st.button(
        "Download missing weights",
        type="primary",
        icon=":material/download:",
        width="stretch",
    ):
        try:
            with st.spinner("Downloading model weights..."):
                detector.model_manager.download_all()
            st.session_state.pop("model_verify", None)
            st.cache_resource.clear()
            st.rerun()
        except ModelError as exc:
            error_card(st, "Download failed", str(exc))
            logger.warning("Model download failed: %s", exc)


def _model_grid(detector: MarineDetector, specs: list[ModelSpec], ready: set) -> None:
    """Two-column card grid of the core registered models."""
    for row_start in range(0, len(specs), 2):
        row_specs = specs[row_start : row_start + 2]
        columns = st.columns(2)
        for column, spec in zip(columns, row_specs):
            with column:
                _model_card(detector, spec, spec.key in ready)
        st.write("")


def _model_card(detector: MarineDetector, spec: ModelSpec, available: bool) -> None:
    """One model card: identity, status, metadata and actions."""
    status = status_badge(available)
    classes = _class_names(detector, spec, available)

    metadata: list[tuple[str, str]] = [
        ("Identifier", spec.key),
        ("Classes", str(len(classes)) if classes else str(len(spec.classes))),
        (
            "Recommended threshold",
            f"{spec.recommended_confidence:.3f}" if spec.recommended_confidence else "—",
        ),
        ("Version", spec.version or "—"),
        ("Source", spec.source or "—"),
        ("License", spec.license or "—"),
        ("Type", "YOLOv8 (Ultralytics)"),
    ]

    body = (
        f"<p class='be-card-sub'>{esc(spec.summary or spec.description)}</p>"
        f"<div style='margin:.3rem 0 .5rem'>{status} "
        f"{badge('Custom', 'warn') if spec.custom else badge('Built-in', 'info')}</div>"
    )
    if classes:
        body += f"<div style='margin-bottom:.4rem'>{chips(classes)}</div>"
    body += (
        "<dl class='be-kv'>"
        + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in metadata)
        + "</dl>"
    )
    if spec.homepage:
        body += f"<p class='be-card-sub'>Reference: {esc(spec.homepage)}</p>"

    st.markdown(
        f"<div class='be-card'><div class='be-card-title'>{span('psychology')}"
        f"{esc(spec.display_name)}</div>{body}</div>",
        unsafe_allow_html=True,
    )

    if available:
        if st.button(
            "Use this model",
            icon=":material/check_circle:",
            key=f"use_{spec.key}",
            width="stretch",
        ):
            st.session_state["model_selection"] = spec.key
            state.navigate("detect")
    elif spec.is_downloadable:
        if st.button(
            "Download weights",
            icon=":material/download:",
            key=f"dl_{spec.key}",
            width="stretch",
        ):
            _download_one(detector, spec)
    else:
        st.caption("Not installed · place your weights and register them (below).")


def _download_one(detector: MarineDetector, spec: ModelSpec) -> None:
    try:
        with st.spinner(f"Downloading {spec.display_name}..."):
            detector.model_manager.download(spec.key)
        st.session_state.pop("model_verify", None)
        st.cache_resource.clear()
        st.rerun()
    except ModelError as exc:
        error_card(st, "Download failed", str(exc))
        logger.warning("Model download failed: %s", exc)


def _class_names(detector: MarineDetector, spec: ModelSpec, available: bool) -> list[str]:
    """Class names of a model, without loading weights unnecessarily."""
    if spec.classes:
        return list(spec.classes)
    if not available:
        return []
    try:
        names = detector.model_manager.get_class_names(spec.key)
        return [names[index] for index in sorted(names)]
    except Exception:  # noqa: BLE001 - card must render even if loading fails
        logger.debug("Could not read class names for %s", spec.key, exc_info=True)
        return []


# --------------------------------------------------------------------------- #
# Additional / Indian aquatic species models
# --------------------------------------------------------------------------- #
def _verified(detector: MarineDetector, spec: ModelSpec) -> dict:
    """Load + inference status for one model, cached per session.

    The cache key carries the weight file's mtime so a re-downloaded file is
    re-verified. Models known to need training / be incompatible answer
    immediately without loading anything (see ``ModelManager.verify``).
    """
    try:
        path = detector.model_manager.resolve_path(spec.key)
        stamp = path.stat().st_mtime_ns if path.is_file() else 0
    except Exception:  # noqa: BLE001 - never break the page
        stamp = 0

    cache = st.session_state.setdefault("model_verify", {})
    entry = cache.get(spec.key)
    if entry and entry.get("stamp") == stamp:
        return entry["result"]

    result = detector.verify_model(spec.key)
    cache[spec.key] = {"stamp": stamp, "result": result}
    return result


def _additional_section(detector: MarineDetector, registry: dict, ready: set) -> None:
    """'Additional Indian Aquatic Species Models' - optional, opt-in models."""
    specs = state.additional_specs(registry)
    if not specs:
        return

    st.markdown("### Additional Indian Aquatic Species Models")
    ready_count = sum(1 for spec in specs if spec.key in ready)
    notice(
        st,
        "info",
        "Optional, opt-in models",
        f"{ready_count} of {len(specs)} additional model(s) are installed. "
        "They are never run by 'Auto' - select one explicitly (or use "
        "'Every installed model') in Detect. Status reflects a real inference "
        "test, not just the presence of a config file.",
    )

    for row_start in range(0, len(specs), 2):
        row_specs = specs[row_start : row_start + 2]
        columns = st.columns(2)
        for column, spec in zip(columns, row_specs):
            with column:
                _additional_card(detector, spec, spec.key in ready)
        st.write("")


def _additional_card(detector: MarineDetector, spec: ModelSpec, available: bool) -> None:
    """Card for one additional model, with an honest readiness status."""
    if spec.readiness:
        result = {"status": spec.readiness, "ok": False, "error": None, "classes": []}
    else:
        result = _verified(detector, spec)
    status = str(result.get("status") or spec.readiness_status(available))
    classes = list(result.get("classes") or spec.classes)

    metadata: list[tuple[str, str]] = [
        ("Architecture", spec.architecture),
        ("Habitat", spec.habitat or "—"),
        ("Classes", str(len(classes)) if classes else "—"),
        ("Recommended threshold", f"{spec.recommended_confidence:.3f}"),
        ("Weights", "present" if available else "not installed"),
        ("Source", spec.source or "—"),
        ("License", spec.license or "—"),
    ]

    body = (
        f"<p class='be-card-sub'>{esc(spec.summary or spec.description)}</p>"
        f"<div style='margin:.3rem 0 .5rem'>{model_status_badge(status)} "
        f"{badge('Additional', 'info')}</div>"
    )
    if classes:
        body += f"<div style='margin-bottom:.4rem'>{chips(classes)}</div>"
    body += (
        "<dl class='be-kv'>"
        + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in metadata)
        + "</dl>"
    )
    if spec.status_note:
        body += f"<p class='be-card-sub'>{esc(spec.status_note)}</p>"
    if result.get("error"):
        body += f"<p class='be-card-sub'>Inference error: {esc(str(result['error']))}</p>"
    if spec.homepage:
        body += f"<p class='be-card-sub'>Reference: {esc(spec.homepage)}</p>"

    st.markdown(
        f"<div class='be-card'><div class='be-card-title'>{span(spec.icon)}"
        f"{esc(spec.display_name)}</div>{body}</div>",
        unsafe_allow_html=True,
    )

    if status == "ready":
        if st.button(
            "Use this model",
            icon=":material/check_circle:",
            key=f"use_{spec.key}",
            width="stretch",
        ):
            st.session_state["model_selection"] = spec.key
            state.navigate("detect")
    elif status == "not_installed" and spec.is_downloadable:
        if st.button(
            "Download weights",
            icon=":material/download:",
            key=f"dl_{spec.key}",
            width="stretch",
        ):
            _download_one(detector, spec)
    else:
        st.caption("Not runnable yet - see the note above.")


# --------------------------------------------------------------------------- #
# Custom / regional models
# --------------------------------------------------------------------------- #
def _custom_section(detector: MarineDetector, registry: dict) -> None:
    """Explain the extension seam for regional and future custom models."""
    custom = [
        spec
        for spec in registry.values()
        if spec.custom and spec.category not in state.ADDITIONAL_CATEGORIES
    ]

    st.markdown("### Regional and custom models")
    if custom:
        notice(
            st,
            "info",
            f"{len(custom)} custom model(s) registered",
            "They appear as cards above and can be selected in Detect. "
            "They were added through models/custom/registry.json.",
        )
    else:
        notice(
            st,
            "info",
            "No user custom model registered",
            "BlueEye supports regional / local species models through "
            "models/custom/registry.json, but no user model is registered. "
            "The optional pretrained models above are listed separately. "
            "See docs/indian_biodiversity.md for the Indian freshwater / "
            "coastal roadmap and which models still need training.",
        )

    with st.expander("How to add a regional or custom model", expanded=False):
        st.markdown(
            """
**BlueEye does not hard-code its models.** Any YOLOv8 checkpoint can be added
without touching application code:

1. Train (or obtain) a `.pt` model — see `docs/regional_models.md` for the full
   dataset → training → evaluation workflow.
2. Copy the weights under `models/`, for example
   `models/regional/karnataka/BlueEyeRegional.pt`.
3. Describe it in `models/custom/registry.json` (a commented template lives at
   `models/custom/registry.json.example`).
4. Restart BlueEye — the model appears here, in Detect and in the CLI. It runs
   in `auto` mode unless its entry sets `"auto": false`.

Example entry:

```json
{
  "models": [
    {
      "id": "regional_karnataka",
      "name": "Karnataka Coastal Species",
      "weights": "regional/karnataka/BlueEyeRegional.pt",
      "classes": ["tuna", "mackerel", "shark"],
      "version": "0.1",
      "type": "yolov8",
      "recommended_confidence": 0.5,
      "source": "Trained in-house on a local dataset"
    }
  ]
}
```

Registry file: `PROJECT_ROOT / """ + CUSTOM_REGISTRY_FILE + """.` Invalid entries are
logged and skipped — a broken file never stops the built-in models from
working.

**Indian species** (freshwater fish, Gangetic river dolphin, freshwater turtle,
gharial and coastal species): ready-to-edit dataset configs live in
`training/datasets/`, and the verified datasets, licences and remaining gaps
are documented in `docs/indian_biodiversity.md`. BlueEye ships **no** Indian
weights — each model must be trained and evaluated before it is registered.
"""
        )

    example_path = PROJECT_ROOT / "models" / "custom" / "registry.json.example"
    if example_path.exists():
        download_button(
            st,
            "Download registry.json.example",
            example_path,
            "application/json",
        )


__all__ = ["render"]
