"""Models page: registry cards, honest status and installation actions.

Cards are generated from :meth:`app.detection.model_manager.ModelManager.registry`,
so a model registered in ``models/custom/registry.json`` appears here with
no code changes. Models are grouped so users can tell apart the original
built-in models, additional pretrained additions, locally trained models,
models that still need training and incompatible formats.

BlueEye never claims a model exists, is trained or was evaluated when it was
not: status always comes from the real registry + on-disk weights, and
evaluation metrics are shown only when the registry actually records them.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging

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

#: Human labels for the readiness statuses (used by the filter + summaries).
STATUS_LABEL = {
    "ready": "Ready",
    "not_installed": "Not installed",
    "needs_training": "Needs training",
    "incompatible": "Incompatible",
}


def render(detector: MarineDetector, settings) -> None:
    """Render the model catalogue."""
    registry = state.registry(detector)
    ready = set(state.available_keys(detector))
    counts = _status_counts(registry, ready)

    hero(
        st,
        "Model catalogue",
        "Aquatic Species Models",
        "Friendly names, real status and honest provenance for every model "
        "BlueEye can run.",
        icon="psychology",
    )

    metrics_row(
        st,
        [
            ("Registered", len(registry), "models in the registry"),
            ("Ready", counts["ready"], "weights load + inference tested"),
            ("Needs training", counts["needs_training"], "no usable weights yet"),
            ("Incompatible", counts["incompatible"], "unsupported format"),
        ],
    )
    st.caption(
        f"Inference device: {str(detector.device).upper()} · "
        f"{counts['not_installed']} model(s) not installed · weights are fetched "
        "on demand or placed by you, never fabricated."
    )
    st.write("")

    visible = _apply_filters(registry, ready)
    if not visible:
        notice(
            st,
            "info",
            "No models match the filters",
            "Adjust or clear the readiness / habitat filters to see the catalogue.",
        )
        _custom_explainer()
        return

    missing_downloadable = [
        spec
        for key, spec in registry.items()
        if key not in ready and spec.is_downloadable
    ]
    if missing_downloadable:
        _download_panel(detector, missing_downloadable)

    originals, additional, local, needs_training, incompatible = _categorise(visible, ready)

    _grid_section(detector, "Original models", originals, ready, _model_card)
    _additional_section(detector, additional, ready)
    _local_section(detector, local, ready)
    _training_section(detector, needs_training, ready)
    _incompatible_section(detector, incompatible, ready)
    _custom_explainer()


# --------------------------------------------------------------------------- #
# Categorisation / filtering
# --------------------------------------------------------------------------- #
def _status_counts(registry: dict[str, ModelSpec], ready: set) -> dict[str, int]:
    counts = {key: 0 for key in STATUS_LABEL}
    for spec in registry.values():
        counts[spec.readiness_status(spec.key in ready)] += 1
    return counts


def _apply_filters(registry: dict[str, ModelSpec], ready: set) -> list[ModelSpec]:
    """Filter the catalogue by readiness status and habitat (real fields only)."""
    habitats = sorted({spec.habitat for spec in registry.values() if spec.habitat})

    filter_columns = st.columns([1, 1])
    with filter_columns[0]:
        status_filter = st.multiselect(
            "Filter by readiness",
            options=list(STATUS_LABEL),
            format_func=lambda value: STATUS_LABEL[value],
            default=[],
            key="models_status_filter",
        )
    with filter_columns[1]:
        habitat_filter = st.multiselect(
            "Filter by habitat",
            options=habitats,
            default=[],
            key="models_habitat_filter",
        )

    selected = []
    for spec in registry.values():
        if status_filter and spec.readiness_status(spec.key in ready) not in status_filter:
            continue
        if habitat_filter and spec.habitat not in habitat_filter:
            continue
        selected.append(spec)
    return selected


def _categorise(
    specs: list[ModelSpec], ready: set
) -> tuple[list[ModelSpec], list[ModelSpec], list[ModelSpec], list[ModelSpec], list[ModelSpec]]:
    """Split specs into the five honest buckets shown on the page."""
    originals: list[ModelSpec] = []
    additional: list[ModelSpec] = []
    local: list[ModelSpec] = []
    needs_training: list[ModelSpec] = []
    incompatible: list[ModelSpec] = []

    for spec in specs:
        status = spec.readiness_status(spec.key in ready)
        if not spec.custom:
            originals.append(spec)
        elif status == "needs_training":
            needs_training.append(spec)
        elif status == "incompatible":
            incompatible.append(spec)
        elif spec.category in state.ADDITIONAL_CATEGORIES:
            additional.append(spec)
        else:
            local.append(spec)
    return originals, additional, local, needs_training, incompatible


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _grid_section(detector, title, specs, ready, card_fn) -> None:
    if not specs:
        return
    st.markdown(f"### {title}")
    _grid(detector, specs, ready, card_fn)


def _grid(detector, specs, ready, card_fn) -> None:
    for row_start in range(0, len(specs), 2):
        row_specs = specs[row_start : row_start + 2]
        columns = st.columns(2)
        for column, spec in zip(columns, row_specs):
            with column:
                card_fn(detector, spec, spec.key in ready)
        st.write("")


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


def _additional_section(detector: MarineDetector, specs: list[ModelSpec], ready: set) -> None:
    """Additional pretrained (and download-on-demand) aquatic models."""
    if not specs:
        return

    st.markdown("### Additional pretrained models")
    st.caption(
        "Optional, opt-in models. They are never run by 'Auto' - select one "
        "explicitly (or 'Every installed model') in Detect. Status reflects a "
        "real load + inference test, not the presence of a config file."
    )
    _grid(detector, specs, ready, _additional_card)


def _local_section(detector: MarineDetector, specs: list[ModelSpec], ready: set) -> None:
    st.markdown("### Locally trained / custom models")
    if specs:
        notice(
            st,
            "info",
            f"{len(specs)} custom model(s) registered",
            "Added through models/custom/registry.json and selectable in Detect.",
        )
        _grid(detector, specs, ready, _additional_card)
    else:
        notice(
            st,
            "info",
            "No locally trained model registered",
            "Train a model and register it in models/custom/registry.json; it "
            "will appear here with its real metrics. See docs/regional_models.md.",
        )


def _training_section(detector: MarineDetector, specs: list[ModelSpec], ready: set) -> None:
    if not specs:
        return
    st.markdown("### Models that require training")
    st.caption(
        "No trustworthy pretrained weights for these targets were found. They "
        "stay here - honestly - until a licensed dataset is obtained and a "
        "model is genuinely trained and evaluated."
    )
    _grid(detector, specs, ready, _additional_card)


def _incompatible_section(detector: MarineDetector, specs: list[ModelSpec], ready: set) -> None:
    if not specs:
        return
    st.markdown("### Incompatible model formats")
    st.caption(
        "Published in a format the current Ultralytics pipeline cannot run "
        "(for example a device-specific NPU export). Kept visible so the gap "
        "is clear rather than hidden."
    )
    _grid(detector, specs, ready, _additional_card)


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #
def _model_card(detector: MarineDetector, spec: ModelSpec, available: bool) -> None:
    """Card for an original / built-in model: identity, status, metadata."""
    status = status_badge(available)
    classes = _class_names(detector, spec, available)

    body = (
        f"<p class='be-card-sub'>{esc(spec.summary or spec.description)}</p>"
        f"<div style='margin:.3rem 0 .5rem'>{status} "
        f"{badge('Original', 'info')}</div>"
    )
    body += _class_block(classes, spec)
    body += _metadata_block(
        [
            ("Identifier", spec.key),
            ("Architecture", spec.architecture),
            ("Habitat", spec.habitat or "—"),
            ("Recommended threshold", _threshold(spec)),
            ("Version", spec.version or "—"),
            ("Source", spec.source or "—"),
            ("License", spec.license or "—"),
        ]
    )
    body += _metrics_block(spec)
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
            st.session_state["detect_media"] = "Image"
            state.navigate("detect_image")
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


def _additional_card(detector: MarineDetector, spec: ModelSpec, available: bool) -> None:
    """Card for an additional / training / incompatible model, with a real status."""
    if spec.readiness:
        result = {"status": spec.readiness, "ok": False, "error": None, "classes": []}
    else:
        result = _verified(detector, spec)
    status = str(result.get("status") or spec.readiness_status(available))
    classes = list(result.get("classes") or spec.classes)

    label = {
        "additional": ("Additional", "info"),
        "indian": ("Indian target", "warn"),
    }.get(spec.category, ("Custom", "info"))

    body = (
        f"<p class='be-card-sub'>{esc(spec.summary or spec.description)}</p>"
        f"<div style='margin:.3rem 0 .5rem'>{model_status_badge(status)} "
        f"{badge(label[0], label[1])}</div>"
    )
    body += _class_block(classes, spec)
    body += _metadata_block(
        [
            ("Architecture", spec.architecture),
            ("Habitat", spec.habitat or "—"),
            ("Recommended threshold", _threshold(spec)),
            ("Weights", "present" if available else "not installed"),
            ("Source", spec.source or "—"),
            ("License", spec.license or "—"),
        ]
    )
    body += _metrics_block(spec)
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
            st.session_state["detect_media"] = "Image"
            state.navigate("detect_image")
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


def _class_block(classes: list[str], spec: ModelSpec) -> str:
    if classes:
        return f"<div style='margin-bottom:.4rem'>{chips(classes)}</div>"
    known = spec.classes
    if known:
        return f"<div style='margin-bottom:.4rem'>{chips(known)}</div>"
    return ""


def _metadata_block(rows: list[tuple[str, str]]) -> str:
    return (
        "<dl class='be-kv'>"
        + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows)
        + "</dl>"
    )


def _metrics_block(spec: ModelSpec) -> str:
    """Evaluation metrics - only when genuinely recorded for the model."""
    if not spec.metrics:
        return ""
    items = " ".join(
        badge(f"{label}: {value}", "ok") for label, value in spec.metrics
    )
    return (
        f"<div style='margin:.3rem 0'><span class='be-card-sub'>Evaluation: </span>"
        f"{items}</div>"
    )


def _threshold(spec: ModelSpec) -> str:
    return f"{spec.recommended_confidence:.3f}" if spec.recommended_confidence else "—"


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


# --------------------------------------------------------------------------- #
# Custom model extension seam
# --------------------------------------------------------------------------- #
def _custom_explainer() -> None:
    """Explain how to register a regional / custom model."""
    with st.expander("How to add a regional or custom model", expanded=False):
        st.markdown(
            """
**BlueEye does not hard-code its models.** Any supported checkpoint can be
added without touching application code:

1. Train (or obtain) a `.pt` model — see `docs/regional_models.md` for the full
   dataset → training → evaluation workflow.
2. Copy the weights under `models/`, for example
   `models/regional/karnataka/BlueEyeRegional.pt`.
3. Describe it in `models/custom/registry.json` (a commented template lives at
   `models/custom/registry.json.example`). Add a `"metrics"` object to show
   evaluation results on this page.
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
      "category": "regional",
      "habitat": "Marine (Karnataka coast)",
      "metrics": {"precision": "0.81", "recall": "0.74", "mAP@50": "0.79"},
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
