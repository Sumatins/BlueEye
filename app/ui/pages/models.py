"""Models page: the single source of truth for the model catalogue.

Cards are generated from :meth:`app.detection.model_manager.ModelManager.registry`,
so a model registered in ``models/custom/registry.json`` appears here with no
code changes. Models are grouped into three honest buckets:

* **Ready and verified** - weights exist *and* a real load + inference test
  succeeded in this session;
* **Downloadable but not yet verified** - an official checkpoint URL exists but
  the weights are not installed here yet;
* **Unavailable or incompatible** - no usable weights, or an unsupported
  format. Entries that cannot run live in
  ``models/custom/inactive_models.json`` and never appear here.

BlueEye never claims a model exists, loads, runs or was evaluated when it was
not: status always comes from the real registry, the on-disk weights and a real
inference test. Evaluation metrics are shown only when the registry records
them.

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

#: Runtime label per Ultralytics loader.
RUNTIME_LABEL = {"yolo": "Ultralytics YOLO", "rtdetr": "Ultralytics RT-DETR"}


def render(detector: MarineDetector, settings) -> None:
    """Render the model catalogue."""
    registry = state.registry(detector)
    available = set(state.available_keys(detector))

    # Real load + inference status for every installed model (cached per
    # session, keyed by the weight file's mtime). Uninstalled / overridden
    # entries answer immediately without loading anything.
    results = {key: _verified(detector, spec) for key, spec in registry.items()}
    ready_all, downloadable_all, unavailable_all = _categorise(
        list(registry.values()), results
    )

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
            ("Registered", len(registry), "models in the active registry"),
            ("Ready & verified", len(ready_all), "weights load + inference tested"),
            ("Downloadable", len(downloadable_all), "official weights not installed"),
            ("Unavailable", len(unavailable_all), "training or format required"),
        ],
    )
    st.caption(
        f"Inference device: {str(detector.device).upper()} · a model is marked "
        "'Ready' only after a real load and inference test - never from the file "
        "merely existing on disk."
    )
    st.write("")

    visible = _apply_filters(registry, results, available)
    if not visible:
        notice(
            st,
            "info",
            "No models match the filters",
            "Adjust or clear the readiness / habitat filters to see the catalogue.",
        )
        _custom_explainer()
        return

    missing_downloadable = list(downloadable_all)
    if missing_downloadable:
        _download_panel(detector, missing_downloadable)

    ready_specs, downloadable_specs, unavailable_specs = _categorise(visible, results)
    _section(detector, "Ready and verified", ready_specs, available, results)
    _downloadable_section(detector, downloadable_specs, available, results)
    _unavailable_section(detector, unavailable_specs, available, results)
    _custom_explainer()


# --------------------------------------------------------------------------- #
# Categorisation / filtering
# --------------------------------------------------------------------------- #
def _apply_filters(
    registry: dict[str, ModelSpec], results: dict[str, dict], available: set
) -> list[ModelSpec]:
    """Filter the catalogue by verified status and habitat (real fields only)."""
    habitats = sorted({spec.habitat for spec in registry.values() if spec.habitat})

    filter_columns = st.columns([1, 1])
    with filter_columns[0]:
        status_filter = st.multiselect(
            "Filter by status",
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
    for key, spec in registry.items():
        status = results.get(key, {}).get("status") or spec.readiness_status(key in available)
        if status_filter and status not in status_filter:
            continue
        if habitat_filter and spec.habitat not in habitat_filter:
            continue
        selected.append(spec)
    return selected


def _categorise(
    specs: list[ModelSpec], results: dict[str, dict]
) -> tuple[list[ModelSpec], list[ModelSpec], list[ModelSpec]]:
    """Split specs into ready / downloadable / unavailable buckets."""
    ready: list[ModelSpec] = []
    downloadable: list[ModelSpec] = []
    unavailable: list[ModelSpec] = []
    for spec in specs:
        status = str(results.get(spec.key, {}).get("status") or "")
        if status == "ready":
            ready.append(spec)
        elif status == "not_installed" and spec.is_downloadable:
            downloadable.append(spec)
        else:
            unavailable.append(spec)
    return ready, downloadable, unavailable


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _section(detector, title, specs, available, results) -> None:
    if not specs:
        return
    st.markdown(f"### {title}")
    _grid(detector, specs, available, results)


def _downloadable_section(detector, specs, available, results) -> None:
    if not specs:
        return
    st.markdown("### Downloadable but not yet verified")
    st.caption(
        "An official checkpoint is available on demand but is not installed "
        "here, so it has not been tested. BlueEye never ships or fabricates "
        "weights."
    )
    _grid(detector, specs, available, results)


def _unavailable_section(detector, specs, available, results) -> None:
    if not specs:
        return
    st.markdown("### Unavailable or incompatible")
    st.caption(
        "No usable weights are installed (or the published artefact is in an "
        "unsupported format), so these are not selectable in Detect. They stay "
        "visible so the gap is clear rather than hidden."
    )
    _grid(detector, specs, available, results)


def _grid(detector, specs, available, results) -> None:
    for row_start in range(0, len(specs), 2):
        row_specs = specs[row_start : row_start + 2]
        columns = st.columns(2)
        for column, spec in zip(columns, row_specs):
            with column:
                _card(detector, spec, spec.key in available, results.get(spec.key, {}))
        st.write("")


# --------------------------------------------------------------------------- #
# Card
# --------------------------------------------------------------------------- #
def _card(detector: MarineDetector, spec: ModelSpec, available: bool, result: dict) -> None:
    """One model card with identity, real status, metadata and an action."""
    status = str(result.get("status") or spec.readiness_status(available))
    classes = list(result.get("classes") or spec.classes)
    error = result.get("error")

    image_verified = status == "ready"
    video_verified = bool(spec.video_verified)

    body = f"<p class='be-card-sub'>{esc(spec.summary or spec.description)}</p>"
    body += (
        f"<div style='margin:.3rem 0 .5rem'>{model_status_badge(status)} "
        f"{badge(spec.key, 'info')}</div>"
    )
    if classes:
        body += f"<div style='margin-bottom:.4rem'>{chips(classes)}</div>"
    body += _metadata_block(
        [
            ("Architecture", spec.architecture),
            ("Runtime", RUNTIME_LABEL.get(spec.loader, spec.loader)),
            ("Habitat", spec.habitat or "—"),
            ("Recommended threshold", _threshold(spec)),
            ("Weights", "present" if available else "not installed"),
            ("Source", spec.source or "—"),
            ("License", spec.license or "—"),
            ("Image inference", "verified" if image_verified else "not verified"),
            ("Video inference", "verified" if video_verified else "not tested here"),
            ("Last inference test", _test_summary(status, error)),
        ]
    )
    body += _metrics_block(spec)
    if spec.limitations:
        body += f"<p class='be-card-sub'>Limitations: {esc(spec.limitations)}</p>"
    if spec.status_note and spec.status_note != spec.limitations:
        body += f"<p class='be-card-sub'>{esc(spec.status_note)}</p>"
    if error:
        body += f"<p class='be-card-sub'>Test error: {esc(str(error))}</p>"

    st.markdown(
        f"<div class='be-card'><div class='be-card-title'>{_icon_html(spec.icon)}"
        f"{esc(spec.display_name)}</div>{body}</div>",
        unsafe_allow_html=True,
    )
    if spec.homepage:
        st.caption(f"Source: {spec.homepage}")

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
        st.caption("Not runnable here - see the status and note above.")


def _test_summary(status: str, error) -> str:
    if status == "ready":
        return "load + inference succeeded"
    if status == "not_installed":
        return "not installed"
    if status == "needs_training":
        return "no trained weights yet"
    if status == "incompatible":
        return esc(str(error)) if error else "unsupported format"
    return "not tested"


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


def _icon_html(icon: str) -> str:
    """Render a spec icon: built-ins use emoji, custom models Material names."""
    if not icon:
        return ""
    if icon.isascii():
        return span(icon)
    return f"<span aria-hidden='true'>{esc(icon)}</span> "


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
def _download_panel(detector: MarineDetector, downloadable: list[ModelSpec]) -> None:
    notice(
        st,
        "info",
        "Some weights are not installed",
        f"{len(downloadable)} official model(s) can be downloaded on demand. "
        "BlueEye never ships or fabricates weights.",
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
   evaluation results on this page, and `"video_verified": true` only once you
   have actually tested it on a video.
4. Restart BlueEye — the model appears here, in Detect and in the CLI.

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
are documented in `docs/indian_biodiversity.md`. These are inactive research
targets — no Indian weights are shipped, and they are kept out of the active
registry until a real model is trained and evaluated. See
`models/custom/inactive_models.json`.
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
