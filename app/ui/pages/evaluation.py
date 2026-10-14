"""Research Evaluation Center: run the real evaluation pipeline from the UI.

This page is the interface over :mod:`app.evaluation` and the *same* code path
as ``scripts/run_evaluation.py``. It lets an authenticated user:

* register a labelled YOLO dataset (auto-discovered or an explicit path) and
  validate it *before* anything is run;
* select compatible, installed models and configure a real run;
* execute the run in the background, watch its lifecycle and, if needed,
  cancel it;
* keep a dataset/model pair under **continuous evaluation** - a schedule
  re-runs when the dataset or the weights change, on an interval, or both;
* inspect genuinely measured metrics, per-class results and real plots;
* export publication-ready Markdown / LaTeX / CSV tables, dataset cards, a
  provenance sheet and a self-contained printable report;
* revisit, compare and export experiments after a restart.

Honesty rules enforced here
--------------------------
* No dataset is bundled or downloaded. When none is found the page explains how
  to configure one and stays **pending** - it never invents a score.
* Metrics are only shown when a real run produced them, and the wording makes
  clear they were *measured locally* on the named dataset/split.
* Published reference values are never mixed in, and one model's weights are
  never used to stand in for another's.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from app.detection.detector import MarineDetector
from app.evaluation.compat import compare_classes
from app.evaluation.discovery import (
    available_splits,
    default_search_roots,
    discover_datasets,
    find_data_yaml,
)
from app.evaluation.environment import (
    collect_environment,
    evaluator_version,
    git_commit,
)
from app.evaluation.experiments import (
    ACTIVE_STATUSES,
    EXPERIMENT_STATUSES,
    EvaluationStore,
    ExperimentRecord,
    file_fingerprint,
)
from app.evaluation.jobs import (
    ExperimentRequest,
    is_running,
    request_cancel,
    start_experiment,
)
from app.evaluation import scheduler as eval_scheduler
from app.evaluation.publication import (
    HEADLINE_COLUMNS,
    PER_CLASS_COLUMNS,
    build_bundle,
    headline_rows,
    per_class_rows,
    render_html as render_publication_html,
)
from app.evaluation.schedules import (
    MAX_INTERVAL_MINUTES,
    MIN_INTERVAL_MINUTES,
    TRIGGER_LABELS,
)
from app.evaluation.validation import (
    DatasetValidation,
    DatasetValidationError,
    validate_dataset,
)
from app.ui import state
from app.ui.components import (
    badge,
    download_button,
    empty_state,
    error_card,
    hero,
    metrics_row,
    notice,
)

logger = logging.getLogger(__name__)

#: Default number of images per evaluation (keeps the UI responsive).
DEFAULT_LIMIT = 25

#: Top-level sections of the research workspace.
SECTIONS = ("Run", "Experiments", "Schedules", "Datasets", "Compare", "Export")

#: Badge kind for each experiment status.
_STATUS_KIND = {
    "configured": "info",
    "validating": "warn",
    "queued": "warn",
    "running": "warn",
    "completed": "ok",
    "partially_completed": "warn",
    "failed": "error",
    "cancelled": "off",
}

_MIME = {
    ".json": "application/json",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".html": "text/html",
    ".png": "image/png",
    ".log": "text/plain",
}

#: Friendly labels for the files a run can produce.
_ARTIFACT_LABELS = {
    "results.json": "Results (JSON)",
    "environment.json": "Environment (JSON)",
    "metrics.csv": "Metrics (CSV)",
    "per_class.csv": "Per-class (CSV)",
    "summary.md": "Summary (Markdown)",
    "report.html": "Report (HTML)",
}

#: Human wording for the stages the worker records while a run is live.
_STAGE_LABELS = {
    "preparing": "Preparing",
    "loading dataset": "Loading dataset",
    "loading model": "Loading model",
    "inference": "Running inference",
    "metric calculation": "Calculating metrics",
    "report": "Writing report",
}

#: How often the live progress block re-reads the job state (seconds).
PROGRESS_REFRESH_SECONDS = 2.0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render(detector: MarineDetector, settings, user=None) -> None:
    """Render the research evaluation workspace for the signed-in account."""
    hero(
        st,
        "Measured locally, never fabricated",
        "Research Evaluation Center",
        "Register a labelled dataset, run the real detectors and record "
        "precision, recall and mAP with full provenance.",
        icon="science",
    )
    if user is None:
        st.caption("Sign in to configure, run and store an evaluation.")
        return

    user_id = user.id
    store = state.evaluation_store(settings)

    _overview(settings, store, user_id)

    goto = st.session_state.pop("eval_goto", None)
    if goto in SECTIONS:
        st.session_state["eval_section"] = goto
    if st.session_state.get("eval_section") not in SECTIONS:
        st.session_state["eval_section"] = "Run"
    section = st.radio(
        "Section",
        SECTIONS,
        key="eval_section",
        horizontal=True,
        label_visibility="collapsed",
    )

    if section == "Run":
        _run_section(detector, settings, store, user_id)
    elif section == "Experiments":
        _experiments_section(settings, store, user_id)
    elif section == "Schedules":
        _schedules_section(detector, settings, store, user_id)
    elif section == "Datasets":
        _datasets_section(settings, store, user_id)
    elif section == "Export":
        _export_section(settings, store, user_id)
    else:
        _compare_section(settings, store, user_id)


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def _overview(settings, store: EvaluationStore, user_id: str) -> None:
    stats = store.stats(user_id)
    metrics_row(
        st,
        [
            ("Saved datasets", stats["datasets"], "registered configs"),
            ("Experiments", stats["experiments"], "recorded for you"),
            ("In progress", stats["active"], "running / queued"),
            ("With results", stats["completed"] + stats["partial"], "completed"),
        ],
    )
    last = stats.get("last_completed")
    if last is not None:
        st.caption(
            f"Last completed: **{last.name}** · {last.status} · "
            f"{last.completed_at or 'unknown time'}"
        )
    else:
        st.caption(
            "No experiment has completed yet. Configure a dataset and run one "
            "to produce the first real result set."
        )

    _scheduler_caption(settings)

    columns = st.columns(3)
    with columns[0]:
        if st.button(
            "Run an evaluation",
            icon=":material/play_arrow:",
            key="eval_goto_run",
            width="stretch",
        ):
            st.session_state["eval_goto"] = "Run"
    with columns[1]:
        if st.button(
            "Register a dataset",
            icon=":material/database:",
            key="eval_goto_datasets",
            width="stretch",
        ):
            st.session_state["eval_goto"] = "Datasets"
    with columns[2]:
        if st.button(
            "Compare experiments",
            icon=":material/compare_arrows:",
            key="eval_goto_compare",
            width="stretch",
        ):
            st.session_state["eval_goto"] = "Compare"

    columns2 = st.columns(2)
    with columns2[0]:
        if st.button(
            "Schedule continuous evaluation",
            icon=":material/autorenew:",
            key="eval_goto_schedules",
            width="stretch",
        ):
            st.session_state["eval_goto"] = "Schedules"
    with columns2[1]:
        if st.button(
            "Export tables for a paper",
            icon=":material/table_view:",
            key="eval_goto_export",
            width="stretch",
        ):
            st.session_state["eval_goto"] = "Export"


# --------------------------------------------------------------------------- #
# Dataset options shared by the Run and Datasets sections
# --------------------------------------------------------------------------- #
@dataclass
class _DatasetOption:
    """One dataset the user could evaluate."""

    data_yaml: Path
    name: str
    class_names: tuple[str, ...]
    source: str  # "registered" | "discovered"
    splits: tuple[str, ...] = ()


def _dataset_options(
    settings, store: EvaluationStore, user_id: str
) -> list[_DatasetOption]:
    """Registered configs first, then anything auto-discovered (de-duplicated)."""
    options: dict[str, _DatasetOption] = {}
    for record in store.list_datasets(user_id):
        path = Path(record.data_yaml)
        options[str(path)] = _DatasetOption(
            data_yaml=path,
            name=record.name,
            class_names=record.class_names,
            source="registered",
            splits=record.splits,
        )
    for info in discover_datasets(default_search_roots(settings)):
        key = str(info.data_yaml)
        options.setdefault(
            key,
            _DatasetOption(
                data_yaml=info.data_yaml,
                name=info.name,
                class_names=info.class_names,
                source="discovered",
            ),
        )
    return list(options.values())


# --------------------------------------------------------------------------- #
# Run section
# --------------------------------------------------------------------------- #
def _run_section(detector, settings, store: EvaluationStore, user_id: str) -> None:
    notice(
        st,
        "info",
        "What an experiment measures",
        "Each selected model is run independently over the same dataset split "
        "with its own weights. Metrics describe only the evaluated dataset and "
        "this machine; models are never merged into a single score.",
        icon="info",
    )

    available = state.available_keys(detector)
    options = _dataset_options(settings, store, user_id)

    if not options:
        _pending_state(settings)
        if not available:
            _weights_notice()
        _recent_panel(settings, store, user_id)
        return

    if not available:
        _weights_notice()
        _recent_panel(settings, store, user_id)
        return

    registry = state.registry(detector)
    labels = {
        str(option.data_yaml): (
            f"{option.name} · {len(option.class_names)} classes · {option.source}"
        )
        for option in options
    }
    left, right = st.columns([2, 1])
    with left:
        dataset_key = st.selectbox(
            "Dataset",
            options=list(labels),
            format_func=lambda key: labels.get(key, key),
            key="eval_dataset",
        )
    option = next(item for item in options if str(item.data_yaml) == dataset_key)
    splits = list(available_splits(option.data_yaml) or option.splits or ())
    if not splits:
        splits = ["test"]
    with right:
        split = st.selectbox(
            "Split",
            options=splits,
            index=(splits.index("test") if "test" in splits else 0),
            key="eval_split",
        )

    if option.class_names:
        st.caption("Dataset classes: " + ", ".join(option.class_names))

    selections = st.multiselect(
        "Models to evaluate",
        options=available,
        default=available[: min(2, len(available))],
        format_func=lambda key: state.model_label(key, registry.get(key), True),
        key="eval_models",
        help="Each model is evaluated independently on the same split.",
    )
    if selections:
        _compat_panel(registry, selections, option.class_names)

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        use_default = st.checkbox(
            "Use each model's recommended threshold",
            value=True,
            key="eval_use_default",
        )
    with c2:
        confidence = st.slider(
            "Confidence override",
            min_value=0.01,
            max_value=0.99,
            value=0.5,
            step=0.01,
            disabled=use_default,
            key="eval_confidence",
        )
    with c3:
        limit = st.number_input(
            "Max images (0 = all)",
            min_value=0,
            max_value=1_000_000,
            value=DEFAULT_LIMIT,
            step=5,
            key="eval_limit",
        )

    c4, c5 = st.columns([1, 2])
    with c4:
        iou = st.slider(
            "IoU threshold for figures",
            min_value=0.50,
            max_value=0.95,
            value=0.50,
            step=0.05,
            key="eval_iou",
            help=(
                "Standard COCO mAP@0.5 and mAP@0.5:0.95 are always computed. "
                "This threshold only controls the PR / F1 / confusion figures."
            ),
        )
    with c5:
        plots = st.checkbox(
            "Write plots and charts (requires matplotlib)",
            value=True,
            key="eval_plots",
        )

    name = st.text_input(
        "Experiment name",
        value=f"Evaluation {datetime.now():%Y-%m-%d %H:%M}",
        key="eval_name",
    )
    purpose = st.text_input(
        "Research purpose / question (optional)",
        key="eval_purpose",
        placeholder="e.g. compare fish detection recall across two models",
    )
    notes = st.text_area(
        "Notes (optional)",
        key="eval_notes",
        placeholder="Anything a reader should know about this run.",
        height=80,
    )

    active = store.list_experiments(user_id, statuses=ACTIVE_STATUSES, limit=10)
    if active:
        _active_panel(settings, store, user_id, active)

    if st.button(
        "Run evaluation",
        type="primary",
        icon=":material/play_arrow:",
        key="eval_run",
        disabled=bool(active),
    ):
        if not selections:
            error_card(
                st, "Select at least one model", "Pick one or more models to evaluate."
            )
        elif not option.class_names:
            error_card(
                st,
                "Dataset has no class names",
                "Its data.yaml does not define 'names', so labels cannot be scored.",
            )
        else:
            _start_run(
                detector=detector,
                settings=settings,
                store=store,
                user_id=user_id,
                option=option,
                split=split,
                selections=list(selections),
                confidence=None if use_default else float(confidence),
                iou=float(iou),
                limit=int(limit or 0),
                plots=bool(plots),
                name=name.strip() or f"Evaluation {datetime.now():%Y-%m-%d %H:%M}",
                purpose=purpose.strip(),
                notes=notes.strip(),
            )

    _recent_panel(settings, store, user_id)


def _compat_panel(registry: dict, selections: list[str], dataset_classes: tuple) -> None:
    """Show, per model, whether it can honestly be scored on this dataset."""
    rows = []
    for key in selections:
        spec = registry.get(key)
        model_classes = (
            tuple(spec.classes) if spec is not None and spec.classes else ()
        )
        if dataset_classes and model_classes:
            compat = compare_classes(dataset_classes, model_classes)
            overlap = (
                f"{len(compat.supported)} shared"
                if compat.comparable
                else "none - not comparable"
            )
        else:
            overlap = "unknown until weights load"
        rows.append(
            {
                "Model": state.model_label(key, spec, True),
                "Architecture": getattr(spec, "architecture", "") or "unknown",
                "Known classes": len(model_classes) or "unknown",
                "Overlap with dataset": overlap,
                "Limitations": (getattr(spec, "limitations", "") or "").strip()
                or "none recorded",
            }
        )
    if rows:
        with st.expander("Model compatibility and limitations", expanded=False):
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption(
                "A model with no shared class names is skipped at run time: "
                "BlueEye never invents a score for an incompatible pair."
            )


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse a stored UTC timestamp, tolerating aware/naive variants."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                parsed = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _progress_view(record) -> dict:
    """Derive display values from a record's *stored* progress counters."""
    progress = record.progress or {}
    stage = str(progress.get("stage") or "")
    done = progress.get("images_done")
    total = progress.get("images_total")
    completed = progress.get("completed")
    requested = progress.get("requested")

    fraction: float | None = None
    counter_label = ""
    if (
        stage in ("inference", "metric calculation")
        and isinstance(total, int)
        and total > 0
        and isinstance(done, int)
    ):
        fraction = done / total
        counter_label = f"{done}/{total} images"
    elif isinstance(completed, int) and isinstance(requested, int) and requested > 0:
        fraction = completed / requested
        counter_label = f"{completed}/{requested} models"

    elapsed: float | None = None
    eta: float | None = None
    started = _parse_timestamp(record.started_at)
    if started is not None:
        elapsed = (datetime.now(tz=timezone.utc) - started).total_seconds()
        if elapsed > 0 and fraction is not None and 0 < fraction < 1:
            eta = elapsed * (1 - fraction) / fraction

    return {
        "stage": stage,
        "stage_label": _STAGE_LABELS.get(stage, stage),
        "model": progress.get("model"),
        "model_index": progress.get("model_index"),
        "requested": requested,
        "counter_label": counter_label,
        "fraction": fraction,
        "elapsed": elapsed,
        "eta": eta,
    }


def _render_progress(record, live: bool) -> None:
    """Render one experiment's genuine progress (bar + counters + timing)."""
    view = _progress_view(record)
    fraction = view["fraction"]
    if isinstance(fraction, float):
        st.progress(min(max(fraction, 0.0), 1.0))

    parts: list[str] = []
    if view["stage_label"]:
        parts.append(view["stage_label"])
    model = view["model"]
    if model:
        if isinstance(view["model_index"], int) and isinstance(view["requested"], int):
            parts.append(f"model {model} ({view['model_index']}/{view['requested']})")
        else:
            parts.append(f"model {model}")
    if view["counter_label"]:
        parts.append(view["counter_label"])
    if view["elapsed"] is not None:
        parts.append(f"elapsed {_format_duration(view['elapsed'])}")
    if view["eta"] is not None:
        parts.append(f"~{_format_duration(view['eta'])} remaining")
    if parts:
        st.caption(" · ".join(parts))
    if not live:
        st.caption(
            "No live worker is running in this process (the app may have "
            "restarted). Cancel it to unblock new runs."
        )


@st.fragment(run_every=PROGRESS_REFRESH_SECONDS)
def _live_progress(store: EvaluationStore, user_id: str, experiment_id: str) -> None:
    """Auto-refreshing progress for one running experiment.

    Re-reads the persisted row so the numbers come from the worker, not from
    the page that happened to start the run.
    """
    record = store.get_experiment(experiment_id, user_id)
    if record is None:
        return
    _render_progress(record, is_running(experiment_id))


def _active_panel(settings, store, user_id, active: list) -> None:
    st.markdown("#### Runs in progress")
    for record in active:
        running = is_running(record.id)
        st.markdown(
            badge(record.status, _STATUS_KIND.get(record.status, "info"), dot=True)
            + f" **{record.name}**",
            unsafe_allow_html=True,
        )
        _live_progress(store, user_id, record.id)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Cancel run", key=f"eval_cancel_{record.id}", width="stretch"):
                request_cancel(record.id)
                store.request_cancel(record.id, user_id)
                if not running:
                    store.set_status(
                        record.id,
                        user_id,
                        "cancelled",
                        error="Cancelled before completion.",
                    )
                st.rerun()
        with c2:
            if not running and st.button(
                "Mark as failed",
                key=f"eval_interrupt_{record.id}",
                width="stretch",
            ):
                store.set_status(
                    record.id,
                    user_id,
                    "failed",
                    error="No live worker: the run was interrupted (app restart).",
                )
                st.rerun()
    st.caption(
        "Progress updates live from the background worker. A finished run can be "
        "reopened from the Experiments section."
    )


def _start_run(
    *,
    detector,
    settings,
    store: EvaluationStore,
    user_id: str,
    option: _DatasetOption,
    split: str,
    selections: list[str],
    confidence: float | None,
    iou: float,
    limit: int,
    plots: bool,
    name: str,
    purpose: str,
    notes: str,
) -> None:
    # Server-side re-check: the button is disabled while a run is active, but a
    # double submit or a second tab must not start a second worker for the same
    # account. One evaluation at a time keeps progress and metrics honest.
    if store.list_experiments(user_id, statuses=ACTIVE_STATUSES, limit=1):
        error_card(
            st,
            "A run is already in progress",
            "BlueEye runs one evaluation at a time. Cancel it or wait for it to "
            "finish (the Experiments section shows live progress), then retry.",
        )
        return

    with st.spinner("Validating the dataset before running..."):
        try:
            validation = validate_dataset(option.data_yaml, split=split)
        except DatasetValidationError as exc:
            error_card(st, "Dataset config could not be read", str(exc))
            return

    _show_validation(validation)
    if not validation.ready:
        error_card(
            st,
            "Dataset is not ready",
            "BlueEye will not score a dataset it cannot validate. Resolve the "
            "errors above and try again.",
        )
        return

    record = store.register_dataset(
        user_id,
        option.name,
        option.data_yaml,
        class_names=validation.class_names,
        splits=validation.splits_with_images,
        fingerprint=validation.fingerprint,
        validation=validation.as_dict(),
    )

    registry = state.registry(detector)
    fingerprints = {}
    for key in selections:
        spec = registry.get(key)
        if spec is not None:
            fingerprints[key] = file_fingerprint(
                detector.model_manager.resolve_path(spec.key)
            )

    parameters = {
        "split": split,
        "confidence": confidence,
        "use_recommended_threshold": confidence is None,
        "iou_threshold_for_figures": iou,
        "max_images": limit,
        "plots": plots,
        "device": str(getattr(detector, "device", "") or "auto"),
        "note": (
            "Input resolution, preprocessing and batch size are the "
            "application's Ultralytics defaults; the detector pipeline is "
            "unchanged."
        ),
    }

    experiment = store.create_experiment(
        user_id,
        name,
        notes=notes,
        purpose=purpose,
        dataset_id=record.id,
        dataset_name=record.name,
        data_yaml=str(option.data_yaml),
        split=split,
        class_names=validation.class_names,
        dataset_fingerprint=validation.fingerprint,
        model_keys=tuple(selections),
        model_fingerprints=fingerprints,
        evaluator_version=evaluator_version(),
        parameters=parameters,
        environment=collect_environment(),
        git_commit=git_commit(),
        status="queued",
    )

    output_dir = state.evaluation_run_dir(settings, user_id, experiment.id)
    request = ExperimentRequest(
        experiment_id=experiment.id,
        user_id=user_id,
        data_yaml=str(option.data_yaml),
        split=split,
        model_keys=tuple(selections),
        confidence=confidence,
        iou_threshold=iou,
        limit=limit,
        output_dir=output_dir,
        make_plots=plots,
        parameters=parameters,
    )
    start_experiment(store, request, detector=detector)
    st.session_state["eval_last_run"] = experiment.id
    st.session_state["eval_goto"] = "Experiments"
    st.rerun()


def _recent_panel(settings, store: EvaluationStore, user_id: str) -> None:
    st.markdown("### Recent experiments")
    experiments = store.list_experiments(user_id, limit=25)
    if not experiments:
        empty_state(
            st,
            "science",
            "No experiment yet",
            "Configure the run above. Your first real result set will appear here.",
        )
        return
    last = st.session_state.get("eval_last_run")
    labels = {
        record.id: f"{record.name} · {record.status} · {record.created_at}"
        for record in experiments
    }
    index = next(
        (i for i, record in enumerate(experiments) if record.id == last), 0
    )
    selected = st.selectbox(
        "Experiment",
        options=[record.id for record in experiments],
        index=index,
        format_func=lambda key: labels.get(key, key),
        key="eval_recent",
    )
    record = store.get_experiment(selected, user_id)
    if record is not None:
        _render_experiment(settings, store, user_id, record)


# --------------------------------------------------------------------------- #
# Datasets section
# --------------------------------------------------------------------------- #
def _datasets_section(settings, store: EvaluationStore, user_id: str) -> None:
    st.markdown("### Register a dataset")
    st.caption(
        "BlueEye never downloads data. Point it at a YOLO dataset you already "
        "have, or use one it discovered under data/raw, data/datasets, "
        "data/input or datasets/."
    )
    path_text = st.text_input(
        "Path to a dataset folder or data.yaml",
        key="eval_ds_path",
        placeholder="data/raw/my_dataset  or  data/raw/my_dataset/data.yaml",
    )
    name_text = st.text_input("Display name (optional)", key="eval_ds_name")
    if st.button(
        "Validate and register", icon=":material/fact_check:", key="eval_ds_register"
    ):
        _register_from_path(store, user_id, path_text, name_text)

    st.markdown("### Saved datasets")
    registered = store.list_datasets(user_id)
    if not registered:
        empty_state(
            st,
            "dataset",
            "No saved dataset",
            "Register one above, or let BlueEye discover a prepared dataset.",
        )
    else:
        for record in registered:
            with st.expander(f"{record.name} · {record.data_yaml}", expanded=False):
                st.caption(
                    f"{len(record.class_names)} classes · splits: "
                    f"{', '.join(record.splits) or 'unknown'} · "
                    f"fingerprint {record.fingerprint[:12] or 'n/a'}"
                )
                if record.class_names:
                    st.caption("Classes: " + ", ".join(record.class_names))
                splits = (record.validation or {}).get("splits")
                if splits:
                    st.dataframe(splits, hide_index=True, width="stretch")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(
                        "Re-validate",
                        key=f"eval_ds_revalidate_{record.id}",
                        width="stretch",
                    ):
                        _revalidate(store, user_id, record)
                with c2:
                    if st.button(
                        "Delete",
                        key=f"eval_ds_delete_{record.id}",
                        width="stretch",
                    ):
                        store.delete_dataset(record.id, user_id)
                        st.rerun()

    discovered = discover_datasets(default_search_roots(settings))
    st.markdown("### Auto-discovered datasets")
    if not discovered:
        st.caption(
            "None found. Searched: "
            + ", ".join(str(root) for root in default_search_roots(settings))
        )
    else:
        st.dataframe(
            [
                {
                    "Name": info.name,
                    "Config": str(info.data_yaml),
                    "Classes": len(info.class_names),
                    "Splits": ", ".join(available_splits(info.data_yaml)) or "none",
                }
                for info in discovered
            ],
            hide_index=True,
            width="stretch",
        )
        labels = {str(info.data_yaml): info.name for info in discovered}
        selected = st.selectbox(
            "Validate a discovered dataset",
            options=list(labels),
            format_func=lambda key: labels.get(key, key),
            key="eval_ds_discovered",
        )
        if st.button(
            "Validate", icon=":material/fact_check:", key="eval_ds_validate_discovered"
        ):
            _validate_and_show(Path(selected))


def _register_from_path(store: EvaluationStore, user_id, path_text, name_text) -> None:
    if not path_text.strip():
        error_card(st, "No path given", "Enter a dataset folder or data.yaml path.")
        return
    candidate = Path(path_text.strip())
    if not candidate.is_absolute():
        from app.config import PROJECT_ROOT

        candidate = PROJECT_ROOT / candidate
    if candidate.is_dir():
        config = find_data_yaml(candidate)
        if config is None:
            error_card(
                st,
                "No dataset config found",
                f"{candidate} does not contain data.yaml / dataset.yaml / data.yml.",
            )
            return
        candidate = config
    if not candidate.is_file():
        error_card(st, "Path not found", f"{candidate} does not exist.")
        return

    validation = _validate_and_show(candidate)
    if validation is None or not validation.ready:
        return
    record = store.register_dataset(
        user_id,
        name_text.strip() or validation.name,
        candidate,
        class_names=validation.class_names,
        splits=validation.splits_with_images,
        fingerprint=validation.fingerprint,
        validation=validation.as_dict(),
    )
    st.success(
        f"Registered '{record.name}' with {len(record.class_names)} classes "
        f"across splits: {', '.join(record.splits) or 'none'}."
    )


def _revalidate(store: EvaluationStore, user_id: str, record) -> None:
    validation = _validate_and_show(Path(record.data_yaml))
    if validation is None:
        return
    store.register_dataset(
        user_id,
        record.name,
        record.data_yaml,
        class_names=validation.class_names,
        splits=validation.splits_with_images,
        fingerprint=validation.fingerprint,
        validation=validation.as_dict(),
        notes=record.notes,
    )
    st.success("Re-validated and updated.")


def _validate_and_show(path: Path) -> DatasetValidation | None:
    with st.spinner(f"Validating {path.name}..."):
        try:
            validation = validate_dataset(path)
        except DatasetValidationError as exc:
            error_card(st, "Dataset config could not be read", str(exc))
            return None
    _show_validation(validation)
    return validation


def _show_validation(validation: DatasetValidation) -> None:
    st.markdown(f"#### Validation · {validation.name}")
    kind = "ok" if validation.ready else "error"
    label = "ready to evaluate" if validation.ready else "not ready"
    st.markdown(badge(label, kind, dot=True), unsafe_allow_html=True)
    summary = [
        {
            "Split": report.name,
            "Images": report.num_images,
            "Labels": report.num_labels,
            "Objects": report.num_objects,
            "Missing labels": report.missing_labels,
            "Malformed": report.malformed_rows,
            "Out-of-range": report.out_of_range,
            "Bad class ids": report.bad_class_ids,
        }
        for report in validation.reports
    ]
    if summary:
        st.dataframe(summary, hide_index=True, width="stretch")
    st.caption(
        "Fingerprint "
        + (validation.fingerprint[:16] or "n/a")
        + " · classes: "
        + (", ".join(validation.class_names) or "none")
    )
    _issue_table("Errors", validation.errors, expanded=True)
    _issue_table("Warnings", validation.warnings, expanded=False)
    if not validation.errors and not validation.warnings:
        st.caption("No problems found.")


def _issue_table(title: str, issues, expanded: bool) -> None:
    if not issues:
        return
    with st.expander(f"{title} ({len(issues)})", expanded=expanded):
        st.dataframe(
            [
                {
                    "Code": issue.code,
                    "Detail": issue.message,
                    "File": issue.path or "",
                }
                for issue in issues[:200]
            ],
            hide_index=True,
            width="stretch",
        )


# --------------------------------------------------------------------------- #
# Experiments section
# --------------------------------------------------------------------------- #
def _experiments_section(settings, store: EvaluationStore, user_id: str) -> None:
    st.markdown("### Your experiments")
    experiments = store.list_experiments(user_id, limit=200)
    if not experiments:
        empty_state(
            st,
            "history",
            "No experiments stored",
            "Run an evaluation from the Run section; it is saved here and "
            "survives a restart.",
        )
        return

    filter_col, refresh_col = st.columns([3, 1])
    with filter_col:
        statuses = st.multiselect(
            "Filter by status",
            options=list(EXPERIMENT_STATUSES),
            default=[],
            key="eval_filter_status",
        )
    with refresh_col:
        st.write("")
        if st.button("Refresh", key="eval_refresh", width="stretch"):
            st.rerun()
    if statuses:
        experiments = [item for item in experiments if item.status in statuses]
    if not experiments:
        st.caption("No experiment matches that filter.")
        return

    labels = {
        record.id: f"{record.name} · {record.status} · {record.created_at}"
        for record in experiments
    }
    selected = st.selectbox(
        "Experiment",
        options=[record.id for record in experiments],
        format_func=lambda key: labels.get(key, key),
        key="eval_selected",
    )
    record = store.get_experiment(selected, user_id)
    if record is None:
        st.caption("That experiment is no longer available.")
        return
    _render_experiment(settings, store, user_id, record)

    with st.expander("Delete this experiment", expanded=False):
        st.caption("Removes the saved record. Generated files on disk are untouched.")
        if st.button("Delete experiment", key=f"eval_delete_{record.id}"):
            store.delete_experiment(record.id, user_id)
            st.rerun()


def _render_experiment(
    settings, store: EvaluationStore, user_id: str, record
) -> None:
    st.markdown(
        badge(record.status, _STATUS_KIND.get(record.status, "info"), dot=True)
        + f" &nbsp; **{record.name}**",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Dataset: {record.dataset_name} · split `{record.split}` · "
        f"fingerprint {record.dataset_fingerprint[:12] or 'n/a'} · "
        f"created {record.created_at}"
    )
    if record.purpose:
        st.markdown(f"**Purpose:** {record.purpose}")
    if record.notes:
        st.markdown(f"**Notes:** {record.notes}")

    if not record.is_terminal:
        _active_panel(settings, store, user_id, [record])

    if record.error:
        error_card(st, "Run problems", record.error, log=False)
    for warning in record.warnings:
        st.caption(f"warning: {warning}")

    if not record.results:
        st.caption(
            "No measured results are stored for this experiment yet."
            + ("" if record.is_terminal else " It is still running.")
        )
        _reproducibility(record)
        return

    _results_summary(record)
    _per_class(record)
    _visual_analysis(settings, user_id, record)
    _diagnostics(record)
    _artifacts(settings, user_id, record)
    _reproducibility(record)


def _results_summary(record) -> None:
    st.markdown("#### Summary")
    rows = []
    for result in record.results:
        summary = result.summary or {}
        rows.append(
            {
                "Model": result.model,
                "Status": result.status,
                "Images": summary.get("images", "n/a"),
                "Objects": summary.get("objects", "n/a"),
                "Precision": _num(summary.get("precision")),
                "Recall": _num(summary.get("recall")),
                "mAP@0.5": _num(summary.get("mAP50")),
                "mAP@0.5:0.95": _num(summary.get("mAP50_95")),
                "s/image": _num(summary.get("seconds_per_image")),
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "Measured locally on the named dataset/split - not values from the "
        "model authors. Precision/recall are at the evaluator's IoU 0.5 "
        "operating point; mAP@0.5:0.95 averages IoU 0.50-0.95."
    )


def _per_class(record) -> None:
    rows = []
    for result in record.results:
        for metric in result.metrics.get("classes", []):
            rows.append(
                {
                    "Model": result.model,
                    "Class": metric.get("name", "?"),
                    "GT": metric.get("gt", 0),
                    "Pred": metric.get("predictions", 0),
                    "Precision": _num(metric.get("precision")),
                    "Recall": _num(metric.get("recall")),
                    "AP@0.5": _num(metric.get("AP50")),
                    "AP@0.5:0.95": _num(metric.get("AP50_95")),
                }
            )
    if rows:
        st.markdown("#### Per-class results")
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            "Classes with no ground-truth boxes in this split are reported with "
            "an undefined AP (shown as 0), never a guessed value."
        )


def _visual_analysis(settings, user_id: str, record) -> None:
    figures = [
        artifact
        for artifact in record.artifacts
        if artifact.path.lower().endswith(".png")
        and state.is_owned_path(settings, user_id, artifact.path)
        and Path(artifact.path).is_file()
    ]
    if not figures:
        return
    st.markdown("#### Visual analysis")
    st.caption("Charts generated from this run's real predictions and ground truth.")
    columns = st.columns(2)
    for index, artifact in enumerate(figures):
        with columns[index % 2]:
            try:
                st.image(
                    Path(artifact.path).read_bytes(),
                    caption=artifact.name,
                    width="stretch",
                )
            except OSError:
                continue


def _diagnostics(record) -> None:
    st.markdown("#### Diagnostics")
    issues = False
    for result in record.results:
        if not result.comparable:
            issues = True
            st.caption(
                f"{result.model}: no shared class names with the dataset; the "
                "result is kept for the record but is not a valid score."
            )
        for note in result.notes:
            issues = True
            st.caption(f"{result.model}: {note}")
    for warning in record.warnings:
        issues = True
        st.caption(f"warning: {warning}")
    if record.error:
        issues = True
        st.caption(f"error: {record.error}")
    if not issues:
        st.caption("No diagnostics were recorded for this experiment.")


def _artifacts(settings, user_id: str, record) -> None:
    reports = [
        artifact
        for artifact in record.artifacts
        if not artifact.path.lower().endswith(".png")
        and state.is_owned_path(settings, user_id, artifact.path)
        and Path(artifact.path).is_file()
    ]
    if not reports:
        return
    st.markdown("#### Export artifacts")
    columns = st.columns(2)
    for index, artifact in enumerate(sorted(reports, key=lambda item: item.name)):
        path = Path(artifact.path)
        label = _ARTIFACT_LABELS.get(artifact.name, artifact.name)
        with columns[index % 2]:
            download_button(
                st,
                label,
                path,
                _MIME.get(path.suffix.lower(), "application/octet-stream"),
            )


def _reproducibility(record) -> None:
    st.markdown("#### Reproducibility")
    st.dataframe(
        [
            {"Field": "Experiment id", "Value": record.id},
            {"Field": "Dataset", "Value": record.dataset_name},
            {"Field": "Config path", "Value": record.data_yaml},
            {"Field": "Split", "Value": record.split},
            {"Field": "Fingerprint", "Value": record.dataset_fingerprint or "n/a"},
            {"Field": "Models", "Value": ", ".join(record.model_keys)},
            {"Field": "Evaluator", "Value": record.evaluator_version or "n/a"},
            {"Field": "Git commit", "Value": record.git_commit or "unknown"},
            {"Field": "Started", "Value": record.started_at or "n/a"},
            {"Field": "Completed", "Value": record.completed_at or "n/a"},
        ],
        hide_index=True,
        width="stretch",
    )
    if record.parameters:
        st.caption("Parameters")
        st.json(record.parameters)
    if record.model_fingerprints:
        st.caption("Model weight fingerprints")
        st.json(record.model_fingerprints)
    st.code(_rerun_command(record), language="powershell")
    st.caption(
        "Weights and environment may change; BlueEye records them so a rerun "
        "can be judged, but does not claim perfect reproducibility."
    )


def _rerun_command(record) -> str:
    """A CLI command that reproduces this run from its *saved* configuration.

    Built from the persisted experiment row (dataset, split, models and the
    stored ``parameters``), never from the current form values, and only with
    flags the CLI actually supports.
    """
    parameters = record.parameters or {}
    model_flags = (
        " ".join(f"--model {key}" for key in record.model_keys)
        if record.model_keys
        else "--model all"
    )
    parts = [
        "python scripts/run_evaluation.py",
        f'--data "{record.data_yaml}"',
        f"--split {record.split or 'test'}",
        model_flags,
    ]
    limit = parameters.get("max_images")
    if isinstance(limit, int) and limit > 0:
        parts.append(f"--limit {limit}")
    iou = parameters.get("iou_threshold_for_figures")
    if isinstance(iou, (int, float)):
        parts.append(f"--iou {float(iou):g}")
    confidence = parameters.get("confidence")
    if isinstance(confidence, (int, float)):
        parts.append(f"--confidence {float(confidence):g}")
    device = parameters.get("device")
    if isinstance(device, str) and device and device != "auto":
        parts.append(f"--device {device}")
    if parameters.get("plots"):
        parts.append("--plots")
    parts.append(f"--record --owner {record.user_id}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Continuous / scheduled evaluation
# --------------------------------------------------------------------------- #
def _scheduler_caption(settings) -> None:
    """One honest line about the background worker in the overview."""
    if not getattr(settings, "scheduler_enabled", True):
        st.caption(
            "Continuous evaluation worker: **disabled** "
            "(`BLUEEYE_SCHEDULER=off`). Schedules still run when you start them."
        )
        return
    status = eval_scheduler.status()
    if status["running"]:
        st.caption(
            f"Continuous evaluation worker: **running** · last check "
            f"{status['last_tick'] or 'starting...'} · {status['summary']}"
        )
    else:
        st.caption(
            "Continuous evaluation worker: **starting** - add a schedule to keep "
            "a dataset and model pair under evaluation."
        )


def _schedules_section(
    detector: MarineDetector, settings, store: EvaluationStore, user_id: str
) -> None:
    st.markdown("### Continuous research evaluation")
    notice(
        st,
        "info",
        "Re-run automatically, and record the history",
        "A schedule re-runs a real evaluation while BlueEye is running: when the "
        "dataset or the model weights change, on a fixed interval, or both. Every "
        "run is an ordinary experiment with full provenance - nothing here is "
        "simulated, and a change is only ever detected from the real files on disk.",
        icon="autorenew",
    )
    _scheduler_caption(settings)

    sched_store = state.schedule_store(settings)
    options = _dataset_options(settings, store, user_id)
    available = state.available_keys(detector)
    registry = state.registry(detector)

    if not options:
        _pending_state(settings)
    elif not available:
        _weights_notice()
    else:
        _schedule_builder(
            detector, settings, store, sched_store, user_id, options, available, registry
        )

    st.markdown("### Saved schedules")
    schedules = sched_store.list(user_id)
    if not schedules:
        empty_state(
            st,
            "autorenew",
            "No schedules yet",
            "Create one above to keep a dataset and model pair under continuous "
            "evaluation and build a history of measurements over time.",
        )
        return
    for schedule in schedules:
        _schedule_card(detector, settings, store, sched_store, user_id, schedule)


def _schedule_builder(
    detector,
    settings,
    store: EvaluationStore,
    sched_store,
    user_id: str,
    options: list,
    available: list[str],
    registry: dict,
) -> None:
    """Form that saves a new schedule (validated before it is stored)."""
    with st.expander("Create a schedule", expanded=False):
        labels = {
            str(option.data_yaml): (
                f"{option.name} · {len(option.class_names)} classes · {option.source}"
            )
            for option in options
        }
        name = st.text_input(
            "Schedule name",
            value=f"Continuous evaluation {datetime.now():%Y-%m-%d}",
            key="sched_name",
        )
        left, right = st.columns([2, 1])
        with left:
            dataset_key = st.selectbox(
                "Dataset",
                options=list(labels),
                format_func=lambda key: labels.get(key, key),
                key="sched_dataset",
            )
        option = next(item for item in options if str(item.data_yaml) == dataset_key)
        splits = list(available_splits(option.data_yaml) or option.splits or ())
        if not splits:
            splits = ["test"]
        with right:
            split = st.selectbox(
                "Split",
                options=splits,
                index=(splits.index("test") if "test" in splits else 0),
                key="sched_split",
            )

        selections = st.multiselect(
            "Models to evaluate",
            options=available,
            default=available[: min(2, len(available))],
            format_func=lambda key: state.model_label(key, registry.get(key), True),
            key="sched_models",
        )

        modes = list(TRIGGER_LABELS)
        mode = st.radio(
            "When to run",
            options=modes,
            format_func=lambda value: TRIGGER_LABELS[value],
            horizontal=True,
            key="sched_mode",
        )
        interval = st.number_input(
            "Interval (minutes)",
            min_value=MIN_INTERVAL_MINUTES,
            max_value=MAX_INTERVAL_MINUTES,
            value=60,
            step=15,
            disabled=(mode == "on_change"),
            key="sched_interval",
            help="Used by the 'interval' and 'both' triggers.",
        )

        c1, c2 = st.columns(2)
        with c1:
            use_default = st.checkbox(
                "Use each model's recommended threshold",
                value=True,
                key="sched_default",
            )
        with c2:
            confidence = st.slider(
                "Confidence override",
                min_value=0.01,
                max_value=0.99,
                value=0.5,
                step=0.01,
                disabled=use_default,
                key="sched_conf",
            )
        c3, c4 = st.columns(2)
        with c3:
            limit = st.number_input(
                "Max images per run (0 = all)",
                min_value=0,
                max_value=1_000_000,
                value=DEFAULT_LIMIT,
                step=5,
                key="sched_limit",
            )
        with c4:
            plots = st.checkbox(
                "Write plots on every run",
                value=True,
                key="sched_plots",
            )
        purpose = st.text_input(
            "Research purpose / question (optional)",
            key="sched_purpose",
            placeholder="e.g. watch recall as the dataset grows",
        )

        if st.button(
            "Save schedule",
            type="primary",
            icon=":material/add_alarm:",
            key="sched_create",
        ):
            if not selections:
                error_card(st, "Select at least one model", "Pick one or more models.")
                return
            try:
                validation = validate_dataset(option.data_yaml, split=split)
            except DatasetValidationError as exc:
                error_card(st, "Dataset config could not be read", str(exc))
                return
            if not validation.ready:
                error_card(
                    st,
                    "Dataset is not ready",
                    "BlueEye will not schedule a dataset it cannot validate. "
                    "Resolve the issues in the Datasets section and try again.",
                )
                return

            parameters = {
                "confidence": None if use_default else float(confidence),
                "use_recommended_threshold": use_default,
                "iou_threshold_for_figures": 0.5,
                "max_images": int(limit or 0),
                "plots": bool(plots),
                "purpose": purpose.strip(),
                "device": str(getattr(detector, "device", "") or "auto"),
            }
            sched_store.create(
                user_id,
                name.strip() or f"Continuous evaluation {datetime.now():%Y-%m-%d}",
                option.data_yaml,
                dataset_name=option.name,
                split=split,
                class_names=validation.class_names,
                model_keys=tuple(selections),
                parameters=parameters,
                trigger_mode=mode,
                interval_minutes=int(interval),
            )
            st.success(
                "Schedule saved. It runs on the next worker check "
                f"({settings.scheduler_tick_seconds}s) when it is due."
            )
            st.rerun()


def _schedule_card(
    detector: MarineDetector,
    settings,
    store: EvaluationStore,
    sched_store,
    user_id: str,
    schedule,
) -> None:
    with st.container(border=True):
        head = st.columns([4, 1, 1])
        with head[0]:
            kind = {
                "completed": "ok",
                "failed": "error",
                "blocked": "warn",
                "queued": "warn",
                "running": "warn",
            }.get(schedule.last_status, "info")
            last_label = schedule.last_status or "never run"
            st.markdown(
                badge(last_label, kind, dot=True) + f" **{schedule.name}**",
                unsafe_allow_html=True,
            )
            st.caption(
                f"{schedule.dataset_name or Path(schedule.data_yaml).name} · split "
                f"`{schedule.split}` · models: {', '.join(schedule.model_keys) or 'none'}"
            )
            trigger = schedule.trigger_label
            if schedule.trigger_mode in ("interval", "both"):
                trigger += f" · every {schedule.interval_minutes} min"
            st.caption(f"Trigger: {trigger} · {schedule.next_due_hint()}")
            st.caption(
                f"Last run: {schedule.last_run_at or 'never'} · "
                f"{schedule.run_count} run(s) recorded"
            )
        with head[1]:
            enabled = st.toggle(
                "Enabled",
                value=schedule.enabled,
                key=f"sched_enabled_{schedule.id}",
                help="A disabled schedule is ignored by the worker.",
            )
            if enabled != schedule.enabled:
                sched_store.set_enabled(schedule.id, user_id, enabled)
                st.rerun()
        with head[2]:
            if st.button(
                "Run now",
                key=f"sched_run_{schedule.id}",
                icon=":material/play_arrow:",
                width="stretch",
            ):
                _run_schedule_now(detector, settings, store, sched_store, schedule)

        _schedule_history(settings, store, sched_store, user_id, schedule)


def _run_schedule_now(
    detector, settings, store: EvaluationStore, sched_store, schedule
) -> None:
    dataset_fp = eval_scheduler.compute_dataset_fingerprint(schedule.data_yaml)
    model_fps = eval_scheduler.compute_model_fingerprints(
        detector, schedule.model_keys
    )
    try:
        experiment_id = eval_scheduler.start_scheduled_run(
            store=store,
            settings=settings,
            sched_store=sched_store,
            schedule=schedule,
            detector=detector,
            trigger="manual",
            dataset_fingerprint=dataset_fp,
            model_fingerprints=model_fps,
        )
    except eval_scheduler.ScheduleRunError as exc:
        error_card(st, "Could not start the scheduled run", str(exc))
        return
    st.success(
        f"Started experiment `{experiment_id}` for schedule '{schedule.name}'."
    )
    st.rerun()


def _schedule_history(settings, store: EvaluationStore, sched_store, user_id, schedule) -> None:
    """Results over time for this schedule (real runs only)."""
    with st.expander(f"History · {schedule.run_count} run(s)"):
        experiments = store.list_experiments(user_id, limit=200)
        runs = [
            record
            for record in experiments
            if (record.parameters or {}).get("schedule_id") == schedule.id
        ]
        if not runs:
            st.caption(
                "No run has been recorded for this schedule yet. Use 'Run now' or "
                "wait for the worker to detect a change."
            )
        else:
            table = []
            points: list[float] = []
            for record in reversed(runs):
                best = record.mAP50
                table.append(
                    {
                        "created": record.created_at,
                        "status": record.status,
                        "trigger": (record.parameters or {}).get("trigger", ""),
                        "models": ", ".join(record.model_keys),
                        "mAP@0.5": f"{best:.4f}" if isinstance(best, (int, float)) else "n/a",
                        "experiment": record.id,
                    }
                )
                if isinstance(best, (int, float)):
                    points.append(float(best))
            st.dataframe(table, hide_index=True, width="stretch")
            if len(points) >= 2:
                st.caption("Best mAP@0.5 per recorded run (real measurements only).")
                st.line_chart(
                    {"run": list(range(1, len(points) + 1)), "mAP@0.5": points},
                    x="run",
                    y="mAP@0.5",
                )
            else:
                st.caption(
                    "A trend line appears once two runs with measured mAP are recorded."
                )

        confirm = st.checkbox(
            "I understand this deletes the schedule (not its experiments)",
            key=f"sched_confirm_delete_{schedule.id}",
        )
        if st.button(
            "Delete schedule",
            key=f"sched_delete_{schedule.id}",
            disabled=not confirm,
        ):
            sched_store.delete(schedule.id, user_id)
            st.rerun()


# --------------------------------------------------------------------------- #
# Publication export
# --------------------------------------------------------------------------- #
_PUBLICATION_MIME = {
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".tex": "text/plain",
    ".html": "text/html",
}


def _export_cell(key: str, value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _export_section(settings, store: EvaluationStore, user_id: str) -> None:
    st.markdown("### Publication export")
    notice(
        st,
        "info",
        "Paper-ready tables, straight from the measurements",
        "Every table is generated from the experiments you select - the same "
        "numbers the workspace shows. Reference values from the literature are "
        "never mixed in, and a value that was not measured stays empty.",
        icon="table_view",
    )

    with_results = [
        record
        for record in store.list_experiments(user_id, limit=200)
        if record.results
    ]
    if not with_results:
        empty_state(
            st,
            "table_view",
            "Nothing to export yet",
            "Run or schedule an evaluation first; export needs at least one "
            "experiment with measured results.",
        )
        return

    labels = {
        record.id: (
            f"{record.name} · {record.status} · {len(record.results)} model(s) · "
            f"{record.created_at}"
        )
        for record in with_results
    }
    chosen = st.multiselect(
        "Experiments to include",
        options=list(labels),
        default=[with_results[0].id],
        format_func=lambda key: labels.get(key, key),
        key="export_experiments",
    )
    if not chosen:
        st.caption("Select at least one experiment to build the tables.")
        return

    selected = [record for record in with_results if record.id in set(chosen)]
    rows = headline_rows(selected)

    st.markdown("#### Headline metrics preview")
    st.dataframe(
        [
            {label: _export_cell(key, row.get(key, "")) for key, label, _ in HEADLINE_COLUMNS}
            for row in rows
        ],
        hide_index=True,
        width="stretch",
    )

    per_class = per_class_rows(selected)
    if per_class:
        with st.expander(f"Per-class metrics preview ({len(per_class)} rows)"):
            st.dataframe(
                [
                    {
                        label: _export_cell(key, row.get(key, ""))
                        for key, label, _ in PER_CLASS_COLUMNS
                    }
                    for row in per_class
                ],
                hide_index=True,
                width="stretch",
            )

    key = hashlib.sha256(",".join(sorted(chosen)).encode("utf-8")).hexdigest()[:10]
    bundle_dir = settings.outputs_dir / "evaluation" / user_id / "publication" / key

    if st.button(
        "Prepare publication bundle",
        type="primary",
        icon=":material/download:",
        key="export_prepare",
    ):
        written = build_bundle(selected, bundle_dir)
        st.session_state["eval_export_bundle"] = {
            "dir": str(bundle_dir),
            "ids": tuple(sorted(chosen)),
        }
        st.success(f"Wrote {len(written)} file(s) to `{bundle_dir}`.")

    bundle = st.session_state.get("eval_export_bundle")
    if not bundle or tuple(bundle.get("ids", ())) != tuple(sorted(chosen)):
        st.caption(
            "Choose the experiments, then prepare the bundle to download the "
            "Markdown, LaTeX, CSV, dataset cards, provenance sheet and a printable "
            "HTML report."
        )
        return

    written = build_bundle(selected, Path(bundle["dir"]))
    st.markdown("#### Downloads")
    columns = st.columns(3)
    for index, (name, path) in enumerate(sorted(written.items())):
        with columns[index % 3]:
            download_button(
                st,
                name,
                path,
                _PUBLICATION_MIME.get(path.suffix.lower(), "application/octet-stream"),
                icon=":material/download:",
            )
    st.caption(f"Bundle directory: `{bundle['dir']}`")

    with st.expander("Preview the printable report", expanded=False):
        components.html(
            render_publication_html(selected), height=720, scrolling=True
        )


# --------------------------------------------------------------------------- #
# Compare section
# --------------------------------------------------------------------------- #
def _compare_section(settings, store: EvaluationStore, user_id: str) -> None:
    st.markdown("### Compare experiments")
    with_results = [
        record
        for record in store.list_experiments(user_id, limit=200)
        if record.results
    ]
    if len(with_results) < 2:
        empty_state(
            st,
            "compare_arrows",
            "Not enough results to compare",
            "At least two experiments with measured results are needed.",
        )
        return

    labels = {
        record.id: f"{record.name} · {record.status} · {record.split}"
        for record in with_results
    }
    chosen = st.multiselect(
        "Experiments to compare",
        options=list(labels),
        default=list(labels)[:2],
        format_func=lambda key: labels.get(key, key),
        key="eval_compare_selection",
    )
    selected = [record for record in with_results if record.id in chosen]
    if len(selected) < 2:
        st.caption("Choose at least two experiments.")
        return

    for warning in _comparability_warnings(selected):
        st.markdown(badge(warning, "warn"), unsafe_allow_html=True)

    rows = []
    for record in selected:
        for result in record.results:
            summary = result.summary or {}
            rows.append(
                {
                    "Experiment": record.name,
                    "Model": result.model,
                    "Dataset": record.dataset_name,
                    "Split": record.split,
                    "Status": result.status,
                    "Precision": _num(summary.get("precision")),
                    "Recall": _num(summary.get("recall")),
                    "mAP@0.5": _num(summary.get("mAP50")),
                    "mAP@0.5:0.95": _num(summary.get("mAP50_95")),
                    "s/image": _num(summary.get("seconds_per_image")),
                }
            )
    st.markdown("#### Headline metrics")
    st.dataframe(rows, hide_index=True, width="stretch")

    per_class_rows = []
    for record in selected:
        for result in record.results:
            for metric in result.metrics.get("classes", []):
                per_class_rows.append(
                    {
                        "Model": f"{record.name}/{result.model}",
                        "Class": metric.get("name", "?"),
                        "GT": metric.get("gt", 0),
                        "AP@0.5": _num(metric.get("AP50")),
                    }
                )
    if per_class_rows:
        st.markdown("#### Per-class AP@0.5")
        st.dataframe(per_class_rows, hide_index=True, width="stretch")

    st.caption(
        "Metric differences are only meaningful when the dataset fingerprint, "
        "split and class mapping agree. BlueEye never declares a universal "
        "'best' model."
    )
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "experiments": [record.as_dict() for record in selected],
        "warnings": _comparability_warnings(selected),
    }
    st.download_button(
        "Download comparison (JSON)",
        data=json.dumps(payload, indent=2).encode("utf-8"),
        file_name="blueeye_comparison.json",
        mime="application/json",
        key="eval_compare_download",
    )


def _comparability_warnings(experiments: list) -> list[str]:
    warnings: list[str] = []
    if len({record.dataset_fingerprint for record in experiments}) > 1:
        warnings.append(
            "These experiments use different datasets (fingerprints differ): "
            "metric differences are not directly comparable."
        )
    splits = {record.split for record in experiments}
    if len(splits) > 1:
        warnings.append(
            "Different splits are compared ("
            + ", ".join(sorted(splits))
            + "); treat the differences with care."
        )
    if len({tuple(record.class_names) for record in experiments}) > 1:
        warnings.append(
            "The dataset class mappings differ between experiments; per-class "
            "comparisons may be invalid."
        )
    return warnings


# --------------------------------------------------------------------------- #
# Pending state (no dataset)
# --------------------------------------------------------------------------- #
def _weights_notice() -> None:
    error_card(
        st,
        "No model weights installed",
        "Evaluation needs at least one installed model. Download the core "
        "weights from the AI Models page first.",
    )


def _pending_state(settings) -> None:
    notice(
        st,
        "info",
        "No dataset configured yet",
        "BlueEye found no labelled dataset, so any experiment here stays "
        "pending. Register one in the Datasets section, or prepare one and it "
        "will be discovered automatically - BlueEye never scores data it did "
        "not measure.",
        icon="info",
    )
    with st.expander("How to prepare a licensed dataset", expanded=True):
        st.markdown("**1. Convert / register an existing YOLO export**")
        st.code(
            "python scripts/prepare_dataset.py <path-to-export> --name my_dataset",
            language="powershell",
        )
        st.markdown(
            "Datasets are discovered under `data/raw/`, `data/datasets/`, "
            "`data/input/` and `datasets/`."
        )
        st.markdown("**2. Evaluate from the command line (same code path)**")
        st.code(
            "python scripts/run_evaluation.py --data data/raw/my_dataset/data.yaml "
            "--split test --model all --plots",
            language="powershell",
        )
        st.markdown(
            "Documented dataset sources and their licences live in "
            "`docs/evaluation.md`. BlueEye never downloads unverified data and "
            "never reports a metric it did not measure."
        )
    st.caption(
        "Searched: "
        + ", ".join(str(root) for root in default_search_roots(settings))
    )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _num(value):
    """Round a metric for display, or return a readable placeholder."""
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "n/a"


__all__ = ["render"]
