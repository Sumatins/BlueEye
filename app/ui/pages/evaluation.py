"""Research Evaluation Center: run the real evaluation pipeline from the UI.

This page is the interface over :mod:`app.evaluation` and the *same* code path
as ``scripts/run_evaluation.py``. It runs the unmodified detectors over a
locally prepared labelled dataset, checks class compatibility, computes
precision / recall / mAP@0.5 / mAP@0.5:0.95 and writes JSON / CSV / Markdown
results (plus optional charts).

Honesty rules enforced here
---------------------------
* No dataset is bundled or downloaded. When none is found the page clearly
  explains how to configure one and stays **pending** - it never invents a
  score.
* Metrics are only shown when a real run produced them, and the wording makes
  clear they were *measured locally* and are not comparable across different
  datasets.
* Published reference values are never mixed in.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

from app.detection.detector import MarineDetector
from app.evaluation import (
    DatasetError,
    collect_environment,
    detector_predictor,
    limit_split,
    load_split,
    model_class_names,
    run_evaluation,
    write_report,
)
from app.evaluation.discovery import (
    DatasetInfo,
    available_splits,
    default_search_roots,
    discover_datasets,
)
from app.ui import state
from app.ui.components import (
    badge,
    download_button,
    error_card,
    hero,
    metrics_row,
    notice,
)

logger = logging.getLogger(__name__)

#: Default number of images per evaluation (keeps the UI responsive).
DEFAULT_LIMIT = 25


def render(detector: MarineDetector, settings, user=None) -> None:
    """Render the evaluation console for the signed-in account."""
    user_id = user.id if user else None

    hero(
        st,
        "Measured locally, never fabricated",
        "Research Evaluation Center",
        "Run the real detectors over a labelled dataset and record "
        "precision, recall and mAP with full provenance.",
        icon="science",
    )
    if user_id is None:
        st.caption("Sign in to run and store an evaluation.")
        return

    datasets = discover_datasets(default_search_roots(settings))
    if datasets:
        _controls(detector, settings, user_id, datasets)
    else:
        _pending_state(settings)
    # Stored results stay visible even if the dataset is no longer on disk.
    _stored_runs(settings, user_id)


# --------------------------------------------------------------------------- #
# Pending / no-dataset state
# --------------------------------------------------------------------------- #
def _pending_state(settings) -> None:
    """Explain clearly that the experiment cannot run yet (no dataset)."""
    notice(
        st,
        "warning",
        "Experiment pending - no labelled dataset found",
        "Metrics require ground-truth bounding boxes. BlueEye has not found a "
        "prepared dataset in the project's data directories, so no score can "
        "be reported.",
        icon="hourglass_top",
    )
    with st.expander("How to configure a dataset", expanded=True):
        st.markdown("**1. Prepare a licensed dataset in YOLO format**")
        st.code(
            "python scripts/prepare_dataset.py <path-to-export> --name my_dataset",
            language="powershell",
        )
        st.markdown(
            "Datasets are discovered automatically under `data/raw/`, "
            "`data/datasets/`, `data/input/` and `datasets/`."
        )
        st.markdown(
            "**2. Evaluate from the command line (equivalent to this page)**"
        )
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
    scanned = ", ".join(str(root) for root in default_search_roots(settings))
    st.caption(f"Searched: {scanned}")
    st.caption("The rest of BlueEye (detection, comparison, history) stays fully usable.")


# --------------------------------------------------------------------------- #
# Run controls
# --------------------------------------------------------------------------- #
def _controls(
    detector: MarineDetector, settings, user_id: str, datasets: list[DatasetInfo]
) -> None:
    st.markdown("### Configure an evaluation")
    notice(
        st,
        "info",
        "What this measures",
        "Each selected model is run independently on the same split with its "
        "own recommended thresholds (or your override). Metrics are measured "
        "on the machine shown and are not comparable across different "
        "datasets.",
        icon="info",
    )

    available = state.available_keys(detector)
    if not available:
        error_card(
            st,
            "No model weights installed",
            "Evaluation needs at least one installed model. Download the core "
            "weights from the AI Models page first.",
        )
        return

    registry = state.registry(detector)
    labels = {info.data_yaml: f"{info.name} · {info.num_classes} classes" for info in datasets}

    left, right = st.columns([2, 1])
    with left:
        dataset_yaml = st.selectbox(
            "Dataset",
            options=[info.data_yaml for info in datasets],
            format_func=lambda path: labels.get(path, str(path)),
            key="eval_dataset",
        )
    splits = list(available_splits(dataset_yaml)) or ["test"]
    with right:
        split = st.selectbox(
            "Split",
            options=splits,
            index=(splits.index("test") if "test" in splits else 0),
            key="eval_split",
        )

    info = next((item for item in datasets if item.data_yaml == dataset_yaml), None)
    if info is not None and info.class_names:
        st.caption("Dataset classes: " + ", ".join(info.class_names))

    selections = st.multiselect(
        "Models to evaluate",
        options=available,
        default=available[: min(2, len(available))],
        format_func=lambda key: state.model_label(key, registry.get(key), True),
        key="eval_models",
        help="Each model is evaluated independently on the same split.",
    )

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
            max_value=100000,
            value=DEFAULT_LIMIT,
            step=5,
            key="eval_limit",
        )

    plots = st.checkbox(
        "Write comparison charts (requires matplotlib)",
        value=True,
        key="eval_plots",
    )

    if st.button("Run evaluation", type="primary", icon=":material/play_arrow:", key="eval_run"):
        if not selections:
            error_card(st, "Select at least one model", "Pick one or more models to evaluate.")
        else:
            _execute(
                detector,
                settings,
                user_id,
                Path(dataset_yaml),
                split,
                list(selections),
                None if use_default else float(confidence),
                int(limit or 0),
                bool(plots),
            )


# --------------------------------------------------------------------------- #
# Execute
# --------------------------------------------------------------------------- #
def _eval_root(settings, user_id) -> Path:
    return settings.outputs_dir / "evaluation" / (user_id or "cli")


def _execute(
    detector: MarineDetector,
    settings,
    user_id,
    dataset_yaml: Path,
    split: str,
    selections: list[str],
    confidence: float | None,
    limit: int,
    plots: bool,
) -> None:
    try:
        dataset = load_split(dataset_yaml, split=split)
    except DatasetError as exc:
        error_card(st, "Could not load the dataset split", str(exc))
        return
    dataset = limit_split(dataset, limit)

    st.markdown("### Run")
    metrics_row(
        st,
        [
            ("Images", dataset.num_images, "in this split"),
            ("Labelled objects", dataset.num_objects, "ground-truth boxes"),
            ("Dataset classes", len(dataset.class_names), "declared in data.yaml"),
            ("Models", len(selections), "selected"),
        ],
    )
    for note in dataset.notes:
        st.caption(f"note: {note}")

    total = max(1, dataset.num_images * len(selections))
    progress = st.progress(0.0, text="Starting evaluation...")
    counter = {"done": 0}
    evaluations = []
    registry = state.registry(detector)

    for key in selections:
        spec = registry.get(key)
        label = spec.display_name if spec is not None else key
        try:
            classes = model_class_names(detector, key)
            base = detector_predictor(detector, key, confidence=confidence)
        except Exception as exc:  # noqa: BLE001 - report and continue
            error_card(st, f"Could not prepare {label}", str(exc))
            continue

        def _wrapped(path, _base=base, _label=label):
            predictions = _base(path)
            counter["done"] += 1
            progress.progress(
                min(counter["done"] / total, 1.0),
                text=f"{_label}: {counter['done']}/{total} images",
            )
            return predictions

        try:
            evaluations.append(run_evaluation(label, _wrapped, dataset, classes))
        except Exception as exc:  # noqa: BLE001 - one model must not kill the run
            error_card(st, f"Evaluation failed for {label}", str(exc))

    if not evaluations:
        error_card(st, "Nothing to report", "No model could be evaluated.")
        return

    out_dir = _eval_root(settings, user_id) / datetime.now().strftime("%Y%m%d-%H%M%S")
    environment = collect_environment()
    config = {
        "data_yaml": str(dataset_yaml),
        "split": dataset.split,
        "limit": limit,
        "selections": selections,
        "confidence_override": confidence,
        "device": detector.device,
        "note": (
            "Input resolution and preprocessing are Ultralytics' defaults used "
            "by the application itself; the detector pipeline is unchanged."
        ),
    }
    write_report(out_dir, evaluations, environment=environment, config=config)

    if plots:
        try:
            from app.evaluation.plots import plot_all, plot_examples

            plot_all(evaluations, out_dir)
            for evaluation in evaluations:
                plot_examples(evaluation, dataset, out_dir / "examples")
        except Exception as exc:  # noqa: BLE001 - plotting is optional
            st.caption(f"Charts skipped: {exc}")

    st.session_state["eval_last_run"] = str(out_dir)
    st.rerun()


# --------------------------------------------------------------------------- #
# Stored runs
# --------------------------------------------------------------------------- #
def _stored_runs(settings, user_id) -> None:
    root = _eval_root(settings, user_id)
    runs = sorted(
        (path for path in root.glob("*/results.json") if path.is_file()),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    if not runs:
        st.caption(
            "No evaluation has been stored yet for this account. Configure one "
            "above to produce the first real result set."
        )
        return

    st.markdown("### Stored evaluation runs")
    last = st.session_state.get("eval_last_run")
    index = next((i for i, path in enumerate(runs) if str(path.parent) == last), 0)
    labels = {path: path.parent.name for path in runs}
    selected = st.selectbox(
        "Stored run",
        options=runs,
        index=index,
        format_func=lambda path: labels[path],
        key="eval_stored",
    )
    _render_run(selected.parent, settings, user_id)


def _render_run(out_dir: Path, settings, user_id) -> None:
    results_path = Path(out_dir) / "results.json"
    if not state.is_owned_path(settings, user_id, results_path) or not results_path.is_file():
        st.caption("That run's results are no longer available on this device.")
        return
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        st.caption(f"That run's results could not be read: {exc}")
        return

    st.markdown("### Results")
    st.caption(
        f"Measured locally · {payload.get('created', 'unknown time')} · "
        "not comparable across different datasets."
    )

    rows = []
    for item in payload.get("summary", []):
        rows.append(
            {
                "Model": item.get("model", "?"),
                "Split": item.get("split", "?"),
                "Status": item.get("status", "?"),
                "Images": item.get("images", 0),
                "Objects": item.get("objects", 0),
                "mAP@0.5": f"{float(item.get('mAP50', 0.0)):.3f}",
                "mAP@0.5:0.95": f"{float(item.get('mAP50_95', 0.0)):.3f}",
                "Precision": f"{float(item.get('precision', 0.0)):.3f}",
                "Recall": f"{float(item.get('recall', 0.0)):.3f}",
                "s/image": f"{float(item.get('seconds_per_image', 0.0)):.3f}",
            }
        )
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")

    for item in payload.get("evaluations", []):
        _render_evaluation(item)

    environment = payload.get("environment") or {}
    if environment:
        hardware = environment.get("hardware", {})
        st.caption(
            "Test conditions: "
            f"{environment.get('platform', 'unknown')} · "
            f"device {hardware.get('gpu') or 'CPU'} · "
            f"CUDA {hardware.get('cuda_available', False)}"
        )

    _figures(Path(out_dir))
    _downloads(Path(out_dir))


def _render_evaluation(item: dict) -> None:
    comparison = item.get("comparison", {})
    with st.expander(
        f"{item.get('model', 'model')} · {item.get('status', 'unknown')}",
        expanded=False,
    ):
        supported = comparison.get("supported", [])
        st.markdown(
            f"**Comparable classes ({len(supported)}):** "
            + (", ".join(supported) if supported else "none")
        )
        unsupported = comparison.get("unsupported_dataset_classes", [])
        if unsupported:
            st.markdown(
                badge(f"{len(unsupported)} dataset class(es) this model cannot detect", "warn"),
                unsafe_allow_html=True,
            )
            st.caption("Excluded from per-class scoring: " + ", ".join(unsupported))
        extra = comparison.get("extra_model_classes", [])
        if extra:
            st.caption("Model classes absent from the dataset (ignored): " + ", ".join(extra))

        classes = (item.get("metrics") or {}).get("classes", [])
        if classes:
            st.dataframe(
                [
                    {
                        "Class": entry.get("name", "?"),
                        "GT": entry.get("gt", 0),
                        "Pred": entry.get("predictions", 0),
                        "Precision": f"{float(entry.get('precision', 0.0)):.3f}",
                        "Recall": f"{float(entry.get('recall', 0.0)):.3f}",
                        "AP50": f"{float(entry.get('AP50', 0.0)):.3f}",
                        "AP50:95": f"{float(entry.get('AP50_95', 0.0)):.3f}",
                    }
                    for entry in classes
                ],
                hide_index=True,
                width="stretch",
            )

        timing = item.get("timing", {})
        st.caption(
            f"{timing.get('images', 0)} images · "
            f"{timing.get('predictions', 0)} predictions · "
            f"{float(timing.get('seconds_per_image', 0.0)):.3f} s/image"
        )
        for note in item.get("notes", []):
            st.caption(f"note: {note}")


def _figures(out_dir: Path) -> None:
    figures = sorted(out_dir.glob("*.png")) + sorted((out_dir / "examples").glob("*.png"))
    if not figures:
        return
    st.markdown("### Charts")
    for figure in figures[:12]:
        try:
            st.image(figure.read_bytes(), caption=figure.name, width="stretch")
        except OSError:
            continue


def _downloads(out_dir: Path) -> None:
    st.markdown("### Export")
    files = (
        ("results.json", "application/json", "Results (JSON)"),
        ("metrics.csv", "text/csv", "Metrics (CSV)"),
        ("per_class.csv", "text/csv", "Per-class (CSV)"),
        ("summary.md", "text/markdown", "Summary (Markdown)"),
    )
    columns = st.columns(len(files))
    for column, (name, mime, label) in zip(columns, files):
        path = out_dir / name
        with column:
            if path.is_file():
                download_button(st, label, path, mime)


__all__ = ["render"]
