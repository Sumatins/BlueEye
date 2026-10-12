"""Background execution of an evaluation experiment.

A run can take minutes and must not block the Streamlit UI. This module owns
the small amount of state needed to start, track and cancel an experiment:

* the experiment row's ``status`` is the single source of truth and lives in
  the persistent store, so an interrupted run is still visible after a restart
  (it shows as *running* and can be re-run or cancelled);
* an in-process ``threading.Event`` per experiment lets a user request
  cancellation, which is noticed between images.

Nothing here computes a metric itself: it delegates to
:mod:`app.evaluation.runner` and stores exactly what that produced. A run that
cannot proceed (missing split, model that fails to load) is recorded as
*failed* with the reason, never as a zero score.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from app.evaluation.compat import compare_classes
from app.evaluation.datasets import DatasetError, limit_split, load_split
from app.evaluation.environment import (
    collect_environment,
    evaluator_version,
    git_commit,
)
from app.evaluation.experiments import EvaluationStore
from app.evaluation.report import write_report
from app.evaluation.runner import (
    InferenceError,
    detector_predictor,
    model_class_names,
    run_evaluation,
)

logger = logging.getLogger(__name__)


class CancelSignal(Exception):
    """Internal: raised inside a predictor when a run is cancelled."""


_CANCELS: dict[str, threading.Event] = {}
_THREADS: dict[str, threading.Thread] = {}
_LOCK = threading.Lock()


def is_running(experiment_id: str) -> bool:
    """True when a job for this experiment is alive in *this* process.

    A ``running`` row with no live thread means the process that owned it was
    restarted: the UI reports that rather than pretending it is still active.
    """
    with _LOCK:
        thread = _THREADS.get(experiment_id)
    return bool(thread and thread.is_alive())


def request_cancel(experiment_id: str) -> None:
    """Ask a running experiment to stop after the current image."""
    with _LOCK:
        event = _CANCELS.get(experiment_id)
        if event is not None:
            event.set()


def is_cancelling(experiment_id: str) -> bool:
    with _LOCK:
        event = _CANCELS.get(experiment_id)
        return bool(event and event.is_set())


def _cancel_event(experiment_id: str) -> threading.Event:
    with _LOCK:
        event = _CANCELS.get(experiment_id)
        if event is None:
            event = threading.Event()
            _CANCELS[experiment_id] = event
        return event


def _forget(experiment_id: str) -> None:
    with _LOCK:
        _CANCELS.pop(experiment_id, None)


@dataclass
class ExperimentRequest:
    """Everything the background job needs to reproduce a run."""

    experiment_id: str
    user_id: str
    data_yaml: str
    split: str
    model_keys: tuple[str, ...]
    confidence: float | None
    iou_threshold: float
    limit: int | None
    output_dir: Path
    make_plots: bool
    parameters: dict = field(default_factory=dict)


def _guarded(predict, event: threading.Event):
    def _run(image_path):
        if event.is_set():
            raise CancelSignal()
        return predict(image_path)

    return _run


def _final_status(complete: int, requested: int, cancelled: bool) -> str:
    if cancelled:
        return "cancelled"
    if requested and complete >= requested:
        return "completed"
    if complete > 0:
        return "partially_completed"
    return "failed"


def run_experiment(
    store: EvaluationStore, request: ExperimentRequest, *, detector
) -> str:
    """Execute one experiment synchronously and return its final status.

    The store row must already exist (created by the caller with the full
    configuration). This function transitions it through *running* to a
    terminal state and records results, artifacts and provenance.
    """
    event = _cancel_event(request.experiment_id)
    warnings: list[str] = []
    errors: list[str] = []

    if event.is_set():
        store.set_status(
            request.experiment_id,
            request.user_id,
            "cancelled",
            error="Cancelled before starting.",
        )
        _forget(request.experiment_id)
        return "cancelled"

    store.set_status(request.experiment_id, request.user_id, "running")

    try:
        dataset = load_split(request.data_yaml, request.split)
    except DatasetError as exc:
        store.set_status(
            request.experiment_id,
            request.user_id,
            "failed",
            error=f"Could not load the '{request.split}' split: {exc}",
        )
        _forget(request.experiment_id)
        return "failed"

    if request.limit:
        dataset = limit_split(dataset, request.limit)

    out_dir = Path(request.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluations = []
    figures: list[tuple[str, Path]] = []
    complete = 0
    requested = len(request.model_keys)

    from app.evaluation import plots as plots_module

    for key in request.model_keys:
        if event.is_set():
            break
        try:
            classes = model_class_names(detector, key)
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal
            errors.append(f"Model '{key}' could not be loaded: {exc}")
            continue

        compatibility = compare_classes(dataset.class_names, classes)
        if not compatibility.comparable:
            warnings.append(
                f"Model '{key}' shares no class names with this dataset, so it "
                "was not evaluated (the metrics would not be a valid "
                "comparison)."
            )
            continue

        predict = detector_predictor(
            detector, model_selection=key, confidence=request.confidence
        )
        try:
            evaluation = run_evaluation(
                key, _guarded(predict, event), dataset, classes
            )
        except CancelSignal:
            break
        except InferenceError as exc:
            if event.is_set():
                break
            errors.append(f"Model '{key}' failed: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal
            if event.is_set():
                break
            errors.append(f"Model '{key}' failed: {exc}")
            continue

        store.add_model_result(request.experiment_id, request.user_id, evaluation)
        evaluations.append(evaluation)
        complete += 1
        store.set_status(
            request.experiment_id,
            request.user_id,
            "running",
            progress={"completed": complete, "requested": requested},
        )

        if request.make_plots:
            for path in plots_module.plot_evaluation_figures(
                evaluation, out_dir, iou_threshold=request.iou_threshold
            ):
                figures.append((path.stem.replace("_", " "), path))
            for path in plots_module.plot_examples(
                evaluation, dataset, out_dir / "examples", limit=4
            ):
                figures.append((f"examples/{path.stem}", path))

    if request.make_plots and len(evaluations) > 1:
        chart = plots_module.plot_metric_comparison(
            evaluations, out_dir / "metric_comparison.png"
        )
        if chart is not None:
            figures.append(("model comparison", chart))

    cancelled = event.is_set()
    _forget(request.experiment_id)

    # Persist the report from the *actual* evaluations produced this run.
    environment = collect_environment()
    config = {
        "experiment": request.experiment_id,
        "dataset": dataset.name,
        "data_yaml": request.data_yaml,
        "split": request.split,
        "models": list(request.model_keys),
        "parameters": request.parameters,
        "evaluator": evaluator_version(),
        "iou_threshold_for_figures": request.iou_threshold,
    }
    try:
        experiment_row = store.get_experiment(
            request.experiment_id, request.user_id
        )
        written = write_report(
            out_dir,
            evaluations,
            environment=environment,
            config=config,
            plots=figures,
            experiment=experiment_row.as_dict() if experiment_row else None,
        )
        report_paths = list(written.values())
    except Exception as exc:  # noqa: BLE001 - storage of the run still matters
        logger.exception("Could not write evaluation report: %s", exc)
        errors.append(f"Report generation failed: {exc}")
        report_paths = []

    final = _final_status(complete, len(request.model_keys), cancelled)
    store.set_status(
        request.experiment_id,
        request.user_id,
        final,
        error="; ".join(errors),
        warnings=tuple(warnings),
    )

    for path in [*report_paths, *(figure for _, figure in figures)]:
        try:
            store.add_artifact(
                request.experiment_id,
                request.user_id,
                path,
                name=path.name,
                role="plot" if path.suffix.lower() == ".png" else "report",
            )
        except OSError:
            continue
    return final


def start_experiment(
    store: EvaluationStore, request: ExperimentRequest, *, detector
) -> threading.Thread:
    """Run :func:`run_experiment` on a daemon thread and return the thread."""
    thread = threading.Thread(
        target=run_experiment,
        args=(store, request),
        kwargs={"detector": detector},
        name=f"evaluation-{request.experiment_id}",
        daemon=True,
    )
    with _LOCK:
        for finished in [
            key for key, item in _THREADS.items() if not item.is_alive()
        ]:
            _THREADS.pop(finished, None)
        _THREADS[request.experiment_id] = thread
    thread.start()
    return thread
