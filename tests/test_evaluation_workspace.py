"""Unit tests for the Research Evaluation Center workspace.

These cover the pieces added for the workspace: read-only dataset validation,
the persistent experiment store, the background job lifecycle (with a fake
detector - no weights), the self-contained HTML report and the comparison
warnings. Nothing here loads real model weights or touches the network.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

from app.evaluation.experiments import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    DatasetRecord,
    EvaluationStore,
    ExperimentRecord,
)
from app.evaluation.jobs import (
    ExperimentRequest,
    _Progress,
    _final_status,
    is_running,
    run_experiment,
)
from app.evaluation.validation import validate_dataset


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def make_dataset(root: Path, *, label: str = "0 0.5 0.5 0.2 0.2\n", names=("fish",)) -> Path:
    """A tiny but real one-image YOLO dataset (test split only)."""
    (root / "images" / "test").mkdir(parents=True)
    (root / "labels" / "test").mkdir(parents=True)
    Image.new("RGB", (32, 32), (10, 40, 90)).save(root / "images" / "test" / "a.jpg")
    (root / "labels" / "test" / "a.txt").write_text(label, encoding="utf-8")
    lines = ["path: .", "test: images/test", "names:"]
    lines += [f"  {index}: {name}" for index, name in enumerate(names)]
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root / "data.yaml"


class _Detection:
    def __init__(self, class_name, confidence, bbox):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox


class _Result:
    def __init__(self, detections):
        self.detections = detections


class _FakeDetector:
    """A stand-in detector: real predictor plumbing, no real weights."""

    def __init__(self, classes=("fish",)):
        self.classes = tuple(classes)

    def get_class_names(self, model_selection=None):
        return {(model_selection or "fake"): dict(enumerate(self.classes))}

    def predict_image(self, image, model_selection=None, confidence=None, source_name=""):
        if model_selection == "broken":
            raise RuntimeError("weights for 'broken' are unavailable")
        return _Result([_Detection("fish", 0.9, (1, 1, 20, 20))])


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validation_accepts_a_clean_dataset(tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "ds")
    result = validate_dataset(data_yaml, split="test")
    assert result.ready
    assert not result.errors
    assert result.class_names == ("fish",)
    assert result.splits_with_images == ("test",)
    assert result.fingerprint


def test_validation_flags_a_malformed_row(tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "ds", label="0 0.5 0.5 0.2\n")
    result = validate_dataset(data_yaml, split="test")
    assert not result.ready
    assert any(issue.code == "label_malformed" for issue in result.errors)
    assert any(issue.path and issue.path.endswith("a.txt") for issue in result.errors)


def test_validation_flags_an_out_of_range_box(tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "ds", label="0 1.5 0.5 0.2 0.2\n")
    result = validate_dataset(data_yaml, split="test")
    assert not result.ready
    assert any(issue.code == "label_out_of_range" for issue in result.errors)


def test_validation_flags_an_unknown_class_id(tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "ds", label="7 0.5 0.5 0.2 0.2\n")
    result = validate_dataset(data_yaml, split="test")
    assert not result.ready
    assert any(issue.code == "label_bad_class" for issue in result.errors)


def test_validation_warns_when_a_label_is_missing(tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "ds")
    (tmp_path / "ds" / "labels" / "test" / "a.txt").unlink()
    result = validate_dataset(data_yaml, split="test")
    # A missing label is a valid negative image (warning, not an error).
    assert result.ready
    assert any(issue.code == "missing_label" for issue in result.warnings)


def test_validation_flags_nc_mismatch(tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "ds")
    data_yaml.write_text(
        data_yaml.read_text(encoding="utf-8") + "nc: 3\n", encoding="utf-8"
    )
    result = validate_dataset(data_yaml, split="test")
    assert not result.ready
    assert any(issue.code == "nc_mismatch" for issue in result.errors)


def test_validation_reports_a_missing_requested_split(tmp_path: Path) -> None:
    data_yaml = make_dataset(tmp_path / "ds")
    result = validate_dataset(data_yaml, split="train")
    assert not result.ready
    assert any(issue.code == "requested_split_missing" for issue in result.errors)


# --------------------------------------------------------------------------- #
# Persistent store
# --------------------------------------------------------------------------- #
def _register_owned(store: EvaluationStore, user_id: str, data_yaml: Path) -> DatasetRecord:
    validation = validate_dataset(data_yaml, split="test")
    return store.register_dataset(
        user_id,
        "tiny",
        data_yaml,
        class_names=validation.class_names,
        splits=validation.splits_with_images,
        fingerprint=validation.fingerprint,
        validation=validation.as_dict(),
    )


def test_store_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "eval.db"
    data_yaml = make_dataset(tmp_path / "ds")
    store = EvaluationStore(db)
    record = _register_owned(store, "u1", data_yaml)
    experiment = store.create_experiment(
        "u1", "run", dataset_id=record.id, data_yaml=str(data_yaml), split="test"
    )

    reopened = EvaluationStore(db)
    loaded = reopened.get_experiment(experiment.id, "u1")
    assert loaded is not None
    assert loaded.name == "run"
    assert [item.id for item in reopened.list_datasets("u1")] == [record.id]


def test_store_is_owner_scoped(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path / "eval.db")
    data_yaml = make_dataset(tmp_path / "ds")
    record = _register_owned(store, "u1", data_yaml)
    experiment = store.create_experiment("u1", "run", dataset_id=record.id)

    assert store.get_experiment(experiment.id, "u2") is None
    assert store.get_dataset(record.id, "u2") is None
    assert store.list_experiments("u2") == ()
    assert store.delete_experiment(experiment.id, "u2") is False


def test_store_records_results_and_artifacts(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path / "eval.db")
    data_yaml = make_dataset(tmp_path / "ds")
    record = _register_owned(store, "u1", data_yaml)
    experiment = store.create_experiment("u1", "run", dataset_id=record.id)

    detector = _FakeDetector()
    request = ExperimentRequest(
        experiment_id=experiment.id,
        user_id="u1",
        data_yaml=str(data_yaml),
        split="test",
        model_keys=("fake",),
        confidence=None,
        iou_threshold=0.5,
        limit=None,
        output_dir=tmp_path / "out",
        make_plots=False,
    )
    status = run_experiment(store, request, detector=detector)
    assert status == "completed"

    loaded = store.get_experiment(experiment.id, "u1")
    assert loaded is not None
    assert loaded.status in TERMINAL_STATUSES
    assert loaded.status == "completed"
    assert len(loaded.results) == 1
    assert loaded.results[0].summary["images"] == 1
    assert loaded.results[0].model == "fake"
    artifact_names = {artifact.name for artifact in loaded.artifacts}
    assert "results.json" in artifact_names
    assert (tmp_path / "out" / "report.html").is_file()


def test_run_marks_partial_success_when_a_model_fails(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path / "eval.db")
    data_yaml = make_dataset(tmp_path / "ds")
    record = _register_owned(store, "u1", data_yaml)
    experiment = store.create_experiment("u1", "run", dataset_id=record.id)

    request = ExperimentRequest(
        experiment_id=experiment.id,
        user_id="u1",
        data_yaml=str(data_yaml),
        split="test",
        model_keys=("fake", "broken"),
        confidence=None,
        iou_threshold=0.5,
        limit=None,
        output_dir=tmp_path / "out",
        make_plots=False,
    )
    status = run_experiment(store, request, detector=_FakeDetector())
    assert status == "partially_completed"
    loaded = store.get_experiment(experiment.id, "u1")
    assert loaded is not None
    assert loaded.status == "partially_completed"
    assert "broken" in loaded.error
    assert len(loaded.results) == 1


def test_run_fails_when_the_split_cannot_be_loaded(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path / "eval.db")
    experiment = store.create_experiment("u1", "run")
    request = ExperimentRequest(
        experiment_id=experiment.id,
        user_id="u1",
        data_yaml=str(tmp_path / "missing.yaml"),
        split="test",
        model_keys=("fake",),
        confidence=None,
        iou_threshold=0.5,
        limit=None,
        output_dir=tmp_path / "out",
        make_plots=False,
    )
    assert run_experiment(store, request, detector=_FakeDetector()) == "failed"
    loaded = store.get_experiment(experiment.id, "u1")
    assert loaded is not None and loaded.status == "failed" and loaded.error


def _stored_progress(store: EvaluationStore, experiment_id: str) -> dict:
    record = store.get_experiment(experiment_id, "u1")
    assert record is not None
    return record.progress


def test_progress_reporter_throttles_but_forces_stage(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path / "eval.db")
    experiment = store.create_experiment("u1", "run")

    reporter = _Progress(store, experiment.id, "u1", 2, min_interval=60.0)
    reporter.stage("inference", model="fake", images_done=1, images_total=10)
    assert _stored_progress(store, experiment.id)["images_done"] == 1

    # A counter update inside the throttle window is deliberately not written...
    reporter.update(images_done=2)
    assert _stored_progress(store, experiment.id)["images_done"] == 1

    # ...but a stage transition is always flushed immediately.
    reporter.stage("report", images_done=10)
    stored = _stored_progress(store, experiment.id)
    assert stored["stage"] == "report"
    assert stored["images_done"] == 10


def test_run_experiment_persists_live_stage_and_counters(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path / "eval.db")
    data_yaml = make_dataset(tmp_path / "ds")
    record = _register_owned(store, "u1", data_yaml)
    experiment = store.create_experiment("u1", "run", dataset_id=record.id)

    request = ExperimentRequest(
        experiment_id=experiment.id,
        user_id="u1",
        data_yaml=str(data_yaml),
        split="test",
        model_keys=("fake",),
        confidence=None,
        iou_threshold=0.5,
        limit=None,
        output_dir=tmp_path / "out",
        make_plots=False,
    )
    assert run_experiment(store, request, detector=_FakeDetector()) == "completed"

    progress = _stored_progress(store, experiment.id)
    assert progress["requested"] == 1
    assert progress["completed"] == 1
    assert progress["images_total"] == 1
    assert progress["stage"] == "report"


def test_progress_view_derives_images_percent_elapsed_and_eta() -> None:
    from app.ui.pages import evaluation as page

    started = (datetime.now(tz=timezone.utc) - timedelta(seconds=20)).strftime(
        "%Y-%m-%dT%H:%M:%S.%f%z"
    )
    record = types.SimpleNamespace(
        progress={
            "stage": "inference",
            "images_done": 5,
            "images_total": 10,
            "completed": 0,
            "requested": 2,
        },
        started_at=started,
    )
    view = page._progress_view(record)
    assert view["fraction"] == pytest.approx(0.5)
    assert view["counter_label"] == "5/10 images"
    assert view["stage_label"] == "Running inference"
    assert view["elapsed"] == pytest.approx(20, abs=5)
    assert view["eta"] == pytest.approx(20, abs=6)  # roughly as long again


def test_progress_view_falls_back_to_models_without_eta() -> None:
    from app.ui.pages import evaluation as page

    record = types.SimpleNamespace(
        progress={"stage": "loading model", "completed": 1, "requested": 4},
        started_at=None,
    )
    view = page._progress_view(record)
    assert view["fraction"] == pytest.approx(0.25)
    assert view["counter_label"] == "1/4 models"
    assert view["elapsed"] is None
    assert view["eta"] is None


def test_final_status_matrix() -> None:
    assert _final_status(2, 2, False) == "completed"
    assert _final_status(1, 2, False) == "partially_completed"
    assert _final_status(0, 2, False) == "failed"
    assert _final_status(2, 2, True) == "cancelled"
    assert is_running("exp_does_not_exist") is False
    assert "queued" in ACTIVE_STATUSES


def test_run_can_be_cancelled_between_images(tmp_path: Path, monkeypatch) -> None:
    import app.evaluation.jobs as jobs

    # Two images so cancellation is noticed after the first one.
    data_yaml = make_dataset(tmp_path / "ds")
    Image.new("RGB", (32, 32), (5, 5, 5)).save(
        tmp_path / "ds" / "images" / "test" / "b.jpg"
    )
    (tmp_path / "ds" / "labels" / "test" / "b.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    store = EvaluationStore(tmp_path / "eval.db")
    record = _register_owned(store, "u1", data_yaml)
    experiment = store.create_experiment("u1", "cancel me", dataset_id=record.id)

    experiment_id = experiment.id

    class _CancellingDetector(_FakeDetector):
        def predict_image(self, *args, **kwargs):
            jobs.request_cancel(experiment_id)
            return _Result([_Detection("fish", 0.9, (1, 1, 20, 20))])

    request = ExperimentRequest(
        experiment_id=experiment.id,
        user_id="u1",
        data_yaml=str(data_yaml),
        split="test",
        model_keys=("fake",),
        confidence=None,
        iou_threshold=0.5,
        limit=None,
        output_dir=tmp_path / "out",
        make_plots=False,
    )
    assert jobs.run_experiment(store, request, detector=_CancellingDetector()) == "cancelled"
    loaded = store.get_experiment(experiment.id, "u1")
    assert loaded is not None and loaded.status == "cancelled"


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def test_plots_are_written_and_embedded_in_the_html_report(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path / "eval.db")
    data_yaml = make_dataset(tmp_path / "ds")
    record = _register_owned(store, "u1", data_yaml)
    experiment = store.create_experiment("u1", "plotted", dataset_id=record.id)

    request = ExperimentRequest(
        experiment_id=experiment.id,
        user_id="u1",
        data_yaml=str(data_yaml),
        split="test",
        model_keys=("fake",),
        confidence=None,
        iou_threshold=0.5,
        limit=None,
        output_dir=tmp_path / "out",
        make_plots=True,
    )
    assert run_experiment(store, request, detector=_FakeDetector()) == "completed"

    figures = sorted(path.name for path in (tmp_path / "out").glob("*.png"))
    assert figures, "no real figures were written"
    html = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    loaded = store.get_experiment(experiment.id, "u1")
    assert loaded is not None
    assert any(artifact.role == "plot" for artifact in loaded.artifacts)


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def test_html_report_is_self_contained_and_cites_the_experiment(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path / "eval.db")
    data_yaml = make_dataset(tmp_path / "ds")
    record = _register_owned(store, "u1", data_yaml)
    experiment = store.create_experiment("u1", "My Reef Run", dataset_id=record.id)
    request = ExperimentRequest(
        experiment_id=experiment.id,
        user_id="u1",
        data_yaml=str(data_yaml),
        split="test",
        model_keys=("fake",),
        confidence=None,
        iou_threshold=0.5,
        limit=None,
        output_dir=tmp_path / "out",
        make_plots=False,
    )
    run_experiment(store, request, detector=_FakeDetector())

    html = (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    assert "Metric definitions" in html
    assert "My Reef Run" in html
    assert "Scope" in html
    assert "mAP@0.5" in html
    # The report carries its own metric definitions so a number cannot be
    # mistaken for something it is not.
    assert "intersection-over-union" in html


# --------------------------------------------------------------------------- #
# Comparison warnings
# --------------------------------------------------------------------------- #
def _record(**overrides) -> ExperimentRecord:
    base = dict(
        id="exp_x",
        user_id="u1",
        name="run",
        status="completed",
        dataset_fingerprint="abc",
        split="test",
        class_names=("fish",),
    )
    base.update(overrides)
    return ExperimentRecord(**base)


def test_comparability_warnings_flag_differences() -> None:
    from app.ui.pages.evaluation import _comparability_warnings

    same = [_record(), _record(id="exp_y", name="run2")]
    assert _comparability_warnings(same) == []

    different = [
        _record(),
        _record(id="exp_y", dataset_fingerprint="zzz", split="val", class_names=("crab",)),
    ]
    warnings = "\n".join(_comparability_warnings(different)).lower()
    assert "different datasets" in warnings
    assert "different splits" in warnings
    assert "class mappings differ" in warnings


# --------------------------------------------------------------------------- #
# CLI reproduction command
# --------------------------------------------------------------------------- #
def test_rerun_command_matches_cli_end_to_end() -> None:
    """The displayed command must use flags the CLI actually accepts."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_evaluation_cli_workspace",
        Path(__file__).resolve().parents[1] / "scripts" / "run_evaluation.py",
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    parser = cli.build_parser()

    from app.ui.pages import evaluation as page

    record = types.SimpleNamespace(
        data_yaml="data/raw/tiny/data.yaml",
        split="test",
        model_keys=("fish_inv", "megafauna"),
        user_id="u1",
        parameters={
            "max_images": 25,
            "iou_threshold_for_figures": 0.75,
            "confidence": 0.4,
            "device": "cpu",
            "plots": True,
        },
    )
    command = page._rerun_command(record)

    assert '--data "data/raw/tiny/data.yaml"' in command
    assert "--split test" in command
    assert "--model fish_inv --model megafauna" in command
    assert "--limit 25" in command
    assert "--iou 0.75" in command
    assert "--confidence 0.4" in command
    assert "--device cpu" in command
    assert "--plots" in command
    assert "--record --owner u1" in command

    # The generated command really parses with the CLI's own argument parser.
    import shlex

    argv = shlex.split(command)
    assert argv[0] == "python"
    args = parser.parse_args(argv[2:])
    assert args.data == "data/raw/tiny/data.yaml"
    assert args.split == "test"
    assert args.model == ["fish_inv", "megafauna"]
    assert args.limit == 25
    assert args.iou == pytest.approx(0.75)
    assert args.confidence == pytest.approx(0.4)
    assert args.device == "cpu"
    assert args.plots is True
    assert args.record is True
    assert args.owner == "u1"


def test_rerun_command_omits_unset_settings() -> None:
    from app.ui.pages import evaluation as page

    record = types.SimpleNamespace(
        data_yaml="d/data.yaml",
        split="val",
        model_keys=(),
        user_id="u2",
        parameters={},
    )
    command = page._rerun_command(record)
    assert "--model all" in command
    assert "--limit" not in command
    assert "--iou" not in command
    assert "--confidence" not in command
    assert "--device" not in command
    assert "--plots" not in command
