"""Tests for the reproducible evaluation pipeline (``app.evaluation``).

These tests never touch the network and never load real weights: the metric
maths, the dataset reader, the class-compatibility check and the report
writers are all exercised with synthetic data and a stub predictor.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from app.evaluation import (
    DatasetError,
    DetectionBox,
    box_iou,
    compare_classes,
    evaluate,
    limit_split,
    load_split,
    parse_label_file,
    render_summary,
    run_evaluation,
    write_report,
)
from app.evaluation.datasets import class_names_from_yaml, read_yaml


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _box(class_id: int, x1: float, y1: float, x2: float, y2: float, score: float = 1.0):
    return DetectionBox(class_id, x1, y1, x2, y2, score)


def test_box_iou_identical_and_disjoint():
    a = _box(0, 0, 0, 10, 10)
    assert box_iou(a, _box(0, 0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou(a, _box(0, 20, 20, 30, 30)) == 0.0


def test_box_iou_partial_overlap():
    a = _box(0, 0, 0, 10, 10)
    b = _box(0, 5, 0, 15, 10)  # half the area overlaps
    assert box_iou(a, b) == pytest.approx(1 / 3)


def test_perfect_predictions_score_one():
    gt = {"a.jpg": [_box(0, 0, 0, 10, 10), _box(1, 20, 20, 40, 40)]}
    pred = {"a.jpg": [_box(0, 0, 0, 10, 10, 0.9), _box(1, 20, 20, 40, 40, 0.8)]}
    result = evaluate(gt, pred, class_names={0: "fish", 1: "shark"})
    assert result.map50 == pytest.approx(1.0)
    assert result.map50_95 == pytest.approx(1.0)
    assert [c.name for c in result.classes] == ["fish", "shark"]


def test_no_predictions_is_zero_with_note():
    gt = {"a.jpg": [_box(0, 0, 0, 10, 10)]}
    result = evaluate(gt, {"a.jpg": []}, class_names={0: "fish"})
    assert result.map50 == 0.0
    assert any("no predictions" in note.lower() for note in result.notes)


def test_no_ground_truth_is_zero_with_note():
    result = evaluate({"a.jpg": []}, {"a.jpg": [_box(0, 0, 0, 10, 10)]})
    assert result.map50 == 0.0
    assert any("ground-truth" in note.lower() for note in result.notes)


def test_false_positive_lowers_precision():
    gt = {"a.jpg": [_box(0, 0, 0, 10, 10)]}
    pred = {
        "a.jpg": [
            _box(0, 0, 0, 10, 10, 0.9),
            _box(0, 50, 50, 60, 60, 0.8),  # a false positive
        ]
    }
    result = evaluate(gt, pred, class_names={0: "fish"})
    assert result.map50 == pytest.approx(1.0)  # true positive is still found first
    fish = result.classes[0]
    assert fish.num_pred == 2  # the false positive is still counted
    assert fish.num_gt == 1


# --------------------------------------------------------------------------- #
# Class compatibility
# --------------------------------------------------------------------------- #
def test_compare_classes_matches_and_reports_gaps():
    compat = compare_classes(["Fish", "shark", "turtle"], ["fish", "shark", "ray"])
    assert compat.comparable
    assert compat.supported == ("Fish", "shark")
    assert compat.unsupported == ("turtle",)
    assert compat.extra_model == ("ray",)


def test_compare_classes_refuses_empty_overlap():
    compat = compare_classes(["dolphin"], ["fish", "shark"])
    assert not compat.comparable
    assert "not comparable" in compat.summary.lower()


# --------------------------------------------------------------------------- #
# Dataset reading
# --------------------------------------------------------------------------- #
def _make_dataset(root: Path, names=("fish", "shark")) -> Path:
    for part in ("images/test", "labels/test"):
        (root / part).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), "blue").save(root / "images" / "test" / "a.jpg")
    Image.new("RGB", (200, 100), "green").save(root / "images" / "test" / "b.jpg")
    # a.jpg: one fish centred, one shark to the right
    (root / "labels" / "test" / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.8 0.5 0.1 0.1\n", encoding="utf-8"
    )
    # b.jpg: one fish
    (root / "labels" / "test" / "b.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    lines = ["path: .", "train: images/test", "val: images/test", "test: images/test", "names:"]
    for index, name in enumerate(names):
        lines.append(f"  {index}: {name}")
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def test_parse_label_file_converts_to_pixels(tmp_path):
    label = tmp_path / "l.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    boxes = parse_label_file(label, 100, 200)
    assert len(boxes) == 1
    box = boxes[0]
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((40, 80, 60, 120))


def test_load_split_reads_images_and_ground_truth(tmp_path):
    data_yaml = _make_dataset(tmp_path / "ds")
    dataset = load_split(data_yaml, split="test")
    assert dataset.num_images == 2
    assert dataset.num_objects == 3
    assert dataset.class_names == ("fish", "shark")
    assert dataset.class_counts() == {"fish": 2, "shark": 1}


def test_load_split_refuses_missing_split(tmp_path):
    root = tmp_path / "mini"
    (root / "images" / "test").mkdir(parents=True)
    (root / "labels" / "test").mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(root / "images" / "test" / "a.jpg")
    (root / "labels" / "test" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text("path: .\ntest: images/test\nnames:\n  0: fish\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_split(data_yaml, split="train")


def test_limit_split_restricts_images(tmp_path):
    dataset = load_split(_make_dataset(tmp_path / "ds"), split="test")
    limited = limit_split(dataset, 1)
    assert limited.num_images == 1
    assert "Limited" in limited.notes[-1]


def test_class_names_and_yaml_helpers(tmp_path):
    data_yaml = _make_dataset(tmp_path / "ds")
    data = read_yaml(data_yaml)
    assert class_names_from_yaml(data) == ("fish", "shark")
    assert class_names_from_yaml({"names": {1: "b", 0: "a"}}) == ("a", "b")


# --------------------------------------------------------------------------- #
# Full evaluation + reporting (stub predictor, no weights)
# --------------------------------------------------------------------------- #
def _perfect_predictor(dataset):
    def _predict(image_path):
        boxes = dataset.ground_truth[image_path.name]
        return [
            (dataset.class_names[box.class_id], 0.9, box.as_xyxy())
            for box in boxes
        ]

    return _predict


def test_run_evaluation_with_stub_predictor(tmp_path):
    dataset = load_split(_make_dataset(tmp_path / "ds"), split="test")
    evaluation = run_evaluation(
        "stub", _perfect_predictor(dataset), dataset, dataset.class_names
    )
    assert evaluation.status == "complete"
    assert evaluation.metrics.map50 == pytest.approx(1.0)
    assert evaluation.images_evaluated == 2
    assert evaluation.num_predictions == 3


def test_run_evaluation_reports_genuine_progress(tmp_path):
    dataset = load_split(_make_dataset(tmp_path / "ds"), split="test")
    events: list[tuple[str, int, int]] = []
    evaluation = run_evaluation(
        "stub",
        _perfect_predictor(dataset),
        dataset,
        dataset.class_names,
        on_progress=lambda stage, done, total: events.append((stage, done, total)),
    )
    assert evaluation.images_evaluated == 2

    inference = [event for event in events if event[0] == "inference"]
    # Starts at zero and advances one *real* image at a time to the real total.
    assert inference[0] == ("inference", 0, 2)
    assert inference[-1] == ("inference", 2, 2)
    assert [done for _, done, _ in inference] == [0, 1, 2]
    assert all(total == 2 for _, _, total in inference)
    # The last notification marks the real metric phase, still with real counts.
    assert events[-1] == ("metric calculation", 2, 2)


def test_run_evaluation_marks_not_comparable(tmp_path):
    dataset = load_split(_make_dataset(tmp_path / "ds"), split="test")
    evaluation = run_evaluation(
        "wrong",
        lambda path: [("dolphin", 0.9, (0, 0, 5, 5))],
        dataset,
        ["dolphin"],
    )
    assert evaluation.status == "not_comparable"
    assert any("not a valid comparison" in note for note in evaluation.notes)


def test_write_report_creates_all_files(tmp_path):
    dataset = load_split(_make_dataset(tmp_path / "ds"), split="test")
    evaluation = run_evaluation(
        "stub", _perfect_predictor(dataset), dataset, dataset.class_names
    )
    out_dir = tmp_path / "out"
    written = write_report(
        out_dir,
        [evaluation],
        environment={"platform": "test", "packages": {"numpy": "x"}},
        config={"split": "test"},
    )
    for key in ("results_json", "metrics_csv", "per_class_csv", "summary_md"):
        assert written[key].is_file()

    payload = json.loads(written["results_json"].read_text(encoding="utf-8"))
    assert payload["evaluations"][0]["metrics"]["mAP50"] == pytest.approx(1.0)

    with written["metrics_csv"].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["model"] == "stub"
    assert rows[0]["status"] == "complete"

    summary = written["summary_md"].read_text(encoding="utf-8")
    assert "reproduced locally" in summary
    assert "stub" in summary


def test_render_summary_lists_unsupported_classes(tmp_path):
    dataset = load_split(_make_dataset(tmp_path / "ds"), split="test")
    evaluation = run_evaluation(
        "fish_only",
        lambda path: [("fish", 0.9, (0, 0, 5, 5))],
        dataset,
        ["fish"],
    )
    text = render_summary([evaluation])
    assert "shark" in text
    assert "cannot detect" in text


def test_match_stats_counts_tp_miss_and_false_positive():
    from app.evaluation.plots import _match_stats

    ground_truth = [_box(0, 0, 0, 10, 10), _box(0, 50, 50, 60, 60)]
    predictions = [_box(0, 0, 0, 10, 10, 0.9), _box(0, 100, 100, 110, 110, 0.8)]
    assert _match_stats(ground_truth, predictions) == (1, 1, 1)


def test_plot_examples_writes_figures(tmp_path):
    pytest.importorskip("matplotlib")
    from app.evaluation.plots import plot_examples

    dataset = load_split(_make_dataset(tmp_path / "ds"), split="test")
    evaluation = run_evaluation(
        "stub", _perfect_predictor(dataset), dataset, dataset.class_names
    )
    written = plot_examples(evaluation, dataset, tmp_path / "figs", limit=2)
    assert written and all(path.is_file() for path in written)


# --------------------------------------------------------------------------- #
# CLI (scripts/run_evaluation.py) — the honest "pending" behaviour
# --------------------------------------------------------------------------- #
def _load_cli():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_evaluation_cli",
        Path(__file__).resolve().parents[1] / "scripts" / "run_evaluation.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_is_pending_without_a_dataset(capsys):
    cli = _load_cli()
    assert cli.main([]) == 3  # pending, not an error
    output = capsys.readouterr().out
    assert "No labelled dataset" in output
    assert "PENDING" in output


def test_cli_rejects_a_missing_config(tmp_path):
    cli = _load_cli()
    assert cli.main(["--data", str(tmp_path / "nope.yaml")]) == 2


def test_resolve_data_yaml_finds_config_in_directory(tmp_path):
    cli = _load_cli()
    (tmp_path / "data.yaml").write_text("names: [fish]\n", encoding="utf-8")
    resolved = cli._resolve_data_yaml(str(tmp_path))
    assert resolved == tmp_path / "data.yaml"


def test_resolve_data_yaml_dir_without_config_raises(tmp_path):
    cli = _load_cli()
    (tmp_path / "ds").mkdir()
    with pytest.raises(DatasetError):
        cli._resolve_data_yaml(str(tmp_path / "ds"))
