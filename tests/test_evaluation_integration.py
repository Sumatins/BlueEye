"""Integration test: the evaluation pipeline with real weights.

Runs only when the official marine-detect weights are present; otherwise it is
skipped - weights are never fabricated for testing. CPU execution is used, no
GPU required.

The sample image is the upstream parrotfish photograph. We deliberately do
**not** invent annotations for it, so the ground truth is empty and the metric
is honestly reported as 0. This test verifies the *pipeline* (the real detector
is executed, every detection is mapped into the dataset's class space, and the
report files are written) - not an accuracy number.

Run explicitly with::

    pytest -m integration

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import pytest

from app.config import PROJECT_ROOT, get_settings
from app.detection.detector import MarineDetector
from app.detection.model_manager import ModelManager
from app.evaluation import (
    collect_environment,
    detector_predictor,
    load_split,
    model_class_names,
    run_evaluation,
    write_report,
)

SAMPLE_IMAGE = PROJECT_ROOT / "assets" / "images" / "input_folder" / "regq.jpg"


def _core_weights_present() -> bool:
    if not SAMPLE_IMAGE.is_file():
        return False
    manager = ModelManager()
    return all(manager.is_available(key) for key in ("fish_inv", "megafauna"))


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _core_weights_present(),
        reason="core weights / sample image unavailable (run scripts/download_models.py)",
    ),
]


def test_evaluation_pipeline_with_real_weights(tmp_path: Path) -> None:
    """The real detector flows through the pipeline and the reports are written.

    Ground truth is intentionally empty (no fabricated annotations), so the
    metric is 0 — the assertion is about the *plumbing*, not accuracy.
    """
    # A one-image dataset built from the real sample. The label file is empty:
    # BlueEye never invents annotations, so AP is honestly undefined/0.
    root = tmp_path / "ds"
    (root / "images" / "test").mkdir(parents=True)
    (root / "labels" / "test").mkdir(parents=True)
    shutil.copyfile(SAMPLE_IMAGE, root / "images" / "test" / SAMPLE_IMAGE.name)
    (root / "labels" / "test" / f"{SAMPLE_IMAGE.stem}.txt").write_text("", encoding="utf-8")

    settings = get_settings()
    detector = MarineDetector(settings=settings)
    classes = model_class_names(detector, "auto")
    assert classes, "the auto selection should expose class names"

    lines = ["path: .", "test: images/test", "names:"]
    lines += [f"  {index}: {name}" for index, name in enumerate(classes)]
    data_yaml = root / "data.yaml"
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dataset = load_split(data_yaml, split="test")
    assert dataset.num_images == 1
    assert dataset.num_objects == 0  # no annotations were invented

    # Reference count: run the detector directly on the same image.
    image = cv2.imread(str(root / "images" / "test" / SAMPLE_IMAGE.name))
    assert image is not None
    reference = detector.predict_image(image, model_selection="auto")
    expected = len(reference.detections)
    assert expected >= 1  # the upstream parrotfish sample contains a scaridae fish

    evaluation = run_evaluation(
        "auto", detector_predictor(detector, "auto"), dataset, classes
    )
    assert evaluation.status == "complete"  # classes are shared
    assert evaluation.images_evaluated == 1
    assert evaluation.inference_seconds > 0
    assert evaluation.num_predictions == expected  # every detection was mapped
    assert evaluation.metrics.map50 == 0.0  # no ground truth -> honestly 0
    assert any("ground-truth" in note.lower() for note in evaluation.notes)

    written = write_report(
        tmp_path / "reports",
        [evaluation],
        environment=collect_environment(),
        config={"split": "test"},
    )
    payload = json.loads(written["results_json"].read_text(encoding="utf-8"))
    assert payload["evaluations"][0]["timing"]["predictions"] == expected
    assert written["summary_md"].read_text(encoding="utf-8")
