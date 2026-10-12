"""BlueEye reproducible evaluation pipeline.

This sub-package implements the *research* side of BlueEye: a configurable,
dependency-light way to run the existing (unmodified) detectors over a
labelled dataset, check that the model and the dataset annotations are
compatible, and compute standard object-detection metrics
(precision, recall, mAP@0.5, mAP@0.5:0.95) together with inference timing
and provenance.

Nothing here invents numbers: when a checkpoint or a labelled dataset is
missing, the experiment is reported as *pending* and the result files say
so. The pretrained weights are only ever read.

Modules
-------
``metrics``
    IoU matching, per-class AP, mAP@0.5 / mAP@0.5:0.95.
``datasets``
    YOLO dataset discovery, ``data.yaml`` parsing and ground-truth loading.
``compat``
    Dataset <-> model class compatibility (never a silent remap).
``environment``
    Hardware / software provenance for the "test conditions" column.
``runner``
    Orchestration: run a model over a split with an injectable predictor.
``report``
    JSON / CSV / Markdown writers.
``plots``
    Optional matplotlib figures (metric comparison, per-class).

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from app.evaluation.compat import ClassCompat, compare_classes
from app.evaluation.datasets import (
    DatasetError,
    DatasetSplit,
    class_names_from_yaml,
    limit_split,
    load_data_yaml,
    load_split,
    parse_label_file,
    read_yaml,
)
from app.evaluation.environment import collect_environment
from app.evaluation.metrics import (
    IOU_THRESHOLDS,
    ClassMetrics,
    DetectionBox,
    EvaluationResult,
    box_iou,
    evaluate,
)
from app.evaluation.report import render_summary, write_report
from app.evaluation.runner import (
    InferenceError,
    ModelEvaluation,
    detector_predictor,
    model_class_names,
    run_evaluation,
    run_many,
    summarise,
)

__all__ = [
    "IOU_THRESHOLDS",
    "ClassCompat",
    "ClassMetrics",
    "DatasetError",
    "DatasetSplit",
    "DetectionBox",
    "EvaluationResult",
    "InferenceError",
    "ModelEvaluation",
    "box_iou",
    "class_names_from_yaml",
    "collect_environment",
    "compare_classes",
    "detector_predictor",
    "evaluate",
    "limit_split",
    "load_data_yaml",
    "load_split",
    "model_class_names",
    "parse_label_file",
    "read_yaml",
    "render_summary",
    "run_evaluation",
    "run_many",
    "summarise",
    "write_report",
]
