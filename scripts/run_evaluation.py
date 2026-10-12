#!/usr/bin/env python
"""BlueEye research evaluation: compare models on a labelled dataset.

This is the reproducible pipeline behind Phase 3. It runs one or more of the
*installed, unmodified* BlueEye detectors over a labelled YOLO dataset split,
checks that the model and the annotations are class-compatible, and writes
precision / recall / mAP@0.5 / mAP@0.5:0.95, per-class metrics, inference
timing and the full software/hardware context.

Nothing is fabricated. If there is no labelled dataset the script prints the
setup guide and exits with code 3 (the experiment is *pending*); it never
invents a score. The pretrained weights are only ever read.

Examples::

    # List the models and what they can detect
    python scripts/run_evaluation.py --list-models

    # Compare the two core models on your own prepared test split
    python scripts/run_evaluation.py \
        --data data/raw/my_dataset/data.yaml --split test --model all

    # A quick smoke run over the first 25 images, with charts
    python scripts/run_evaluation.py --data data/raw/my_dataset/data.yaml \
        --model fish_inv --limit 25 --plots

    # Also persist the run so it appears in the Research Evaluation Center
    python scripts/run_evaluation.py --data data/raw/my_dataset/data.yaml \
        --model all --record --owner <account-id>

Exit codes: 0 success, 1 some models failed (partial), 2 invalid input
(missing/unreadable config), 3 pending (no dataset / no model evaluated).

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.evaluation import (  # noqa: E402
    DatasetError,
    EvaluationStore,
    InferenceError,
    collect_environment,
    detector_predictor,
    file_fingerprint,
    limit_split,
    load_split,
    model_class_names,
    run_evaluation,
    summarise,
    validate_dataset,
    write_report,
)
from app.evaluation.environment import evaluator_version, git_commit  # noqa: E402

SETUP_GUIDE = """\
No labelled dataset was provided, so no metrics can be computed.

BlueEye does not ship datasets and never invents evaluation numbers. To run a
real comparison:

  1. Obtain a licensed dataset with bounding-box labels in YOLO format -
     see docs/evaluation.md for documented sources and their licences.
  2. Prepare it with:
         python scripts/prepare_dataset.py <path-to-export> --name my_dataset
  3. Evaluate:
         python scripts/run_evaluation.py --data data/raw/my_dataset/data.yaml \\
             --split test --model all

The experiment above is marked PENDING until real labelled data is available.

Available datasets (documented, not bundled): docs/evaluation.md
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible BlueEye model comparison on a labelled dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to a dataset data.yaml (or a directory containing one).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Detector selection (auto/all/<key>). Repeatable. Default: all.",
    )
    parser.add_argument(
        "--out", default=None, help="Output directory (default: outputs/evaluation/<ts>)."
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Evaluate only the first N images (0 = all).",
    )
    parser.add_argument("--device", default=None, help="cpu / 0 / cuda (overrides config).")
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override the per-model confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help=(
            "IoU threshold for the PR / F1 / confusion figures. The reported "
            "mAP@0.5 and mAP@0.5:0.95 always use the standard COCO thresholds."
        ),
    )
    parser.add_argument("--plots", action="store_true", help="Write matplotlib charts.")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Persist the run in the evaluation database (Research Evaluation Center).",
    )
    parser.add_argument(
        "--owner",
        default="cli",
        help="Owner id recorded for --record (default is the CLI sentinel).",
    )
    parser.add_argument("--name", default=None, help="Experiment name (default: timestamp).")
    parser.add_argument("--notes", default="", help="Optional experiment notes.")
    parser.add_argument(
        "--list-models", action="store_true", help="List models and their classes, then exit."
    )
    return parser


def _resolve_data_yaml(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        for name in ("data.yaml", "dataset.yaml", "data.yml"):
            candidate = path / name
            if candidate.is_file():
                return candidate
        raise DatasetError(f"No data.yaml found inside {path}")
    return path


def _list_models(detector) -> int:
    print("Installed BlueEye models and their classes:")
    for entry in detector.model_status():
        status = "ready" if entry.get("available") else "not installed"
        classes = entry.get("classes") or []
        print(f"\n- {entry.get('key')}  [{status}]  {entry.get('name', '')}")
        print(f"    classes: {', '.join(classes) if classes else '(none declared)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.device:
        os.environ["DEVICE"] = args.device

    settings = get_settings()
    settings.ensure_dirs()

    from app.detection.detector import MarineDetector

    detector = MarineDetector(settings=settings)

    if args.list_models:
        return _list_models(detector)

    if not args.data:
        print(SETUP_GUIDE)
        return 3

    try:
        data_yaml = _resolve_data_yaml(args.data)
    except DatasetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not data_yaml.is_file():
        print(f"Error: dataset config not found: {data_yaml}", file=sys.stderr)
        return 2

    # Validate the dataset *before* running: the CLI refuses to score a dataset
    # it cannot honestly read, and prints the errors that block readiness.
    try:
        validation = validate_dataset(data_yaml, split=args.split)
    except Exception as exc:  # noqa: BLE001 - unreadable config is reported
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    for issue in validation.errors:
        location = f" ({issue.path})" if issue.path else ""
        print(f"  error: [{issue.code}] {issue.message}{location}", file=sys.stderr)
    for issue in validation.warnings:
        location = f" ({issue.path})" if issue.path else ""
        print(f"  warning: [{issue.code}] {issue.message}{location}", file=sys.stderr)
    if not validation.ready:
        print(
            f"Error: dataset '{validation.name}' is not ready for the "
            f"'{args.split}' split. Resolve the errors above.",
            file=sys.stderr,
        )
        return 2

    try:
        dataset = load_split(data_yaml, split=args.split)
    except DatasetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("\n" + SETUP_GUIDE)
        return 3

    dataset = limit_split(dataset, args.limit)
    print(
        f"Dataset '{dataset.name}' split '{dataset.split}': "
        f"{dataset.num_images} image(s), {dataset.num_objects} labelled object(s)."
    )
    print(f"Dataset classes: {', '.join(dataset.class_names) or '(unknown)'}")
    for note in dataset.notes:
        print(f"  note: {note}")

    selections = args.model or ["all"]
    evaluations = []
    resolved_keys: list[str] = []
    failures = 0
    for selection in selections:
        try:
            keys = detector.resolve_keys(selection)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"- model '{selection}': skipped ({exc})")
            failures += 1
            continue
        # Each model is evaluated *independently* on the same split; scores are
        # never merged across models.
        for key in keys:
            print(f"\nRunning {key} on {dataset.num_images} image(s) ...")
            try:
                classes = model_class_names(detector, key)
                predictor = detector_predictor(
                    detector, key, confidence=args.confidence
                )
                evaluation = run_evaluation(key, predictor, dataset, classes)
            except (InferenceError, Exception) as exc:  # noqa: BLE001 - continue
                print(f"  skipped '{key}': {exc}", file=sys.stderr)
                failures += 1
                continue
            headline = summarise(evaluation)
            print(
                f"  mAP@0.5={headline['mAP50']:.4f}  "
                f"mAP@0.5:0.95={headline['mAP50_95']:.4f}  "
                f"P={headline['precision']:.4f} R={headline['recall']:.4f} "
                f"({headline['seconds_per_image']:.3f}s/image)"
            )
            for note in evaluation.notes:
                print(f"  note: {note}")
            evaluations.append(evaluation)
            resolved_keys.append(key)

    if not evaluations:
        print("No model could be evaluated. Nothing was written.", file=sys.stderr)
        return 3

    output_dir = (
        Path(args.out)
        if args.out
        else settings.outputs_dir / "evaluation" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    environment = collect_environment()
    config = {
        "data_yaml": str(data_yaml),
        "split": dataset.split,
        "limit": args.limit,
        "selections": selections,
        "models": resolved_keys,
        "confidence_override": args.confidence,
        "iou_threshold_for_figures": args.iou,
        "device": detector.device,
        "note": (
            "Input resolution and preprocessing are Ultralytics' defaults used "
            "by the application itself; the detector pipeline is unchanged."
        ),
    }

    # Charts first so the HTML report can embed them.
    figure_pairs: list[tuple[str, Path]] = []
    if args.plots:
        from app.evaluation import plots as plots_module

        for evaluation in evaluations:
            for path in plots_module.plot_evaluation_figures(
                evaluation, output_dir, iou_threshold=args.iou
            ):
                figure_pairs.append((path.stem.replace("_", " "), path))
            for path in plots_module.plot_examples(
                evaluation, dataset, output_dir / "examples", limit=4
            ):
                figure_pairs.append((f"examples/{path.stem}", path))
        if len(evaluations) > 1:
            chart = plots_module.plot_metric_comparison(
                evaluations, output_dir / "metric_comparison.png"
            )
            if chart is not None:
                figure_pairs.append(("model comparison", chart))

    experiment_payload = None
    store = None
    experiment_id = None
    owner = args.owner
    dataset_fingerprint = validation.fingerprint
    if args.record:
        store = EvaluationStore(settings.evaluation_db_path)
        dataset_record = store.register_dataset(
            owner,
            dataset.name,
            data_yaml,
            class_names=validation.class_names,
            splits=validation.splits_with_images,
            fingerprint=dataset_fingerprint,
            validation=validation.as_dict(),
        )
        fingerprints: dict = {}
        for key in resolved_keys:
            try:
                fingerprints[key] = file_fingerprint(
                    detector.model_manager.resolve_path(key)
                )
            except Exception:  # noqa: BLE001 - provenance is best-effort
                continue
        exp_name = args.name or f"CLI evaluation {dataset.name} {dataset.split}"
        record = store.create_experiment(
            owner,
            exp_name,
            notes=args.notes,
            purpose="Command-line evaluation (scripts/run_evaluation.py).",
            dataset_id=dataset_record.id,
            dataset_name=dataset.name,
            data_yaml=str(data_yaml),
            split=dataset.split,
            class_names=dataset.class_names,
            dataset_fingerprint=dataset_fingerprint,
            model_keys=tuple(resolved_keys),
            model_fingerprints=fingerprints,
            evaluator_version=evaluator_version(),
            parameters=config,
            environment=environment,
            git_commit=git_commit(),
            status="running",
        )
        experiment_id = record.id
        experiment_payload = record.as_dict()

    written = write_report(
        output_dir,
        evaluations,
        environment=environment,
        config=config,
        plots=figure_pairs,
        experiment=experiment_payload,
    )

    if store is not None and experiment_id is not None:
        for evaluation in evaluations:
            store.add_model_result(experiment_id, owner, evaluation)
        final = "partially_completed" if failures else "completed"
        store.set_status(experiment_id, owner, final)
        for path in [*written.values(), *(item for _, item in figure_pairs)]:
            try:
                store.add_artifact(
                    experiment_id,
                    owner,
                    path,
                    name=path.name,
                    role="plot" if path.suffix.lower() == ".png" else "report",
                )
            except OSError:
                continue
        print(f"\nRecorded experiment {experiment_id} (owner: {owner}).")

    print(f"\nWrote report to: {output_dir}")
    for key, path in written.items():
        print(f"  {key}: {path}")
    for caption, path in figure_pairs:
        print(f"  figure ({caption}): {path}")

    if failures:
        print(
            f"\nWarning: {failures} selection(s) could not be evaluated; the "
            "results above are a partial success.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
