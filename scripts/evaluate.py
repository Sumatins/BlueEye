#!/usr/bin/env python
"""Evaluate a YOLOv8 checkpoint on a labelled test/validation split.

Reports precision, recall, mAP@50 and mAP@50:95 for the given dataset.

Example:
    python scripts/evaluate.py \
        --weights models/fish_inv/FishInv.pt \
        --data data/processed/my_dataset/data.yaml \
        --split val

If no labelled dataset is available these metrics cannot be computed -
BlueEye never fabricates evaluation numbers. The upstream marine-detect
README publishes the metrics measured by its authors on their own test
sets; those are *reference* values, not BlueEye measurements.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.utils.file_utils import unique_output_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a marine-life detector (precision/recall/mAP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="Path to the .pt checkpoint.")
    parser.add_argument("--data", required=True, help="Path to the dataset data.yaml.")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="", help="cpu / 0 / '' = auto.")
    parser.add_argument("--save-report", action="store_true", help="Save metrics as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Error: 'ultralytics' is not installed. Run 'pip install -r requirements.txt'.",
            file=sys.stderr,
        )
        return 3

    if not Path(args.weights).is_file():
        print(f"Error: weights not found: {args.weights}", file=sys.stderr)
        return 2
    if not Path(args.data).is_file():
        print(f"Error: dataset config not found: {args.data}", file=sys.stderr)
        return 2

    print(f"Evaluating {args.weights} on split '{args.split}' of {args.data} ...")
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        verbose=True,
    )

    summary = {
        "weights": str(args.weights),
        "dataset": str(args.data),
        "split": args.split,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    print("\nEvaluation metrics:")
    print(f"  Precision     : {summary['precision']:.4f}")
    print(f"  Recall        : {summary['recall']:.4f}")
    print(f"  mAP@50        : {summary['map50']:.4f}")
    print(f"  mAP@50:95     : {summary['map50_95']:.4f}")

    if args.save_report:
        settings = get_settings()
        settings.ensure_dirs()
        out = unique_output_path(settings.reports_output_dir, "evaluation", ".json")
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"  report saved  : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
