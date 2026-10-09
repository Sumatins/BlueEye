#!/usr/bin/env python
"""Train / fine-tune a YOLOv8 marine-life detector on your own dataset.

BlueEye works out of the box with the pretrained marine-detect weights;
training is optional and only needed when you add new data or species.

Dataset format (Ultralytics YOLO):
    dataset/
      ├── data.yaml
      ├── images/train/  ├── images/val/  (├── images/test/)
      └── labels/train/  ├── labels/val/  (├── labels/test/)

Example:
    python scripts/train.py \
        --data data/processed/my_dataset/data.yaml \
        --model yolov8n.pt \
        --epochs 50 --imgsz 640 --batch 16 --device cpu

Metrics (precision, recall, mAP@50, mAP@50:95) are printed by Ultralytics
and stored under runs/detect/<name>/.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 model for marine-life detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="Path to the dataset data.yaml.")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model: a .pt checkpoint (e.g. our fish_inv weights) or an "
        "Ultralytics model name (yolov8n.pt, ...).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 = auto).")
    parser.add_argument(
        "--device",
        default="",
        help="cpu, 0, 0,1, ... (empty = auto: CUDA when available).",
    )
    parser.add_argument("--project", default="runs/detect", help="Output project directory.")
    parser.add_argument("--name", default="blueeye", help="Run name.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    parser.add_argument("--workers", type=int, default=4, help="Data-loader workers.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Error: 'ultralytics' is not installed. "
            "Run 'pip install -r requirements.txt'.",
            file=sys.stderr,
        )
        return 3

    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"Error: dataset config not found: {data_path}", file=sys.stderr)
        return 2

    print(f"Training {args.model} on {data_path} for {args.epochs} epochs...")
    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
    )
    print("\nTraining finished. Results:")
    print(f"  runs directory: {Path(results.save_dir)}")
    print("  metrics (precision, recall, mAP@50, mAP@50:95) are in results.csv")
    print("  and best.pt / last.pt weights were saved in the same directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
