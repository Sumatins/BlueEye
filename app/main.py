"""BlueEye command line interface.

Examples
--------
Detect marine life in an image (uses each model's recommended threshold)::

    python -m app.main --image sample.jpg

Detect in a video with a custom confidence threshold and enhancement::

    python -m app.main --video dive.mp4 --confidence 0.5 --enhance

Model management::

    python -m app.main --list-models
    python -m app.main --download-models

Exit codes: 0 success, 2 input/validation error, 3 model error,
4 processing error, 1 unexpected error.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from app import __version__
from app.config import get_settings, setup_logging
from app.detection.detector import MarineDetector
from app.detection.model_manager import ModelError, download_progress_printer
from app.processing.enhancement import EnhancementConfig
from app.processing.image_processor import ImageProcessor, ProcessingError
from app.utils.file_utils import ensure_output_dirs, human_size
from app.utils.validation import ValidationError

logger = logging.getLogger("blueeye")

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_VALIDATION = 2
EXIT_MODEL = 3
EXIT_PROCESSING = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description=(
            "BlueEye - AI-powered marine life detection for underwater "
            "images and videos (YOLOv8 / PyTorch / OpenCV)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, help="Path to an input image (jpg/png/webp).")
    parser.add_argument("--video", type=str, help="Path to an input video (mp4/avi/mov/mkv).")
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence threshold in [0, 1]. Default: each model's "
        "recommended value (FishInv 0.523 / MegaFauna 0.546).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model selection: 'auto' (default) runs the core models "
        "(fish_inv + megafauna); 'all' also runs the additional installed "
        "models. You can also pass the id of any registered model (e.g. "
        "fish_inv, megafauna, aquatic_brackish). Run --list-models to see "
        "every id and its status.",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Apply the optional underwater image enhancement before detection.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for annotated results (default: outputs/images, outputs/videos).",
    )
    parser.add_argument("--no-json", action="store_true", help="Do not write a JSON report.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: auto, cpu, cuda (auto uses CUDA when available).",
    )
    parser.add_argument("--list-models", action="store_true", help="Show registered models.")
    parser.add_argument(
        "--verify-models",
        action="store_true",
        help="Load every model and run a real inference test, then report status.",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download any missing model weights and exit.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    parser.add_argument("--version", action="version", version=f"BlueEye {__version__}")
    return parser


def _print_detections(detections) -> None:
    if not detections:
        print("  (no detections above the confidence threshold)")
        return
    for rank, det in enumerate(detections, start=1):
        print(
            f"  {rank}. {det.display_name:<34} {det.confidence_percent:>7}"
            f"   bbox={list(det.bbox)}   model={det.model}"
        )


def _print_stats(stats: dict) -> None:
    print("\nDetection summary")
    print(f"  Total objects      : {stats.get('total_detections', 0)}")
    print(f"  Unique species     : {stats.get('unique_classes', 0)}")
    per_class = stats.get("detections_per_class", {}) or {}
    for name, count in per_class.items():
        print(f"    - {name}: {count}")
    avg = float(stats.get("average_confidence", 0.0)) * 100
    maximum = float(stats.get("max_confidence", 0.0)) * 100
    print(f"  Average confidence : {avg:.1f}%")
    print(f"  Max confidence     : {maximum:.1f}%")


def _cmd_list_models(detector: MarineDetector) -> int:
    labels = {
        "ready": "READY",
        "not_installed": "NOT INSTALLED",
        "needs_training": "NEEDS TRAINING",
        "incompatible": "INCOMPATIBLE",
    }
    print("Registered models:\n")
    for entry in detector.model_status():
        status = labels.get(str(entry.get("status")), "?")
        print(f"  [{entry['key']}] {entry['name']} - {entry['description']}")
        print(f"      status      : {status}")
        print(f"      architecture: {entry.get('architecture')}")
        print(f"      habitat     : {entry.get('habitat') or '-'}")
        print(f"      path        : {entry['path']}")
        print(f"      recommended : {entry['recommended_confidence']}")
        if entry.get("license"):
            print(f"      license     : {entry['license']}")
        if entry.get("classes"):
            print(f"      classes     : {', '.join(entry['classes'])}")
        print()
    return EXIT_OK


def _cmd_verify_models(detector: MarineDetector) -> int:
    """Run a real load + inference smoke test for every registered model."""
    labels = {
        "ready": "READY",
        "not_installed": "NOT INSTALLED",
        "needs_training": "NEEDS TRAINING",
        "incompatible": "INCOMPATIBLE",
    }
    print("Verifying each model with a real inference test...\n")
    for entry in detector.model_status():
        key = entry["key"]
        result = detector.verify_model(key)
        status = labels.get(str(result.get("status")), "?")
        detail = f" - {result['error']}" if result.get("error") else ""
        classes = ", ".join(result.get("classes") or entry.get("classes") or [])
        print(f"  [{key}] {entry['name']}: {status}{detail}")
        if classes and result.get("status") == "ready":
            print(f"      classes: {classes}")
    print()
    return EXIT_OK


def _cmd_download_models(detector: MarineDetector) -> int:
    manager = detector.model_manager
    missing = manager.missing_models()
    if not missing:
        print("All model weights are already downloaded.")
        return EXIT_OK
    for spec in missing:
        print(f"Downloading {spec.display_name} ...")
        path = manager.download(spec.key, progress=download_progress_printer(spec.key))
        print(f"  saved to {path}\n")
    print("Done.")
    return EXIT_OK


def _run_image(args: argparse.Namespace, detector: MarineDetector) -> int:
    settings = ensure_output_dirs(get_settings())
    processor = ImageProcessor(detector, settings=settings)
    result = processor.process(
        input_path=args.image,
        model_selection=args.model,
        confidence=args.confidence,
        enhancement=EnhancementConfig(enabled=True) if args.enhance else None,
        output_dir=args.output_dir,
        save_json=not args.no_json,
    )

    print(f"\nInput     : {result.input_path}")
    print(f"Models    : {', '.join(result.detection.models_used)}")
    threshold = result.detection.confidence_threshold
    if threshold is None:
        applied = ", ".join(
            f"{key}={value:g}" for key, value in result.detection.thresholds.items()
        )
        print(f"Threshold : recommended ({applied})")
    else:
        print(f"Threshold : {threshold:g}")
    print(f"Enhanced  : {'yes' if result.enhanced else 'no'}")
    print(f"Time      : {result.elapsed_seconds:.2f} s")
    print("\nDetected marine life:")
    _print_detections(result.detection.sorted_detections())
    _print_stats(result.stats)
    print(f"\nAnnotated image : {result.output_path}")
    if result.report_path:
        print(f"JSON report     : {result.report_path}")
    return EXIT_OK


def _run_video(args: argparse.Namespace, detector: MarineDetector) -> int:
    settings = ensure_output_dirs(get_settings())

    def _progress(done: int, total: Optional[int]) -> None:
        if total:
            percent = min(done * 100 / total, 100.0)
            sys.stdout.write(f"\r  Processing frames: {done}/{total} ({percent:.0f}%)")
        else:
            sys.stdout.write(f"\r  Processing frames: {done}")
        sys.stdout.flush()
        if total and done >= total:
            sys.stdout.write("\n")

    output_path = None
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        from app.utils.file_utils import unique_output_path

        output_path = unique_output_path(out_dir, "detection", ".mp4")

    result = detector.predict_video(
        input_path=args.video,
        output_path=output_path,
        model_selection=args.model,
        confidence=args.confidence,
        enhancement=EnhancementConfig(enabled=True) if args.enhance else None,
        progress_callback=_progress,
        save_json=not args.no_json,
    )
    sys.stdout.write("\n")

    print(f"\nInput     : {result.info.path}")
    print(
        f"Video     : {result.info.width}x{result.info.height}, "
        f"{result.info.fps:.2f} fps, {result.info.frame_count or '?'} frames, "
        f"{human_size(result.info.file_size_bytes)}"
    )
    print(f"Models    : {', '.join(result.models_used)}")
    threshold = "recommended" if args.confidence is None else f"{args.confidence:g}"
    print(f"Threshold : {threshold}")
    print(f"Enhanced  : {'yes' if result.enhanced else 'no'}")
    print(f"Time      : {result.elapsed_seconds:.2f} s ({result.processing_fps:.1f} fps)")

    print("\nDetected marine life (aggregated over all frames):")
    _print_stats(result.statistics)
    print(f"\nAnnotated video : {result.output_path}")
    if result.report_path:
        print(f"JSON report     : {result.report_path}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging("DEBUG" if args.verbose else get_settings().log_level)

    if args.confidence is not None and not 0.0 <= args.confidence <= 1.0:
        print("Error: --confidence must be between 0.0 and 1.0.", file=sys.stderr)
        return EXIT_VALIDATION

    if args.device:
        get_settings().default_device = args.device

    detector = MarineDetector()

    try:
        if args.list_models:
            return _cmd_list_models(detector)
        if args.verify_models:
            return _cmd_verify_models(detector)
        if args.download_models:
            return _cmd_download_models(detector)
        if args.image and args.video:
            print("Error: use either --image or --video, not both.", file=sys.stderr)
            return EXIT_VALIDATION
        if args.image:
            return _run_image(args, detector)
        if args.video:
            return _run_video(args, detector)

        parser.print_help()
        print(
            "\nNo input given. Example:\n"
            "  python -m app.main --image assets/images/input_folder/regq.jpg"
        )
        return EXIT_OK

    except (ValidationError, FileNotFoundError) as exc:
        logger.error("Input error: %s", exc)
        print(f"Input error: {exc}", file=sys.stderr)
        return EXIT_VALIDATION
    except ModelError as exc:
        logger.error("Model error: %s", exc)
        print(f"Model error: {exc}", file=sys.stderr)
        return EXIT_MODEL
    except ProcessingError as exc:
        logger.error("Processing error: %s", exc)
        print(f"Processing error: {exc}", file=sys.stderr)
        return EXIT_PROCESSING
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level guard
        logger.exception("Unexpected error")
        hint = " Re-run with --verbose for details." if not args.verbose else ""
        print(f"Unexpected error: {exc}.{hint}", file=sys.stderr)
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
