#!/usr/bin/env python
"""Download the official pretrained marine-detect model weights.

The weights are NOT bundled with BlueEye. They are fetched on demand from
the official release links published in the upstream marine-detect README
(https://github.com/Orange-OpenSource/marine-detect).

Usage:
    python scripts/download_models.py            # download everything missing
    python scripts/download_models.py fish_inv   # download one model

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings, setup_logging  # noqa: E402
from app.detection.model_manager import (  # noqa: E402
    MODEL_REGISTRY,
    ModelError,
    ModelManager,
    download_progress_printer,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        choices=list(MODEL_REGISTRY),
        help="Specific model key(s) to download (default: all missing).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    args = parser.parse_args(argv)

    setup_logging("DEBUG" if args.verbose else get_settings().log_level)
    manager = ModelManager(get_settings())
    keys = args.models or list(MODEL_REGISTRY)

    try:
        for key in keys:
            print(f"Fetching {MODEL_REGISTRY[key].display_name} ...")
            path = manager.download(key, progress=download_progress_printer(key))
            print(f"  -> {path} ({path.stat().st_size / 1e6:.1f} MB)")
    except ModelError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3

    print("All requested models are available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
