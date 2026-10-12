"""Capture the software and hardware context of an evaluation run.

Evaluation numbers are only meaningful together with the machine and the
package versions that produced them. This module records that context so
every result file is self-describing (see the *test conditions* column of
the comparison table).

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from datetime import datetime


def _version(module_name: str) -> str:
    """Installed version of ``module_name``, or a readable placeholder."""
    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - a missing optional package is not fatal
        return "not installed"
    return str(getattr(module, "__version__", "unknown"))


def _hardware() -> dict:
    """Best-effort CPU / GPU description (never raises)."""
    info: dict = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gpu": None,
        "cuda_available": False,
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - torch is optional for reporting
        pass
    return info


def collect_environment() -> dict:
    """Return a JSON-serialisable snapshot of the runtime environment."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hardware": _hardware(),
        "packages": {
            "torch": _version("torch"),
            "torchvision": _version("torchvision"),
            "ultralytics": _version("ultralytics"),
            "numpy": _version("numpy"),
            "opencv-python": _version("cv2"),
            "Pillow": _version("PIL"),
        },
    }
