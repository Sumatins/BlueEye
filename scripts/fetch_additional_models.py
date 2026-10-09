"""Download the additional (verified) aquatic-species models for BlueEye.

These are genuine, publicly published checkpoints - never fabricated. Each is
fetched from the pinned Hugging Face repository and its SHA-256 is verified
before it is accepted, so a truncated or substituted file is rejected.

The two core models (fish_inv, megaFauna) are handled separately by
``scripts/download_models.py``. These additional models are optional: the
application runs without them.

Usage::

    python scripts/fetch_additional_models.py
    python scripts/fetch_additional_models.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402

#: Verified, publicly downloadable additional models.
#: ``sha256`` pins the exact artefact that was verified for this project.
MODELS: list[dict[str, str]] = [
    {
        "key": "aquatic_brackish",
        "repo": "dronefreak/brackish-yolov8s",
        "file": "best.pt",
        "dest": "additional/aquatic_brackish/BlueEyeAquaticBrackish.pt",
        "sha256": "2a96029dbbe0477e3cf6dc9f0427d0400037c884ff4d4b4573e4dfe0de86090b",
        "license": "AGPL-3.0",
    },
    {
        "key": "underwater_fish",
        "repo": "akridge/yolo8-fish-detector-grayscale",
        "file": "yolov8n_fish_trained.pt",
        "dest": "additional/underwater_fish/BlueEyeUnderwaterFish.pt",
        "sha256": "2e3d70a04c5a5fc7aa1dce0d0e38d10169d32fe23c9d5314d7f0309be0eb36f3",
        "license": "US Government work (public domain / royalty-free)",
    },
    {
        "key": "aquarium_marine",
        "repo": "Kanagavel/aquarium-rtdetr",
        "file": "best.pt",
        "dest": "additional/aquarium_rtdetr/BlueEyeAquariumMarine.pt",
        "sha256": "02c14f958e4d7a5bfb90fc082201d13d813d6d597f5fd65fa75738b95334b72d",
        "license": "AGPL-3.0",
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=180) as response, open(partial, "wb") as handle:  # noqa: S310 - pinned HTTPS
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    partial.replace(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args(argv)

    models_dir = get_settings().models_dir
    failures = 0
    for entry in MODELS:
        destination = models_dir / entry["dest"]
        expected = entry["sha256"]
        if destination.is_file() and not args.force:
            if _sha256(destination) == expected:
                print(f"[{entry['key']}] already present and verified: {destination}")
                continue
            print(f"[{entry['key']}] present but hash mismatch; re-downloading.")
        url = f"https://huggingface.co/{entry['repo']}/resolve/main/{entry['file']}"
        print(f"[{entry['key']}] downloading {url}")
        try:
            _download(url, destination)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[{entry['key']}] FAILED: {exc}")
            failures += 1
            continue
        actual = _sha256(destination)
        if actual != expected:
            destination.unlink(missing_ok=True)
            print(f"[{entry['key']}] FAILED hash check (got {actual[:12]}...); file removed.")
            failures += 1
            continue
        print(f"[{entry['key']}] ok ({destination.stat().st_size / 1e6:.1f} MB), license {entry['license']}")

    if failures:
        print(f"\n{failures} model(s) could not be fetched.")
        return 1
    print("\nAll additional models are installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
