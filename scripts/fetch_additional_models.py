"""Download the additional (verified) aquatic-species models for BlueEye.

These are genuine, publicly published checkpoints - never fabricated. Each is
fetched from a pinned URL and its SHA-256 is verified before it is accepted, so
a truncated or substituted file is rejected. When a checkpoint is distributed
inside a zip archive the member is extracted first and the extracted file is
verified.

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
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402

#: Verified, publicly downloadable additional models.
#: ``sha256`` pins the exact artefact that was verified for this project.
#: ``member`` (optional) extracts one file from a zip download before hashing.
MODELS: list[dict[str, str]] = [
    {
        "key": "aquatic_brackish",
        "url": "https://huggingface.co/dronefreak/brackish-yolov8s/resolve/main/best.pt",
        "dest": "additional/aquatic_brackish/BlueEyeAquaticBrackish.pt",
        "sha256": "2a96029dbbe0477e3cf6dc9f0427d0400037c884ff4d4b4573e4dfe0de86090b",
        "license": "AGPL-3.0",
    },
    {
        "key": "underwater_fish",
        "url": "https://huggingface.co/akridge/yolo8-fish-detector-grayscale/resolve/main/yolov8n_fish_trained.pt",
        "dest": "additional/underwater_fish/BlueEyeUnderwaterFish.pt",
        "sha256": "2e3d70a04c5a5fc7aa1dce0d0e38d10169d32fe23c9d5314d7f0309be0eb36f3",
        "license": "US Government work (public domain / royalty-free)",
    },
    {
        "key": "aquarium_marine",
        "url": "https://huggingface.co/Kanagavel/aquarium-rtdetr/resolve/main/best.pt",
        "dest": "additional/aquarium_rtdetr/BlueEyeAquariumMarine.pt",
        "sha256": "02c14f958e4d7a5bfb90fc082201d13d813d6d597f5fd65fa75738b95334b72d",
        "license": "AGPL-3.0",
    },
    {
        "key": "obsea_mediterranean",
        "url": "https://zenodo.org/api/records/14910365/files/yolov8x_21sp_5364img.pt/content",
        "dest": "additional/obsea_mediterranean/BlueEyeObseaMediterranean.pt",
        "sha256": "12793170b79108bfbbaf730192a4f2d2e404414b00427f4bb148a46dc92262ab",
        "license": "CC-BY-4.0",
    },
    {
        "key": "community_fish",
        "url": "https://github.com/filippovarini/community-fish-detector/releases/download/cfd-1.00-yolov12x/cfd-yolov12x-1.00.pt",
        "dest": "additional/community_fish/BlueEyeCommunityFish.pt",
        "sha256": "0b259afb3dca1d1d7f9d842ebf136884b2e3706b151172b266bb24e07f2b8e98",
        "license": "AGPL-3.0 (model inference)",
    },
    {
        "key": "fishial_detector",
        "url": "https://storage.googleapis.com/fishial-ml-resources/detector_v26_n3.zip",
        "dest": "additional/fishial_detector/BlueEyeFishialDetector.pt",
        "sha256": "5b786b334355fdb0c3faa9d375d70de24f3535ee6f11e9c3e26ec2e90810c03f",
        "member": "model.pt",
        "license": "MIT",
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=600) as response:  # noqa: S310 - pinned HTTPS
        return response.read()


def _download(url: str, member: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _fetch_bytes(url)
    if member:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            data = archive.read(member)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.write_bytes(data)
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
        print(f"[{entry['key']}] downloading {entry['url']}")
        try:
            _download(entry["url"], entry.get("member", ""), destination)
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
        print(
            f"[{entry['key']}] ok ({destination.stat().st_size / 1e6:.1f} MB), "
            f"license {entry['license']}"
        )

    if failures:
        print(f"\n{failures} model(s) could not be fetched.")
        return 1
    print("\nAll additional models are installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
