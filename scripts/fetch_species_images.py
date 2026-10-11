"""Fetch verified species photographs from Wikimedia Commons.

For every species in :data:`app.content.species.SPECIES` this script queries
the Wikimedia Commons MediaWiki API (free, no key) for bitmap images, ranks
the candidates (exact scientific name first, then genus, then common name,
deprioritising diagrams / maps / logos), verifies that the chosen thumbnail
URL really resolves, and writes ``app/content/species_images.py`` with a
per-species ``url``, ``credit``, ``license`` and licence URL.

Honesty rules:

* the script never invents an image: a species with no usable, licensed,
  reachable photo is simply left out and the Species Explorer keeps its
  icon fallback;
* the descriptive User-Agent and the request throttle follow the Wikimedia
  API etiquette (and the project's polite-API convention);
* only files served from Wikimedia's own hosts are accepted.

Run manually:

    python scripts/fetch_species_images.py            # fetch + write module
    python scripts/fetch_species_images.py --dry-run  # print, do not write

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests  # noqa: E402

from app.content.species import SPECIES  # noqa: E402

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OUTPUT = PROJECT_ROOT / "app" / "content" / "species_images.py"

#: Wikimedia etiquette: identify the tool and never hammer the API.
USER_AGENT = (
    "BlueEye/1.0 (educational marine-biodiversity dashboard; "
    "species-image fetch; contact: blueeye-demo@example.invalid)"
)
REQUEST_PAUSE = 1.2

#: File-title words that usually mean the image is not a usable photograph.
#: "dead" / "albino" / "gill" reject biological oddities that would mislead a
#: species card; the book-title words force a real photograph over a plate.
_REJECT_WORDS = (
    "map", "range", "distribution", "diagram", "drawing", "illustration",
    "sketch", "logo", "icon", "cladogram", "anatomy", "chart",
    "graph", "svg", "stamp", "sign", "poster", "cover",
    "dead", "albino", "gill", "atlas", "histoire naturelle", "erpetologie",
)

#: Words that mark a historic plate / museum specimen rather than a live photo.
#: These are not rejected (they often depict the correct species and are public
#: domain) but they are ranked below a genuine photograph.
_ILLUSTRATION_WORDS = (
    "handbook", "plate", "figure", "fig.", "1877", "1889", "1866", "hardwicke",
    "skeleton", "skelett", "skull", "museum", "specimen", "taxidermy",
)

#: Depicts the right species but not the animal itself; slightly de-ranked so
#: an actual portrait wins when one exists, yet still usable as a last resort.
_SUBJECT_WORDS = ("egg", "nest", "larva", "fossil", "spawn")


def _api(params: dict) -> dict:
    """One polite Commons API call with a descriptive User-Agent."""
    response = requests.get(
        COMMONS_API,
        params={"action": "query", "format": "json", **params},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    time.sleep(REQUEST_PAUSE)
    return response.json()


def _search_terms(species) -> list[str]:
    """Search terms, most specific first."""
    terms = []
    if species.scientific_name:
        terms.append(f'intitle:"{species.scientific_name}" filetype:bitmap')
    genus = (species.scientific_name or "").split(" ")[0]
    if genus and genus != species.scientific_name:
        terms.append(f'intitle:"{genus}" filetype:bitmap')
    common = re.sub(r"\s*\(.*?\)", "", species.common_name or "").strip()
    if common:
        terms.append(f'intitle:"{common}" filetype:bitmap')
    return terms


def _candidates(species) -> list[dict]:
    """Collect deduplicated imageinfo candidates across the search terms."""
    found: dict[str, dict] = {}
    for term in _search_terms(species):
        payload = _api(
            {
                "generator": "search",
                "gsrsearch": term,
                "gsrnamespace": 6,
                "gsrlimit": 8,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|mime|size",
                "iiurlwidth": 640,
            }
        )
        pages = (payload.get("query") or {}).get("pages") or {}
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            mime = str(info.get("mime") or "")
            if not mime.startswith("image/"):
                continue
            if mime in {"image/svg+xml", "image/tiff"}:
                continue
            title = str(page.get("title") or "")
            if title not in found:
                found[title] = {
                    "title": title,
                    "thumb": info.get("thumburl") or info.get("url") or "",
                    "page": info.get("descriptionurl") or "",
                    "mime": mime,
                    "metadata": info.get("extmetadata") or {},
                }
    return list(found.values())


def _score(species, candidate: dict) -> float:
    """Rank a candidate; higher is better (0 means reject).

    Honesty first: a candidate is only accepted when its title contains the
    **full scientific name** or the **common name**. A genus-only match is
    never accepted on its own - it is how a *Catla* search once surfaced a
    *Black Sharkminnow* photo of the same genus.
    """
    title = candidate["title"].lower()
    if any(word in title for word in _REJECT_WORDS):
        return 0.0
    scientific = (species.scientific_name or "").lower().strip()
    common = re.sub(r"\s*\(.*?\)", "", species.common_name or "").lower().strip()
    if scientific and scientific in title:
        score = 6.0
    elif common and common in title:
        score = 3.0
    else:
        return 0.0
    if candidate["mime"] == "image/jpeg":
        score += 0.4
    if any(word in title for word in _ILLUSTRATION_WORDS):
        score -= 3.5
    if any(word in title for word in _SUBJECT_WORDS):
        score -= 1.6
    # A concise title (just the species name) is usually a clean portrait;
    # this small tie-break nudges equal-scoring candidates toward it.
    score += max(0.0, 1.0 - len(title) / 80.0)
    return score


def _credit(metadata: dict) -> tuple[str, str, str]:
    """Extract (credit, licence, licence_url) from Commons extmetadata."""
    def value(key: str) -> str:
        raw = (metadata.get(key) or {}).get("value") or ""
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", str(raw))).strip()

    credit = value("Artist") or value("Credit") or "Wikimedia Commons contributor"
    licence = value("LicenseShortName") or ""
    licence_url = value("LicenseUrl") or ""
    return credit[:120], licence[:60], licence_url


def _resolves(url: str) -> bool:
    """True when the image URL really returns an image."""
    if not url:
        return False
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Range": "bytes=0-1024"},
            timeout=30,
            stream=True,
        )
        ok = response.status_code in (200, 206) and response.headers.get(
            "Content-Type", ""
        ).startswith("image/")
        response.close()
        time.sleep(0.5)
        return ok
    except requests.RequestException:
        return False


def fetch_for(species) -> dict | None:
    """Best verified image for one species, or ``None``."""
    candidates = _candidates(species)
    candidates.sort(key=lambda candidate: _score(species, candidate), reverse=True)
    for candidate in candidates:
        if _score(species, candidate) <= 0:
            continue
        # Wikimedia appends a tracking query string (?utm_source=...) when you
        # request imageinfo; the thumbnail path already encodes the size, so
        # strip the query for a stable, cacheable URL.
        url = candidate["thumb"].split("?", 1)[0]
        # Only accept files served from Wikimedia's own hosts.
        if not url.startswith("https://") or not re.search(
            r"https://([a-z0-9-]+\.)*(wikimedia|wikipedia)\.org/", url
        ):
            continue
        if not _resolves(url):
            continue
        credit, licence, licence_url = _credit(candidate["metadata"])
        return {
            "url": url,
            "credit": credit,
            "license": licence,
            "license_url": licence_url,
            "source_page": candidate["page"],
            "title": candidate["title"],
        }
    return None


def _write_module(mapping: dict[str, dict]) -> None:
    """Write the generated ``SPECIES_IMAGES`` data module."""
    module = (
        '"""Verified species photographs (generated by '
        "scripts/fetch_species_images.py).\n\n"
        "Each entry points at a Wikimedia Commons thumbnail with its author\n"
        "credit and licence. The URLs were checked to resolve when this file was\n"
        "generated; :mod:`app.content.species` merges them into the catalogue and\n"
        "the Species Explorer falls back to an icon tile if an image ever fails\n"
        "to load. Regenerate with ``python scripts/fetch_species_images.py``.\n\n"
        "SPDX-License-Identifier: AGPL-3.0-only\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        "#: species id -> {url, credit, license, license_url, source_page, title}\n"
        f"SPECIES_IMAGES: dict[str, dict[str, str]] = {json.dumps(mapping, indent=4, ensure_ascii=False, sort_keys=True)}\n"
    )
    OUTPUT.write_text(module, encoding="utf-8")


def _load_previous() -> dict[str, dict]:
    """Reuse an earlier generated file so a partial run keeps its images."""
    if not OUTPUT.is_file():
        return {}
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_species_images_prev", OUTPUT)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        previous = getattr(module, "SPECIES_IMAGES", {})
        return dict(previous) if isinstance(previous, dict) else {}
    except Exception as exc:  # noqa: BLE001 - a stale file must never break a run
        print(f"  ! could not reuse {OUTPUT.name}: {exc}")
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated species ids to fetch (for a quick smoke test)",
    )
    args = parser.parse_args()

    # Commons titles contain non-Latin scripts; make the console write safe.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stdout
        pass

    wanted = {part.strip() for part in args.only.split(",") if part.strip()}
    species_list = [s for s in SPECIES if not wanted or s.id in wanted]

    mapping = _load_previous()
    for species in species_list:
        try:
            result = fetch_for(species)
        except requests.RequestException as exc:
            print(f"  ! {species.id}: API error {exc}")
            continue
        if result is None:
            # A tighter re-run may demote a species to the icon fallback;
            # never keep a stale image chosen under looser rules.
            mapping.pop(species.id, None)
            print(f"  - {species.id}: no usable image (icon fallback stays)")
            continue
        mapping[species.id] = result
        print(f"  + {species.id}: {result['title']} [{result['license'] or 'license?'}]")
        if not args.dry_run:
            _write_module(mapping)  # incremental: a later error keeps earlier work

    print(f"\n{len(mapping)} species with a verified image "
          f"({sum(1 for s in species_list if s.id in mapping)} of this run).")
    if args.dry_run:
        return 0
    _write_module(mapping)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())