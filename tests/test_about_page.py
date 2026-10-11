"""Tests for the About page presentation.

The About page must read as *our* project rather than a repository showcase.
It keeps the application introduction, purpose, features, workflow and
technology, but the visible "Licence and attribution" block has been removed.
Legal notices (LICENSE, SPDX headers, README attribution) must remain in the
repository - only the About page UI drops the section.

These are structural tests; they do not start Streamlit.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from app.config import PROJECT_ROOT

_ABOUT_SOURCE = (PROJECT_ROOT / "app" / "ui" / "pages" / "about.py").read_text(
    encoding="utf-8"
)


def test_about_page_has_the_required_sections() -> None:
    for marker in (
        "About BlueEye",
        "Our purpose",
        "Key features",
        "How it works",
        "Technology",
        "Responsible use",
    ):
        assert marker in _ABOUT_SOURCE, f"missing About section: {marker}"


def test_about_page_removes_the_licence_section() -> None:
    """The visible "Licence and attribution" UI block must be gone."""
    assert "Licence and attribution" not in _ABOUT_SOURCE
    assert "def _licence" not in _ABOUT_SOURCE
    assert "_licence()" not in _ABOUT_SOURCE


def test_about_page_is_not_a_repository_showcase() -> None:
    for removed in (
        "github.com/Orange-OpenSource",
        "View marine-detect source",
        "Our current models",
        "Future scope",
        "REPORT_DATASETS",
        "Project objective",
    ):
        assert removed not in _ABOUT_SOURCE, f"About still shows: {removed}"


def test_repository_legal_notices_are_preserved() -> None:
    """Removing the About UI section must not touch the legal files."""
    license_file = PROJECT_ROOT / "LICENSE"
    assert license_file.is_file(), "LICENSE file is missing"
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Orange OpenSource" in readme
    assert "marine-detect" in readme
    # Source files keep their SPDX licence headers.
    assert "SPDX-License-Identifier: AGPL-3.0-only" in _ABOUT_SOURCE


def test_about_technology_claims_are_backed_by_requirements() -> None:
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for token in ("streamlit", "ultralytics", "torch", "opencv", "numpy", "pillow"):
        assert token in requirements, f"requirements missing {token}"
        assert token in _ABOUT_SOURCE.lower(), f"About omits real dependency {token}"
