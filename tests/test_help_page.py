"""Tests for the Help & Documentation page.

These are structural tests: they check the page is registered in the navigation
under the Information group and that it documents the real workflow (uploading
images, analysing videos, reading results and model limitations). They do not
start Streamlit.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

from app.config import PROJECT_ROOT
from app.ui import theme
from app.ui.streamlit_app import PAGE_MODULES

_HELP_SOURCE = (PROJECT_ROOT / "app" / "ui" / "pages" / "help_page.py").read_text(
    encoding="utf-8"
)


def test_help_page_is_registered_in_navigation() -> None:
    ids = [page_id for page_id, _, _ in theme.PAGES]
    assert "help" in ids
    assert "help" in PAGE_MODULES


def test_help_page_is_under_information() -> None:
    grouped = {title: ids for title, ids in theme.NAV_GROUPS}
    assert "help" in grouped["Information"]


def test_help_page_documents_the_real_workflow() -> None:
    for marker in (
        "Help & Documentation",
        "Quick start",
        "Detecting an image",
        "Analysing a video",
        "Reading the results",
        "Models and their limits",
        "Troubleshooting",
        "Further documentation",
    ):
        assert marker in _HELP_SOURCE, f"missing help section: {marker}"


def test_help_page_references_real_configuration() -> None:
    """The guidance uses the real extensions and limits, not invented ones."""
    for token in ("IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS", "max_image_mb", "max_video_mb"):
        assert token in _HELP_SOURCE, f"help page omits real setting: {token}"


def test_every_page_has_a_non_empty_label_and_icon() -> None:
    for page_id, label, icon in theme.PAGES:
        assert page_id and label.strip() and icon.strip()
