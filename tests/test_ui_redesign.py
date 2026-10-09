"""Tests for the dashboard redesign, sidebar cleanup and model-card metadata.

These are structural tests: they assert the navigation entry points exist and
stay wired to the right pages, that the compact sidebar has dropped the
per-model list, and that genuinely recorded evaluation metrics flow through
the registry into the Models page. Nothing here starts Streamlit or loads a
model.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import json

from app.config import PROJECT_ROOT, Settings
from app.detection.model_manager import load_custom_specs
from app.ui import theme
from app.ui.streamlit_app import PAGE_MODULES


def test_nav_offers_separate_image_and_video_entry_points() -> None:
    page_ids = [page_id for page_id, _, _ in theme.PAGES]
    assert "detect_image" in page_ids
    assert "detect_video" in page_ids
    # The old combined "detect" id is gone.
    assert "detect" not in page_ids


def test_required_navigation_links_are_present() -> None:
    labels = [label for _, label, _ in theme.PAGES]
    for required in ("Dashboard", "Detect Image", "Detect Video", "Species Explorer",
                     "Models", "Analytics", "History", "About"):
        assert required in labels


def test_sidebar_groups_match_the_brief() -> None:
    grouped = {title: ids for title, ids in theme.NAV_GROUPS}
    assert grouped["Overview"] == ("dashboard",)
    assert grouped["Detection"] == ("detect_image", "detect_video")
    assert grouped["Explore"] == ("species", "models", "analytics")
    assert grouped["Activity"] == ("history",)
    assert grouped["Information"] == ("about",)


def test_detect_pages_preselect_media() -> None:
    assert theme.DETECT_PAGES == {"detect_image": "Image", "detect_video": "Video"}


def test_every_page_has_a_render_module() -> None:
    for page_id, _, _ in theme.PAGES:
        assert page_id in PAGE_MODULES
    # Both entry points share the Detect workflow module.
    assert PAGE_MODULES["detect_image"] is PAGE_MODULES["detect_video"]


def test_nav_groups_cover_every_page_exactly_once() -> None:
    grouped = [page_id for _, ids in theme.NAV_GROUPS for page_id in ids]
    assert sorted(grouped) == sorted(page_id for page_id, _, _ in theme.PAGES)


def test_sidebar_status_is_single_line() -> None:
    """The sidebar must not list individual model names any more."""
    source = (PROJECT_ROOT / "app" / "ui" / "streamlit_app.py").read_text(encoding="utf-8")
    status_block = source.split("def _sidebar_status")[1].split("def main")[0]
    assert "spec.display_name" not in status_block
    assert "models ready" not in status_block
    assert "Inference ready" in status_block


def test_dashboard_has_required_sections() -> None:
    source = (PROJECT_ROOT / "app" / "ui" / "pages" / "dashboard.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        "Understand aquatic life",
        "Detect an Image",
        "Analyse a Video",
        "Explore BlueEye",
        "Biodiversity Analytics",
        "How BlueEye works",
        "Capabilities",
        "Responsible use",
    ):
        assert marker in source
    # The negative-capabilities block and the duplicated history list are gone.
    for removed in ("Recent detections", "Observation overview", "What BlueEye does not do"):
        assert removed not in source


def test_registry_metrics_are_parsed(tmp_path) -> None:
    models_dir = tmp_path / "models"
    custom = models_dir / "custom"
    custom.mkdir(parents=True)
    (custom / "registry.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "regional_demo",
                        "name": "Regional Demo",
                        "weights": "regional/demo.pt",
                        "classes": ["fish"],
                        "metrics": {"mAP@50": "0.81", "precision": "0.79"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    spec = load_custom_specs(models_dir)["regional_demo"]
    assert ("mAP@50", "0.81") in spec.metrics
    assert ("precision", "0.79") in spec.metrics
