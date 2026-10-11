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
    for required in (
        "Dashboard",
        "Detect an Image",
        "Analyse a Video",
        "Detection History",
        "Species Explorer",
        "AI Models",
        "Analytics",
        "About BlueEye",
        "Help & Documentation",
    ):
        assert required in labels


def test_sidebar_groups_match_the_brief() -> None:
    grouped = {title: ids for title, ids in theme.NAV_GROUPS}
    assert grouped["Main"] == ("dashboard", "detect_image", "detect_video")
    assert grouped["Explore"] == ("history", "species", "models", "analytics")
    assert grouped["Information"] == ("about", "help")


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


def test_sidebar_is_clean_of_status_and_repository_text() -> None:
    """The sidebar keeps only brand + navigation.

    The readiness badge, the "models on the Models page" caption and the
    marine-detect credit must all be gone, and they must not be replaced by
    another status panel or an orphaned download button.
    """
    source = (PROJECT_ROOT / "app" / "ui" / "streamlit_app.py").read_text(encoding="utf-8")
    sidebar_block = source.split("def _sidebar")[1].split("def main")[0]
    for removed in (
        "Inference ready",
        "No models installed",
        "Model availability, classes and licences are on the Models page.",
        "marine-detect by Orange OpenSource",
        "sidebar_download",
        "download_all",
        "_sidebar_status",
    ):
        assert removed not in sidebar_block, f"sidebar still contains: {removed}"
    # The removal must not leave a leftover status function or dead CSS class.
    assert "def _sidebar_status" not in source
    assert "be-side-status" not in (PROJECT_ROOT / "app" / "ui" / "theme.py").read_text(
        encoding="utf-8"
    )
    # Brand and every navigation entry are still there.
    for kept in ("BlueEye", "See beneath the surface", "Navigation"):
        assert kept in sidebar_block, f"sidebar lost: {kept}"


def test_dashboard_has_required_sections() -> None:
    source = (PROJECT_ROOT / "app" / "ui" / "pages" / "dashboard.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        # A. header + status
        "BlueEye Dashboard",
        "Detect and explore aquatic animals",
        # B. summary cards
        "Detection runs",
        "Images analysed",
        "Videos analysed",
        "Unique species",
        "AI models available",
        # C. primary actions
        "Start a detection",
        "Detect an Image",
        "Analyse a Video",
        # D. recent activity
        "Recent activity",
        "Open Detection History",
        # E. model status
        "Model status",
        "Inference device",
        # retained value sections
        "How BlueEye works",
        "Explore aquatic biodiversity",
        "Responsible observation",
    ):
        assert marker in source, f"missing dashboard section: {marker}"
    # The negative-capabilities block and the old "Capabilities" narrative are
    # gone, and the dashboard must not invent a "recent detections" list or
    # hard-coded demo statistics.
    for removed in (
        "Observation overview",
        "What BlueEye does not do",
        "### Capabilities",
        "Responsible use",
        "sample_count",
        "demo_count",
    ):
        assert removed not in source, f"dashboard still contains: {removed}"


def test_dashboard_reads_real_history_and_registry() -> None:
    """The dashboard summary must come from real sources, never hard-coded."""
    source = (PROJECT_ROOT / "app" / "ui" / "pages" / "dashboard.py").read_text(
        encoding="utf-8"
    )
    # History + model registry are the only data sources.
    assert "load_entries" in source
    assert "summarize" in source
    assert "state.registry" in source
    assert "state.available_keys" in source


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
