"""Tests for the dashboard redesign, unified navigation and model-card metadata.

These are structural tests: they assert the navigation offers exactly one
AI Detection entry point (image + video live in that one page), that the
sidebar grouping is intact, that the dashboard is display-only and built
from real history/registry data, and that genuinely recorded evaluation
metrics flow through the registry into the Models page. Nothing here starts
Streamlit or loads a model.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import json

from app.config import PROJECT_ROOT, Settings
from app.detection.model_manager import load_custom_specs
from app.ui import theme
from app.ui.streamlit_app import PAGE_MODULES


def test_nav_offers_one_unified_detection_entry_point() -> None:
    """Exactly one sidebar entry - "AI Detection" - leads to the Detect page."""
    page_ids = [page_id for page_id, _, _ in theme.PAGES]
    assert page_ids.count("detect") == 1
    # The separate image / video entries are gone from the sidebar.
    assert "detect_image" not in page_ids
    assert "detect_video" not in page_ids


def test_required_navigation_links_are_present() -> None:
    labels = [label for _, label, _ in theme.PAGES]
    for required in (
        "Dashboard",
        "AI Detection",
        "Detection History",
        "Species Explorer",
        "AI Models",
        "Analytics",
        "About BlueEye",
        "Help & Documentation",
    ):
        assert required in labels
    assert labels.count("AI Detection") == 1
    # The Map section was removed from the navigation entirely.
    assert "Map" not in labels


def test_sidebar_groups_match_the_brief() -> None:
    grouped = {title: ids for title, ids in theme.NAV_GROUPS}
    assert grouped["Main"] == ("dashboard", "detect", "observations")
    assert grouped["Explore"] == ("history", "species", "models", "analytics")
    assert grouped["Information"] == ("about", "help")


def test_detect_pages_preselect_media() -> None:
    assert theme.DETECT_PAGES == {"detect_image": "Image", "detect_video": "Video"}


def test_every_page_has_a_render_module() -> None:
    for page_id, _, _ in theme.PAGES:
        assert page_id in PAGE_MODULES
    # The legacy entry points still resolve to the shared AI Detection module.
    assert PAGE_MODULES["detect_image"] is PAGE_MODULES["detect"]
    assert PAGE_MODULES["detect_video"] is PAGE_MODULES["detect"]


def test_nav_groups_cover_every_page_exactly_once() -> None:
    grouped = [page_id for _, ids in theme.NAV_GROUPS for page_id in ids]
    assert sorted(grouped) == sorted(page_id for page_id, _, _ in theme.PAGES)


def test_map_section_is_removed() -> None:
    """The Map section is gone from navigation, routing and source code."""
    page_ids = [page_id for page_id, _, _ in theme.PAGES]
    assert "map" not in page_ids
    assert "map" not in PAGE_MODULES
    grouped = [page_id for _, ids in theme.NAV_GROUPS for page_id in ids]
    assert "map" not in grouped
    # No dead map-only code remains on disk.
    assert not (PROJECT_ROOT / "app" / "ui" / "pages" / "map_page.py").exists()
    assert not (PROJECT_ROOT / "app" / "content" / "regions.py").exists()
    # The shell no longer imports or wires the removed page.
    shell = (PROJECT_ROOT / "app" / "ui" / "streamlit_app.py").read_text(
        encoding="utf-8"
    )
    assert "map_page" not in shell
    assert '"map"' not in shell


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
    # 1. cinematic hero band
    for marker in (
        "BLUEEYE · MARINE INTELLIGENCE",
        "See Beneath the Surface",
        "Discover marine life through AI-powered detection",
        "Explore Marine Species",
    ):
        assert marker in source, f"missing hero section: {marker}"
    # 2. four real statistic cards
    for marker in (
        "Total Detection Sessions",
        "Unique Species Identified",
        "Total Organisms Detected",
        "Available AI Models",
    ):
        assert marker in source, f"missing statistic card: {marker}"
    # 3. latest detection comparison with honest empty state
    for marker in (
        "Latest Detection",
        "See how AI helps identify marine life",
        "Original Image",
        "Detected Result",
        "Detection Details",
        "No detections yet",
    ):
        assert marker in source, f"missing latest-detection section: {marker}"
    # 4. Indian marine biodiversity driven by the real catalogue
    for marker in (
        "Indian Marine Biodiversity",
        "Explore species found in Indian coastal waters",
        "green_sea_turtle",
        "get_species",
    ):
        assert marker in source, f"missing biodiversity section: {marker}"
    # 5. overview charts and 6. recent discoveries
    for marker in (
        "Marine Biodiversity Overview",
        "Detection activity from your stored runs",
        "Recent Discoveries",
        "No discoveries yet",
    ):
        assert marker in source, f"missing dashboard section: {marker}"
    # The previous dashboard's welcome/cards/preview layout is gone.
    for removed in (
        "Good to see you!",
        "New Detection",
        "Quick actions",
        "Detection Preview",
        "Recent Detection Runs",
        "sample_count",
        "demo_count",
    ):
        assert removed not in source, f"dashboard still contains: {removed}"


def test_dashboard_has_no_direct_detection_workspace() -> None:
    """The dashboard must not embed a detection workspace.

    Uploading, model selection, enhancement and the start button all belong to
    the AI Detection page; the dashboard only navigates there.
    """
    source = (PROJECT_ROOT / "app" / "ui" / "pages" / "dashboard.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "st.file_uploader",
        "st.selectbox",
        "st.toggle",
        "st.checkbox",
        "st.slider",
        "st.select_slider",
        "st.progress",
        "Start detection",
        "detector.detect",
    ):
        assert forbidden not in source, f"dashboard still contains: {forbidden}"


def test_dashboard_reads_real_history() -> None:
    """Dashboard numbers, panels and charts come from real history only."""
    source = (PROJECT_ROOT / "app" / "ui" / "pages" / "dashboard.py").read_text(
        encoding="utf-8"
    )
    assert "load_entries" in source
    assert "history_path" in source
    assert "summarize" in source
    # Every navigation target is an existing page/workflow.
    for target in ('"detect"', '"species"', '"history"', '"analytics"'):
        assert target in source, f"missing navigation target: {target}"
    # The comparison uses a verified class -> species map, never a fake image.
    assert "_CLASS_SPECIES" in source
    for fabricated in ("fake", "demo_count", "sample_count", "sample_image"):
        assert fabricated not in source


def test_dashboard_biodiversity_icons_are_valid_material_symbols() -> None:
    """No invalid Material Symbols names in the biodiversity cards.

    Names that are not real ligatures (for example ``turtle`` / ``ecosystem``)
    render as literal words in the icon font and overflow the small cards -
    exactly the stray oversized text that appeared on the dashboard.
    """
    from app.ui.pages import dashboard

    source = (PROJECT_ROOT / "app" / "ui" / "pages" / "dashboard.py").read_text(
        encoding="utf-8"
    )
    assert '"turtle"' not in source
    assert '"ecosystem"' not in source
    # Every decorative species icon is a real Material Symbols name.
    assert set(dashboard._MINI_ICONS.values()) <= {
        "water_drop",
        "water",
        "waves",
        "pets",
    }


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


# --------------------------------------------------------------------------- #
# Authentication / account page structure
# --------------------------------------------------------------------------- #
def test_account_page_is_in_the_navigation() -> None:
    page_ids = [page_id for page_id, _, _ in theme.PAGES]
    labels = [label for _, label, _ in theme.PAGES]
    assert "account" in page_ids
    assert "Account" in labels
    grouped = {title: ids for title, ids in theme.NAV_GROUPS}
    assert grouped["Account"] == ("account",)


def test_every_page_render_accepts_a_user_argument() -> None:
    """Private pages must receive the authenticated user from the shell."""
    import inspect

    pages_dir = PROJECT_ROOT / "app" / "ui" / "pages"
    for page_id, _, _ in theme.PAGES:
        module = PAGE_MODULES[page_id]
        signature = inspect.signature(module.render)
        assert "user" in signature.parameters, f"{module.__name__}.render lacks 'user'"


def test_private_pages_are_gated_and_public_pages_are_open() -> None:
    from app.ui import auth_ui

    assert auth_ui.PRIVATE_PAGES == {
        "dashboard",
        "detect",
        "observations",
        "history",
        "analytics",
        "compare",
        "evaluation",
        "account",
    }
    # Species / models / about / help must remain browsable anonymously.
    public = {"species", "models", "about", "help"}
    assert public.isdisjoint(auth_ui.PRIVATE_PAGES)


def test_research_pages_are_in_the_navigation() -> None:
    """The comparison and evaluation consoles ship as first-class pages."""
    page_ids = [page_id for page_id, _, _ in theme.PAGES]
    labels = [label for _, label, _ in theme.PAGES]
    assert "compare" in page_ids
    assert "evaluation" in page_ids
    assert "Model Comparison" in labels
    assert "Research Evaluation" in labels
    grouped = {title: ids for title, ids in theme.NAV_GROUPS}
    assert grouped["Research"] == ("compare", "evaluation")
    # Both resolve to real page modules.
    assert PAGE_MODULES["compare"] is not PAGE_MODULES["evaluation"]


def test_auth_ui_uses_the_real_backend() -> None:
    source = (PROJECT_ROOT / "app" / "ui" / "auth_ui.py").read_text(encoding="utf-8")
    # The UI authenticates exclusively through the real service.
    assert "AuthService" in source
    assert ".authenticate(" in source
    assert ".login(" in source
    assert ".register(" in source
    assert ".logout(" in source
    # Sessions are an opaque server-side token, never a client-supplied identity.
    assert "be_auth_token" in source
    # Logout clears private UI state instead of leaking the previous account.
    assert "clear_user_state" in source
    # No fake/demo credentials may be baked into the UI.
    for fabricated in ("demo_password", "admin123", "password123", "test_account"):
        assert fabricated not in source

