"""BlueEye web UI (Streamlit).

Run from the project root with::

    streamlit run app/ui/streamlit_app.py

The shell owns page navigation, the sidebar and the design system; every
page module in :mod:`app.ui.pages` renders the actual content and calls the
existing detection backend unchanged.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow "streamlit run app/ui/streamlit_app.py" from the project root.
if __package__ in (None, ""):  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging  # noqa: E402

import streamlit as st  # noqa: E402

from app import __version__  # noqa: E402
from app.ui import state  # noqa: E402
from app.ui.pages import (  # noqa: E402
    about,
    analytics,
    dashboard,
    detect,
    help_page,
    history_page,
    models,
    species,
)
from app.ui.icons import span  # noqa: E402
from app.ui.theme import DETECT_PAGES, PAGES, apply_theme  # noqa: E402

logger = logging.getLogger(__name__)

#: Page id -> module exposing ``render(detector, settings)``.
#: ``detect_image`` / ``detect_video`` share the Detect workflow module.
PAGE_MODULES = {
    "dashboard": dashboard,
    "detect_image": detect,
    "detect_video": detect,
    "species": species,
    "models": models,
    "analytics": analytics,
    "history": history_page,
    "about": about,
    "help": help_page,
}

# Re-exported for backwards compatibility with earlier versions of this file.
get_detector = state.get_detector
get_settings_cached = state.get_settings_cached


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _sidebar() -> str:
    """Render the brand and navigation; returns the active page.

    The sidebar is deliberately minimal - logo, project name, tagline and the
    grouped navigation. Model counts, readiness badges, repository credits and
    technical notes live on the pages that own them (Models, About), never
    here.
    """
    with st.sidebar:
        # Brand block. Section grouping and icons come from the design
        # system (theme.PAGES / theme.NAV_GROUPS) - the nav stays the single
        # existing radio so navigation mechanics are unchanged.
        st.markdown(
            '<div class="be-brand"><div class="row">'
            f'<div class="mark">{span("visibility")}</div>'
            '<div><p class="n">BlueEye</p>'
            '<p class="t">See beneath the surface</p></div></div>'
            '<div class="be-brand-rule"></div></div>',
            unsafe_allow_html=True,
        )

        state.apply_pending_navigation()
        page = st.radio(
            "Navigation",
            options=[page_id for page_id, _, _ in PAGES],
            format_func=lambda page_id: next(
                label for pid, label, _ in PAGES if pid == page_id
            ),
            label_visibility="collapsed",
            key="nav_sel",
            index=0,
            help="Jump between BlueEye sections.",
        )
    return page


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="BlueEye - Aquatic Life Detection",
        page_icon=":material/waves:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    settings = state.get_settings_cached()
    detector = state.get_detector()
    state.init_session(settings)

    page = _sidebar()

    # "Detect an Image" / "Analyse a Video" are two entry points into the one
    # Detect workflow; preselect the media type before the page renders.
    media = DETECT_PAGES.get(page)
    if media:
        st.session_state["detect_media"] = media

    module = PAGE_MODULES.get(page, dashboard)
    module.render(detector, settings)

    st.divider()
    st.caption(f"BlueEye v{__version__} · See beneath the surface")


if __name__ == "__main__":
    main()
