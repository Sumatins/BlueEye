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
from app.ui import auth_ui, state  # noqa: E402
from app.ui.pages import (  # noqa: E402
    about,
    account,
    analytics,
    compare,
    dashboard,
    detect,
    evaluation,
    help_page,
    history_page,
    models,
    observations,
    species,
)
from app.ui.icons import span  # noqa: E402
from app.ui.theme import DETECT_PAGES, PAGES, apply_theme  # noqa: E402

logger = logging.getLogger(__name__)

#: Page id -> module exposing ``render(detector, settings, user=None)``.
#: The legacy ``detect_image`` / ``detect_video`` ids are kept here so any
#: stale reference still resolves to the same unified Detect workflow.
PAGE_MODULES = {
    "dashboard": dashboard,
    "detect": detect,
    "detect_image": detect,
    "detect_video": detect,
    "species": species,
    "models": models,
    "analytics": analytics,
    "observations": observations,
    "compare": compare,
    "evaluation": evaluation,
    "history": history_page,
    "about": about,
    "help": help_page,
    "account": account,
}

# Re-exported for backwards compatibility with earlier versions of this file.
get_detector = state.get_detector
get_settings_cached = state.get_settings_cached


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _sidebar(user=None) -> str:
    """Render the brand, navigation and identity; returns the active page.

    The sidebar is deliberately minimal - logo, project name, tagline, the
    grouped navigation and a small account block at the bottom. Model counts,
    readiness badges, repository credits and technical notes live on the pages
    that own them (Models, About), never here.
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

        # Legacy "Detect an Image" / "Analyse a Video" ids now resolve to the
        # one unified AI Detection page with the matching media preselected.
        legacy_media = DETECT_PAGES.get(st.session_state.get("nav_sel"))
        if legacy_media:
            st.session_state["detect_media"] = legacy_media
            st.session_state["nav_sel"] = "detect"

        # HTML deep links (?page=... [&species=...]) used by dashboard CTAs.
        requested = st.query_params.get("page")
        if isinstance(requested, str) and requested:
            if requested in DETECT_PAGES:  # old ?page=detect_image / detect_video
                st.session_state["detect_media"] = DETECT_PAGES[requested]
                requested = "detect"
            if requested in {page_id for page_id, _, _ in PAGES}:
                st.session_state["nav_sel"] = requested
                species_id = st.query_params.get("species")
                if requested == "species" and isinstance(species_id, str) and species_id:
                    st.session_state["species_selected"] = species_id
                st.query_params.clear()

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

        auth_ui.sidebar_account(user)
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

    # Resolve the authenticated account (server-side token validation). This
    # happens before navigation so every page receives a real user object or
    # is treated as anonymous - the browser never supplies an identity.
    user = auth_ui.bootstrap()

    page = _sidebar(user)

    module = PAGE_MODULES.get(page, dashboard)
    if user is None:
        if page in auth_ui.PRIVATE_PAGES:
            # Private workspace pages show the sign-in gate for guests.
            auth_ui.render_gate()
        else:
            # Public pages (Species / Map / Models / About / Help) stay
            # browsable with a gentle sign-in prompt.
            auth_ui.render_guest_banner()
            module.render(detector, settings)
    else:
        module.render(detector, settings, user)

    st.divider()
    st.caption(f"BlueEye v{__version__} · See beneath the surface")


if __name__ == "__main__":
    main()
