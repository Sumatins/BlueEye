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
from app.detection.model_manager import ModelError  # noqa: E402
from app.ui import state  # noqa: E402
from app.ui.pages import (  # noqa: E402
    about,
    analytics,
    dashboard,
    detect,
    history_page,
    models,
)
from app.ui.components import status_badge  # noqa: E402
from app.ui.icons import span  # noqa: E402
from app.ui.theme import PAGES, apply_theme  # noqa: E402

logger = logging.getLogger(__name__)

#: Page id -> module exposing ``render(detector, settings)``.
PAGE_MODULES = {
    "dashboard": dashboard,
    "detect": detect,
    "models": models,
    "analytics": analytics,
    "history": history_page,
    "about": about,
}

# Re-exported for backwards compatibility with earlier versions of this file.
get_detector = state.get_detector
get_settings_cached = state.get_settings_cached


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _sidebar(detector, settings) -> str:
    """Render brand + navigation + system status; returns the active page."""
    with st.sidebar:
        # Brand block. Section grouping and icons come from the design
        # system (theme.PAGES / theme.NAV_GROUPS) - the nav stays the single
        # existing radio so navigation mechanics are unchanged.
        st.markdown(
            '<div class="be-brand"><div class="row">'
            f'<div class="mark">{span("visibility")}</div>'
            '<div><p class="n">BlueEye</p>'
            '<p class="t">AI-Powered Marine Life Detection</p></div></div>'
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

        st.write("")
        _sidebar_status(detector)
        st.caption(
            "Models: marine-detect by Orange OpenSource (AGPL-3.0). "
            "Weights are downloaded on demand - never fabricated."
        )
    return page


def _sidebar_status(detector) -> None:
    """Compact model/device status, with a download action when empty."""
    registry = state.registry(detector)
    ready = set(state.available_keys(detector))

    st.markdown(
        f'<div class="be-side-head">{span("monitoring")}<span>System status</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Device: `{detector.device}` · models ready: {len(ready)}/{len(registry)}")

    rows = "".join(
        f"<div style='margin:.15rem 0'>{status_badge(key in ready)} "
        f"<span style='color:#dcebf6;font-size:.86rem'>{spec.display_name}</span></div>"
        for key, spec in registry.items()
    )
    st.markdown(rows, unsafe_allow_html=True)

    if not ready:
        if st.button(
            "Download model weights",
            type="primary",
            icon=":material/download:",
            width="stretch",
            key="sidebar_download",
        ):
            try:
                with st.spinner("Downloading model weights (~87 MB per model)..."):
                    detector.model_manager.download_all()
                st.cache_resource.clear()
                st.rerun()
            except ModelError as exc:
                logger.warning("Model download failed: %s", exc)
                st.error(str(exc))
        st.caption("Or run `python scripts/download_models.py` in the project directory.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="BlueEye - AI Marine Life Detection",
        page_icon=":material/waves:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    settings = state.get_settings_cached()
    detector = state.get_detector()
    state.init_session(settings)

    page = _sidebar(detector, settings)
    module = PAGE_MODULES.get(page, dashboard)
    module.render(detector, settings)

    st.divider()
    st.caption(
        f"BlueEye v{__version__} · AI-Powered Marine Life Detection · "
        "built on marine-detect (Orange OpenSource, AGPL-3.0-only)."
    )


if __name__ == "__main__":
    main()
