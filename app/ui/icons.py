"""BlueEye icon language: Material Symbols Rounded (one library, everywhere).

Streamlit ships the Material Symbols Rounded webfont in its own static
assets, so these icons render offline with no extra dependency and match
the icons Streamlit uses inside its widgets (uploader, help tooltips, ...).

Two ways to use an icon:

* :func:`material` returns the ``":material/name:"`` token accepted by the
  ``icon=`` parameter of ``st.button`` / ``st.download_button``.
* :func:`span` returns an inline HTML ``<span>`` for markup BlueEye authors
  itself (cards, notices, empty states, navigation); the styling lives in
  :mod:`app.ui.theme` as the ``.be-ic`` class.

Icon names are snake_case Material Symbols ligatures such as
``"bar_chart"`` or ``"center_focus_strong"``.

Presentation only: nothing in this module touches detection behaviour.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

#: Font family registered by Streamlit's bundled stylesheet.
FONT_FAMILY = "Material Symbols Rounded"


def material(name: str) -> str:
    """Format ``name`` for Streamlit widget ``icon=`` parameters."""
    return f":material/{name}:"


def span(name: str, modifier: str = "") -> str:
    """Inline Material icon as HTML for BlueEye-authored markup.

    ``modifier`` maps to a ``be-ic--<modifier>`` class defined in the
    theme (for example ``"lg"`` or ``"tile"``).
    """
    classes = "be-ic" + (f" be-ic--{modifier}" if modifier else "")
    return f'<span class="{classes}" aria-hidden="true">{name}</span>'


__all__ = ["FONT_FAMILY", "material", "span"]
