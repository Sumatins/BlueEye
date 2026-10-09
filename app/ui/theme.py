"""BlueEye visual design system (ocean theme).

All styling lives here so that page code stays free of CSS and the look
remains consistent across every page (single source of truth for colours,
radii, shadows and spacing). The palette is deliberately restrained:
deep-ocean navy, cyan/teal accents and light surfaces - appropriate for a
scientific tool rather than a flashy demo.

The icon language is Material Symbols Rounded (see :mod:`app.ui.icons`);
navigation icons and section headers are generated from :data:`PAGES` and
:data:`NAV_GROUPS` so the CSS can never drift from the navigation data.

Nothing here changes detection behaviour: it is presentation only.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import html

import streamlit as st

#: Ordered navigation entries: (page id, label, Material icon name).
#:
#: ``detect_image`` and ``detect_video`` both open the same Detect workflow
#: with the matching media type preselected, so the sidebar can offer direct
#: "Detect Image" / "Detect Video" entry points without duplicating code.
PAGES: tuple[tuple[str, str, str], ...] = (
    ("dashboard", "Dashboard", "home"),
    ("detect_image", "Detect Image", "image"),
    ("detect_video", "Detect Video", "videocam"),
    ("species", "Species Explorer", "menu_book"),
    ("models", "Models", "psychology"),
    ("analytics", "Analytics", "bar_chart"),
    ("history", "History", "history"),
    ("about", "About", "info"),
)

#: Sidebar section grouping: (section title, page ids in display order).
#: The ids must match :data:`PAGES`; section headers are rendered by CSS.
NAV_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Overview", ("dashboard",)),
    ("Detection", ("detect_image", "detect_video")),
    ("Explore", ("species", "models", "analytics")),
    ("Activity", ("history",)),
    ("Information", ("about",)),
)

#: Page ids that render the shared Detect workflow and the media type each
#: one preselects. Consumed by the application shell.
DETECT_PAGES: dict[str, str] = {"detect_image": "Image", "detect_video": "Video"}

#: Widget label of the Detect page media selector (CSS hooks the segmented
#: control by this aria-label - keep it in sync with detect.py).
MEDIA_RADIO_LABEL = "Media type"


def _nav_css() -> str:
    """Generate per-item sidebar icon and section-header rules.

    Streamlit radio options cannot carry HTML, so the icon of each item is
    injected as a CSS ``::before`` content (a Material Symbols ligature) and
    each section header as an absolutely positioned ``::after`` above the
    first item of the group. Both are derived from :data:`PAGES` and
    :data:`NAV_GROUPS`, so adding a page here updates the sidebar CSS too.
    """
    selector = 'section[data-testid="stSidebar"] label[data-testid="stRadioOption"]'
    lines: list[str] = []
    for index, (_page_id, _label, icon) in enumerate(PAGES, start=1):
        lines.append(f"{selector}:nth-of-type({index})::before {{ content: \"{icon}\"; }}")

    seen: set[str] = set()
    for index, (page_id, _label, _icon) in enumerate(PAGES, start=1):
        title = next((t for t, ids in NAV_GROUPS if page_id in ids), None)
        if title is None or title in seen:
            continue
        seen.add(title)
        # Reserve room above the item for the floating section header.
        margin = "1.05rem" if index == 1 else "1.5rem"
        lines.append(f"{selector}:nth-of-type({index}) {{ margin-top: {margin}; }}")
        lines.append(f"{selector}:nth-of-type({index})::after {{ content: \"{title}\"; }}")
    return "\n".join(lines)


CSS = """
<style>
/* ------------------------------------------------------------------ */
/* Design tokens                                                        */
/* ------------------------------------------------------------------ */
:root {
  --be-deep:      #041428;
  --be-navy:      #07203c;
  --be-blue:      #0b3a66;
  --be-cyan:      #0891b2;
  --be-cyan-lit:  #22d3ee;
  --be-teal:      #0d9488;
  --be-surface:   #ffffff;
  --be-surface-2: #f2f8fc;
  --be-border:    #dbe7f0;
  --be-text:      #0f2437;
  --be-muted:     #55697d;
  --be-ok:        #0f7b5f;
  --be-ok-bg:     #e3f7f0;
  --be-warn:      #8a5a00;
  --be-warn-bg:   #fdf3dd;
  --be-off:       #6b7b8c;
  --be-off-bg:    #eef3f7;
  --be-danger:    #a3243b;
  --be-danger-bg: #fdeaed;
  --be-radius:    14px;
  --be-radius-sm: 9px;
  --be-shadow:    0 1px 2px rgba(7, 32, 60, .05), 0 10px 26px rgba(7, 32, 60, .07);
}

/* ------------------------------------------------------------------ */
/* Typography                                                           */
/* ------------------------------------------------------------------ */
html, body, [class*="css"], .stApp {
  font-family: "Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
}
.stMarkdown p, .stMarkdown li { color: var(--be-text); }
[data-testid="stMain"] [data-testid="stMarkdown"] h3 {
  font-weight: 650; letter-spacing: -.01em; margin-top: 1.1rem;
}

/* ------------------------------------------------------------------ */
/* Icons (Material Symbols Rounded - bundled with Streamlit)           */
/* ------------------------------------------------------------------ */
.be-ic {
  font-family: "Material Symbols Rounded";
  font-weight: normal;
  font-style: normal;
  font-feature-settings: 'liga';
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
  line-height: 1;
  letter-spacing: normal;
  text-transform: none;
  display: inline-block;
  white-space: nowrap;
  word-wrap: normal;
  direction: ltr;
  -webkit-font-smoothing: antialiased;
  user-select: none;
  vertical-align: -.2em;
  font-size: 1.05rem;
  color: var(--be-cyan);
}
.be-ic--lg  { font-size: 1.4rem; }
.be-ic--xl  { font-size: 2.1rem; color: #0e7ba8; }

/* ------------------------------------------------------------------ */
/* Cards                                                                */
/* ------------------------------------------------------------------ */
.be-card {
  background: var(--be-surface);
  border: 1px solid var(--be-border);
  border-radius: var(--be-radius);
  box-shadow: var(--be-shadow);
  padding: 1.15rem 1.25rem;
  margin-bottom: .6rem;
}
.be-card--flush { padding: .85rem 1rem; }
.be-card--muted { background: var(--be-surface-2); box-shadow: none; }

.be-card-title {
  font-size: .95rem;
  font-weight: 600;
  color: var(--be-text);
  margin: 0 0 .35rem 0;
}
.be-card-title .be-ic { margin-right: .35rem; vertical-align: -.18em; }
.be-card-sub { color: var(--be-muted); font-size: .82rem; margin: 0 0 .6rem 0; }

/* Icon + text block used by model and information cards. */
.be-iblock { display: flex; align-items: flex-start; gap: .75rem; }
.be-iblock .tile {
  flex: none; width: 38px; height: 38px; border-radius: 11px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #e2f4fb, #d5edf9);
  border: 1px solid #c4e4f3;
}
.be-iblock .tile .be-ic { color: #0e7ba8; font-size: 1.3rem; }
.be-iblock .body { min-width: 0; flex: 1; }
.be-iblock .body .t {
  display: flex; align-items: center; flex-wrap: wrap; gap: .5rem;
  font-size: .98rem; font-weight: 650; color: var(--be-text); margin: 0;
}
.be-iblock .body .s { color: var(--be-muted); font-size: .85rem; margin: .12rem 0 0; }
.be-iblock .body .row { display: flex; flex-wrap: wrap; gap: .3rem; align-items: center; margin-top: .5rem; }

/* ------------------------------------------------------------------ */
/* Hero band                                                            */
/* ------------------------------------------------------------------ */
.be-hero {
  background: linear-gradient(115deg, var(--be-deep) 0%, var(--be-navy) 45%, var(--be-blue) 100%);
  border-radius: var(--be-radius);
  padding: 1.6rem 1.7rem;
  margin-bottom: 1rem;
  color: #ffffff;
}
.be-hero h1 { color: #ffffff; margin: 0; font-size: 1.9rem; letter-spacing: .2px; }
.be-hero p  { color: #c9e2f3; margin: .35rem 0 0 0; font-size: .95rem; }
.be-hero .be-tag {
  color: #7fdcf0; font-size: .78rem; letter-spacing: 1.4px;
  text-transform: uppercase; font-weight: 600;
}
.be-hero .be-tag .be-ic { font-size: .95rem; color: #7fdcf0; margin-right: .35rem; vertical-align: -.18em; }

/* ------------------------------------------------------------------ */
/* Metrics                                                              */
/* ------------------------------------------------------------------ */
.be-metric {
  background: var(--be-surface);
  border: 1px solid var(--be-border);
  border-radius: var(--be-radius-sm);
  box-shadow: var(--be-shadow);
  padding: .85rem 1rem;
  height: 100%;
}
.be-metric .k { color: var(--be-muted); font-size: .76rem; text-transform: uppercase; letter-spacing: .6px; }
.be-metric .v { color: var(--be-text); font-size: 1.55rem; font-weight: 650; line-height: 1.25; font-variant-numeric: tabular-nums; }
.be-metric .h { color: var(--be-muted); font-size: .76rem; }
.be-metric--accent { border-left: 4px solid var(--be-cyan); }

/* ------------------------------------------------------------------ */
/* Badges                                                               */
/* ------------------------------------------------------------------ */
.be-badge {
  display: inline-block;
  padding: .18rem .58rem;
  border-radius: 999px;
  font-size: .74rem;
  font-weight: 600;
  line-height: 1.5;
  white-space: nowrap;
}
.be-badge--ok   { background: var(--be-ok-bg);   color: var(--be-ok); }
.be-badge--warn { background: var(--be-warn-bg); color: var(--be-warn); }
.be-badge--off  { background: var(--be-off-bg);  color: var(--be-off); }
.be-badge--info { background: #e2f4fb;           color: #075985; }
.be-badge .be-dot {
  display: inline-block; width: .5em; height: .5em; border-radius: 50%;
  background: currentColor; margin-right: .42rem; vertical-align: .06em;
}

/* ------------------------------------------------------------------ */
/* Key/value rows and chips                                            */
/* ------------------------------------------------------------------ */
.be-kv { display: grid; grid-template-columns: 42% 58%; gap: .28rem .5rem; font-size: .84rem; margin: .3rem 0 .5rem; }
.be-kv dt { color: var(--be-muted); }
.be-kv dd { color: var(--be-text); margin: 0; overflow-wrap: anywhere; }

.be-chip {
  display: inline-block;
  background: var(--be-surface-2);
  border: 1px solid var(--be-border);
  color: var(--be-blue);
  border-radius: 999px;
  padding: .1rem .55rem;
  margin: 0 .25rem .3rem 0;
  font-size: .74rem;
}

/* ------------------------------------------------------------------ */
/* Notices                                                              */
/* ------------------------------------------------------------------ */
.be-notice {
  border-radius: var(--be-radius-sm);
  padding: .8rem 1rem;
  font-size: .88rem;
  border: 1px solid transparent;
}
.be-notice h3 { margin: 0 0 .3rem 0; font-size: .98rem; }
.be-notice h3 .be-ic { font-size: 1.05rem; vertical-align: -.17em; margin-right: .4rem; color: inherit; }
.be-notice p  { margin: 0; }
.be-notice--error   { background: var(--be-danger-bg); border-color: #f3c9d1; color: var(--be-danger); }
.be-notice--warning { background: var(--be-warn-bg);   border-color: #f0ddb0; color: var(--be-warn); }
.be-notice--info    { background: #e2f4fb;             border-color: #c3e6f5; color: #075985; }

/* ------------------------------------------------------------------ */
/* Empty states                                                         */
/* ------------------------------------------------------------------ */
.be-empty { text-align: center; padding: .6rem .5rem .3rem; }
.be-empty .tile {
  width: 64px; height: 64px; border-radius: 18px;
  display: flex; align-items: center; justify-content: center;
  margin: 0 auto .65rem;
  background: linear-gradient(135deg, #e6f5fc, #d7eefa);
  border: 1px solid #c8e6f4;
}
.be-empty .tile .be-ic { font-size: 2rem; color: #0e7ba8; }
.be-empty .be-card-title { font-size: 1.02rem; margin-top: .1rem; }
.be-empty p { max-width: 460px; margin: .1rem auto 0; }

/* ------------------------------------------------------------------ */
/* Sidebar brand                                                        */
/* ------------------------------------------------------------------ */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--be-deep) 0%, var(--be-navy) 100%);
  border-right: 1px solid rgba(255, 255, 255, .06);
}
section[data-testid="stSidebar"] .stMarkdown p { color: #d7e6f2; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #8fb0c9; }

.be-brand { padding: .2rem 0 .35rem 0; }
.be-brand .row { display: flex; align-items: center; gap: .7rem; }
.be-brand .mark {
  flex: none; width: 40px; height: 40px; border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, var(--be-blue) 0%, var(--be-cyan) 100%);
  box-shadow: 0 6px 16px rgba(8, 145, 178, .35);
}
.be-brand .mark .be-ic { color: #ffffff; font-size: 1.45rem; font-variation-settings: 'FILL' 1, 'wght' 500, 'GRAD' 0, 'opsz' 24; }
.be-brand .n { color: #ffffff; font-size: 1.3rem; font-weight: 700; letter-spacing: .3px; margin: 0; line-height: 1.15; }
.be-brand .t { color: #7fdcf0; font-size: .74rem; margin: .05rem 0 0 0; }
.be-brand-rule {
  height: 3px; width: 46px; border-radius: 3px;
  background: linear-gradient(90deg, var(--be-cyan), var(--be-teal));
  margin: .55rem 0 0 0;
}

/* ------------------------------------------------------------------ */
/* Sidebar navigation                                                   */
/* ------------------------------------------------------------------ */
section[data-testid="stSidebar"] [role="radiogroup"] { gap: .06rem; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  position: relative;
  border-radius: 10px;
  padding: .42rem .65rem;
  cursor: pointer;
  transition: background-color .15s ease;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]::before {
  font-family: "Material Symbols Rounded";
  font-weight: normal;
  font-style: normal;
  font-feature-settings: 'liga';
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
  font-size: 1.12rem;
  line-height: 1;
  color: #8fb3cc;
  flex: none;
  margin-right: .65rem;
  transition: color .15s ease;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div {
  margin-left: 0 !important;
  flex: 1 1 auto;
  min-width: 0;
}
/* Hide the native radio circle: the icon is the affordance now. */
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div > div:first-child {
  display: none !important;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {
  color: #cfe2ef; font-size: .93rem; font-weight: 500; margin: 0;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover { background: rgba(255, 255, 255, .06); }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover p { color: #ffffff; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) {
  background: linear-gradient(90deg, rgba(8, 145, 178, .32), rgba(8, 145, 178, .10));
  box-shadow: inset 3px 0 0 var(--be-cyan-lit);
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) p {
  color: #ffffff; font-weight: 600;
}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked)::before {
  color: #67e8f9;
  font-variation-settings: 'FILL' 1, 'wght' 500, 'GRAD' 0, 'opsz' 24;
}
/* Section headers float above the first item of each nav group
   (content is generated from NAV_GROUPS; they are not interactive). */
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]::after {
  content: "";
  position: absolute;
  top: -1.05rem;
  left: .72rem;
  font-size: .62rem;
  font-weight: 700;
  letter-spacing: .13em;
  text-transform: uppercase;
  color: #5d7f9c;
  white-space: nowrap;
  pointer-events: none;
}
.be-side-head {
  display: flex; align-items: center; gap: .45rem;
  color: #7fa3c0; font-size: .76rem; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase;
  margin: .2rem 0 .35rem;
}
.be-side-head .be-ic { color: #67e8f9; font-size: .95rem; }

/* Compact single-line readiness indicator (model detail lives on Models). */
.be-side-status {
  display: flex; align-items: center; gap: .45rem;
  font-size: .84rem; font-weight: 600;
  border-radius: 9px; padding: .4rem .6rem; margin: .2rem 0 .35rem;
}
.be-side-status .be-ic { font-size: 1rem; }
.be-side-status--ok {
  color: #7ff0cf;
  background: rgba(16, 185, 129, .13);
  border: 1px solid rgba(16, 185, 129, .28);
}
.be-side-status--ok .be-ic { color: #7ff0cf; }
.be-side-status--warn {
  color: #ffd9a1;
  background: rgba(234, 179, 8, .13);
  border: 1px solid rgba(234, 179, 8, .3);
}
.be-side-status--warn .be-ic { color: #f6c86a; }

/* ------------------------------------------------------------------ */
/* Segmented control (Detect page media selector)                      */
/* ------------------------------------------------------------------ */
[data-testid="stMain"] [data-testid="stRadioGroup"][aria-orientation="horizontal"] {
  display: inline-flex;
  flex-wrap: wrap;
  gap: .22rem;
  background: var(--be-surface-2);
  border: 1px solid var(--be-border);
  border-radius: 999px;
  padding: .25rem;
}
[data-testid="stMain"] [data-testid="stRadioGroup"][aria-orientation="horizontal"] label[data-testid="stRadioOption"] {
  display: flex;
  align-items: center;
  position: relative;
  border-radius: 999px;
  padding: .4rem 1.05rem;
  cursor: pointer;
  transition: background-color .15s ease, box-shadow .15s ease;
}
[data-testid="stMain"] [data-testid="stRadioGroup"][aria-orientation="horizontal"] label[data-testid="stRadioOption"] > div {
  margin-left: 0 !important;
}
[data-testid="stMain"] [data-testid="stRadioGroup"][aria-orientation="horizontal"] label[data-testid="stRadioOption"] > div > div > div:first-child {
  display: none !important;
}
[data-testid="stMain"] [data-testid="stRadioGroup"][aria-orientation="horizontal"] label[data-testid="stRadioOption"]::before {
  font-family: "Material Symbols Rounded";
  font-weight: normal;
  font-style: normal;
  font-feature-settings: 'liga';
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
  font-size: 1.05rem;
  line-height: 1;
  color: #5b7488;
  margin-right: .42rem;
}
[data-testid="stMain"] [data-testid="stRadioGroup"][aria-orientation="horizontal"] label[data-testid="stRadioOption"] p {
  color: var(--be-muted); font-size: .88rem; font-weight: 600; margin: 0;
}
[data-testid="stMain"] [data-testid="stRadioGroup"][aria-orientation="horizontal"] label[data-testid="stRadioOption"]:hover {
  background: #ffffff;
}
[data-testid="stMain"] [data-testid="stRadioGroup"][aria-orientation="horizontal"] label[data-testid="stRadioOption"]:has(input:checked) {
  background: linear-gradient(135deg, var(--be-blue) 0%, var(--be-cyan) 100%);
  box-shadow: 0 2px 8px rgba(11, 58, 102, .28);
}
[data-testid="stMain"] [data-testid="stRadioGroup"][aria-orientation="horizontal"] label[data-testid="stRadioOption"]:has(input:checked) p {
  color: #ffffff;
}
[data-testid="stMain"] [data-testid="stRadioGroup"][aria-orientation="horizontal"] label[data-testid="stRadioOption"]:has(input:checked)::before {
  color: #a5ecfb;
}
/* Media-type specific glyphs (keep in sync with detect.MEDIA_MODES). */
[data-testid="stMain"] [data-testid="stRadioGroup"][aria-label="MEDIA_RADIO"] label:nth-of-type(1)::before { content: "image"; }
[data-testid="stMain"] [data-testid="stRadioGroup"][aria-label="MEDIA_RADIO"] label:nth-of-type(2)::before { content: "videocam"; }

/* ------------------------------------------------------------------ */
/* File uploader (polished dropzone - widget behaviour unchanged)      */
/* ------------------------------------------------------------------ */
[data-testid="stFileUploader"] { width: 100% !important; }
[data-testid="stFileUploaderDropzone"] {
  border: 2px dashed #a7cfe3 !important;
  background: linear-gradient(180deg, #f7fbfe 0%, #eef6fb 100%) !important;
  border-radius: 16px !important;
  padding: 1.7rem 1.25rem !important;
  transition: border-color .15s ease, background-color .15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--be-cyan) !important;
  background: linear-gradient(180deg, #f2fafd 0%, #e9f5fb 100%) !important;
}
[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] {
  color: #5d748a; font-size: .82rem;
}
[data-testid="stFileUploaderDropzone"] button {
  background: linear-gradient(135deg, var(--be-blue) 0%, var(--be-cyan) 100%) !important;
  border: none !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
  padding: .5rem 1.15rem !important;
  box-shadow: 0 4px 12px rgba(8, 145, 178, .28) !important;
  transition: filter .15s ease, box-shadow .15s ease !important;
}
[data-testid="stFileUploaderDropzone"] button:hover { filter: brightness(1.08); }

/* ------------------------------------------------------------------ */
/* Buttons                                                              */
/* ------------------------------------------------------------------ */
[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, var(--be-blue) 0%, var(--be-cyan) 100%) !important;
  border: 1px solid transparent !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
  box-shadow: 0 6px 16px rgba(8, 145, 178, .25);
  transition: filter .15s ease, box-shadow .15s ease, transform .15s ease;
}
[data-testid="stBaseButton-primary"]:hover {
  filter: brightness(1.07);
  box-shadow: 0 8px 20px rgba(8, 145, 178, .33);
  transform: translateY(-1px);
  border-color: transparent !important;
}
[data-testid="stBaseButton-primary"]:active { transform: translateY(0); }
[data-testid="stBaseButton-secondary"] {
  border: 1px solid var(--be-border) !important;
  background: #ffffff !important;
  color: var(--be-blue) !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
}
[data-testid="stBaseButton-secondary"]:hover {
  border-color: #a9cbdf !important;
  background: var(--be-surface-2) !important;
}

/* ------------------------------------------------------------------ */
/* Steps                                                                */
/* ------------------------------------------------------------------ */
.be-step { display: flex; align-items: center; gap: .55rem; margin: .1rem 0 .5rem; }
.be-step .n {
  background: linear-gradient(135deg, var(--be-blue), var(--be-cyan));
  color: #fff; border-radius: 999px;
  width: 22px; height: 22px; display: inline-flex; align-items: center;
  justify-content: center; font-size: .74rem; font-weight: 700; flex: none;
}
.be-step .t { font-size: .95rem; font-weight: 600; color: var(--be-text); }

/* ------------------------------------------------------------------ */
/* Content grids and workflow flow (About page)                        */
/* ------------------------------------------------------------------ */
.be-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
  gap: .7rem;
  margin-bottom: .6rem;
}
.be-tile {
  background: var(--be-surface);
  border: 1px solid var(--be-border);
  border-radius: var(--be-radius-sm);
  box-shadow: var(--be-shadow);
  padding: .95rem 1.05rem;
}
.be-tile .ic {
  width: 34px; height: 34px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, var(--be-blue) 0%, var(--be-cyan) 100%);
  margin-bottom: .6rem;
}
.be-tile .ic .be-ic { color: #ffffff; font-size: 1.15rem; }
.be-tile h4 { margin: 0 0 .18rem 0; font-size: .95rem; color: var(--be-text); }
.be-tile p { margin: 0; color: var(--be-muted); font-size: .82rem; }
.be-tile ul { margin: .3rem 0 0; padding-left: 1.05rem; }
.be-tile li { color: var(--be-muted); font-size: .82rem; margin-bottom: .2rem; }

.be-flow { display: flex; flex-wrap: wrap; align-items: stretch; gap: .5rem; margin: .3rem 0 .7rem; }
.be-flow .fstep {
  display: flex; align-items: center; gap: .55rem;
  background: var(--be-surface);
  border: 1px solid var(--be-border);
  border-radius: 10px;
  box-shadow: var(--be-shadow);
  padding: .55rem .8rem;
  max-width: 100%;
}
.be-flow .fstep .txt { font-size: .86rem; font-weight: 600; color: var(--be-text); }
.be-flow .farr { align-self: center; flex: none; }
.be-flow .farr .be-ic { color: #9db4c6; font-size: 1rem; }

/* ------------------------------------------------------------------ */
/* Media previews                                                       */
/* ------------------------------------------------------------------ */
[data-testid="stImage"] img {
  border-radius: 12px;
  border: 1px solid var(--be-border);
  box-shadow: var(--be-shadow);
}
[data-testid="stVideo"] video {
  border-radius: 12px;
  border: 1px solid var(--be-border);
  background: var(--be-deep);
}

/* ------------------------------------------------------------------ */
/* Tables                                                               */
/* ------------------------------------------------------------------ */
[data-testid="stDataFrame"] div[data-testid="stHorizontalScrollbar"] { visibility: hidden; }

/* ------------------------------------------------------------------ */
/* Responsive                                                           */
/* ------------------------------------------------------------------ */
@media (max-width: 900px) {
  .be-hero { padding: 1.1rem 1.15rem; }
  .be-hero h1 { font-size: 1.5rem; }
  .be-metric .v { font-size: 1.3rem; }
  .be-card { padding: .9rem 1rem; }
  [data-testid="stFileUploaderDropzone"] { padding: 1.2rem .9rem !important; }
  .be-grid { grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
  .be-flow .fstep { flex: 1 1 100%; }
  .be-flow .farr { display: none; }
}
</style>
"""

#: CSS with generated navigation rules substituted in.
CSS = CSS.replace(
    'aria-label="MEDIA_RADIO"',
    f'aria-label="{MEDIA_RADIO_LABEL}"',
).replace(
    "</style>",
    _nav_css() + "\n</style>",
)


def apply_theme() -> None:
    """Inject the BlueEye design system (call once per page render)."""
    st.markdown(CSS, unsafe_allow_html=True)


def esc(value: object) -> str:
    """HTML-escape a value for safe interpolation into templates."""
    return html.escape(str(value))
