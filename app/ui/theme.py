"""BlueEye visual design system (marine theme).

All styling lives here so that page code stays free of CSS and the look
remains consistent across every page (single source of truth for colours,
radii, shadows and spacing). The palette is a deliberate deep-ocean marine
intelligence look: near-black navy page, ocean-blue panels, cyan/teal
accents and light text - premium for a demo, legible for a scientific tool.

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
#: There is exactly **one** detection entry point: the unified ``detect``
#: page (labelled "AI Detection") handles both image and video analysis
#: through its own media selector. The old ``detect_image`` /
#: ``detect_video`` ids are kept in :data:`DETECT_PAGES` purely as a legacy
#: redirect: if anything still asks for them, the shell opens ``detect`` with
#: the matching media type preselected.
PAGES: tuple[tuple[str, str, str], ...] = (
    ("dashboard", "Dashboard", "home"),
    ("detect", "AI Detection", "center_focus_strong"),
    ("observations", "My Observations", "bookmarks"),
    ("history", "Detection History", "history"),
    ("species", "Species Explorer", "menu_book"),
    ("models", "AI Models", "psychology"),
    ("analytics", "Analytics", "bar_chart"),
    ("compare", "Model Comparison", "compare_arrows"),
    ("evaluation", "Research Evaluation", "science"),
    ("about", "About BlueEye", "info"),
    ("help", "Help & Documentation", "help"),
    ("account", "Account", "account_circle"),
)

#: Sidebar section grouping: (section title, page ids in display order).
#: The ids must match :data:`PAGES`; section headers are rendered by CSS.
NAV_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Main", ("dashboard", "detect", "observations")),
    ("Explore", ("history", "species", "models", "analytics")),
    ("Research", ("compare", "evaluation")),
    ("Information", ("about", "help")),
    ("Account", ("account",)),
)

#: Legacy page ids that resolve to the unified Detect workflow, mapped to the
#: media type they preselect. Consumed by the application shell when a stale
#: ``nav_sel`` or ``?page=`` deep link still names one of them.
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
  --be-deep:      #030b18;
  --be-navy:      #061a30;
  --be-blue:      #0b3a66;
  --be-cyan:      #0891b2;
  --be-cyan-lit:  #22d3ee;
  --be-teal:      #0d9488;
  --be-surface:   #0b223d;
  --be-surface-2: #0f2c4e;
  --be-border:    rgba(125, 195, 235, .16);
  --be-text:      #e9f4fb;
  --be-muted:     #8fb0c9;
  --be-ok:        #5eead4;
  --be-ok-bg:     rgba(20, 184, 166, .16);
  --be-warn:      #fcd34d;
  --be-warn-bg:   rgba(245, 158, 11, .15);
  --be-off:       #c3d4e2;
  --be-off-bg:    rgba(148, 163, 184, .15);
  --be-danger:    #fda4af;
  --be-danger-bg: rgba(225, 29, 72, .16);
  --be-radius:    14px;
  --be-radius-sm: 9px;
  --be-shadow:    0 1px 2px rgba(0, 0, 0, .3), 0 12px 28px rgba(0, 0, 0, .35);
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
.be-ic--xl  { font-size: 2.1rem; color: #7dd3fc; }

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
  background: linear-gradient(135deg, rgba(34, 211, 238, .16), rgba(34, 211, 238, .05));
  border: 1px solid rgba(34, 211, 238, .22);
}
.be-iblock .tile .be-ic { color: #7dd3fc; font-size: 1.3rem; }
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
.be-badge--info { background: rgba(34, 211, 238, .14); color: #7dd3fc; }
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
  background: rgba(255, 255, 255, .05);
  border: 1px solid var(--be-border);
  color: #a5d8ea;
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
.be-notice--error   { background: var(--be-danger-bg); border-color: rgba(244, 63, 94, .3);  color: var(--be-danger); }
.be-notice--warning { background: var(--be-warn-bg);   border-color: rgba(245, 158, 11, .3); color: var(--be-warn); }
.be-notice--info    { background: rgba(14, 116, 144, .18); border-color: rgba(34, 211, 238, .25); color: #7fe0f2; }

/* ------------------------------------------------------------------ */
/* Empty states                                                         */
/* ------------------------------------------------------------------ */
.be-empty { text-align: center; padding: .6rem .5rem .3rem; }
.be-empty .tile {
  width: 64px; height: 64px; border-radius: 18px;
  display: flex; align-items: center; justify-content: center;
  margin: 0 auto .65rem;
  background: linear-gradient(135deg, rgba(34, 211, 238, .16), rgba(34, 211, 238, .05));
  border: 1px solid rgba(34, 211, 238, .22);
}
.be-empty .tile .be-ic { font-size: 2rem; color: #7dd3fc; }
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

/* ------------------------------------------------------------------ */
/* Sidebar balance (brand -> navigation only)                          */
/* ------------------------------------------------------------------ */
.be-brand { margin-bottom: .35rem; }

/* ------------------------------------------------------------------ */
/* Landing page (dashboard)                                            */
/* ------------------------------------------------------------------ */
.be-section-lead {
  color: var(--be-muted);
  font-size: .9rem;
  margin: -.35rem 0 .9rem;
}

/* Action card used by the dashboard navigation grid. */
.be-nav-card {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 200px;
  background: var(--be-surface);
  border: 1px solid var(--be-border);
  border-radius: var(--be-radius);
  box-shadow: var(--be-shadow);
  padding: 1.05rem 1.1rem;
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.be-nav-card:hover {
  transform: translateY(-2px);
  border-color: #2c6f96;
  box-shadow: 0 8px 22px rgba(0, 0, 0, .35), 0 18px 40px rgba(0, 0, 0, .35);
}
.be-nav-card .ic {
  width: 42px; height: 42px; border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, rgba(34, 211, 238, .16), rgba(34, 211, 238, .05));
  border: 1px solid rgba(34, 211, 238, .22);
  margin-bottom: .65rem;
}
.be-nav-card .ic .be-ic { color: #7dd3fc; font-size: 1.35rem; }
.be-nav-card h4 { margin: 0 0 .3rem 0; font-size: 1rem; color: var(--be-text); }
.be-nav-card p { margin: 0; color: var(--be-muted); font-size: .83rem; line-height: 1.45; }
.be-nav-card .be-ic--inline { color: var(--be-cyan); }

/* Species preview card (dashboard -> Species Explorer). */
.be-species-card { min-height: 168px; display: flex; flex-direction: column; }
.be-species-card .be-card-title { line-height: 1.3; }
.be-species-card .be-chip { margin-top: .15rem; }

/* Species photographs with a graceful icon fallback. A broken image URL
   reveals the icon tile instead of the browser's broken-image glyph. */
.be-photo {
  position: relative;
  aspect-ratio: 4 / 3;
  border-radius: var(--be-radius-sm);
  overflow: hidden;
  background: linear-gradient(135deg, rgba(11, 58, 102, .92), rgba(8, 32, 54, .96));
  border: 1px solid var(--be-border);
  margin-bottom: .55rem;
}
.be-photo img {
  position: absolute; inset: 0;
  width: 100%; height: 100%;
  object-fit: cover;
  display: block;
}
.be-photo-fallback {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  color: #7fdcf0;
}
.be-photo-fallback .be-ic { font-size: 2.1rem; }
.be-photo--missing img { display: none; }
.be-photo:not(.be-photo--missing) .be-photo-fallback { display: none; }
.be-photo--wide { aspect-ratio: 21 / 9; margin-bottom: .35rem; }

/* Scientific name shown under the main heading in a detail hero. */
.be-hero .sci { color: #a9d6ec; font-size: 1.02rem; margin-top: .15rem; }

/* Species detail lists (facts / verified references). */
.be-facts, .be-reflist { margin: .1rem 0 .6rem; padding-left: 1.15rem; }
.be-facts li, .be-reflist li { color: var(--be-text); font-size: .88rem; margin-bottom: .3rem; }
.be-reflist a { color: #67e8f9; text-decoration: none; border-bottom: 1px solid rgba(103, 232, 249, .35); }
.be-reflist a:hover { color: var(--be-cyan); border-bottom-color: var(--be-cyan); }

/* Subtle responsible-use band. */
.be-responsible {
  display: flex; align-items: flex-start; gap: .7rem;
  background: var(--be-surface-2);
  border: 1px solid var(--be-border);
  border-left: 3px solid var(--be-cyan);
  border-radius: var(--be-radius-sm);
  padding: .85rem 1rem;
  font-size: .86rem; color: var(--be-muted);
}
.be-responsible .be-ic { color: var(--be-cyan); font-size: 1.2rem; flex: none; }
.be-responsible b { color: var(--be-text); }

/* ------------------------------------------------------------------ */
/* Dashboard (product landing page)                                    */
/* ------------------------------------------------------------------ */
/* Status row inside the dashboard header band (real readiness + device). */
.be-hero .be-statusrow {
  display: flex; flex-wrap: wrap; align-items: center; gap: .5rem;
  margin-top: .85rem;
}
.be-hero .be-statusrow .dev { color: #9fc3da; font-size: .8rem; }

/* Short muted lead under a section heading. */
.be-dash-lead { color: var(--be-muted); font-size: .9rem; margin: -.2rem 0 .85rem; }

/* Responsive statistic grid for the dashboard summary cards. */
.be-statgrid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
  gap: .7rem;
  margin-bottom: .35rem;
}
.be-statgrid .be-metric { margin: 0; }

/* Primary action panels (Detect an Image / Analyse a Video). */
.be-action {
  display: flex; flex-direction: column; height: 100%;
  background: var(--be-surface);
  border: 1px solid var(--be-border);
  border-top: 3px solid var(--be-cyan);
  border-radius: var(--be-radius);
  box-shadow: var(--be-shadow);
  padding: 1.1rem 1.15rem .95rem;
}
.be-action .ic {
  width: 44px; height: 44px; border-radius: 13px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, var(--be-blue) 0%, var(--be-cyan) 100%);
  margin-bottom: .7rem;
}
.be-action .ic .be-ic { color: #ffffff; font-size: 1.4rem; }
.be-action h4 { margin: 0 0 .3rem 0; font-size: 1.05rem; color: var(--be-text); }
.be-action p { margin: 0 0 .85rem 0; color: var(--be-muted); font-size: .86rem; line-height: 1.5; }
.be-action .spacer { flex: 1 1 auto; }

/* Recent-activity feed rows. */
.be-feed { display: flex; flex-direction: column; gap: .5rem; }
.be-feed .item {
  display: flex; align-items: flex-start; gap: .7rem;
  background: var(--be-surface);
  border: 1px solid var(--be-border);
  border-radius: var(--be-radius-sm);
  padding: .6rem .75rem;
}
.be-feed .item .tile {
  flex: none; width: 34px; height: 34px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, rgba(34, 211, 238, .16), rgba(34, 211, 238, .05));
  border: 1px solid rgba(34, 211, 238, .22);
}
.be-feed .item .tile .be-ic { color: #7dd3fc; font-size: 1.15rem; }
.be-feed .item .body { min-width: 0; flex: 1; }
.be-feed .item .fn { font-size: .9rem; font-weight: 600; color: var(--be-text); overflow-wrap: anywhere; }
.be-feed .item .meta { color: var(--be-muted); font-size: .78rem; margin-top: .1rem; }
.be-feed .item .species { margin-top: .3rem; }
.be-feed .item .st { flex: none; }

/* Model-status rows (dashboard right column). */
.be-modelrow {
  display: flex; align-items: center; gap: .55rem;
  padding: .48rem .1rem;
  border-bottom: 1px dashed var(--be-border);
}
.be-modelrow:last-child { border-bottom: none; }
.be-modelrow .nm { flex: 1 1 auto; min-width: 0; font-size: .9rem; font-weight: 600; color: var(--be-text); overflow-wrap: anywhere; }
.be-modelrow .nm .sub { display: block; color: var(--be-muted); font-size: .76rem; font-weight: 400; }

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
  background: rgba(255, 255, 255, .07);
}
[data-testid="stMain"] [data-testid="stRadioGroup"][aria-orientation="horizontal"] label[data-testid="stRadioOption"]:has(input:checked) {
  background: linear-gradient(135deg, var(--be-blue) 0%, var(--be-cyan) 100%);
  box-shadow: 0 2px 10px rgba(8, 145, 178, .4);
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
  border: 2px dashed rgba(125, 195, 235, .35) !important;
  background: linear-gradient(180deg, rgba(13, 36, 62, .85) 0%, rgba(9, 26, 45, .85) 100%) !important;
  border-radius: 16px !important;
  padding: 1.7rem 1.25rem !important;
  transition: border-color .15s ease, background-color .15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--be-cyan-lit) !important;
  background: linear-gradient(180deg, rgba(15, 44, 78, .9) 0%, rgba(10, 30, 52, .9) 100%) !important;
}
[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] {
  color: #9fc3da; font-size: .82rem;
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
  background: rgba(255, 255, 255, .05) !important;
  color: var(--be-text) !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
}
[data-testid="stBaseButton-secondary"]:hover {
  border-color: #2c6f96 !important;
  background: rgba(255, 255, 255, .09) !important;
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
.be-flow .fstep .n {
  flex: none; width: 22px; height: 22px; border-radius: 999px;
  display: inline-flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, var(--be-blue), var(--be-cyan));
  color: #ffffff; font-size: .72rem; font-weight: 700;
}
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
/* Dashboard - cinematic hero band                                      */
/* ------------------------------------------------------------------ */
.be-mhero {
  position: relative;
  overflow: hidden;
  border-radius: var(--be-radius);
  border: 1px solid rgba(125, 195, 235, .18);
  box-shadow: var(--be-shadow);
  min-height: 330px;
  padding: 2.6rem 2.2rem 2.1rem;
  margin-bottom: .4rem;
  background-size: cover;
  background-position: center 28%;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
}
.be-mhero .eyebrow {
  display: inline-flex; align-items: center; gap: .45rem;
  color: #67e8f9; font-size: .78rem; font-weight: 600;
  letter-spacing: .24em; text-transform: uppercase; margin-bottom: .5rem;
}
.be-mhero .eyebrow .be-ic { font-size: 1rem; }
.be-mhero h1 {
  color: #ffffff; margin: 0 0 .4rem 0;
  font-size: 2.55rem; font-weight: 750; line-height: 1.08;
  letter-spacing: .4px; text-shadow: 0 2px 20px rgba(0, 0, 0, .45);
}
.be-mhero p {
  color: #d3eaf8; font-size: 1.02rem; line-height: 1.5;
  max-width: 640px; margin: 0 0 1.25rem 0;
}
.be-cta {
  display: inline-flex; align-items: center; gap: .55rem;
  width: fit-content;
  background: linear-gradient(135deg, var(--be-blue) 0%, var(--be-cyan) 100%);
  border: 1px solid rgba(103, 232, 249, .35);
  color: #ffffff; font-weight: 600; font-size: .95rem; text-decoration: none;
  padding: .72rem 1.45rem; border-radius: 12px;
  box-shadow: 0 8px 22px rgba(8, 145, 178, .4);
  transition: filter .15s ease, transform .15s ease, box-shadow .15s ease;
}
.be-cta:hover { filter: brightness(1.08); transform: translateY(-1px); }
.be-cta .arr { font-size: 1.1rem; line-height: 1; }

/* Dashboard - marine statistic cards */
.be-mstat {
  display: flex; align-items: center; gap: .85rem;
  background: var(--be-surface);
  border: 1px solid var(--be-border);
  border-radius: var(--be-radius-sm);
  box-shadow: var(--be-shadow);
  padding: .95rem 1rem;
  height: 100%;
}
.be-mstat .ic {
  flex: none; width: 46px; height: 46px; border-radius: 13px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, rgba(11, 58, 102, .7), rgba(8, 145, 178, .35));
  border: 1px solid rgba(34, 211, 238, .25);
}
.be-mstat .ic .be-ic { color: #67e8f9; font-size: 1.5rem; }
.be-mstat .bd { min-width: 0; }
.be-mstat .v {
  font-size: 1.6rem; font-weight: 700; line-height: 1.2;
  color: var(--be-text); font-variant-numeric: tabular-nums;
}
.be-mstat .k {
  color: var(--be-muted); font-size: .76rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .6px;
}
.be-mstat .h { color: var(--be-muted); font-size: .74rem; margin-top: .1rem; }
.be-mstatgrid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: .7rem;
  margin-bottom: .95rem;
}

/* Dashboard - latest detection comparison */
.be-cmp {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) minmax(0, 1.02fr);
  gap: .85rem;
  align-items: stretch;
  margin-bottom: .5rem;
}
.be-cmp .panel {
  background: var(--be-surface);
  border: 1px solid var(--be-border);
  border-radius: var(--be-radius-sm);
  box-shadow: var(--be-shadow);
  overflow: hidden;
  display: flex; flex-direction: column;
}
.be-cmp .panel .media {
  flex: 1 1 auto;
  display: flex; align-items: center; justify-content: center;
  background: radial-gradient(circle at 70% 20%, #0a243e 0%, #061423 70%);
  min-height: 200px;
}
.be-cmp .panel .media img { display: block; width: 100%; height: auto; }
.be-cmp .panel .media .ph { color: var(--be-muted); font-size: .86rem; padding: 1rem; text-align: center; }
.be-cmp .panel .lbl {
  display: flex; align-items: center; gap: .4rem;
  padding: .5rem .8rem;
  border-top: 1px solid var(--be-border);
  background: rgba(255, 255, 255, .025);
  font-size: .74rem; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: var(--be-muted);
}
.be-cmp .panel .lbl .be-ic { font-size: 1rem; color: #7dd3fc; }
.be-cmp .arrow { display: flex; align-items: center; justify-content: center; }
.be-cmp .arrow .be-ic { font-size: 1.7rem; color: #67e8f9; }
.be-cmp .panel--c .detail { padding: .8rem 1rem; }
.be-cmp .panel--c .detail .be-kv { margin-top: .15rem; }
.be-cmp .panel--c .detail .be-link { display: inline-flex; align-items: center; gap: .3rem; margin-top: .5rem; }
.be-cmp .panel--c .lbl { border-top: none; border-bottom: 1px solid var(--be-border); }
.be-cmp .panel--span { grid-column: 1 / -1; }

/* Dashboard - text links & small "view" links */
.be-link {
  color: #67e8f9; font-size: .85rem; font-weight: 600; text-decoration: none;
  border-bottom: 1px solid rgba(103, 232, 249, .4);
}
.be-link:hover { border-bottom-color: transparent; }
.be-view {
  color: #67e8f9; font-size: .8rem; font-weight: 600; text-decoration: none;
  white-space: nowrap; margin-left: .4rem;
}
.be-view:hover { text-decoration: underline; }

/* Dashboard - Indian marine biodiversity */
.be-bio-featured {
  display: flex; align-items: flex-start; gap: 1.05rem;
  background: var(--be-surface);
  border: 1px solid var(--be-border);
  border-left: 4px solid var(--be-cyan);
  border-radius: var(--be-radius);
  box-shadow: var(--be-shadow);
  padding: 1.05rem 1.2rem;
  margin-bottom: .7rem;
}
.be-bio-featured .photo {
  flex: none; width: 88px; height: 88px; border-radius: 18px;
  display: flex; align-items: center; justify-content: center;
  background: radial-gradient(circle at 30% 25%, rgba(34, 211, 238, .28), rgba(11, 58, 102, .55));
  border: 1px solid rgba(34, 211, 238, .3);
}
/* Let the text column shrink and wrap rather than overflow the card. */
.be-bio-featured > div:not(.photo) { min-width: 0; }
.be-bio-featured .photo .be-ic { font-size: 2.7rem; color: #a5ecfb; }
.be-bio-featured h4 { margin: 0 0 .12rem 0; font-size: 1.14rem; color: var(--be-text); }
.be-bio-featured .sci { color: #7dd3fc; font-size: .9rem; font-style: italic; margin: 0 0 .35rem 0; }
.be-bio-featured .meta { color: var(--be-muted); font-size: .84rem; line-height: 1.5; margin: .2rem 0 .55rem; }
.be-bio-featured .badges { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .6rem; }
.be-bio-mini {
  display: flex; flex-direction: column; gap: .3rem;
  background: var(--be-surface);
  border: 1px solid var(--be-border);
  border-radius: var(--be-radius-sm);
  box-shadow: var(--be-shadow);
  padding: .85rem .95rem;
  height: 100%;
}
.be-bio-mini .row { display: flex; align-items: center; gap: .6rem; }
.be-bio-mini .row .ic {
  flex: none; width: 36px; height: 36px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, rgba(11, 58, 102, .7), rgba(8, 145, 178, .35));
  border: 1px solid rgba(34, 211, 238, .22);
}
.be-bio-mini .row .ic .be-ic { color: #7dd3fc; font-size: 1.15rem; }
.be-bio-mini h4 { margin: 0; font-size: .95rem; line-height: 1.25; color: var(--be-text); }
.be-bio-mini .sci { color: var(--be-muted); font-size: .78rem; font-style: italic; margin: 0; }
.be-bio-mini .st { margin-top: auto; padding-top: .35rem; }
.be-biogrid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: .7rem;
  margin-bottom: .95rem;
}

/* ------------------------------------------------------------------ */
/* Authentication gate + account page                                  */
/* ------------------------------------------------------------------ */
.be-auth { text-align: center; margin: 2.2rem 0 .6rem; }
.be-auth .be-auth-mark {
  width: 64px; height: 64px; margin: 0 auto .65rem;
  display: flex; align-items: center; justify-content: center;
  border-radius: 18px; background: linear-gradient(135deg, #0b3a66, #0891b2);
  border: 1px solid rgba(34, 211, 238, .35);
  box-shadow: var(--be-shadow);
}
.be-auth .be-auth-mark .be-ic { color: #cffafe; font-size: 2.2rem; }
.be-auth h1 { margin: 0; font-size: 1.9rem; letter-spacing: -.02em; color: var(--be-text); }
.be-auth .be-auth-sub { color: var(--be-muted); margin: .3rem 0 0; font-size: .95rem; }
.be-auth-title {
  display: flex; align-items: center; gap: .55rem;
  font-weight: 650; font-size: 1.05rem; color: var(--be-text); margin-bottom: .4rem;
}
.be-auth-title .be-ic { color: #67e8f9; font-size: 1.25rem; }
.be-value { font-size: 1.05rem; color: var(--be-text); margin: .15rem 0 0; }
.be-side-acct {
  margin: 1rem 0 .2rem; padding: .7rem .8rem;
  background: var(--be-surface); border: 1px solid var(--be-border);
  border-radius: var(--be-radius-sm);
}
.be-side-acct-head { display: flex; align-items: center; gap: .6rem; }
.be-side-acct-head .be-ic { color: #67e8f9; font-size: 1.6rem; }
.be-side-acct-head .n { margin: 0; color: var(--be-text); font-weight: 600; font-size: .9rem; overflow-wrap: anywhere; }
.be-side-acct-head .e { margin: 0; color: var(--be-muted); font-size: .76rem; overflow-wrap: anywhere; }

/* Account page identity banner + aligned summary cards. */
.be-identity {
  display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
  background: var(--be-surface); border: 1px solid var(--be-border);
  border-radius: var(--be-radius); box-shadow: var(--be-shadow);
  padding: 1.1rem 1.25rem; margin-bottom: .8rem;
}
.be-identity .avatar {
  flex: none; width: 58px; height: 58px; border-radius: 16px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #0b3a66, #0891b2);
  border: 1px solid rgba(34, 211, 238, .35);
}
.be-identity .avatar .be-ic { font-size: 2rem; color: #cffafe; }
.be-identity .who { min-width: 0; flex: 1 1 auto; }
.be-identity .who h4 { margin: 0; font-size: 1.18rem; color: var(--be-text); overflow-wrap: anywhere; }
.be-identity .who p { margin: .12rem 0 0; color: var(--be-muted); font-size: .9rem; overflow-wrap: anywhere; }
.be-identity .state { flex: none; }
.be-infogrid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem;
  margin-bottom: .8rem;
}
.be-infogrid .be-card { height: 100%; margin-bottom: 0; }
.be-infogrid .be-card .be-value { margin-top: .1rem; }

/* ------------------------------------------------------------------ */
/* Responsive                                                           */
/* ------------------------------------------------------------------ */
/* Comparison and stat grids adapt to medium screens first. */
@media (max-width: 1300px) {
  .be-cmp {
    grid-template-columns: 1fr;
    gap: .7rem;
  }
  .be-cmp .arrow { display: none; }
  .be-mstatgrid, .be-biogrid { grid-template-columns: repeat(2, 1fr); }
  .be-infogrid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 900px) {
  .be-hero { padding: 1.1rem 1.15rem; }
  .be-hero h1 { font-size: 1.5rem; }
  .be-metric .v { font-size: 1.3rem; }
  .be-card { padding: .9rem 1rem; }
  .be-nav-card { min-height: 0; }
  .be-species-card { min-height: 0; }
  .be-statgrid { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
  [data-testid="stFileUploaderDropzone"] { padding: 1.2rem .9rem !important; }
  .be-grid { grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
  .be-flow .fstep { flex: 1 1 100%; }
  .be-flow .farr { display: none; }

  /* Marine hero tightens up on small screens. */
  .be-mhero { min-height: 0; padding: 1.7rem 1.3rem 1.4rem; }
  .be-mhero h1 { font-size: 1.85rem; }
  .be-mhero p { font-size: .93rem; }
  .be-bio-featured { flex-direction: column; }
  .be-bio-featured .photo { width: 64px; height: 64px; }
  .be-bio-featured .photo .be-ic { font-size: 2rem; }

  /* Stack main-area columns so cards, the hero and the workflow stay usable
     on narrow screens instead of being squeezed side by side. */
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    flex: 1 1 100% !important;
    width: 100% !important;
    min-width: 0 !important;
  }
}

@media (max-width: 640px) {
  .be-mstatgrid, .be-biogrid, .be-infogrid { grid-template-columns: 1fr; }
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
