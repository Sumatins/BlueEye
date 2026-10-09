"""About page: the BlueEye project - purpose, objective, workflow,
technology, models, future scope and open-source attribution.

The narrative follows the BlueEye project report (introduction, project
foundation, implementation and future scope). Claims about the running
system come from the live registry and the actual pipeline; nothing is
invented. Attribution to Orange OpenSource's marine-detect is preserved
in full.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import streamlit as st

from app import __version__
from app.detection.detector import MarineDetector
from app.ui import state
from app.ui.components import badge, chips, hero, metrics_row, notice
from app.ui.icons import span
from app.ui.theme import esc

UPSTREAM = "https://github.com/Orange-OpenSource/marine-detect"

#: Datasets referenced by the project report (for future training work).
REPORT_DATASETS: tuple[tuple[str, str], ...] = (
    ("Ticon shark dataset", "https://universe.roboflow.com/ticon-dataset/shark-ibmby"),
    ("Shark species dataset", "https://universe.roboflow.com/rizal-fadia-al-fikri/shark_species"),
    ("Zebra shark dataset", "https://universe.roboflow.com/minhajul-arefin/zebra_shark"),
    ("Fish dataset", "https://universe.roboflow.com/roboflow-gw7yv/fish-yzfml"),
    ("Count-a-manta dataset", "https://universe.roboflow.com/le-wagon-w02yl/count-a-manta"),
    ("Oz Fish dataset (AIMS/UWA/Curtin)", "https://doi.org/10.25845/5e28f062c5097"),
    ("DePondFi / Orange Chromide dataset (CC BY 4.0)", "https://data.mendeley.com/datasets/7w45jx35hd/1"),
    ("Underwater Species Dataset NR (CC BY 4.0)", "https://data.mendeley.com/datasets/4tp83br92z/1"),
)


def render(detector: MarineDetector, settings) -> None:
    """Render the project description and attribution."""
    hero(
        st,
        "About the project",
        "BlueEye",
        "AI-Powered Marine Life Detection - a deep-learning system that "
        "identifies marine organisms in underwater images and videos.",
        icon="info",
    )

    metrics_row(
        st,
        [
            ("Version", __version__, "BlueEye release"),
            ("Detector", "YOLOv8", "Ultralytics / PyTorch"),
            ("Models", len(state.registry(detector)), "in the registry"),
            ("Device", str(detector.device).upper(), "auto-selected"),
        ],
    )
    st.write("")

    _why()
    _objective()
    _workflow()
    _technology()
    _capabilities()
    _models(detector)
    _limitations()
    _future()
    _attribution()


# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #
def _tile(icon: str, title: str, body: str) -> str:
    """One labelled icon tile for the content grids."""
    return (
        '<div class="be-tile">'
        f'<div class="ic">{span(icon)}</div>'
        f"<h4>{esc(title)}</h4><p>{esc(body)}</p></div>"
    )


def _flow_step(icon: str, text: str) -> str:
    """One step of the 'How BlueEye works' workflow strip."""
    return f'<div class="fstep">{span(icon)}<span class="txt">{esc(text)}</span></div>'


def _flow_arrow() -> str:
    return f'<div class="farr">{span("arrow_forward")}</div>'


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _why() -> None:
    st.markdown("### Why BlueEye?")
    st.markdown(
        """
The oceans cover more than 70% of the Earth's surface and host an
extraordinary range of life, from microscopic plankton to large marine
mammals. Monitoring those ecosystems is inherently difficult: the vastness
of the ocean, limited accessibility and complex underwater conditions make
traditional identification - manual observation by experts - slow, costly
and prone to human error.

Marine ecosystems also face growing pressure from climate change,
pollution, overfishing and habitat destruction, which makes continuous,
large-scale monitoring increasingly important. BlueEye applies computer
vision and deep learning to assist with:
"""
    )
    st.markdown(
        """
- **Marine biodiversity monitoring** - turning raw underwater imagery into
  structured observations that can be reviewed over time.
- **Conservation** - repeatable automated analysis for reef and habitat
  surveys.
- **Research** - a reproducible, scriptable detection pipeline (web UI,
  CLI and API over the same code).
- **Fisheries support** - automated species identification that can assist
  sustainable fisheries management.
- **Automated species identification** - replacing slow manual review with
  model-assisted labelling.
"""
    )


def _objective() -> None:
    st.markdown("### Project objective")
    st.markdown(
        """
The objective of BlueEye is to automatically recognize and classify marine
life from underwater images using deep learning - improving on the speed
and cost of manual methods while dealing with the practical difficulties
of underwater imagery. The system is intended to support large-scale
marine ecosystem monitoring, assist in the identification of endangered
species and help track biodiversity.
"""
    )
    st.markdown(
        chips(
            (
                "Noise handling",
                "Low visibility",
                "Color distortion",
                "Large-scale monitoring",
                "Endangered-species identification",
                "Biodiversity tracking",
            )
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        "Aimed at research, fisheries management and environmental "
        "conservation while contributing to the advancement of marine science."
    )


def _workflow() -> None:
    st.markdown("### How BlueEye works")
    steps = (
        ("upload", "Upload"),
        ("tune", "Image / video processing"),
        ("center_focus_strong", "YOLOv8 detection"),
        ("label", "Species identification"),
        ("grid_on", "Boxes + confidence"),
        ("photo_library", "Annotated result"),
    )
    parts: list[str] = []
    for index, (icon, text) in enumerate(steps):
        if index:
            parts.append(_flow_arrow())
        parts.append(_flow_step(icon, text))
    st.markdown(f'<div class="be-flow">{"".join(parts)}</div>', unsafe_allow_html=True)
    st.markdown(
        """
Input files are validated first (extension, size, decodability), then -
optionally - passed through underwater enhancement (colour correction,
CLAHE contrast, denoising; off by default, as it is not guaranteed to
improve accuracy). The selected YOLOv8 model runs on the CPU or an NVIDIA
GPU, boxes and labels with confidence scores are drawn around every
detection, and the run is written to `outputs/` as an annotated
image/video plus a JSON report.

Videos are processed frame by frame and re-encoded as H.264 for browser
playback. The same pipeline serves this interface, the CLI
(`python -m app.main`) and the `scripts/` utilities, so results are
reproducible across entry points.
"""
    )


def _technology() -> None:
    st.markdown("### Technology")
    st.markdown(
        '<div class="be-grid">'
        + _tile(
            "center_focus_strong",
            "YOLOv8",
            "Object detection - finds and classifies marine species in images and video frames in real time.",
        )
        + _tile(
            "psychology",
            "PyTorch",
            "Deep learning - runs the model backend on CPU or GPU.",
        )
        + _tile(
            "movie",
            "OpenCV",
            "Image & video processing - decoding frames, resizing and encoding outputs.",
        )
        + _tile(
            "code",
            "Python",
            "Application & AI pipeline - connects the interface, CLI and detection code.",
        )
        + _tile(
            "calculate",
            "NumPy",
            "Data processing - array operations behind statistics and reports.",
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Runs on ordinary hardware: a CPU works (8-16 GB RAM recommended); "
        "an NVIDIA GPU speeds up detection and training. Docker, Conda or "
        "venv are used to set up the environment."
    )


def _capabilities() -> None:
    st.markdown("### What BlueEye provides")
    st.markdown(
        '<div class="be-grid">'
        + _tile(
            "web",
            "Web interface",
            "Upload media, choose a model, review results and download outputs.",
        )
        + _tile(
            "terminal",
            "CLI and API",
            "`python -m app.main` runs the same pipeline headlessly for scripting.",
        )
        + _tile(
            "data_object",
            "JSON reports",
            "Every run writes an annotated output and a structured report.",
        )
        + _tile(
            "insights",
            "History and analytics",
            "Finished runs are stored locally and summarized - nothing is uploaded.",
        )
        + _tile(
            "hub",
            "Model registry",
            "Built-in and custom YOLOv8 models load through one registry, extensible for regional models.",
        )
        + _tile(
            "memory",
            "CPU / GPU selection",
            "The device is detected automatically; CUDA is used when available.",
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def _models(detector: MarineDetector) -> None:
    """List the registered models with their honest provenance."""
    st.markdown("### Our current models")
    registry = state.registry(detector)
    ready = set(state.available_keys(detector))

    rows: list[str] = []
    for spec in registry.values():
        status = "Ready" if spec.key in ready else "Not installed"
        kind = badge("Custom", "warn") if spec.custom else badge("Built-in", "info")
        summary = spec.summary or spec.description
        rows.append(
            '<div class="be-card be-card--flush"><div class="be-iblock">'
            f'<div class="tile">{span("psychology")}</div><div class="body">'
            f'<p class="t">{esc(spec.display_name)}'
            f'{badge(status, "ok" if spec.key in ready else "off", dot=True)}{kind}</p>'
            f'<p class="s">{esc(summary)}</p>'
            f'<div class="row">{chips((f"{len(spec.classes)} classes", spec.key))}</div>'
            "</div></div></div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)

    st.markdown(
        """
BlueEye is the **application**; the pretrained models above are part of
the **marine-detect** project by Orange OpenSource. BlueEye did not train
them - weights are downloaded on demand from the official links and are
never fabricated or modified. Model cards, status badges and download
actions live on the **Models** page.
"""
    )


def _limitations() -> None:
    st.markdown("### Limitations")
    st.markdown(
        """
- Detection quality depends entirely on the training data of the selected
  model; it is not guaranteed to generalise to every sea, season or camera.
- There is **no region-specific model** shipped with BlueEye. Support for
  adding one (registry + training workflow) is documented in
  `docs/regional_models.md`, but no model is claimed until real weights are
  trained or sourced and verified.
- Underwater enhancement is optional preprocessing, not an accuracy boost.
- Very long videos are processed frame by frame in memory; extremely large
  files may take a long time on CPU.
"""
    )


def _future() -> None:
    st.markdown("### Future scope")
    notice(
        st,
        "info",
        "Directions for future work",
        "The items below are future possibilities identified in the project "
        "report - they are not implemented features of the current system.",
    )
    st.markdown(
        '<div class="be-grid">'
        + _tile(
            "trending_up",
            "Improved detection models",
            "Training YOLOv8 on larger, more diverse marine datasets, covering a greater variety of species including rare and endangered ones.",
        )
        + _tile(
            "developer_board",
            "Edge deployment",
            "Real-time deployment on NVIDIA Jetson Nano or Raspberry Pi for continuous monitoring without human involvement.",
        )
        + _tile(
            "show_chart",
            "Behaviour and population analysis",
            "Behaviour analysis, population tracking and migration-pattern identification.",
        )
        + _tile(
            "cloud",
            "Cloud integration",
            "Large-scale storage, sharing and analysis of detection data on cloud platforms.",
        )
        + _tile(
            "auto_fix_high",
            "Underwater image handling",
            "Further improvements for low visibility and difficult lighting conditions.",
        )
        + _tile(
            "devices",
            "Mobile accessibility",
            "Extending access beyond the desktop web interface to mobile applications.",
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    links = " · ".join(f"[{name}]({url})" for name, url in REPORT_DATASETS)
    st.caption(f"Datasets referenced in the project report for future training: {links}")


def _attribution() -> None:
    st.markdown("### Built with open-source technology")
    st.markdown(
        f"""
BlueEye builds on the **marine-detect** project by **Orange OpenSource**
(Orange Business Services SA), published at [{UPSTREAM}]({UPSTREAM}).

- The pretrained *Fish & Invertebrates* and *MegaFauna* weights, the model
  architecture choices and the original inference flow come from that
  project. BlueEye **did not create** those models.
- marine-detect is licensed under **AGPL-3.0-only**; BlueEye's own `app/`
  package carries the same `SPDX-License-Identifier: AGPL-3.0-only` header
  and the upstream code under `src/marine_detect/` is retained untouched
  for attribution.
- See `README.md` -> *Credits and licence* for full details.
"""
    )
    st.markdown(
        badge("AGPL-3.0-only", "warn") + " " + badge("YOLOv8 / Ultralytics", "info"),
        unsafe_allow_html=True,
    )
    st.link_button(
        "View marine-detect source",
        UPSTREAM,
        icon=":material/open_in_new:",
        width="stretch",
    )


__all__ = ["render"]
