"""Help and documentation page.

Plain-language instructions for the workflow BlueEye actually implements:
uploading an image, analysing a video, reading the results, and understanding
what the models can and cannot tell you. Everything here describes real
behaviour of the running application (see :mod:`app.ui.pages.detect` and
:mod:`app.detection.model_manager`); nothing is invented.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import streamlit as st

from app.config import IMAGE_EXTENSIONS, PROJECT_ROOT, VIDEO_EXTENSIONS
from app.detection.detector import MarineDetector
from app.ui.components import download_button, hero, notice
from app.ui.icons import span
from app.ui.theme import esc

#: Repository documentation files referenced from this page (if present).
_DOC_FILES: tuple[tuple[str, str, str], ...] = (
    (
        "README.md",
        "Project README",
        "Installation, model download and the full command-line reference.",
    ),
    (
        "docs/indian_biodiversity.md",
        "Indian biodiversity notes",
        "The verified datasets, licences and remaining gaps for Indian species.",
    ),
    (
        "docs/regional_models.md",
        "Regional model guide",
        "The full dataset -> training -> evaluation workflow for adding a model.",
    ),
)


def render(detector: MarineDetector, settings, user=None) -> None:
    """Render the help and documentation content."""
    hero(
        st,
        "Guide",
        "Help & Documentation",
        "How to detect an image, analyse a video, read the results and understand "
        "the models' limits.",
        icon="help",
    )

    _quick_start(settings)
    _image_guide(settings)
    _video_guide(settings)
    _results_guide()
    _models_guide(detector)
    _troubleshooting(detector)
    _further_docs()


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _quick_start(settings) -> None:
    st.markdown("### Quick start")
    st.markdown(
        '<div class="be-flow">'
        f'<div class="fstep"><span class="n">1</span><span class="txt">Upload an image or video</span></div>'
        f'<span class="farr">{span("arrow_forward")}</span>'
        f'<div class="fstep"><span class="n">2</span><span class="txt">Choose a model</span></div>'
        f'<span class="farr">{span("arrow_forward")}</span>'
        f'<div class="fstep"><span class="n">3</span><span class="txt">Set the threshold</span></div>'
        f'<span class="farr">{span("arrow_forward")}</span>'
        f'<div class="fstep"><span class="n">4</span><span class="txt">Review the result</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Every finished run is stored locally and can be revisited in Detection "
        "History and summarised in Analytics. Nothing is uploaded to a server."
    )


def _image_guide(settings) -> None:
    st.markdown("### Detecting an image")
    st.markdown(
        '<div class="be-grid">'
        + _tile(
            "upload",
            "1. Upload",
            "Open Detect an Image and add an underwater photo. Accepted formats: "
            f"{', '.join(sorted(IMAGE_EXTENSIONS))}, up to {settings.max_image_mb} MB.",
        )
        + _tile(
            "psychology",
            "2. Choose a model",
            "'Auto' runs the two core models; 'Every installed model' also runs "
            "the additional aquatic models; or pick one model explicitly.",
        )
        + _tile(
            "tune",
            "3. Set the threshold",
            "Detections below the confidence threshold are ignored. Leave the "
            "toggle on to use each model's recommended threshold.",
        )
        + _tile(
            "rocket_launch",
            "4. Run",
            "Press 'Detect Aquatic Life'. BlueEye draws the boxes it found and "
            "writes an annotated image plus a JSON report.",
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def _video_guide(settings) -> None:
    st.markdown("### Analysing a video")
    st.markdown(
        "Switch the media type to **Video** on the Detect page (or open "
        "*Analyse a Video* in the sidebar), then upload a clip. BlueEye accepts "
        f"{', '.join(sorted(VIDEO_EXTENSIONS))} up to {settings.max_video_mb} MB and "
        "processes it frame by frame, showing real progress as it goes. The "
        "output is a browser-playable annotated video and a JSON report that "
        "aggregates the detections across frames."
    )
    notice(
        st,
        "info",
        "Video is heavier than images",
        "Frame-by-frame processing takes considerably longer than a single "
        "image, and the reported species counts summarise frames rather than "
        "unique individuals.",
    )


def _results_guide() -> None:
    st.markdown("### Reading the results")
    st.markdown(
        '<div class="be-grid">'
        + _tile(
            "label",
            "Labels",
            "Each box carries the class name the model predicted - a species or "
            "a broader group, depending on the model.",
        )
        + _tile(
            "speed",
            "Confidence",
            "The model's confidence for each detection, as a percentage. Higher "
            "is more certain, but it is a model estimate, not a guarantee.",
        )
        + _tile(
            "crop_free",
            "Bounding boxes",
            "The rectangle around each detection, so you can see exactly what "
            "the model reacted to.",
        )
        + _tile(
            "description",
            "JSON report",
            "The machine-readable record of the run - detections, scores, boxes "
            "and settings - downloadable from the result panel.",
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "A detection is a model prediction. Treat it as a starting point to "
        "review, not as a confirmed identification."
    )


def _models_guide(detector: MarineDetector) -> None:
    st.markdown("### Models and their limits")
    st.markdown(
        "Each model was trained on its own fixed list of classes - BlueEye never "
        "reports a species outside those classes. Open **AI Models** to see every "
        "registered model, its class list, its licence, and its verified status "
        "(a model is marked ready only after a real load-and-inference test on "
        "this machine)."
    )
    listed = getattr(detector.model_manager, "available_models", None)
    ready = len(listed()) if callable(listed) else 0
    st.markdown(
        f"Currently **{ready}** model(s) have weights installed here, running on "
        f"**{esc(str(detector.device).upper())}**."
    )
    notice(
        st,
        "warning",
        "Honest about scope",
        "BlueEye reports what its selected models were trained to recognise. "
        "Species that fall outside a model's classes cannot be detected, and a "
        "broad class such as 'fish' is not a species-level identification.",
    )


def _troubleshooting(detector: MarineDetector) -> None:
    st.markdown("### Troubleshooting")
    st.markdown(
        '<div class="be-grid">'
        + _tile(
            "cloud_off",
            "No models installed",
            "If no weights are present, open AI Models and download the official "
            "checkpoints - BlueEye never ships fabricated weights.",
        )
        + _tile(
            "report",
            "No organisms detected",
            "Nothing cleared the confidence threshold. Try lowering it, trying a "
            "different model, or checking the input quality.",
        )
        + _tile(
            "block",
            "Unsupported file",
            "Only the listed image and video formats within the size limits are "
            "accepted. Re-export the file to a supported format and retry.",
        )
        + _tile(
            "hourglass_top",
            "Slow video runs",
            "Video processing is frame by frame. Shorter clips and a lower "
            "resolution complete faster; a GPU speeds inference up when present.",
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def _further_docs() -> None:
    st.markdown("### Further documentation")
    rows = ""
    for relative, title, blurb in _DOC_FILES:
        if (PROJECT_ROOT / relative).is_file():
            rows += (
                '<div class="be-modelrow">'
                '<span class="be-ic" aria-hidden="true">description</span>'
                f'<span class="nm">{esc(title)}<span class="sub">{esc(blurb)} '
                f"&#183; {esc(relative)}</span></span></div>"
            )
    if rows:
        st.markdown(rows, unsafe_allow_html=True)
    readme = PROJECT_ROOT / "README.md"
    if readme.is_file():
        download_button(st, "Download README.md", readme, "text/markdown")


# --------------------------------------------------------------------------- #
# Layout helper
# --------------------------------------------------------------------------- #
def _tile(icon: str, title: str, body: str) -> str:
    """One labelled icon tile for the content grids."""
    return (
        '<div class="be-tile">'
        f'<div class="ic">{span(icon)}</div>'
        f"<h4>{esc(title)}</h4><p>{esc(body)}</p></div>"
    )


__all__ = ["render"]
