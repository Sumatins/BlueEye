"""Structured exports for saved observation records.

Everything here is rendered from stored fields only. Exports are produced as
plain strings so they can be attached to a Streamlit download button without
writing temporary files, and they carry an explicit data-limitations note:
ordinary model output is a *prediction*, not a confirmed biological
identification.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import base64
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.detection.results import display_class_name
from app.observations.models import Analysis, Collection

#: Plain-language limitation included in every export.
DATA_LIMITATION = (
    "Labels are outputs of an AI detection model (predictions). They are NOT "
    "confirmed biological identifications and must not be treated as "
    "ground-truth species records. Detection metrics (precision / recall / "
    "mAP) require a labelled dataset and are produced only by the Research "
    "Evaluation pipeline. A model that only supports a broad class (for "
    "example 'Fish') does not establish a species-level identification."
)

#: Observation-level CSV columns.
OBSERVATION_COLUMNS = (
    "analysis_id",
    "kind",
    "source_filename",
    "detected_at",
    "saved_at",
    "project",
    "collections",
    "review_status",
    "models",
    "model_version",
    "confidence_threshold",
    "enhanced",
    "total_detections",
    "unique_classes",
    "average_confidence",
    "max_confidence",
    "frames_processed",
    "location",
    "habitat",
    "notes",
)

#: Detection-level CSV columns.
DETECTION_COLUMNS = (
    "analysis_id",
    "source_filename",
    "detected_at",
    "kind",
    "project",
    "review_status",
    "detection_id",
    "class_name",
    "confidence",
    "bbox",
    "model",
    "frame_index",
    "frame_time",
    "original_label",
    "corrected_label",
    "correction_note",
    "corrected_at",
    "corrected_by",
)


def _generated() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def analysis_to_dict(analysis: Analysis) -> dict[str, Any]:
    """Full JSON export of one saved observation (detections + corrections)."""
    payload = analysis.to_dict()
    payload["record_type"] = "blueeye_observation"
    payload["schema_version"] = 1
    payload["generated"] = _generated()
    payload["data_limitations"] = DATA_LIMITATION
    return payload


def observations_to_dict(
    analyses: Sequence[Analysis], *, title: str = "BlueEye observations"
) -> dict[str, Any]:
    """JSON export of a set of saved observations."""
    return {
        "record_type": "blueeye_observations",
        "schema_version": 1,
        "title": title,
        "generated": _generated(),
        "count": len(analyses),
        "data_limitations": DATA_LIMITATION,
        "observations": [analysis_to_dict(item) for item in analyses],
    }


def observations_csv(analyses: Sequence[Analysis]) -> str:
    """CSV of saved observations (one row per analysis, real stored fields)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(OBSERVATION_COLUMNS)
    for item in analyses:
        writer.writerow(
            [
                item.id,
                item.kind,
                item.source_filename,
                item.detected_at,
                item.created_at,
                item.project,
                "|".join(item.collection_names),
                item.review_status,
                "|".join(item.models),
                item.model_version or "unavailable",
                "" if item.confidence_threshold is None else f"{item.confidence_threshold:.4f}",
                "yes" if item.enhanced else "no",
                item.total_detections,
                item.unique_classes,
                f"{item.average_confidence:.4f}",
                f"{item.max_confidence:.4f}",
                "" if item.frames_processed is None else item.frames_processed,
                item.location,
                item.habitat,
                item.notes,
            ]
        )
    return buffer.getvalue()


def detections_csv(analyses: Sequence[Analysis]) -> str:
    """CSV of individual detections (one row per detected object)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(DETECTION_COLUMNS)
    for item in analyses:
        for det in item.detections:
            latest = det.latest_correction
            writer.writerow(
                [
                    item.id,
                    item.source_filename,
                    item.detected_at,
                    item.kind,
                    item.project,
                    item.review_status,
                    det.id,
                    det.class_name,
                    f"{det.confidence:.4f}",
                    "" if det.bbox is None else " ".join(str(v) for v in det.bbox),
                    det.model,
                    "" if det.frame_index is None else det.frame_index,
                    "" if det.frame_time is None else f"{det.frame_time:.3f}",
                    det.class_name,
                    latest.corrected_label if latest else "",
                    latest.note if latest else "",
                    latest.created_at if latest else "",
                    latest.reviewer_id if latest else "",
                ]
            )
    return buffer.getvalue()


def analysis_csv(analysis: Analysis) -> str:
    """Detection-level CSV for a single observation."""
    return detections_csv([analysis])


def collection_to_dict(collection: Collection, analyses: Sequence[Analysis]) -> dict[str, Any]:
    """JSON export of one collection (survey) and its observations."""
    return {
        "record_type": "blueeye_collection",
        "schema_version": 1,
        "generated": _generated(),
        "collection": collection.to_dict(),
        "count": len(analyses),
        "data_limitations": DATA_LIMITATION,
        "observations": [analysis_to_dict(item) for item in analyses],
    }


def collection_csv(collection: Collection, analyses: Sequence[Analysis]) -> str:
    """Observation-level CSV for one collection."""
    return observations_csv(analyses)


# --------------------------------------------------------------------------- #
# Printable HTML report
# --------------------------------------------------------------------------- #
_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

#: Never embed an evidence image larger than this (keeps reports portable).
_MAX_EMBED_BYTES = 8 * 1024 * 1024


def _embed_image(path: str | Path | None) -> str | None:
    """Base64 data URI for a readable, reasonably sized image file."""
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    try:
        if file_path.stat().st_size > _MAX_EMBED_BYTES:
            return None
        data = file_path.read_bytes()
    except OSError:
        return None
    mime = _MIME_BY_SUFFIX.get(file_path.suffix.lower(), "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _html_escape(value: Any) -> str:
    import html

    return html.escape(str(value if value is not None else ""))


def _detection_rows_html(analysis: Analysis) -> str:
    if not analysis.detections:
        return "<tr><td colspan='6'>No individual objects were recorded for this run.</td></tr>"
    rows = []
    for index, det in enumerate(analysis.detections, start=1):
        latest = det.latest_correction
        original = display_class_name(det.class_name)
        effective = display_class_name(det.effective_label)
        label = effective
        if latest:
            label = f"{effective} (corrected from {original})"
        bbox = "" if det.bbox is None else ", ".join(str(v) for v in det.bbox)
        frame = "" if det.frame_index is None else str(det.frame_index)
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{_html_escape(label)}</td>"
            f"<td>{det.confidence * 100:.1f}%</td>"
            f"<td>{_html_escape(bbox)}</td>"
            f"<td>{_html_escape(frame)}</td>"
            f"<td>{_html_escape(det.model or '-')}</td>"
            "</tr>"
        )
    return "".join(rows)


def _corrections_html(analysis: Analysis) -> str:
    items = [
        "<li>"
        f"Detection {det.id[:8]}: "
        f"<b>{_html_escape(display_class_name(correction.original_label))}</b> "
        f"&rarr; <b>{_html_escape(display_class_name(correction.corrected_label))}</b> "
        f"({_html_escape(correction.created_at)})"
        f"{(' - ' + _html_escape(correction.note)) if correction.note else ''}"
        "</li>"
        for det in analysis.detections
        for correction in det.corrections
    ]
    if not items:
        return "<p>No label corrections were recorded.</p>"
    return "<ul>" + "".join(items) + "</ul>"


def _evidence_html(analysis: Analysis, embed_images: bool) -> str:
    if not embed_images:
        return (
            "<p><em>Evidence images are not embedded in this report. "
            "Download them from the observation first.</em></p>"
        )
    blocks = []
    for title, path in (
        ("Original media", analysis.original_path),
        ("Annotated result", analysis.annotated_path),
    ):
        uri = _embed_image(path)
        if uri:
            blocks.append(
                f"<figure><figcaption>{_html_escape(title)}</figcaption>"
                f"<img src='{uri}' alt='{_html_escape(title)}'/></figure>"
            )
        else:
            blocks.append(
                f"<figure><figcaption>{_html_escape(title)}</figcaption>"
                f"<div class='missing'>Not available for embedding</div></figure>"
            )
    return "<div class='evidence'>" + "".join(blocks) + "</div>"


_HTML_STYLE = """
body { font-family: "Segoe UI", system-ui, Arial, sans-serif; color: #12222f; margin: 32px; }
h1 { color: #0b3a66; }
h2 { color: #0b3a66; border-bottom: 1px solid #cfe0ee; padding-bottom: 4px; margin-top: 28px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; }
th, td { border: 1px solid #cfe0ee; padding: 6px 8px; text-align: left; font-size: 13px; }
th { background: #eef6fb; }
.evidence { display: flex; gap: 16px; flex-wrap: wrap; }
figure { margin: 0; max-width: 48%; }
figure img { max-width: 100%; border: 1px solid #cfe0ee; border-radius: 6px; }
figcaption { font-size: 12px; color: #496579; margin-bottom: 4px; }
.missing { border: 1px dashed #cfe0ee; border-radius: 6px; padding: 24px; color: #7d97aa; }
.limit { background: #fff7e6; border: 1px solid #f0c36d; border-radius: 6px; padding: 12px 14px; }
.meta td { background: #f7fbfe; }
"""


def analysis_report_html(analysis: Analysis, *, embed_images: bool = True) -> str:
    """A self-contained, printable HTML report for one saved observation."""
    meta_rows = "".join(
        f"<tr><th>{_html_escape(key)}</th><td>{_html_escape(value)}</td></tr>"
        for key, value in (
            ("Observation id", analysis.id),
            ("Source file", analysis.source_filename),
            ("Media type", analysis.kind),
            ("Detection timestamp", analysis.detected_at),
            ("Saved", analysis.created_at),
            ("Model(s)", ", ".join(analysis.models) or "unavailable"),
            ("Model version", analysis.model_version or "unavailable"),
            ("Confidence threshold", analysis.confidence_threshold),
            ("Enhanced preprocessing", "yes" if analysis.enhanced else "no"),
            ("Project / survey", analysis.project or "unassigned"),
            ("Collections", ", ".join(analysis.collection_names) or "none"),
            ("Location", analysis.location or "not recorded"),
            ("Habitat", analysis.habitat or "not recorded"),
            ("Review status", analysis.review_status),
        )
    )
    stats_rows = "".join(
        f"<tr><th>{_html_escape(key)}</th><td>{_html_escape(value)}</td></tr>"
        for key, value in (
            ("Total objects", analysis.total_detections),
            ("Distinct classes", analysis.unique_classes),
            ("Average confidence", f"{analysis.average_confidence * 100:.1f}%"),
            ("Max confidence", f"{analysis.max_confidence * 100:.1f}%"),
            ("Frames processed", analysis.frames_processed),
        )
    )
    notes = _html_escape(analysis.notes) or "<em>No notes recorded.</em>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>BlueEye observation {_html_escape(analysis.id[:8])}</title>"
        f"<style>{_HTML_STYLE}</style></head><body>"
        "<h1>BlueEye observation report</h1>"
        f"<p>Generated {_html_escape(_generated())}</p>"
        "<h2>Identification</h2>"
        f"<table class='meta'>{meta_rows}</table>"
        "<h2>Detection evidence</h2>"
        f"{_evidence_html(analysis, embed_images)}"
        "<h2>Summary statistics</h2>"
        f"<table class='meta'>{stats_rows}</table>"
        "<h2>Detected objects</h2>"
        "<table><thead><tr><th>#</th><th>Label</th><th>Confidence</th>"
        "<th>Bounding box (x1,y1,x2,y2)</th><th>Frame</th><th>Model</th></tr></thead>"
        f"<tbody>{_detection_rows_html(analysis)}</tbody></table>"
        "<h2>Corrections</h2>"
        f"{_corrections_html(analysis)}"
        "<h2>User notes</h2>"
        f"<p>{notes}</p>"
        "<h2>Data limitations</h2>"
        f"<div class='limit'>{_html_escape(DATA_LIMITATION)}</div>"
        "</body></html>"
    )


def collection_report_html(
    collection: Collection, analyses: Sequence[Analysis], *, embed_images: bool = True
) -> str:
    """A printable HTML report for a survey / collection of observations."""
    sections = []
    for item in analyses:
        sections.append(
            f"<h3>{_html_escape(item.source_filename)}</h3>"
            f"<p>{_html_escape(item.detected_at)} &middot; {_html_escape(item.kind)} "
            f"&middot; {item.total_detections} object(s) &middot; "
            f"{_html_escape(item.review_status)}</p>"
            f"{_evidence_html(item, embed_images)}"
        )
    summary_rows = "".join(
        f"<tr><td>{_html_escape(item.source_filename)}</td>"
        f"<td>{_html_escape(item.detected_at)}</td>"
        f"<td>{item.total_detections}</td>"
        f"<td>{item.unique_classes}</td>"
        f"<td>{item.average_confidence * 100:.1f}%</td>"
        f"<td>{_html_escape(item.review_status)}</td></tr>"
        for item in analyses
    )
    total_objects = sum(item.total_detections for item in analyses)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>BlueEye survey {_html_escape(collection.name)}</title>"
        f"<style>{_HTML_STYLE}</style></head><body>"
        f"<h1>Survey report: {_html_escape(collection.name)}</h1>"
        f"<p>{_html_escape(collection.description)}</p>"
        f"<p>Generated {_html_escape(_generated())} &middot; "
        f"{len(analyses)} observation(s) &middot; {total_objects} object(s)</p>"
        "<h2>Observations</h2>"
        "<table><thead><tr><th>Source</th><th>Detected</th><th>Objects</th>"
        "<th>Classes</th><th>Avg confidence</th><th>Review</th></tr></thead>"
        f"<tbody>{summary_rows}</tbody></table>"
        "<h2>Evidence</h2>"
        f"{''.join(sections)}"
        "<h2>Data limitations</h2>"
        f"<div class='limit'>{_html_escape(DATA_LIMITATION)}</div>"
        "</body></html>"
    )


def json_dumps(payload: Any) -> str:
    """Consistent, human-readable JSON for any export payload."""
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def iter_detections(analyses: Iterable[Analysis]) -> Iterable[Any]:
    """Flat iterator over every detection across ``analyses``."""
    for item in analyses:
        yield from item.detections


__all__ = [
    "DATA_LIMITATION",
    "DETECTION_COLUMNS",
    "OBSERVATION_COLUMNS",
    "analysis_csv",
    "analysis_report_html",
    "analysis_to_dict",
    "collection_csv",
    "collection_report_html",
    "collection_to_dict",
    "detections_csv",
    "observations_csv",
    "observations_to_dict",
]
