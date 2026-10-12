"""Write evaluation results as JSON, CSV and a readable Markdown summary.

Everything BlueEye writes here is *reproduced locally* from an actual run.
The Markdown summary states the exact conditions (device, package versions,
dataset, split) so a reader can never mistake a local measurement for a
number copied from a paper. Reference values published by a model's authors
are a separate, clearly-labelled category and are never mixed in.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import base64
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from app.evaluation.runner import ModelEvaluation, summarise

METRICS_HEADER = (
    "model",
    "dataset",
    "split",
    "status",
    "images",
    "objects",
    "mAP50",
    "mAP50_95",
    "precision",
    "recall",
    "seconds_per_image",
    "comparable_classes",
    "unsupported_dataset_classes",
)

PER_CLASS_HEADER = (
    "model",
    "class",
    "gt",
    "predictions",
    "precision",
    "recall",
    "AP50",
    "AP50_95",
)


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def _metrics_rows(evaluations: Sequence[ModelEvaluation]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in evaluations:
        headline = summarise(item)
        rows.append(
            [
                item.model,
                item.dataset,
                item.split,
                item.status,
                str(item.images_evaluated),
                str(headline["objects"]),
                f"{headline['mAP50']:.6f}",
                f"{headline['mAP50_95']:.6f}",
                f"{headline['precision']:.6f}",
                f"{headline['recall']:.6f}",
                f"{headline['seconds_per_image']:.4f}",
                str(len(item.compat.supported)),
                "; ".join(item.compat.unsupported),
            ]
        )
    return rows


def _per_class_rows(evaluations: Sequence[ModelEvaluation]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in evaluations:
        for metric in item.metrics.classes:
            rows.append(
                [
                    item.model,
                    metric.name,
                    str(metric.num_gt),
                    str(metric.num_pred),
                    f"{metric.precision:.6f}",
                    f"{metric.recall:.6f}",
                    f"{metric.ap50:.6f}",
                    f"{metric.ap50_95:.6f}",
                ]
            )
    return rows


def write_report(
    output_dir: str | Path,
    evaluations: Sequence[ModelEvaluation],
    environment: dict | None = None,
    config: dict | None = None,
    *,
    plots: Sequence[tuple[str, str | Path]] | None = None,
    experiment: dict | None = None,
) -> dict[str, Path]:
    """Write ``results.json``, ``metrics.csv``, ``per_class.csv`` and ``summary.md``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now().isoformat(timespec="seconds")

    payload = {
        "created": created,
        "config": config or {},
        "environment": environment or {},
        "evaluations": [item.as_dict() for item in evaluations],
        "summary": [summarise(item) for item in evaluations],
    }
    written: dict[str, Path] = {}
    written["results_json"] = out_dir / "results.json"
    written["results_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    written["environment_json"] = out_dir / "environment.json"
    written["environment_json"].write_text(
        json.dumps(environment or {}, indent=2), encoding="utf-8"
    )

    written["metrics_csv"] = out_dir / "metrics.csv"
    _write_csv(written["metrics_csv"], METRICS_HEADER, _metrics_rows(evaluations))

    written["per_class_csv"] = out_dir / "per_class.csv"
    _write_csv(written["per_class_csv"], PER_CLASS_HEADER, _per_class_rows(evaluations))

    written["summary_md"] = out_dir / "summary.md"
    written["summary_md"].write_text(
        render_summary(
            evaluations, environment=environment, config=config, created=created
        ),
        encoding="utf-8",
    )

    written["report_html"] = out_dir / "report.html"
    written["report_html"].write_text(
        render_html_report(
            evaluations,
            environment=environment,
            config=config,
            created=created,
            images=plots,
            experiment=experiment,
        ),
        encoding="utf-8",
    )
    return written


def render_summary(
    evaluations: Sequence[ModelEvaluation],
    environment: dict | None = None,
    config: dict | None = None,
    created: str | None = None,
) -> str:
    """Render the human-readable Markdown summary."""
    environment = environment or {}
    lines: list[str] = []
    lines.append("# BlueEye evaluation summary")
    lines.append("")
    lines.append(f"Generated: {created or datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(
        "All numbers below were **reproduced locally** by BlueEye on the dataset "
        "and split named in each row. They are not copied from a paper and must "
        "not be compared across different datasets as if they were equivalent."
    )
    lines.append("")

    if environment:
        hw = environment.get("hardware", {})
        packages = environment.get("packages", {})
        lines.append("## Test conditions")
        lines.append("")
        lines.append(f"- Platform: `{environment.get('platform', 'unknown')}`")
        lines.append(f"- Python: `{environment.get('python', 'unknown')}`")
        lines.append(f"- Device: `{hw.get('gpu') or 'CPU'}`")
        lines.append(f"- CUDA available: `{hw.get('cuda_available', False)}`")
        for name, version in packages.items():
            lines.append(f"- {name}: `{version}`")
        lines.append("")

    lines.append("## Model comparison")
    lines.append("")
    lines.append(
        "| Model | Dataset | Split | Status | Images | mAP50 | mAP50:95 | "
        "Precision | Recall | s/image |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for item in evaluations:
        headline = summarise(item)
        lines.append(
            f"| {item.model} | {item.dataset} | {item.split} | {item.status} | "
            f"{item.images_evaluated} | {headline['mAP50']:.3f} | "
            f"{headline['mAP50_95']:.3f} | {headline['precision']:.3f} | "
            f"{headline['recall']:.3f} | {headline['seconds_per_image']:.3f} |"
        )
    lines.append("")

    for item in evaluations:
        lines.append(f"## {item.model} on {item.dataset} ({item.split})")
        lines.append("")
        lines.append(f"- Status: **{item.status}**")
        lines.append(
            f"- Comparable classes: {', '.join(item.compat.supported) or 'none'}"
        )
        if item.compat.unsupported:
            lines.append(
                "- Dataset classes this model cannot detect (excluded from "
                f"per-class scoring): {', '.join(item.compat.unsupported)}"
            )
        if item.compat.extra_model:
            lines.append(
                f"- Model classes absent from the dataset (ignored): "
                f"{', '.join(item.compat.extra_model)}"
            )
        lines.append("")
        if item.metrics.classes:
            lines.append("| Class | GT | Pred | Precision | Recall | AP50 | AP50:95 |")
            lines.append("|---|---|---|---|---|---|---|")
            for metric in item.metrics.classes:
                lines.append(
                    f"| {metric.name} | {metric.num_gt} | {metric.num_pred} | "
                    f"{metric.precision:.3f} | {metric.recall:.3f} | "
                    f"{metric.ap50:.3f} | {metric.ap50_95:.3f} |"
                )
            lines.append("")
        for note in item.notes:
            lines.append(f"> {note}")
        lines.append("")

    if config:
        lines.append("## Configuration")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(config, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-contained HTML report
# --------------------------------------------------------------------------- #
#: Plain-language definitions shown in every report so a metric can never be
#: mistaken for something it is not.
METRIC_DEFINITIONS = (
    (
        "Precision",
        "Of the detections the model produced, the proportion that matched a "
        "ground-truth box under the configured matching criteria.",
    ),
    (
        "Recall",
        "Of the ground-truth objects, the proportion the model detected under "
        "those criteria.",
    ),
    (
        "mAP@0.5",
        "Mean average precision using an intersection-over-union (IoU) "
        "threshold of 0.5.",
    ),
    (
        "mAP@0.5:0.95",
        "Mean average precision averaged over IoU thresholds from 0.50 to 0.95 "
        "in steps of 0.05.",
    ),
)

#: A single embedded plot larger than this is skipped (keeps HTML shareable).
_MAX_EMBED_BYTES = 8 * 1024 * 1024

_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       color: #12212e; margin: 2rem auto; max-width: 960px; line-height: 1.5; }
h1, h2, h3 { color: #0b3d59; }
table { border-collapse: collapse; width: 100%; margin: 0.6rem 0 1.2rem; font-size: 0.92rem; }
th, td { border: 1px solid #cfd8e0; padding: 5px 8px; text-align: left; }
th { background: #eaf4fa; }
code, pre { background: #f3f7fa; padding: 2px 4px; border-radius: 4px; }
pre { padding: 10px; overflow-x: auto; }
.muted { color: #5b6b78; }
.caveat { background: #fff6e5; border-left: 4px solid #e0a33c; padding: 10px 14px;
          border-radius: 4px; }
.card { border: 1px solid #cfd8e0; border-radius: 8px; padding: 12px 16px;
        margin: 1rem 0; }
img { max-width: 100%; border: 1px solid #cfd8e0; border-radius: 6px; margin-top: 6px; }
"""


def _fmt(value, digits: int = 3, suffix: str = "") -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{suffix}"
    return "&mdash;"


def _eval_dict(item) -> dict:
    if isinstance(item, dict):
        return item
    as_dict = getattr(item, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    return {}


def _headline(item: dict) -> dict:
    metrics = item.get("metrics") or {}
    classes = metrics.get("classes") or []
    timing = item.get("timing") or {}
    prec = [c.get("precision") for c in classes if c.get("gt", 0) > 0]
    rec = [c.get("recall") for c in classes if c.get("gt", 0) > 0]
    return {
        "mAP50": metrics.get("mAP50"),
        "mAP50_95": metrics.get("mAP50_95"),
        "precision": sum(prec) / len(prec) if prec else 0.0,
        "recall": sum(rec) / len(rec) if rec else 0.0,
        "images": timing.get("images"),
        "objects": sum(int(c.get("gt", 0)) for c in classes),
        "seconds_per_image": timing.get("seconds_per_image"),
    }


def _embed_image(path: str | Path) -> str | None:
    candidate = Path(path)
    try:
        if not candidate.is_file() or candidate.stat().st_size > _MAX_EMBED_BYTES:
            return None
        data = base64.b64encode(candidate.read_bytes()).decode("ascii")
    except OSError:
        return None
    suffix = candidate.suffix.lower().lstrip(".") or "png"
    if suffix == "svg":
        return f"data:image/svg+xml;base64,{data}"
    return f"data:image/{suffix};base64,{data}"


def render_html_report(
    evaluations: Sequence,
    environment: dict | None = None,
    config: dict | None = None,
    created: str | None = None,
    *,
    images: Sequence[tuple[str, str | Path]] | None = None,
    experiment: dict | None = None,
    title: str = "BlueEye research evaluation report",
) -> str:
    """Render a self-contained, printable HTML report.

    Everything is generated from the supplied measurements: no example or
    placeholder score is ever inserted. Plots referenced by ``images`` are
    embedded as base64 when small enough so the file can be archived or shared
    as a single artifact.
    """
    environment = environment or {}
    config = config or {}
    experiment = experiment or {}
    evaluations = [_eval_dict(item) for item in evaluations]
    created = created or datetime.now().isoformat(timespec="seconds")

    lines: list[str] = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        f"<title>{html.escape('BlueEye evaluation report')}</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>BlueEye research evaluation report</h1>",
        f'<p class="muted">Generated: {html.escape(created)}</p>',
    ]

    if experiment:
        name = experiment.get("name") or experiment.get("id") or "experiment"
        lines.append(f"<h2>Experiment: {html.escape(str(name))}</h2>")
        rows = []
        for label, key in (
            ("Experiment id", "id"),
            ("Status", "status"),
            ("Research purpose", "purpose"),
            ("Notes", "notes"),
            ("Git commit", "git_commit"),
        ):
            value = experiment.get(key)
            if value:
                rows.append(
                    f"<tr><th>{html.escape(label)}</th>"
                    f"<td>{html.escape(str(value))}</td></tr>"
                )
        dataset = experiment.get("dataset") if isinstance(experiment.get("dataset"), dict) else {}
        for label, value in (
            ("Dataset", dataset.get("name") or experiment.get("dataset_name")),
            ("Dataset config", dataset.get("data_yaml") or experiment.get("data_yaml")),
            ("Split", dataset.get("split") or experiment.get("split")),
            ("Dataset fingerprint", dataset.get("fingerprint") or experiment.get("dataset_fingerprint")),
        ):
            if value:
                rows.append(
                    f"<tr><th>{html.escape(label)}</th>"
                    f"<td><code>{html.escape(str(value))}</code></td></tr>"
                )
        if rows:
            lines.append("<table>" + "".join(rows) + "</table>")

    lines.append(
        '<div class="caveat"><strong>Scope.</strong> Every number below was '
        "measured locally by BlueEye on the dataset, split and configuration "
        "named in each section. Results describe performance on that evaluated "
        "dataset only and do not automatically establish generalisation to all "
        "aquatic environments.</div>"
    )

    lines.append("<h2>Metric definitions</h2>")
    lines.append("<table><tr><th>Metric</th><th>Definition</th></tr>")
    for name, definition in METRIC_DEFINITIONS:
        lines.append(
            f"<tr><th>{html.escape(name)}</th><td>{html.escape(definition)}</td></tr>"
        )
    lines.append("</table>")

    if environment:
        lines.append("<h2>Environment / test conditions</h2><table>")
        for key, value in environment.items():
            if isinstance(value, dict):
                value = ", ".join(f"{k}={v}" for k, v in value.items())
            lines.append(
                f"<tr><th>{html.escape(str(key))}</th>"
                f"<td>{html.escape(str(value))}</td></tr>"
            )
        lines.append("</table>")

    lines.append("<h2>Model comparison</h2>")
    lines.append(
        "<table><tr><th>Model</th><th>Dataset</th><th>Split</th><th>Status</th>"
        "<th>Images</th><th>Objects</th><th>Precision</th><th>Recall</th>"
        "<th>mAP@0.5</th><th>mAP@0.5:0.95</th><th>s/image</th></tr>"
    )
    for item in evaluations:
        headline = _headline(item)
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('model', '')))}</td>"
            f"<td>{html.escape(str(item.get('dataset', '')))}</td>"
            f"<td>{html.escape(str(item.get('split', '')))}</td>"
            f"<td>{html.escape(str(item.get('status', '')))}</td>"
            f"<td>{_fmt(headline['images'], 0)}</td>"
            f"<td>{headline['objects']}</td>"
            f"<td>{_fmt(headline['precision'])}</td>"
            f"<td>{_fmt(headline['recall'])}</td>"
            f"<td>{_fmt(headline['mAP50'])}</td>"
            f"<td>{_fmt(headline['mAP50_95'])}</td>"
            f"<td>{_fmt(headline['seconds_per_image'], 4)}</td>"
            "</tr>"
        )
    lines.append("</table>")

    for item in evaluations:
        model = str(item.get("model", "model"))
        lines.append(f'<div class="card"><h3>{html.escape(model)}</h3>')
        lines.append(
            f"<p class=\"muted\">Status: {html.escape(str(item.get('status', '')))} "
            f"&middot; dataset: {html.escape(str(item.get('dataset', '')))} "
            f"&middot; split: {html.escape(str(item.get('split', '')))}</p>"
        )
        comparison = item.get("comparison") or {}
        if comparison:
            supported = ", ".join(comparison.get("supported", []) or []) or "none"
            lines.append(
                f"<p>Comparable classes: {html.escape(supported)}</p>"
            )
            unsupported = comparison.get("unsupported_dataset_classes") or []
            if unsupported:
                lines.append(
                    "<p class=\"muted\">Dataset classes this model cannot "
                    f"detect (excluded from per-class scoring): "
                    f"{html.escape(', '.join(unsupported))}</p>"
                )
        classes = (item.get("metrics") or {}).get("classes") or []
        if classes:
            lines.append(
                "<table><tr><th>Class</th><th>GT</th><th>Pred</th>"
                "<th>Precision</th><th>Recall</th><th>AP@0.5</th>"
                "<th>AP@0.5:0.95</th></tr>"
            )
            for metric in classes:
                lines.append(
                    "<tr>"
                    f"<td>{html.escape(str(metric.get('name', '')))}</td>"
                    f"<td>{metric.get('gt', 0)}</td>"
                    f"<td>{metric.get('predictions', 0)}</td>"
                    f"<td>{_fmt(metric.get('precision'))}</td>"
                    f"<td>{_fmt(metric.get('recall'))}</td>"
                    f"<td>{_fmt(metric.get('AP50'))}</td>"
                    f"<td>{_fmt(metric.get('AP50_95'))}</td>"
                    "</tr>"
                )
            lines.append("</table>")
        for note in item.get("notes") or []:
            lines.append(f"<p class=\"muted\">&gt; {html.escape(str(note))}</p>")
        lines.append("</div>")

    if images:
        lines.append("<h2>Plots</h2>")
        for caption, path in images:
            embedded = _embed_image(path)
            if not embedded:
                lines.append(
                    f'<p class="muted">{html.escape(str(caption))}: image not '
                    "embedded (missing or too large).</p>"
                )
                continue
            lines.append(f"<h3>{html.escape(str(caption))}</h3>")
            lines.append(f'<img alt="{html.escape(str(caption))}" src="{embedded}">')

    if config:
        lines.append("<h2>Configuration</h2>")
        lines.append(f"<pre>{html.escape(json.dumps(config, indent=2))}</pre>")

    lines.append(
        '<p class="muted">Produced by BlueEye. Numbers are local measurements, '
        "not reference values from the model authors.</p>"
    )
    lines.append("</body></html>")
    return "\n".join(lines)
