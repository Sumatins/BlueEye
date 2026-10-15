"""Publication-ready exports of stored evaluation results.

Everything here is built from what BlueEye actually *measured* and stored - an
:class:`~app.evaluation.experiments.ExperimentRecord` with its per-model
results. The exports never look up reference numbers from the literature and
never fill a gap with a guess; a value that was not measured stays empty.

The same helpers power the *Export* section of the Research Evaluation Center
and the ``build_bundle`` writer, which produces a folder a researcher can drop
into a paper's supplementary material:

* ``publication_metrics.csv``  - one row per evaluated model;
* ``publication_per_class.csv`` - one row per (model, class);
* ``publication_tables.md``    - Markdown tables + headnotes;
* ``publication_tables.tex``   - booktabs LaTeX tables;
* ``dataset_cards.md``         - a dataset card per experiment;
* ``provenance.md``            - evaluator version, environment, parameters;
* ``publication_report.html``  - one self-contained, printable report.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from app.evaluation.compat import compare_classes

# --------------------------------------------------------------------------- #
# Column definitions: (key, human label, numeric)
# --------------------------------------------------------------------------- #
HEADLINE_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("experiment", "Experiment", False),
    ("model", "Model", False),
    ("dataset", "Dataset", False),
    ("split", "Split", False),
    ("status", "Status", False),
    ("comparable", "Comparable", False),
    ("images", "Images", True),
    ("objects", "Objects", True),
    ("mAP50", "mAP@0.5", True),
    ("mAP50_95", "mAP@0.5:0.95", True),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("f1", "F1", True),
    ("seconds_per_image", "s/image", True),
    ("classes_scored", "Classes scored", True),
    ("classes_unsupported", "Dataset classes unsupported", False),
)

PER_CLASS_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("experiment", "Experiment", False),
    ("model", "Model", False),
    ("dataset", "Dataset", False),
    ("split", "Split", False),
    ("class_name", "Class", False),
    ("gt", "GT", True),
    ("predictions", "Pred", True),
    ("tp", "TP", True),
    ("fp", "FP", True),
    ("fn", "FN", True),
    ("precision", "Precision", True),
    ("recall", "Recall", True),
    ("f1", "F1", True),
    ("AP50", "AP50", True),
    ("AP50_95", "AP50:0.95", True),
)

#: A short honesty note used by every rendered table.
CAVEAT = (
    "All values were measured locally by BlueEye on the dataset and split named "
    "in each row, using the model weights recorded in the provenance sheet. They "
    "are not copied from a paper, and mAP on one dataset is not comparable with "
    "mAP on another."
)


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #
def _num(value, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def headline_rows(experiments: Sequence) -> list[dict]:
    """One row per evaluated model across the given experiments."""
    rows: list[dict] = []
    for experiment in experiments:
        for result in getattr(experiment, "results", ()) or ():
            summary = getattr(result, "summary", {}) or {}
            metrics = getattr(result, "metrics", {}) or {}
            classes = metrics.get("classes") or []
            comparison = compare_classes(
                tuple(getattr(experiment, "class_names", ()) or ()),
                tuple(getattr(result, "model_classes", ()) or ()),
            )
            rows.append(
                {
                    "experiment": getattr(experiment, "name", experiment.id),
                    "experiment_id": experiment.id,
                    "model": getattr(result, "model", ""),
                    "dataset": summary.get("dataset", getattr(experiment, "dataset_name", "")),
                    "split": summary.get("split", getattr(experiment, "split", "")),
                    "status": getattr(result, "status", ""),
                    "comparable": bool(getattr(result, "comparable", True)),
                    "images": summary.get("images", 0),
                    "objects": summary.get("objects", 0),
                    "mAP50": _num(summary.get("mAP50")),
                    "mAP50_95": _num(summary.get("mAP50_95")),
                    "precision": _num(summary.get("precision")),
                    "recall": _num(summary.get("recall")),
                    "f1": _num(summary.get("f1")),
                    "seconds_per_image": _num(summary.get("seconds_per_image")),
                    "classes_scored": len(comparison.supported),
                    "classes_unsupported": "; ".join(comparison.unsupported),
                }
            )
    return rows


def per_class_rows(experiments: Sequence) -> list[dict]:
    """One row per (model, class) for every stored result."""
    rows: list[dict] = []
    for experiment in experiments:
        for result in getattr(experiment, "results", ()) or ():
            metrics = getattr(result, "metrics", {}) or {}
            for item in metrics.get("classes") or ():
                rows.append(
                    {
                        "experiment": getattr(experiment, "name", experiment.id),
                        "model": getattr(result, "model", ""),
                        "dataset": getattr(experiment, "dataset_name", ""),
                        "split": getattr(experiment, "split", ""),
                        "class_name": item.get("name", ""),
                        "gt": item.get("gt", 0),
                        "predictions": item.get("predictions", 0),
                        "tp": int(item.get("tp", 0)),
                        "fp": int(item.get("fp", 0)),
                        "fn": int(item.get("fn", 0)),
                        "precision": _num(item.get("precision")),
                        "recall": _num(item.get("recall")),
                        "f1": _num(item.get("f1")),
                        "AP50": _num(item.get("AP50")),
                        "AP50_95": _num(item.get("AP50_95")),
                    }
                )
    return rows


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _cell(key: str, value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def to_csv(rows: Sequence[dict], columns: Sequence[tuple[str, str, bool]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([label for _, label, _ in columns])
    for row in rows:
        writer.writerow([_cell(key, row.get(key, "")) for key, _, _ in columns])
    return buffer.getvalue()


def to_markdown(
    rows: Sequence[dict],
    columns: Sequence[tuple[str, str, bool]],
    *,
    caption: str | None = None,
) -> str:
    lines: list[str] = []
    if caption:
        lines.append(f"**{caption}**")
        lines.append("")
    lines.append("| " + " | ".join(label for _, label, _ in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append(
            "| " + " | ".join(_cell(key, row.get(key, "")) for key, _, _ in columns) + " |"
        )
    return "\n".join(lines)


def to_latex(
    rows: Sequence[dict],
    columns: Sequence[tuple[str, str, bool]],
    *,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    align = "".join("r" if numeric else "l" for _, _, numeric in columns)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
    ]
    if caption:
        lines.append(r"\caption{" + _latex_escape(caption) + "}")
    if label:
        lines.append(r"\label{" + _latex_escape(label) + "}")
    lines.append(r"\begin{tabular}{" + align + "}")
    lines.append(r"\toprule")
    lines.append(
        " & ".join(_latex_escape(label) for _, label, _ in columns) + r" \\"
    )
    lines.append(r"\midrule")
    for row in rows:
        cells = [
            _latex_escape(_cell(key, row.get(key, ""))) for key, _, _ in columns
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Dataset card / provenance
# --------------------------------------------------------------------------- #
def dataset_card(experiment) -> str:
    """A Markdown dataset card for one experiment (data actually evaluated)."""
    results = getattr(experiment, "results", ()) or ()
    images = sum(int((r.summary or {}).get("images", 0) or 0) for r in results)
    objects = sum(int((r.summary or {}).get("objects", 0) or 0) for r in results)
    classes = ", ".join(getattr(experiment, "class_names", ()) or ()) or "(none declared)"
    lines = [
        f"## Dataset card: {getattr(experiment, 'dataset_name', '') or experiment.name}",
        "",
        f"- **Data config:** `{getattr(experiment, 'data_yaml', '')}`",
        f"- **Split evaluated:** `{getattr(experiment, 'split', '')}`",
        f"- **Classes ({len(getattr(experiment, 'class_names', ()) or ())}):** {classes}",
        f"- **Images evaluated:** {images}",
        f"- **Ground-truth objects:** {objects}",
        f"- **Dataset fingerprint:** `{getattr(experiment, 'dataset_fingerprint', '') or 'n/a'}`",
        f"- **Models evaluated:** {', '.join(getattr(experiment, 'model_keys', ()) or ()) or 'none'}",
        f"- **Registered:** {getattr(experiment, 'created_at', '') or 'unknown'}",
        "",
        "> The class list and object counts come from the dataset's own "
        "`data.yaml` and label files. Licensing and provenance of the media are "
        "the dataset owner's responsibility and are not asserted here.",
        "",
    ]
    return "\n".join(lines)


def provenance(experiments: Sequence) -> str:
    """A Markdown provenance sheet covering every input experiment."""
    lines = [
        "# Provenance sheet",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "> This sheet exists so a reader can reproduce the numbers exactly. It "
        "records the evaluator recipe, the machine and the model weights used.",
        "",
    ]
    seen_env = False
    for experiment in experiments:
        lines.append(f"## {experiment.name} (`{experiment.id}`)")
        lines.append("")
        lines.append(f"- Status: **{experiment.status}**")
        lines.append(f"- Evaluator: `{getattr(experiment, 'evaluator_version', '') or 'unknown'}`")
        lines.append(f"- Git commit: `{getattr(experiment, 'git_commit', '') or 'unknown'}`")
        lines.append(
            f"- Dataset fingerprint: `{getattr(experiment, 'dataset_fingerprint', '') or 'n/a'}`"
        )
        lines.append(f"- Created: {getattr(experiment, 'created_at', '') or 'unknown'}")
        lines.append(f"- Completed: {getattr(experiment, 'completed_at', '') or 'not completed'}")
        lines.append("")
        lines.append("Parameters:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(getattr(experiment, "parameters", {}) or {}, indent=2, default=str))
        lines.append("```")
        lines.append("")
        lines.append("Model weights (fingerprints):")
        lines.append("")
        lines.append("```json")
        lines.append(
            json.dumps(getattr(experiment, "model_fingerprints", {}) or {}, indent=2, default=str)
        )
        lines.append("```")
        lines.append("")
        environment = getattr(experiment, "environment", {}) or {}
        if environment and not seen_env:
            seen_env = True
            lines.append("Test conditions:")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(environment, indent=2, default=str))
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Combined renderers
# --------------------------------------------------------------------------- #
def render_markdown(experiments: Sequence) -> str:
    lines = [
        "# BlueEye publication tables",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"> {CAVEAT}",
        "",
        "## Headline metrics",
        "",
        to_markdown(headline_rows(experiments), HEADLINE_COLUMNS),
        "",
    ]
    per_class = per_class_rows(experiments)
    if per_class:
        lines.append("## Per-class metrics")
        lines.append("")
        lines.append(to_markdown(per_class, PER_CLASS_COLUMNS))
        lines.append("")
    for experiment in experiments:
        lines.append(dataset_card(experiment))
        lines.append("")
    lines.append(provenance(experiments))
    return "\n".join(lines)


def render_latex(experiments: Sequence) -> str:
    head = [
        "% BlueEye publication tables (booktabs).",
        f"% Generated: {datetime.now().isoformat(timespec='seconds')}",
        "% " + CAVEAT,
        "",
        to_latex(
            headline_rows(experiments),
            HEADLINE_COLUMNS,
            caption="BlueEye headline detection metrics (measured locally).",
            label="tab:blueeye-headline",
        ),
        "",
    ]
    per_class = per_class_rows(experiments)
    if per_class:
        head.extend(
            [
                to_latex(
                    per_class,
                    PER_CLASS_COLUMNS,
                    caption="BlueEye per-class detection metrics (measured locally).",
                    label="tab:blueeye-per-class",
                ),
                "",
            ]
        )
    return "\n".join(head)


_HTML_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       color: #12212e; margin: 2rem auto; max-width: 1100px; line-height: 1.5; }
h1, h2, h3 { color: #0b3d59; }
table { border-collapse: collapse; width: 100%; margin: 0.6rem 0 1.2rem; font-size: 0.88rem; }
th, td { border: 1px solid #cfd8e0; padding: 4px 7px; text-align: left; }
th { background: #eaf4fa; }
code, pre { background: #f3f7fa; padding: 2px 4px; border-radius: 4px; }
pre { padding: 10px; overflow-x: auto; }
.caveat { background: #fff6e5; border-left: 4px solid #e0a33c; padding: 10px 14px;
          border-radius: 4px; }
"""


def _html_table(rows: Sequence[dict], columns: Sequence[tuple[str, str, bool]]) -> str:
    parts = ["<table>", "<thead><tr>"]
    for _, label, _ in columns:
        parts.append(f"<th>{html.escape(label)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for key, _, _ in columns:
            parts.append(f"<td>{html.escape(_cell(key, row.get(key, '')))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _embed_image(path: str | Path) -> str | None:
    candidate = Path(path)
    try:
        if not candidate.is_file() or candidate.stat().st_size > 8 * 1024 * 1024:
            return None
        data = base64.b64encode(candidate.read_bytes()).decode("ascii")
    except OSError:
        return None
    suffix = candidate.suffix.lower().lstrip(".") or "png"
    return f"data:image/{suffix};base64,{data}"


def render_html(experiments: Sequence, *, embed_plots: bool = True) -> str:
    """A single self-contained, printable publication report."""
    created = datetime.now().isoformat(timespec="seconds")
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>BlueEye publication report</title>",
        f"<style>{_HTML_CSS}</style></head><body>",
        "<h1>BlueEye publication report</h1>",
        f"<p class='muted'>Generated {html.escape(created)}</p>",
        f"<div class='caveat'>{html.escape(CAVEAT)}</div>",
        "<h2>Headline metrics</h2>",
        _html_table(headline_rows(experiments), HEADLINE_COLUMNS),
    ]
    per_class = per_class_rows(experiments)
    if per_class:
        parts.append("<h2>Per-class metrics</h2>")
        parts.append(_html_table(per_class, PER_CLASS_COLUMNS))

    for experiment in experiments:
        parts.append(f"<h2>{html.escape(experiment.name)}</h2>")
        # Render the dataset card as preformatted Markdown for compactness.
        parts.append(f"<pre>{html.escape(dataset_card(experiment))}</pre>")
        if embed_plots:
            for artifact in getattr(experiment, "artifacts", ()) or ():
                if str(getattr(artifact, "path", "")).lower().endswith(".png"):
                    src = _embed_image(artifact.path)
                    if src:
                        parts.append(
                            f"<h3>{html.escape(getattr(artifact, 'name', 'plot'))}</h3>"
                            f"<img alt='{html.escape(getattr(artifact, 'name', 'plot'))}' src='{src}'>"
                        )
    parts.append(f"<h2>Provenance</h2><pre>{html.escape(provenance(experiments))}</pre>")
    parts.append("</body></html>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Bundle writer
# --------------------------------------------------------------------------- #
def build_bundle(experiments: Sequence, output_dir: str | Path) -> dict[str, Path]:
    """Write the full publication bundle into ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def _write(name: str, text: str) -> None:
        path = out / name
        path.write_text(text, encoding="utf-8")
        written[name] = path

    _write("publication_metrics.csv", to_csv(headline_rows(experiments), HEADLINE_COLUMNS))
    _write("publication_per_class.csv", to_csv(per_class_rows(experiments), PER_CLASS_COLUMNS))
    _write("publication_tables.md", render_markdown(experiments))
    _write("publication_tables.tex", render_latex(experiments))
    _write(
        "publication_experiments.json",
        json.dumps(
            [
                exp.as_dict() if hasattr(exp, "as_dict") else dict(exp)
                for exp in experiments
            ],
            indent=2,
            default=str,
        ),
    )
    _write(
        "dataset_cards.md",
        "\n\n".join(dataset_card(experiment) for experiment in experiments),
    )
    _write("provenance.md", provenance(experiments))
    _write("publication_report.html", render_html(experiments))
    return written


__all__ = [
    "CAVEAT",
    "HEADLINE_COLUMNS",
    "PER_CLASS_COLUMNS",
    "build_bundle",
    "dataset_card",
    "headline_rows",
    "per_class_rows",
    "provenance",
    "render_html",
    "render_latex",
    "render_markdown",
    "to_csv",
    "to_latex",
    "to_markdown",
]
