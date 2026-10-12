# Model evaluation pipeline (Phase 3)

This document describes BlueEye's **reproducible evaluation pipeline**: how to
run the existing (unmodified) detectors over a labelled dataset, what is
measured, which datasets are usable, and — honestly — which experiments are
still **pending** because no labelled Indian-region test set ships with the
project.

> **Status: PENDING.** BlueEye ships **no datasets**. Until a licensed,
> labelled test split is added under `data/raw/`, the pipeline runs, the
> command below explains the required setup, and **no metric is produced**.
> BlueEye never invents a score.

---

## 1. What the pipeline is for

The research question is: *how well do pretrained underwater/ aquatic detectors
generalise to Indian-region imagery?* Answering it requires four things that
BlueEye keeps deliberately separate:

1. **A labelled dataset** with bounding boxes (YOLO format) and a **test split**
   that is never used for training.
2. **Class compatibility** between the model's classes and the dataset's
   annotations — reported, never silently remapped.
3. **Standard metrics** (precision, recall, mAP@0.5, mAP@0.5:0.95), per class.
4. **Full test conditions**: device, package versions, input resolution and the
   exact dataset/split.

## 2. How it works

```
scripts/run_evaluation.py                         app/ui/pages/evaluation.py
        │                                                  │
        ├─ app.evaluation.datasets   read data.yaml + YOLO labels → pixel boxes
        ├─ app.evaluation.validation read-only errors/warnings + fingerprint
        ├─ app.evaluation.compat     match dataset classes ↔ model classes
        ├─ app.evaluation.runner     run MarineDetector unchanged, time it
        ├─ app.evaluation.metrics    IoU matching → AP / mAP@0.5 / mAP@0.5:0.95
        ├─ app.evaluation.environment capture device + package versions
        ├─ app.evaluation.report     write JSON / CSV / Markdown / HTML
        ├─ app.evaluation.plots      optional matplotlib figures
        ├─ app.evaluation.jobs       background run + cancellation (UI)
        └─ app.evaluation.experiments persistent, owner-scoped store (UI)
```

The detector pipeline (`MarineDetector.predict_image`) is used **exactly as the
application uses it** — same per-model confidence thresholds, same clamping,
same cross-model duplicate suppression. The pipeline only *reads* the weights.

## 3. Usage

```bash
# 1. See the installed models and their real class lists
python scripts/run_evaluation.py --list-models

# 2. Prepare a licensed dataset you obtained yourself (CC BY / AGPL, see §5)
#    The path is positional; the output is written to data/raw/my_dataset.
python scripts/prepare_dataset.py "D:/data/<dataset>" --name my_dataset

# 3. Evaluate one or more models on the held-out test split
python scripts/run_evaluation.py --data data/raw/my_dataset/data.yaml \
    --split test --model all --plots

# A quick smoke run over the first 25 images
python scripts/run_evaluation.py --data data/raw/my_dataset/data.yaml \
    --model fish_inv --limit 25

# Persist the run so it appears in the Research Evaluation Center
python scripts/run_evaluation.py --data data/raw/my_dataset/data.yaml \
    --split test --model all --record --owner <account-id> --name "Baseline"
```

`--model` accepts any detector selection (`auto`, `all`, `fish_inv`,
`megafauna`, or a registered additional-model id such as `aquarium_marine`).
When a selection expands to several models they are evaluated **independently**
on the same split — scores are never merged. `--confidence` overrides the
per-model threshold; `--device cpu|0|cuda` overrides the compute device;
`--iou` sets the IoU used for the PR / F1 / confusion figures (the reported
mAP@0.5 and mAP@0.5:0.95 always use the standard COCO thresholds).

`--record` also persists the run (and the validated dataset) in the evaluation
database next to the report files, so it appears in the UI after a restart;
`--owner` names the account it belongs to (defaults to the `cli` sentinel).

`scripts/prepare_dataset.py` accepts the dataset directory either positionally
(recommended) or with `--source`; add `--register` to also validate the
prepared layout and save it in the evaluation database.

Dataset validation is separate from conversion: before a run, the CLI and the
UI call `app.evaluation.validation.validate_dataset`, which reports **errors**
(the dataset cannot be scored: a missing/empty requested split, unreadable
config, malformed label rows, out-of-range boxes, class ids with no name, an
inconsistent `nc`) and **warnings** (a missing label treated as a valid
negative, an unreadable image, a declared-but-absent split, possible
train/test leakage, duplicate class names). Nothing is ever rewritten; the
source is opened read-only, and a dataset is only *ready* when it has no
errors and the requested split contains images.

Exit codes: `0` success, `1` partial success (some models failed, the rest are
recorded), `2` invalid input (missing/unreadable config, dataset not ready),
`3` pending (no dataset, or no model could be evaluated). With no `--data` the
script prints the setup guide and exits `3` (the experiment is **pending**).

## 4. Outputs

Everything is written to `outputs/evaluation/<timestamp>/` (override with
`--out`). The UI writes each run to `outputs/evaluation/<user_id>/<experiment_id>/`.

| File | Contents |
|---|---|
| `results.json` | Full machine-readable record: experiment, config, environment, per-model metrics, per-class metrics, notes, warnings and failed steps |
| `environment.json` | Device, CPU/GPU, Python and package versions |
| `metrics.csv` | One row per model: mAP@0.5, mAP@0.5:0.95, precision, recall, s/image, comparable classes |
| `per_class.csv` | One row per model × class: GT, predictions, precision, recall, AP@0.5, AP@0.5:0.95 |
| `summary.md` | Human-readable summary with the test conditions and comparison table |
| `report.html` | Self-contained report (identification, purpose, dataset + fingerprint, split + class names, model/weights, config, metric definitions + values, per-class, embedded plots, environment, timestamps, scope statement) — shareable and printable to PDF |
| `metric_comparison.png`, `pr_*.png`, `f1_*.png`, `confusion_*.png`, `per_class_*.png` | Figures (only with `--plots` / the UI plots toggle, and when matplotlib is available) |
| `examples/*.png` | Ground truth (green) vs predictions (red) for successes, misses and false positives (`--plots`) |

The UI additionally persists every experiment in its **own** SQLite database
(default `outputs/evaluation/blueeye_evaluation.db`, override with
`BLUEEYE_EVALUATION_DB`), separate from the authentication and observations
databases. Each row stores the owner, name/purpose/notes, dataset + fingerprint,
split, class mapping, model ids + weight fingerprints, evaluator and package
versions, parameters, timestamps, status, metrics, per-class results, artifact
references, warnings/errors and the git commit. It is only ever read and
written through owner-scoped queries.

## 5. Datasets — documented sources and licences

BlueEye ships no datasets and does not download them automatically. The
following are **real, licensed** sources recorded during the Phase 2 research
(see also [`docs/indian_biodiversity.md`](indian_biodiversity.md) §3). Re-check
each licence on the source page before redistributing anything derived from it.

| Dataset | Region / scope | Relevance to India | Licence | Source |
|---|---|---|---|---|
| **Underwater Species Dataset (NR)** | 7 marine classes (seals, dolphins, sea turtles, octopus, seahorse, sharks, whales) | Marine fauna found in Indian waters (not India-collected) | CC BY 4.0 | <https://data.mendeley.com/datasets/4tp83br92z/1> |
| **DePondFi / Orange Chromide** | Freshwater pond fish (*Etroplus maculatus*), 1 class | **South Indian pond/aquaculture** (Tamil Nadu ponds) — *not* marine | CC BY 4.0 | <https://data.mendeley.com/datasets/7w45jx35hd/1> |
| **DePondFi benchmark** | Larger pond-fish benchmark | South Indian freshwater aquaculture | verify at source | doi:10.4218/etrij.2024-0383 |
| **Upstream FishInv / MegaFauna** | Reef fish families + shark/ray/turtle | Reference sets for the built-in models | upstream repo licence | <https://github.com/Orange-OpenSource/marine-detect> |

**Indian ocean/coastal imagery vs South Indian pond imagery.** These are
different domains and must not be compared as if equivalent: the DePondFi data
is freshwater pond/aquaculture imaging from South India, whereas the marine
models (FishInv, MegaFauna, OBSEA, aquarium) are trained on reef/aquarium/
coastal footage. A metric computed on DePondFi measures *freshwater pond
performance*, not Indian-marine generalisation. Report them separately.

> A licence alone is not a dataset. Verify that ground-truth boxes are present
> in YOLO format; `scripts/prepare_dataset.py` refuses unlabelled data (exit
> code 2) rather than silently accepting it.

## 6. Class compatibility (never a silent remap)

Before any metric is computed, `app.evaluation.compat.compare_classes` matches
the dataset's class names against the model's class names by exact
(case-insensitive) name:

* **supported** — a shared class name; these are scored;
* **unsupported** — a dataset class the model cannot emit; these can only ever
  be false negatives, so they are excluded from per-class scoring and listed in
  the report;
* **extra model classes** — the model can emit them but the dataset has no such
  class; they are ignored.

If nothing overlaps, the run is recorded with status `not_comparable` and the
summary states that the metrics are not a valid comparison.

## 7. Metrics

| Metric | Definition |
|---|---|
| **IoU** | Intersection-over-union of a predicted and a ground-truth box |
| **Precision / Recall** | At the model's maximum-F1 operating point at IoU 0.5 |
| **AP@0.5** | Area under the interpolated precision-recall curve at IoU 0.5, per class |
| **AP@0.5:0.95** | Mean of AP over IoU thresholds 0.50, 0.55, …, 0.95 |
| **mAP@0.5 / mAP@0.5:0.95** | Mean of the per-class AP values |

Predictions are matched greedily by descending confidence to the best
still-unmatched ground-truth box of the same class. Classes with ground truth
but no predictions score `0`; a class with no ground truth contributes nothing.

Conventions recorded with every result:

* **Precision / Recall** are reported at the fixed IoU **0.50** (the model's
  best-F1 operating point over its confidence sweep).
* **mAP@0.5** and **mAP@0.5:0.95** always use the standard COCO IoU
  thresholds; the configurable `--iou` / UI IoU only drives the PR, F1 and
  confusion-matrix figures, so a reported mAP can never be silently tied to a
  non-standard threshold.
* **Detection ≠ classification.** These metrics score *boxes and classes*; they
  say nothing about whether a species label is biologically correct.
* Any metric the evaluator could not compute (e.g. AP with no ground truth) is
  recorded as **unavailable/undefined**, never as a fabricated value.

## 7.1 Preprocessing and input resolution — an explicit limitation


The evaluator uses the **application pipeline unchanged**. `predict_image`
does not pass a custom `imgsz`, so Ultralytics' default (**640 px**) is used
for every model, even those that were trained at a different size (OBSEA
`yolov8x` at 1920; Community Fish at 1024). This is recorded in `config` in
`results.json`. If a larger resolution is needed, that is a change to the
inference configuration, not a change the evaluation silently makes.

## 7.2 Published vs reproduced metrics

BlueEye only ever reports a metric it has **reproduced locally** on a named
dataset and split. Numbers published by a model's authors on *their* test sets
are **reference** values and are recorded separately (e.g. the OBSEA model's
own evaluation and the FishInv/MegaFauna upstream README). They are never
copied into a BlueEye result and never compared as if measured under the same
conditions.

## 7.3 Known limitations

* **No Indian-region labelled test set ships with BlueEye** → the pipeline is
  *pending*; no generalisation claim can be made until one is supplied.
* **Dataset/domain mismatch** — a metric on DePondFi (freshwater pond) is not a
  marine result; a metric on the Mediterranean OBSEA set is not an Indian
  result.
* **Resolution** — evaluation runs at Ultralytics' default `imgsz` (`imgsz`
  is not yet exposed through `predict_image`); models trained at 1024/1920 may
  under-perform.
* **Single frame** — the pipeline scores images. For video datasets, split by
  recording first so near-identical frames do not leak across the split.
* **Class coverage** — general single-class models (`community_fish`,
  `underwater_fish`, `fishial_detector`) can only be fairly compared against a
  dataset's `fish` class; they do not identify species.

## 8. Reproducing

```bash
python scripts/run_evaluation.py --data <data.yaml> --split test --model all \
    --out outputs/evaluation/run1 --plots --record --owner <account-id>
```

The `environment.json` and `results.json` written by a run record the device,
package versions, dataset, split and configuration, so the table can always be
traced back to the exact conditions that produced it. With `--record` the same
provenance (plus weight fingerprints and the git commit) is stored in the
evaluation database so the experiment can be re-opened, compared and exported
from the UI after a restart. Reproducibility is *not* claimed to be perfect:
weights, library versions and hardware can change, so a re-run is expected to be
*close*, not bit-identical, and the recorded provenance is what lets a reader
judge that.


The metric code, the dataset reader, the compatibility check and the report
writers are covered by `tests/test_evaluation.py` (synthetic data, no weights,
no network). The dataset validation, persistent store, background-job
lifecycle (with a fake detector), figure/HTML output and comparison warnings are
covered by `tests/test_evaluation_workspace.py`. `tests/test_evaluation_integration.py`
drives the same pipeline with the **real** detector and the upstream sample
image (run with `pytest -m integration`); because no annotations are invented
for that image the metric is honestly `0`, and the test asserts the *plumbing* —
every real detection is mapped and the report files are written.
`tests/test_evaluation_ui_integration.py` drives the real Streamlit page end to
end and polls the persistent store until the background run reaches a terminal
status.

## 8.1 Running it from the web UI

The **Research Evaluation Center** page is the UI over this same pipeline. It is
organised into four sections:

* **Run** — pick a discovered/registered dataset, the split, the models
  (showing task, classes, weights, device, compatibility and limitations; an
  unsupported combination is disabled and explained), the confidence, IoU,
  image limit, plots toggle, experiment name and notes/purpose. Real defaults
  are filled from the registry; the configuration is validated before the run
  starts. Each selected model runs **independently** and its scores are never
  merged.
* **Datasets** — register a dataset by explicit path or `data.yaml`, or use the
  auto-discovered ones; validate it and re-validate it on demand. Errors and
  warnings are listed separately, naming the affected file/class.
* **Experiments** — every stored experiment (yours only), with its status,
  summary, sortable per-class table, real plots, diagnostics, artifact
  downloads and reproducibility details (params, versions, git commit) plus a
  copy-ready re-run command.
* **Compare** — select several *completed* experiments and compare model,
  dataset + fingerprint, split, class mapping, metrics, speed and parameters.
  When the experiments are not equivalent (different dataset fingerprint,
  split or class mapping) the comparison says so and only shows the differences
  that are actually valid; there is deliberately no universal “best".

A run executes on a background thread and moves through
`Configured → Validating → Queued/Running → Completed / Partially Completed /
Failed / Cancelled`, with progress, stored logs/errors and an optional cancel
(picked up between images). The status is persisted in the evaluation database,
so it survives an application restart; a `running` row with no live worker (a
restart happened) is shown as such rather than as still active. Reports can be
downloaded as `results.json` and as the self-contained `report.html`.

When no prepared dataset is found on disk the page reports the experiment as
**pending** and explains how to configure one - it never invents a score.
Results are written under `outputs/evaluation/<user_id>/<experiment_id>/` and
are only visible to the account that produced them. The **Model Comparison**
page follows the same rule: it runs several models on one image and shows their
real detections, confidences and timing side by side without presenting them as
an accuracy comparison.

