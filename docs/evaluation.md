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
scripts/run_evaluation.py
        │
        ├─ app.evaluation.datasets   read data.yaml + YOLO labels → pixel boxes
        ├─ app.evaluation.compat     match dataset classes ↔ model classes
        ├─ app.evaluation.runner     run MarineDetector unchanged, time it
        ├─ app.evaluation.metrics    IoU matching → AP / mAP@0.5 / mAP@0.5:0.95
        ├─ app.evaluation.environment capture device + package versions
        ├─ app.evaluation.report     write JSON / CSV / Markdown
        └─ app.evaluation.plots      optional matplotlib figures
```

The detector pipeline (`MarineDetector.predict_image`) is used **exactly as the
application uses it** — same per-model confidence thresholds, same clamping,
same cross-model duplicate suppression. The pipeline only *reads* the weights.

## 3. Usage

```bash
# 1. See the installed models and their real class lists
python scripts/run_evaluation.py --list-models

# 2. Prepare a licensed dataset you obtained yourself (CC BY / AGPL, see §5)
python scripts/prepare_dataset.py --source "D:/data/<dataset>" --name my_dataset

# 3. Evaluate one or more models on the held-out test split
python scripts/run_evaluation.py --data data/raw/my_dataset/data.yaml \
    --split test --model all --plots

# A quick smoke run over the first 25 images
python scripts/run_evaluation.py --data data/raw/my_dataset/data.yaml \
    --model fish_inv --limit 25
```

`--model` accepts any detector selection (`auto`, `all`, `fish_inv`,
`megafauna`, or a registered additional-model id such as `aquarium_marine`).
`--confidence` overrides the per-model threshold; `--device cpu|0|cuda`
overrides the compute device.

With no `--data` the script prints the setup guide and exits with code `3`
(the experiment is **pending**).

## 4. Outputs

Everything is written to `outputs/evaluation/<timestamp>/` (override with
`--out`):

| File | Contents |
|---|---|
| `results.json` | Full machine-readable record: config, environment, per-model metrics, per-class metrics, notes |
| `environment.json` | Device, CPU/GPU, Python and package versions |
| `metrics.csv` | One row per model: mAP@0.5, mAP@0.5:0.95, precision, recall, s/image, comparable classes |
| `per_class.csv` | One row per model × class: GT, predictions, precision, recall, AP@0.5, AP@0.5:0.95 |
| `summary.md` | Human-readable summary with the test conditions and comparison table |
| `metric_comparison.png`, `per_class_*.png` | Figures (only with `--plots` and when matplotlib is available) |
| `examples/*.png` | Ground truth (green) vs predictions (red) for a few successes, misses and false positives (`--plots`) |

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

## 7.1 Preprocessing and input resolution — an explicit limitation

The evaluator uses the **application pipeline unchanged**. `predict_image`
does not pass a custom `imgsz`, so Ultralytics' default (**640 px**) is used
for every model, even those that were trained at a different size (OBSEA
`yolov8x` at 1920; Community Fish at 1024). This is recorded in `config` in
`results.json`. If a larger resolution is needed, that is a change to the
inference configuration, not a change the evaluation silently makes.

## 7. Published vs reproduced metrics

BlueEye only ever reports a metric it has **reproduced locally** on a named
dataset and split. Numbers published by a model's authors on *their* test sets
are **reference** values and are recorded separately (e.g. the OBSEA model's
own evaluation and the FishInv/MegaFauna upstream README). They are never
copied into a BlueEye result and never compared as if measured under the same
conditions.

## 7. Known limitations

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
    --out outputs/evaluation/run1 --plots
```

The `environment.json` and `results.json` written by a run record the device,
package versions, dataset, split and configuration, so the table can always be
traced back to the exact conditions that produced it.

The metric code, the dataset reader, the compatibility check and the report
writers are covered by `tests/test_evaluation.py` (synthetic data, no weights,
no network). `tests/test_evaluation_integration.py` drives the same pipeline
with the **real** detector and the upstream sample image (run with
`pytest -m integration`); because no annotations are invented for that image
the metric is honestly `0`, and the test asserts the *plumbing* — every real
detection is mapped and the report files are written.
