# Blue Eye: Evaluating Pretrained Deep Learning Models for Aquatic Species Detection in Indian Environments

> **Draft status.** This is a *draft*. Sections that depend on experiments that
> have **not yet been run** are marked **`[PENDING]`** and contain no numbers.
> Where a metric is quoted from the literature or a model card it is labelled
> **`[PUBLISHED]`** and is *not* a Blue Eye measurement. No citation, value,
> dataset or contribution in this draft has been invented; every reference is a
> real artefact with a verifiable URL or DOI. Blue Eye is an applied system and
> an open evaluation pipeline — it does **not** claim a new architecture.

---

## 1. Abstract

Blue Eye ("BlueEye") is an open-source marine and aquatic biodiversity
monitoring application built on the *marine-detect* project ([6]). It performs
image and video object detection with pretrained models and records per-user
detection history. This paper asks a narrow, testable question: *how well do
publicly available pretrained aquatic detectors transfer to Indian-region
imagery, and what evidence exists for their generalisation?* We describe the
Blue Eye system, catalogue the pretrained models it integrates (two core models
plus seven additional verified checkpoints), document the licensed datasets
available for Indian and adjacent regions, and present a reproducible
evaluation pipeline that computes precision, recall, mAP@0.5 and mAP@0.5:0.95
with full test-condition provenance. To date, **no Indian-region labelled test
set ships with Blue Eye**, so the quantitative comparison is **`[PENDING]`**;
we report the evaluation of the pipeline itself and the honest status of each
model rather than any invented score. The contribution is an applied,
reproducible evaluation harness and a candid evidence audit — not a new
detector and not a claim of state-of-the-art performance.

## 2. Keywords

Aquatic object detection · Transfer learning · YOLOv8 · RT-DETR · Indian marine
biodiversity · Reef fish · Sharks and rays · Sea turtles · Model evaluation ·
Reproducibility · Generalisation · mAP.

## 3. Introduction and motivation

India has a long coastline (Arabian Sea, Bay of Bengal, Lakshadweep, Andaman &
Nicobar) and rich but under-documented marine biodiversity. Manual monitoring
from imagery is slow and expensive, and deep-learning detectors can assist.
However, the widely available pretrained detectors are trained on non-Indian
data: Pacific reef surveys, Mediterranean coastal cameras, brackish European
cameras, or general aquarium imagery. Applying such a model to Indian waters is
a transfer-learning problem, and its accuracy cannot be assumed.

Blue Eye is built so that this assumption can be tested rather than presumed.
It is (i) an operational detection application with a single Detection section
for images and video, a real dashboard, and per-user detection history, and
(ii) a reproducible evaluation pipeline that produces standard detection
metrics with full provenance and never fabricates a score.

## 4. Problem statement and research objectives

**Formal problem statement.** Given a set of pretrained detectors `M` and a
labelled dataset `D` with dataset class set `C_D`, and each model `m` with its
own class set `C_m`, define the comparison *only* over the shared classes
`C_D ∩ C_m`. Compute per-class average precision at IoU thresholds
`t ∈ {0.5, 0.55, …, 0.95}`, the means `mAP@0.5` and `mAP@0.5:0.95`, precision
and recall at the best-F1 operating point, and inference time, together with
the test conditions (device, versions, resolution, split). If `C_D ∩ C_m = ∅`
the comparison is **undefined and must be refused**.

**Research objectives.**

1. Preserve and characterise the two upstream models (`fish_inv`, `megafauna`)
   without altering their weights, class mappings or inference behaviour.
2. Integrate additional *verified*, publicly licensed aquatic models as
   independent, selectable adapters, keeping the originals unchanged.
3. Build a reproducible evaluation pipeline that computes the metrics above
   and never invents results.
4. Clearly distinguish *published* metrics from *locally reproduced* metrics.
5. State honestly which evidence for Indian-region generalisation is missing.

## 5. Related work, with verified scholarly references

* **One-stage detectors.** The YOLO family ([1]) underpins the core models; the
  aquarium adapter uses RT-DETR ([3]).
* **Metric conventions.** We follow the COCO evaluation protocol ([2]).
* **Base project.** The two core models come from the *marine-detect* software
  ([6]) — an AGPL-3.0-only repository, i.e. an artefact rather than a
  peer-reviewed result claimed here.
* **General fish detection.** The Community Fish Detection dataset/detector
  ([7]) and the Fishial detector ([8]) provide single-class and general fish
  detection.
* **General animal/object detection.** Open Images V4/V7 ([4]) supplies the
  601-class `wildlife_oi7` adapter.
* **Regional / underwater data.** The Mediterranean OBSEA species detector
  ([11]) and the Underwater Species Dataset ([12]) are region/domain-specific;
  none is Indian.

> **Novelty caveat.** A full systematic literature review has *not* been
> completed. No novelty claim is made; Section 13 records this as a required
> step before submission.

## 6. Dataset sources, licensing, and class distribution

Blue Eye ships **no datasets**. The following are documented, licensed sources.
Licences are those stated by the source; **dataset licences and weight licences
are different things** and both must be checked before redistribution.

| Dataset | Domain / region | Classes | Licence | Source |
|---|---|---|---|---|
| Underwater Species Dataset (NR) | Marine (not India-collected) | 7 (seals, dolphins, sea turtles, octopus, seahorse, sharks, whales) | CC BY 4.0 | Mendeley `4tp83br92z` [12] |
| DePondFi / Orange Chromide | **South Indian freshwater pond/aquaculture** | 1 (`fish`) | CC BY 4.0 | Mendeley `7w45jx35hd` [9] |
| DePondFi benchmark | South Indian freshwater aquaculture | pond fish | verify at source | doi:10.4218/etrij.2024-0383 [10] |
| FishInv / MegaFauna sets | Reef / lagoon; megafauna | 15 + 3 | upstream repository licence | [6] |

**Indian ocean/coastal vs South Indian pond imagery.** These are distinct
domains and must be reported separately. DePondFi is freshwater pond/aquaculture
imaging from South India, whereas every integrated marine model is trained on
reef, aquarium, coastal or brackish footage. A DePondFi metric measures
*freshwater pond* performance, not Indian-marine generalisation.

**Class distribution.** **`[PENDING]`** — to be measured from the prepared
dataset with `scripts/prepare_dataset.py`, which writes per-class counts to
`dataset_info.json`.

## 7. System architecture

Blue Eye is a Python application (`app/`) with a Streamlit web UI, a small
command-line interface, and a single inference entry point
(`MarineDetector`) that the UI, CLI and evaluation pipeline all share.

```
app/
├── detection/     detector + model registry/loading (YOLOv8 & RT-DETR)
├── evaluation/    reproducible evaluation: metrics, datasets, compat, report
├── ui/pages/      dashboard, detection, history, models, account
├── auth/          local accounts, bcrypt hashing, sessions, per-user data
├── content/       species reference material
└── utils/         validation, statistics, file utilities
data/              user inputs (per-user), prepared datasets (none shipped)
models/            core weights (fetched on demand) + custom registry
outputs/           detections, reports, evaluation results, auth database
```

Authentication is enforced on the backend (`app/auth`) — each user can read
only their own profile, detections and history; the evaluation pipeline is
offline and independent of the UI.

## 8. Model descriptions and supported classes

**Core models (unchanged, from marine-detect [6]).**

| id | Classes | Notes |
|---|---|---|
| `fish_inv` | 15 fish & invertebrate classes | Reef fish families + invertebrates |
| `megafauna` | `shark`, `ray`, `turtle` | Marine megafauna |

**Additional verified models (opt-in; each weight is SHA-256-verified).**

| id | Architecture | Classes (exact) | Weight licence | Source |
|---|---|---|---|---|
| `aquatic_brackish` | YOLOv8s | crab, fish, jellyfish, shrimp, small_fish, starfish | AGPL-3.0 | HF `dronefreak/brackish-yolov8s` [13] |
| `underwater_fish` | YOLOv8n | fish | US Gov. work (royalty-free) | HF `akridge/…grayscale` [15] |
| `aquarium_marine` | RT-DETR | fish, jellyfish, penguin, puffin, shark, starfish, stingray | AGPL-3.0 | HF `Kanagavel/aquarium-rtdetr` [14] |
| `obsea_mediterranean` | YOLOv8x | 21 Mediterranean species (incl. `Myliobatidae` rays) | CC BY 4.0 | Zenodo `14910365` [11] |
| `community_fish` | YOLOv12x | fish | AGPL-3.0 | GitHub `filippovarini/community-fish-detector` [7] |
| `fishial_detector` | YOLO26n | Fish | MIT | GitHub `fishial/fish-identification` [8] |
| `wildlife_oi7` | YOLOv8n | 601 Open Images v7 classes (incl. Shark, Sea turtle, Jellyfish, …) | AGPL-3.0 | Ultralytics assets / Open Images [4] |

> **Indian specificity.** *None* of these models is trained for Indian waters.
> `obsea_mediterranean` detects Mediterranean species; `aquatic_brackish` was
> trained on a single Danish brackish camera. They are region *analogues*, not
> Indian models. The single-class models detect a generic fish and identify no
> species.

## 9. Methodology and experimental setup

1. **Input.** A labelled YOLO dataset with `train`/`val`/`test` splits.
2. **Class compatibility.** `app.evaluation.compat` matches dataset and model
   class names exactly (case-insensitive); unsupported dataset classes are
   reported and excluded from per-class scoring; empty overlap is refused.
3. **Inference.** `MarineDetector.predict_image` is called unchanged, with the
   per-model recommended thresholds; the detector's clamping and cross-model
   duplicate suppression apply as in the application.
4. **Resolution.** Ultralytics' default input size is used, because `imgsz` is
   not currently exposed through `predict_image` (documented limitation).
5. **Metrics.** Greedy per-class IoU matching at thresholds 0.5…0.95 → AP,
   mAP@0.5, mAP@0.5:0.95, precision and recall ([2]).
6. **Provenance.** Device, CPU/GPU and package versions are captured per run
   (`environment.json`).
7. **Independence.** A test split is never used for training; for video data,
   frames from one recording must not be split across train and test.

**Hardware / software.** **`[PENDING]`** — captured automatically per run;
no fixed machine is claimed here.

## 10. Evaluation metrics

* **IoU** = area(intersection) / area(union) of two boxes.
* **AP@t** = area under the interpolated precision-recall curve at IoU `t`.
* **mAP@0.5** = mean AP@0.5 over classes.
* **mAP@0.5:0.95** = mean over `t ∈ {0.5, 0.55, …, 0.95}` of mAP@t ([2]).

Precision/recall are reported at the maximum-F1 operating point. The pipeline
is implemented in `app/evaluation/metrics.py` and unit-tested with synthetic
data (`tests/test_evaluation.py`).

## 11. Results and comparison tables

**`[PENDING]`** — no Indian-region labelled test split has been evaluated, so
no local metric exists. The table below is the *template* that
`scripts/run_evaluation.py` fills from a real run; every cell would come from
`metrics.csv`.

| Model | Dataset / split | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | s/image | Device |
|---|---|---|---|---|---|---|---|
| `fish_inv` | `[PENDING]` | — | — | — | — | — | — |
| `megafauna` | `[PENDING]` | — | — | — | — | — | — |
| `obsea_mediterranean` | `[PENDING]` | — | — | — | — | — | — |
| `aquarium_marine` | `[PENDING]` | — | — | — | — | — | — |

**Published vs reproduced.** The `wildlife_oi7` model card reports a low mAP
(≈18.4) on Open Images; that value is **`[PUBLISHED]`**, not a Blue Eye number.
No `[PUBLISHED]` value is placed in the table above, and results from different
datasets are never compared as if directly equivalent.

## 12. Error analysis and discussion

**`[PENDING]`** quantitative error analysis. Qualitatively, the following are
*expected* failure modes that must be measured rather than assumed:

* **Domain shift** — models trained on clear reef/aquarium footage may fail on
  turbid, low-light or green-water Indian coastal imagery.
* **Taxonomic gap** — a generic `fish` class cannot resolve species; a
  Mediterranean `Chromis chromis` class does not correspond to an Indian
  species.
* **Class-level confusion** — `wildlife_oi7` "Shark"/"Sea turtle" are
  class-level detections, not species.
* **Resolution** — models trained at 1024/1920 run at the default size today.
* **Small/occluded objects** — small fish and partial occlusions are the most
  likely misses; per-class AP will expose them once `D` exists.

## 13. Limitations

See also [`docs/evaluation.md`](docs/evaluation.md).

1. **No local Indian-region test set** → the central experiment is pending.
2. **No completed literature review** → no novelty claim.
3. **Input size not configurable** through the current inference entry point.
4. **Domain mismatch** between Indian coastal imagery and all integrated data.
5. **Single-frame scoring**; video datasets need recording-level splitting.
6. **No retraining** was performed in Phases 1–3.

## 14. Conclusion and future work

Blue Eye contributes an operational detection application and a reproducible,
honest evaluation pipeline. The models it integrates are real, licensed and
loadable, but none is India-specific. The immediate next step is to obtain a
licensed, labelled Indian-region test set (coastal/marine) and a separate South
Indian freshwater pond set, prepare them with `scripts/prepare_dataset.py`, and
run `scripts/run_evaluation.py` to fill the tables in Sections 6 and 11. Only
then can a generalisation claim be made. A systematic literature review and,
later, fine-tuning of a detector on Indian data are the following steps.

## 15. References

1. Redmon, J., Divvala, S., Girshick, R., Farhadi, A. (2016). *You Only Look
   Once: Unified, Real-Time Object Detection.* CVPR 2016. arXiv:1506.02640.
2. Lin, T.-Y., Maire, M., Belongie, S., et al. (2014). *Microsoft COCO: Common
   Objects in Context.* ECCV 2014. arXiv:1405.0312.
3. Zhao, Y. et al. (2024). *DETRs Beat YOLOs on Real-time Object Detection.*
   CVPR 2024. arXiv:2304.08069.
4. Kuznetsova, A. et al. (2020). *The Open Images Dataset V4.* International
   Journal of Computer Vision 128, 1956–1981. arXiv:1811.00982.
5. Jocher, G., Chaurasia, A., Qiu, J. (2023). *Ultralytics YOLO* [software].
   <https://github.com/ultralytics/ultralytics> (AGPL-3.0).
6. Orange OpenSource. *marine-detect* [software].
   <https://github.com/Orange-OpenSource/marine-detect> (AGPL-3.0-only).
7. Varini, F. et al. *Community Fish Detection* [software/dataset].
   <https://github.com/filippovarini/community-fish-detector>.
8. Fishial. *fish-identification* [software].
   <https://github.com/fishial/fish-identification> (MIT).
9. Vijayalakshmi & Sasithradevi et al. (2024). *DePondFi / Orange Chromide
   dataset* [dataset]. Mendeley Data `7w45jx35hd`, CC BY 4.0.
   <https://data.mendeley.com/datasets/7w45jx35hd/1>.
10. Mohankumar et al. (2024). *DePondFi benchmark* [article]. ETRI Journal.
    doi:10.4218/etrij.2024-0383.
11. OBSEA / UPC (2025). *Mediterranean species detector (21 classes)*
    [software/weights]. Zenodo record `14910365`, CC BY 4.0.
    <https://zenodo.org/records/14910365>.
12. *Underwater Species Dataset (NR)* [dataset]. Mendeley Data `4tp83br92z`,
    CC BY 4.0. <https://data.mendeley.com/datasets/4tp83br92z/1>.
13. `dronefreak/brackish-yolov8s` [weights]. Hugging Face.
    <https://huggingface.co/dronefreak/brackish-yolov8s>.
14. `Kanagavel/aquarium-rtdetr` [weights]. Hugging Face.
    <https://huggingface.co/Kanagavel/aquarium-rtdetr>.
15. `akridge/yolo8-fish-detector-grayscale` [weights]. Hugging Face (NOAA).
    <https://huggingface.co/akridge/yolo8-fish-detector-grayscale>.

*[1]–[4], the [10] article and the [9] dataset article are peer-reviewed
publications; the remaining entries are software artefacts, weights or
datasets. This list is provisional and must be completed with a systematic
review before submission.*
