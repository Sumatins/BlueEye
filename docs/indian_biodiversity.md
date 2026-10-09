# Indian biodiversity focus — models, datasets and integration plan

BlueEye is a final-year project on **Smart Marine Biodiversity Monitoring
through Deep Learning**. It is built on the open-source
[marine-detect](https://github.com/Orange-OpenSource/marine-detect) project
(Orange OpenSource, AGPL-3.0-only) and ships with two pretrained YOLOv8
models. This document records how BlueEye is extended toward **Indian
freshwater and coastal species**, what was actually verified, and what still
requires training.

It follows one rule throughout:

> **No fabricated weights, no placeholder predictions, no hard-coded
> results.** A species is only listed as detectable if a model actually
> supports that class. If only a dataset exists, BlueEye ships a training
> pipeline — it does not pretend the dataset is a trained model.

---

## 1. What already works (the built-in models)

The two upstream models are preserved unchanged and are the source of truth
for what BlueEye can detect today:

| Model | Key | Classes | Habitat |
|---|---|---|---|
| Fish & Invertebrates | `fish_inv` | 15: `fish`, `serranidae`, `scaridae`, `chaetodontidae`, `lutjanidae`, `muraenidae`, `haemulidae`, `cromileptes_altivelis`, `cheilinus_undulatus`, `bolbometopon_muricatum`, `giant_clam`, `urchin`, `sea_cucumber`, `crown_of_thorns`, `lobster` | Marine (Indo-Pacific reefs) |
| MegaFauna | `megafauna` | 3: `shark`, `ray`, `turtle` | Marine (open water) |

Class names are read from the weights at runtime (`ModelManager.get_class_names`);
the lists above are only a convenience copy. Recommended thresholds: 0.523
(FishInv) and 0.546 (MegaFauna), as published upstream.

> **Registry status (October 2026).** The **active** registry holds five working
> detectors: the two core models above plus the three verified additional
> models in [§2.2](#22-pretrained-aquatic-models-that-were-integrated-verified).
> The six entries that cannot run today (`freshwater_fish`,
> `gangetic_dolphin`, `freshwater_turtle`, `gharial`, `indian_marine`,
> `aquarium_axera`) were moved to
> [`models/custom/inactive_models.json`](../models/custom/inactive_models.json).
> That file is **documentation only** — it is never loaded, nothing from it
> appears in a selector or count, and its provenance is kept so an entry can be
> reactivated once real weights exist.

### 1.1 What MegaFauna already covers for the "future marine" targets

The task asks to check whether MegaFauna already detects the Indian marine
species before building anything new. It **partially** does:

| Requested species | Covered today? | How |
|---|---|---|
| Whale shark | Yes — group level | MegaFauna class `shark` (not species-specific) |
| Marine rays | Yes — group level | MegaFauna class `ray` |
| Olive Ridley sea turtle | Yes — group level | MegaFauna class `turtle` |
| **Octopus** | **No** | Not in FishInv or MegaFauna |

So whale shark / rays / Olive Ridley can be detected **as their group**, not
as a named species, today. A dedicated model is only worth training for
*species-level* output and for **octopus**, which the built-ins miss.

---

## 2. Research summary (per target)

Legend — **Status**: `covered` = already detected (group or species);
`needs training` = a dataset exists, no trustworthy weights;
`needs data` = no verified dataset found, data must be collected;
`architecture ready` = plumbing done, waiting for weights.

| Target | Habitat | Pretrained weights found? | Dataset found | Status |
|---|---|---|---|---|
| A. Freshwater fish | Freshwater (Karnataka & South Indian ponds/rivers) | ❌ none verified | ✅ DePondFi / Orange Chromide (CC BY 4.0) | **needs training** |
| B. Gangetic river dolphin | Freshwater (Ganga/Brahmaputra/Chambal) | ❌ none | ❌ none verified | **needs data** |
| C. Freshwater turtle | Freshwater (rivers/lakes/ponds) | ❌ none | ⚠️ none freshwater-specific & licensed | **needs data** |
| D. Gharial | Freshwater (clear, fast rivers) | ❌ none | ❌ none verified | **needs data** |
| E. Indian marine species | Marine (Arabian Sea / Bay of Bengal) | ❌ none species-specific | ✅ Underwater Species Dataset (NR), CC BY 4.0 | **architecture ready / needs training** |

### 2.1 Indian-species pretrained models: none found

Everything below was searched, but nothing provided **species-specific,
licensed, ready-to-download Indian weights**:

- **AquaYOLO** (Vijayalakshmi et al., *Scientific Reports* 2025) reports
  `aquayolo1/2/3.pt` trained on DePondFi. **No trustworthy public download
  source was found**, so it is not integrated. (A different 2025 "AquaYOLO"
  by Lu et al. targets *sonar* imagery, not pond video.)
- **YOLO-Fish** (tamim662, GPL-3.0) is a **Darknet/YOLOv3** model for a
  single generic `Fish` class — not YOLOv8, not species-specific, and would
  not run through BlueEye's Ultralytics pipeline.
- **Roboflow Universe** projects (dolphin, whale-shark, freshwater fish, …)
  are often unpublished or state no licence; a model may not be registered
  without a verifiable class list and licence, and their weights usually
  require an account/API key rather than a direct public download.
- **Community Fish Detector** (open weights) detects a single generic `fish`
  class across domains — redundant with the built-in `fish` class.

This is deliberate: a fake "India model" would be worse than none.

### 2.2 Pretrained aquatic models that *were* integrated (verified)

Because the Indian-species models above could not be verified, BlueEye was
extended with **six genuinely pretrained, publicly downloadable aquatic
detectors** instead. Each was downloaded, loaded with Ultralytics and run on
a real image before being registered; none is India-specific, and each is
clearly documented as such. They are **opt-in** (`"auto": false`): `auto`
still runs only the two core models, so existing behaviour is unchanged.

| id | Classes | Architecture | Licence | Source | Status |
|---|---|---|---|---|---|
| `aquatic_brackish` | `crab`, `fish`, `jellyfish`, `shrimp`, `small_fish`, `starfish` | YOLOv8s | AGPL-3.0 | [dronefreak/brackish-yolov8s](https://huggingface.co/dronefreak/brackish-yolov8s) | **READY** |
| `aquarium_marine` | `fish`, `jellyfish`, `penguin`, `puffin`, `shark`, `starfish`, `stingray` | RT-DETR | AGPL-3.0 | [Kanagavel/aquarium-rtdetr](https://huggingface.co/Kanagavel/aquarium-rtdetr) | **READY** |
| `underwater_fish` | `fish` | YOLOv8n | US Gov. work (public domain / royalty-free) | [akridge/yolo8-fish-detector-grayscale](https://huggingface.co/akridge/yolo8-fish-detector-grayscale) | **READY** |
| `obsea_mediterranean` | 21 Mediterranean species (groupers, seabreams, wrasse, moray, Myliobatidae, …) | YOLOv8x | CC-BY-4.0 | [OBSEA / Zenodo 14910365](https://zenodo.org/records/14910365) | **READY** |
| `community_fish` | `fish` | YOLOv12x | AGPL-3.0 | [filippovarini/community-fish-detector](https://github.com/filippovarini/community-fish-detector) | **READY** |
| `fishial_detector` | `Fish` | YOLO26n | MIT | [fishial/fish-identification](https://github.com/fishial/fish-identification) | **READY** |

Notes and provenance limits:

- `aquatic_brackish` is trained on one **Danish brackish** camera — an
  analogue for Indian **estuaries/backwaters**, not Indian rivers.
- `aquarium_marine` is **community-contributed** with no published metrics;
  the weights load and detect, but no evaluation numbers are claimed.
- `underwater_fish` is trained on **grayscale** underwater footage and may
  under-detect on colour images.
- `obsea_mediterranean` is the only **species-level** model here, but its 21
  classes are **Mediterranean** reef species, not Indian ones.
- `community_fish` (YOLOv12x) and `fishial_detector` (YOLO26n) each emit a
  single **generic** `fish`/`Fish` class — they are strong fish detectors but
  do **not** identify a species, so they must never be shown as a species
  detection.

Fetch them (SHA-256-pinned) with:

```bash
python scripts/fetch_additional_models.py
```

Weights live under `models/additional/<id>/` and stay out of Git. Verify
every model — core and additional — with:

```bash
python -m app.main --verify-models
```

A fourth candidate, **AXERA-TECH/YOLOv8-Aquarium**, is kept in the inactive
registry as `aquarium_axera` with status **INCOMPATIBLE**: only Axera NPU
(`.axmodel`) exports are published, so it cannot run through the Ultralytics
pipeline. It is not registered as an active model.

---

## 3. Datasets (verified sources and licences)

These are the datasets BlueEye points at for training. None is downloaded
automatically and none is committed to Git.

| Dataset | Scope | Size | Licence | Link |
|---|---|---|---|---|
| **DePondFi / Orange Chromide** (Vijayalakshmi & Sasithradevi, *Data in Brief* 2024) | Freshwater pond fish (*Etroplus maculatus*), 1 class `fish` | 586 images / 10,607 instances | **CC BY 4.0** (Mendeley); article CC BY-NC 4.0 | <https://data.mendeley.com/datasets/7w45jx35hd/1> |
| **Underwater Species Dataset (NR)** | 7 marine classes: seals, dolphins, sea turtles, **octopus**, seahorse, sharks, whales | 1,728 images | **CC BY 4.0** | <https://data.mendeley.com/datasets/4tp83br92z/1> |
| **Community Fish Detection (CFD)** | Single class `fish`, multi-domain | ~2M images / 935K boxes | per constituent dataset (CC / MIT subsets) | <https://lila.science/datasets/community-fish-detection-dataset> |
| **Upstream FishInv / MegaFauna sets** | Reef fish families + shark/ray/turtle | 12,243 + 8,130 images | see upstream repo (AGPL-3.0-only) | <https://github.com/Orange-OpenSource/marine-detect> |
| **DePondFi benchmark** (Mohankumar et al., *ETRI Journal* 2024) | Larger pond-fish benchmark (reported ~8,150 images / ~50k boxes) | — | to verify at source | doi:10.4218/etrij.2024-0383 |

> Re-check the licence on the source page before redistributing any weights
> derived from a dataset. Dataset licences and model-weight licences are not
> the same thing.

### 3.1 Downloadability audit (October 2026)

The licences above are real, but a licence alone is not a dataset. A direct
check of the two Mendeley records found **two concrete blockers** that stop a
genuine training run today:

| Dataset | What was tested | Result |
|---|---|---|
| DePondFi / Orange Chromide | Mendeley public API `…/datasets/7w45jx35hd/files?folder_id=root&version=1` and the page's "Download All" control | The API returns **no files** and "Download All" is **disabled**; the 586 images + labels are not served to anonymous users. **Blocked** until the authors enable public download or the data is obtained directly. |
| Underwater Species Dataset (NR) | Downloaded the published archive (`NR Underwater Species.rar`, 96,651,036 B, sha256 `8840ce88…da258c`) and unpacked it with the 7-Zip RAR codec | The archive contains **1,728 images only** (train 1,291 / valid 294 / test 143) and **no YOLO `.txt` label files** and no `data.yaml`. Filenames hint at the 7 classes (octopus, seals, seahorse, sea turtles, sharks, whales, frame) but without annotations it **cannot** be used for supervised training as-is. **Blocked** until annotations are published or the images are re-annotated. |

Neither blocker is hidden inside the app: `freshwater_fish` and
`indian_marine` stay reported as **NEEDS TRAINING**, exactly as they are. When
usable annotations become available, [`scripts/prepare_dataset.py`](../scripts/prepare_dataset.py)
validates the class list and lays the data out for
[`scripts/train.py`](../scripts/train.py) — it refuses a dataset that has
images but no labels, so an unlabelled archive can never be mistaken for a
trainable one.


---

## 4. How a new model plugs in (no application code changes)

```
User selects a model (UI / CLI / API)
        ↓
Model registry   (built-in MODEL_REGISTRY  +  models/custom/registry.json)
        ↓
ModelSpec  (id, name, weights path, classes, threshold, source, licence)
        ↓
ModelManager.load(key)   →   Ultralytics  YOLO("models/.../YourModel.pt")
        ↓
MarineDetector.predict_image / predict_frame   (single inference path)
        ↓
Existing pipeline → boxes, labels, statistics, annotated image/video, JSON
```

A model registered in `models/custom/registry.json` is automatically
available:

- on the **Models** page (with its real status: *ready* / *not installed* /
  *needs training* / *incompatible*),
- in the **Detect** model selector,
- in the **CLI**: `python -m app.main --image x.jpg --model <id>`.

Two convenience selections exist:

- **`auto`** (default) runs only the **core** models — `fish_inv` +
  `megafauna` — so the original behaviour is preserved exactly;
- **`all`** runs **every installed model** (core *and* additional). When
  several models report the same class for the same object (IoU ≥ 0.7), the
  lower-confidence duplicate is dropped, so combined results stay clean.

Additional models default to opt-in (`"auto": false`); a user-added model
defaults to `auto: true` unless it sets `"auto": false`.

Weights are loaded **once** and cached by `ModelManager`, so video frames
never trigger repeated disk loads. A missing weight file produces a clear
error (exit code 3) and never disables the other working models.

---

## 5. Freshwater vs marine — intended habitat

BlueEye keeps the two environments explicitly separate so a model is never
applied outside the water body it was built for:

| Model (planned id) | Environment | Intended habitat |
|---|---|---|
| `fish_inv` | Marine | Indo-Pacific coral reefs |
| `megafauna` | Marine | Open water (shark / ray / turtle) |
| `aquatic_brackish` | **Brackish / estuarine** | Estuaries and backwaters (trained on a Danish brackish site) |
| `aquarium_marine` | Marine | Reef / aquarium footage (fish, shark, stingray, jellyfish, …) |
| `underwater_fish` | Underwater | Grayscale camera footage (single `fish` class) |

**Inactive target ids** (not registered; preserved in
`models/custom/inactive_models.json`): `freshwater_fish`, `gangetic_dolphin`,
`freshwater_turtle`, `gharial` (all intended for **freshwater**) and
`indian_marine` (intended for the **Arabian Sea / Bay of Bengal**). Because
none has verified weights, none is offered in the model selector.

Not every animal is present in every water body — a freshwater model should
not be expected to work on reef footage, and vice versa.

---

## 6. Training and evaluation

Dataset config templates live in [`training/`](../training/README.md), one per
target. The workflow is:

```bash
# 0. Validate + lay out a downloaded, licensed dataset (refuses unlabelled data)
python scripts/prepare_dataset.py --source "D:/data/pond_fish" \
    --name freshwater_fish --classes fish

# train (transfer-learning from the existing weights, or a small backbone)
python scripts/train.py --data training/datasets/freshwater_fish.yaml \
    --model models/fish_inv/FishInv.pt --epochs 100 --imgsz 640 --batch 16

# evaluate on the held-out test split before claiming any metric
python scripts/evaluate.py \
    --weights runs/detect/blueeye/weights/best.pt \
    --data training/datasets/freshwater_fish.yaml --split test --save-report
```

Metrics reported: **precision, recall, mAP@50, mAP@50:0.95** (from
Ultralytics `model.val`). Split **by sequence, not by frame**, to avoid
inflated scores. Register the finished model in
`models/custom/registry.json`; a generic template is in
`models/custom/registry.json.example`, and the untrained Indian targets (with
their blockers) are preserved in `models/custom/inactive_models.json`.

Because underwater imagery differs strongly from ordinary photographs,
**evaluate on representative underwater footage** whenever possible.

---

## 7. Known limitations

- The built-in `shark` / `ray` / `turtle` classes are **group-level**, not
  species-level: they cannot distinguish a whale shark from another shark, or
  an Olive Ridley from another sea turtle.
- `fish_inv` is trained on Indo-Pacific reef families, **not** on Indian
  freshwater species — accuracy on local fish is unverified.
- The freshwater pond dataset available today (DePondFi/Orange Chromide) is
  **single-species** (`Etroplus maculatus`); it does not provide a general
  Indian freshwater fish classifier.
- **Neither candidate dataset is downloadable as labelled training data today**
  (see §3.1): DePondFi's files are not served to anonymous users, and the NR
  archive ships images without annotations. This — not a lack of code — is why
  `freshwater_fish` and `indian_marine` remain untrained and are kept as
  **inactive** research entries rather than active models.
- **Compute:** the environment this was verified in has **no CUDA GPU** (8 CPU
  cores, ~7.7 GB RAM, `torch … +cpu`). Even with annotations, five 100-epoch
  runs would be impractical here; a GPU (or a reduced schedule documented in
  the training run) is the next required resource.
- Gangetic dolphin, freshwater turtle and gharial have **no verified,
  licensed, ready-to-train detection dataset** found here; they need field /
  UAV imagery to be collected and annotated.
- The three integrated additional models (`aquatic_brackish`,
  `aquarium_marine`, `underwater_fish`) are **not India-specific**. They add
  coverage of generic aquatic groups (fish, shark, stingray, jellyfish, crab,
  shrimp, starfish) but must not be presented as Indian-species models.
- `aquarium_marine` has **no published evaluation metrics** (community model);
  it is marked READY only because the weights load and a real inference test
  succeeds, not because a benchmark was reproduced.
- No metrics are quoted for any Indian model because no Indian model has been
  trained and evaluated yet. Training is done by the user; BlueEye never
  invents numbers.
- Weights and large datasets are deliberately kept out of Git.

---

## 8. References

- Upstream project: <https://github.com/Orange-OpenSource/marine-detect> (AGPL-3.0-only)
- Vijayalakshmi M., Sasithradevi A. (2024). *A comprehensive annotated image
  dataset for real-time fish detection in pond settings.* Data in Brief 57:111007. <https://doi.org/10.1016/j.dib.2024.111007>
- Vijayalakshmi M. et al. (2025). *AquaYOLO: Advanced YOLO-based fish
  detection for optimized aquaculture pond monitoring.* Scientific Reports. <https://www.nature.com/articles/s41598-025-89611-y>
- Mohankumar V. et al. (2024). *A benchmark dataset and ensemble YOLO method
  for pond fish detection.* ETRI Journal. doi:10.4218/etrij.2024-0383
- Naveen P., Rajasekaran T. (2024). *Underwater Species Dataset (NR).* Mendeley Data. <https://data.mendeley.com/datasets/4tp83br92z/1>
- Community Fish Detection dataset: <https://lila.science/datasets/community-fish-detection-dataset>
- Ultralytics YOLOv8: <https://docs.ultralytics.com/models/yolov8/>

### Integrated additional-model sources

- `aquatic_brackish` — DetectionBench / Brackish Underwater: <https://huggingface.co/dronefreak/brackish-yolov8s> (AGPL-3.0)
- `aquarium_marine` — Aquarium RT-DETR: <https://huggingface.co/Kanagavel/aquarium-rtdetr> (AGPL-3.0)
- `underwater_fish` — Grayscale fish detector (NOAA): <https://huggingface.co/akridge/yolo8-fish-detector-grayscale> (US Gov. work / royalty-free)
- `aquarium_axera` — Axera NPU export (registered as INCOMPATIBLE): <https://huggingface.co/AXERA-TECH/YOLOv8-Aquarium>
