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

### Why no pretrained model was integrated

Everything below was searched, but none satisfied **all** of BlueEye's
integration bar (licence permits use, class list known, YOLOv8 `.pt`,
provenance verifiable, tested through BlueEye's own pipeline):

- **AquaYOLO** (Vijayalakshmi et al., *Scientific Reports* 2025) reports
  `aquayolo1/2/3.pt` trained on DePondFi. **No trustworthy public download
  source was found**, so it is not integrated. (A different 2025 "AquaYOLO"
  by Lu et al. targets *sonar* imagery, not pond video.)
- **YOLO-Fish** (tamim662, GPL-3.0) is a **Darknet/YOLOv3** model for a
  single generic `Fish` class — not YOLOv8, not species-specific, and would
  not run through BlueEye's Ultralytics pipeline.
- **Roboflow Universe** projects (dolphin, whale-shark, freshwater fish, …)
  are often unpublished or state no licence; a model may not be registered
  without a verifiable class list and licence.
- **Community Fish Detector** (open weights) detects a single generic `fish`
  class across domains — redundant with the built-in `fish` class.

This is deliberate: a fake "India model" would be worse than none.

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

- on the **Models** page (with its real installed / not-installed status),
- in the **Detect** model selector,
- in **`auto`** mode (runs every installed model and merges results — the
  class sets are disjoint, so there are no duplicates),
- in the **CLI**: `python -m app.main --image x.jpg --model <id>`.

Weights are loaded **once** and cached by `ModelManager`, so video frames
never trigger repeated disk loads. A missing weight file produces a clear
error and never disables the other working models.

---

## 5. Freshwater vs marine — intended habitat

BlueEye keeps the two environments explicitly separate so a model is never
applied outside the water body it was built for:

| Model (planned id) | Environment | Intended habitat |
|---|---|---|
| `fish_inv` | Marine | Indo-Pacific coral reefs |
| `megafauna` | Marine | Open water (shark / ray / turtle) |
| `freshwater_fish` | **Freshwater** | Karnataka & South Indian ponds, lakes, rivers, reservoirs |
| `gangetic_dolphin` | **Freshwater** | Ganga / Brahmaputra / Chambal river systems |
| `freshwater_turtle` | **Freshwater** | Indian rivers, lakes, ponds |
| `gharial` | **Freshwater** | Clear, fast-flowing Indian rivers |
| `indian_marine` | **Marine** | Arabian Sea / Bay of Bengal coastal waters |

Not every animal is present in every water body — a freshwater model should
not be expected to work on reef footage, and vice versa.

---

## 6. Training and evaluation

Dataset config templates live in [`training/`](../training/README.md), one per
target. The workflow is:

```bash
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
`models/custom/registry.json`; templates for the Indian models are in
`models/custom/registry.json.example`.

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
- Gangetic dolphin, freshwater turtle and gharial have **no verified,
  licensed, ready-to-train detection dataset** found here; they need field /
  UAV imagery to be collected and annotated.
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
