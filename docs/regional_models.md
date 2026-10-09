# Regional and custom marine-species models

BlueEye is built so that a **region-specific model** (for example one trained
for Indian coastal waters) can be added later **without rewriting any
application code**. This document records:

1. What "regional support" means in BlueEye today.
2. What was researched about existing public datasets/models — and why
   **no regional model ships with BlueEye**.
3. The full workflow to collect, train, evaluate and register your own.

---

## 1. Status: honest summary

| Item | Status |
|---|---|
| Registry / architecture able to load a regional model | ✅ Ready (`models/custom/registry.json`) |
| Training + evaluation scripts | ✅ Ready (`scripts/train.py`, `scripts/evaluate.py`) |
| Documentation for the full workflow | ✅ This file |
| Indian-biodiversity dataset configs | ✅ Ready (`training/datasets/`) — see [`docs/indian_biodiversity.md`](indian_biodiversity.md) |
| A trained India / Karnataka / Arabian Sea model **inside BlueEye** | ❌ **Does not exist** |

BlueEye does **not** claim any model is trained for India or any specific
coast. Until real weights are trained or sourced, verified and registered, the
*Models* page shows:

> **No regional model installed** — support is ready for a custom model
> integration.

The two models that ship with BlueEye are the pretrained **Fish &
Invertebrates** (15 classes) and **MegaFauna** (3 classes) weights from the
upstream *marine-detect* project. Their class lists already overlap with
species found across the Indo-Pacific (reef fish families such as
Chaetodontidae, Scaridae, Lutjanidae, Serranidae, Muraenidae, Haemulidae,
plus giant clam, urchin, lobster, crown-of-thorns, shark, ray and turtle) —
but they were **not trained on regional data**, so accuracy on local species
is unverified.

---

## 2. Research findings (candidates — none integrated)

A literature/dataset search was performed while building BlueEye. The results
below are **starting points only**. None of them were downloaded, trained,
benchmark-tested or integrated by BlueEye, and their class lists and licences
were **not** fully verified here. Treat every entry as "to be evaluated".

### Public datasets worth evaluating

For the Indian-freshwater / coastal focus, the datasets with a **verified
licence** are catalogued in [`docs/indian_biodiversity.md`](indian_biodiversity.md)
(DePondFi / Orange Chromide and the Underwater Species Dataset, both CC BY 4.0).
The broader candidate list below is marine/general.

| Candidate | Where | Notes |
|---|---|---|
| Marine Fish (object detection) | <https://universe.roboflow.com/marine-fish-detection/marine-fish> | Small Roboflow Universe project (~240 images) with a pre-trained model. Licence, classes and geography **not verified**. |
| Shark IBM | <https://universe.roboflow.com/ticon-dataset/shark-ibmby> | Shark instances; already referenced by the BlueEye report. |
| Shark Species | <https://universe.roboflow.com/rizal-fadia-al-fikri/shark_species> | Shark species detection; referenced by the report. |
| Zebra Shark | <https://universe.roboflow.com/minhajul-arefin/zebra_shark> | Single-species. |
| Fish Dataset | <https://universe.roboflow.com/roboflow-gw7yv/fish-yzfml> | General fish detection. |
| Count-a-Manta | <https://universe.roboflow.com/le-wagon-w02yl/count-a-manta> | Manta rays. |
| OzFish | <https://doi.org/10.25845/5e28f062c5097> | Australian reef fish counts (used upstream by marine-detect). |
| Fish Detection YOLOv8 (GitHub) | <https://github.com/Vinay0905/Fish-Detection-YOLOv8> | Personal project; **licence not verified**. |
| Fish species detector (Kaggle) | <https://www.kaggle.com/code/killa92/map-0-9-fish-species-detector-yolov11/input> | Notebook + weights; licence of the underlying images **not verified**. |
| Indian seafood market species (research paper) | PMC article *"Species identification for Indian seafood markets"* | Peer-reviewed dataset for Indian seafood; check the paper's data-availability statement for access and licence. |

### The integration bar (and what now passes it)

A model may only be registered when **all** of the following are true:

1. **Licence** permits use (and, if weights are redistributed, redistribution).
2. **Class list** is known and documented — no invented species names.
3. **Format** is a YOLO / RT-DETR-compatible checkpoint.
4. **Provenance** can be stated on the Models page.
5. It was **tested** through BlueEye's own pipeline.

None of the Roboflow Universe candidates above states a verifiable licence
*and* class list *and* public weights, so none is registered. Three **general
aquatic** models from Hugging Face *do* pass all five and are now integrated
as optional additions (`aquatic_brackish`, `aquarium_marine`,
`underwater_fish`) — see
[`indian_biodiversity.md`](indian_biodiversity.md). They are **not
India-specific**. For Indian species the architecture stays ready rather than
shipping a fake "India model".

> **Tip:** Roboflow Universe datasets usually state a licence on the project
> page (often CC BY 4.0). If you find one for Indian coastal species, verify
> it, convert it to YOLO format and follow §4 below.

---

## 3. Where a regional model plugs in

```
User selects model in the UI / CLI
        ↓
Model registry  (built-in MODEL_REGISTRY  +  models/custom/registry.json)
        ↓
ModelSpec  (id, name, weights path, classes, threshold, source, licence)
        ↓
ModelManager.load(key)   →  Ultralytics YOLO("models/.../YourModel.pt")
        ↓
Common detector interface (MarineDetector.predict_image / predict_frame)
        ↓
Existing inference pipeline  →  boxes, labels, statistics, outputs
```

The detection engine is unchanged; only a JSON entry and a `.pt` file are
added. See [`models/custom/README.md`](../models/custom/README.md) for the
schema.

---

## 4. Training your own regional model

### 4.1 Collect images

Target environments: Arabian Sea, Bay of Bengal, Karnataka / Kerala / Goa /
Tamil Nadu coasts, Lakshadweep, Andaman & Nicobar.

- ROV / dive / fishing-vessel footage frames (watch duplicates).
- Aim for **diverse conditions**: turbidity, depth, lighting, seasons, angles.
- As a rough minimum for fine-tuning, **1–2k annotated images per class**
  gives usable results; fewer works only with transfer learning from the
  existing BlueEye weights.
- Record provenance and, where people appear, consent/privacy notes.

### 4.2 Label

- Tools: **Label Studio**, **CVAT**, **Roboflow**, or any YOLO-capable editor.
- Draw tight boxes; use a consistent class vocabulary (decide snake_case ids
  once: `tuna`, `mackerel`, `shark`, …).
- Add difficult negatives (empty reefs, sand, divers) labelled with no boxes —
  they reduce false positives.

### 4.3 Convert to YOLO format

Each label file is `images/<name>.txt` with one row per object:

```
<class_id> <x_center> <y_center> <width> <height>     # all normalised 0..1
```

Directory layout (recommended, matching `data/README.md`):

```
data/raw/regional_karnataka/     # or datasets/regional/... — your choice
├── images/
│   ├── train/  *.jpg|png
│   ├── val/    *.jpg|png
│   └── test/   *.jpg|png
├── labels/
│   ├── train/  *.txt
│   ├── val/    *.txt
│   └── test/   *.txt
└── data.yaml
```

`data.yaml`:

```yaml
path: data/raw/regional_karnataka     # absolute or project-relative root
train: images/train
val: images/val
test: images/test

names:
  0: tuna
  1: mackerel
  2: shark
```

Split **by sequence, not by frame** — frames from the same dive video in both
train and val inflate metrics. Typical split: 70 / 20 / 10.

### 4.4 Train

```bash
# Start from a small backbone, or fine-tune from the existing BlueEye weights:
python scripts/train.py \
    --data data/raw/regional_karnataka/data.yaml \
    --model models/fish_inv/FishInv.pt \
    --epochs 100 --batch 16 --device "" \
    --name blueeye_regional

# Or from scratch with an Ultralytics checkpoint:
python scripts/train.py --data ... --model yolov8n.pt --epochs 100
```

Outputs land in `runs/detect/blueeye_regional/weights/best.pt`.

Practical notes:

- `--device ""` = auto (CUDA if available, otherwise CPU; CPU training of a
  small model is feasible but slow).
- Monitor `runs/.../results.csv`; stop when val loss plateaus
  (`--patience` handles early stopping).
- Class imbalance is normal — consider `--model yolov8s.pt` for more capacity.

### 4.5 Evaluate before claiming anything

```bash
python scripts/evaluate.py \
    --weights runs/detect/blueeye_regional/weights/best.pt \
    --data data/raw/regional_karnataka/data.yaml \
    --split test --save-report
```

Record precision, recall, mAP@0.5 and mAP@0.5:0.95 **on the held-out test
split** and keep the JSON report next to the model. Do not quote numbers from
the training split.

Sanity-check in BlueEye:

```bash
python -m app.main --image path/to/local_reef.jpg --model regional_karnataka
```

### 4.6 Register the model

```bash
# 1. place weights
mkdir -p models/regional/karnataka
copy runs\detect\blueeye_regional\weights\best.pt models\regional\karnataka\BlueEyeRegional.pt

# 2. describe them
copy models\custom\registry.json.example models\custom\registry.json
#    → edit id / name / weights / classes / source / license / threshold

# 3. verify + run
python scripts/download_models.py --list
python -m app.main --list-models
python -m streamlit run app/ui/streamlit_app.py     # Models page shows 🟢 Ready
```

The model then appears automatically on the *Models* page, in the *Detect*
selector and in the CLI — with no UI changes. It joins `auto` mode unless its
entry sets `"auto": false`.

---

## 5. Checklist before you call it a "regional model"

- [ ] Weights trained on data from (or representative of) the target region.
- [ ] Licence of the dataset **and** of the resulting weights recorded in `registry.json`.
- [ ] Class names documented and stable.
- [ ] Test-split metrics produced by `scripts/evaluate.py --save-report`.
- [ ] Provenance (`source`, `homepage`) filled in — no vague claims.
- [ ] Works through `python -m app.main --image ... --model <id>`.
- [ ] Models page shows the correct status.

---

## 6. References

- Upstream project: <https://github.com/Orange-OpenSource/marine-detect> (AGPL-3.0-only)
- Ultralytics YOLOv8 docs: <https://docs.ultralytics.com/models/yolov8/>
- Roboflow Universe (dataset search): <https://universe.roboflow.com/search>
- BlueEye model registry: [`models/custom/README.md`](../models/custom/README.md)
- Dataset layout & splitting: [`data/README.md`](../data/README.md)
