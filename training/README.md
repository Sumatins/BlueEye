# BlueEye training pipeline — Indian biodiversity models

BlueEye is a YOLOv8 detector. The two models it ships with (**Fish &
Invertebrates** and **MegaFauna**) come from the upstream
[marine-detect](https://github.com/Orange-OpenSource/marine-detect) project
and are calibrated for Indo-Pacific coral reefs. This folder contains the
dataset configuration and the reproducible workflow to train **additional**
models for Indian freshwater and coastal species.

> **Honesty first.** BlueEye never fabricates weights and never claims a
> species can be detected unless a model actually supports that class. No
> trustworthy *pretrained* YOLOv8 weights for the Indian species below were
> available when this was written, so these models must be **trained** on
> real, licensed, annotated data. The research (datasets, sources, licences
> and gaps) is documented in
> [`docs/indian_biodiversity.md`](../docs/indian_biodiversity.md).

---

## What is here

| Config | Habitat | Target classes | Status |
|---|---|---|---|
| [`datasets/freshwater_fish.yaml`](datasets/freshwater_fish.yaml) | Freshwater (ponds/rivers/reservoirs, South India) | `fish` | needs training (DePondFi / Orange Chromide dataset) |
| [`datasets/gangetic_dolphin.yaml`](datasets/gangetic_dolphin.yaml) | Freshwater (Ganga/Brahmaputra/Chambal) | `gangetic_river_dolphin` | needs data + training |
| [`datasets/freshwater_turtle.yaml`](datasets/freshwater_turtle.yaml) | Freshwater (rivers/lakes/ponds) | `freshwater_turtle` | needs data + training |
| [`datasets/gharial.yaml`](datasets/gharial.yaml) | Freshwater (clear, fast-flowing rivers) | `gharial` | needs data + training |
| [`datasets/indian_marine.yaml`](datasets/indian_marine.yaml) | Marine (Arabian Sea / Bay of Bengal) | `whale_shark`, `octopus`, `ray`, `olive_ridley_turtle` | needs data + training |

Each config follows the standard Ultralytics YOLO layout and uses the same
`scripts/train.py` and `scripts/evaluate.py` that already ship with BlueEye.
They are **templates**: the `path:` points at `data/raw/<name>` which you
populate yourself. Large datasets and trained weights are intentionally
**not** committed to Git.

```
data/raw/<name>/
├── images/
│   ├── train/   *.jpg|png
│   ├── val/     *.jpg|png
│   └── test/    *.jpg|png
├── labels/
│   ├── train/   *.txt     # class x_center y_center width height (normalised)
│   ├── val/     *.txt
│   └── test/    *.txt
```

---

## The five-step workflow

```bash
# 1. Fetch a licensed dataset and convert it to the YOLO layout above.
#    (Sources and licences: docs/indian_biodiversity.md)

# 2. Train (transfer learning from the existing BlueEye weights, or a small
#    Ultralytics backbone). Run from the project root:
python scripts/train.py \
    --data training/datasets/freshwater_fish.yaml \
    --model models/fish_inv/FishInv.pt \
    --epochs 100 --imgsz 640 --batch 16 --device ""

#    Weights land in runs/detect/blueeye/weights/best.pt

# 3. Evaluate on the held-out test split BEFORE claiming anything.
python scripts/evaluate.py \
    --weights runs/detect/blueeye/weights/best.pt \
    --data training/datasets/freshwater_fish.yaml \
    --split test --save-report

# 4. Place the weights under models/ and register them (no code change needed):
#    models/regional/freshwater_fish/BlueEyeFreshwaterFish.pt
#    then add an entry to models/custom/registry.json
#    (template: models/custom/registry.json.example)

# 5. Use it everywhere:
python -m app.main --list-models
python -m app.main --image river.jpg --model freshwater_fish
python -m streamlit run app/ui/streamlit_app.py
```

`--device ""` means auto (CUDA when available, otherwise CPU). CPU training of
a small model is possible but slow; a GPU is recommended.

> **Split by sequence, not by frame.** Frames from the same dive/video must
> not appear in both train and val, otherwise the metrics are inflated.
> Typical split: 70 / 20 / 10.

---

## Dataset sources (verified)

| Dataset | Species / scope | Licence | Link |
|---|---|---|---|
| DePondFi / Orange Chromide | Freshwater pond fish (*Etroplus maculatus*), 586 images, 1 class | CC BY 4.0 (Mendeley) | <https://data.mendeley.com/datasets/7w45jx35hd/1> |
| Underwater Species Dataset (NR) | 7 marine classes incl. **octopus**, sharks, turtles | CC BY 4.0 | <https://data.mendeley.com/datasets/4tp83br92z/1> |
| Community Fish Detection (CFD) | 1 class `fish`, ~2M images, multi-domain | per-dataset (CC/MIT) | <https://lila.science/datasets/community-fish-detection-dataset> |
| Upstream FishInv / MegaFauna sets | Reef fish families + shark/ray/turtle | see upstream repo | <https://github.com/Orange-OpenSource/marine-detect> |

Datasets that were researched but **not** integrated (no verifiable licence,
unpublished, or no public weights) are listed with reasons in
[`docs/indian_biodiversity.md`](../docs/indian_biodiversity.md).

---

## Registering a trained model

A model becomes visible everywhere (Models page, Detect selector, `auto` mode
and the CLI) as soon as its weights exist and it is described in
`models/custom/registry.json`. There is **no application code to change**.
See [`models/custom/README.md`](../models/custom/README.md) for the schema and
[`models/custom/registry.json.example`](../models/custom/registry.json.example)
for ready-to-edit Indian-biodiversity templates.

Once registered, the model can be selected directly:

```bash
python -m app.main --image river.jpg --model <your_model_id>
```
