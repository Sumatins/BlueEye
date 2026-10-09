# BlueEye Datasets

This directory holds data used by BlueEye. Nothing large is downloaded
automatically.

## Layout

```
data/
├── input/        # files uploaded through the web UI (auto-created)
├── raw/          # original, untouched datasets
├── processed/    # cleaned / converted datasets
├── train/        # YOLO training split   (or produced by the dataset yaml)
├── val/          # YOLO validation split
└── test/         # YOLO test split
```

## Adding a marine dataset (YOLO format)

BlueEye trains with the standard Ultralytics YOLO layout:

```
my_dataset/
├── data.yaml
├── images/
│   ├── train/   xxx001.jpg ...
│   ├── val/     yyy001.jpg ...
│   └── test/    zzz001.jpg ...
└── labels/
    ├── train/   xxx001.txt ...   # class x_center y_center width height (normalized)
    ├── val/     yyy001.txt ...
    └── test/    zzz001.txt ...
```

`data.yaml` example:

```yaml
path: data/processed/my_dataset   # dataset root (relative to project root)
train: images/train
val: images/val
test: images/test

names:
  0: shark
  1: fish
  2: turtle
```

### Datasets referenced by the BlueEye report

| Dataset | Source |
| --- | --- |
| Shark Dataset | https://universe.roboflow.com/ticon-dataset/shark-ibmby |
| Shark Species | https://universe.roboflow.com/rizal-fadia-al-fikri/shark_species |
| Zebra Shark | https://universe.roboflow.com/minhajul-arefin/zebra_shark |
| Fish Dataset | https://universe.roboflow.com/roboflow-gw7yv/fish-yzfml |
| Count-a-Manta | https://universe.roboflow.com/le-wagon-w02yl/count-a-manta |
| OzFish | https://doi.org/10.25845/5e28f062c5097 |

The upstream *marine-detect* project also publishes two annotated YOLO
datasets (FishInv, MegaFauna) - see the links in the upstream README:
https://github.com/Orange-OpenSource/marine-detect

**None of these datasets are downloaded automatically.** Fetch them
yourself, convert them to the layout above, then train with:

```bash
python scripts/train.py --data data/processed/my_dataset/data.yaml --epochs 50
python scripts/evaluate.py --weights runs/detect/blueeye/best.pt --data data/processed/my_dataset/data.yaml
```
