# BlueEye

**See beneath the surface. Understand aquatic life. Support biodiversity.**
BlueEye is an end-to-end, AI-assisted application that detects and classifies
aquatic organisms in underwater images and supported videos using YOLOv8 /
RT-DETR deep learning.

> *"Detect and identify aquatic organisms from underwater images and videos
> using deep learning."*

BlueEye is an academic final-year engineering project (Department of CSE,
KIT, Tiptur — 2025-26). It is built on the open-source
[marine-detect](https://github.com/Orange-OpenSource/marine-detect) project
by Orange OpenSource, which provides the two pretrained YOLOv8 model
weights used here.

---

## Overview

Aquatic ecosystems are difficult and expensive to monitor manually. Underwater
images suffer from low lighting, noise, colour distortion, blur and low
visibility, and many species look very similar. BlueEye automates
species identification so that researchers can analyse large volumes of
underwater imagery for **biodiversity monitoring, conservation, fisheries
management and aquatic research**.

BlueEye accepts an image or a video, runs YOLOv8 object detection, draws
labelled bounding boxes with confidence scores, and produces a results
dashboard with statistics and a machine-readable JSON report.

## Features

- **Aquatic-biodiversity dashboard** — a branded hero with real *Detect an
  Image* / *Analyse a Video* actions, an "Explore BlueEye" launcher, a
  "How BlueEye works" explainer, a capabilities section and a responsible-use
  note, all driven by real navigation (no invented metrics, no model loading
  on load)
- **Professional multi-page web UI** — Dashboard, Detect Image, Detect Video,
  Species Explorer, Models, Analytics, History and About, with a compact
  grouped sidebar, one consistent icon set (Material Symbols Rounded), a
  deep-ocean design system, real metadata cards, status badges, honest empty
  states, loading states, friendly error cards and a responsive layout
- **Species Explorer** — an educational reference library of 38 aquatic
  species (Indian freshwater and coastal species first), with search by
  common/scientific name, habitat and group filters, a detail view, and an
  honest per-species "can BlueEye detect this?" note (direct class support /
  broad-category only / not currently supported) — see
  [`app/content/species.py`](app/content/species.py)
- **Image detection** — upload JPG/JPEG/PNG/WEBP, get an annotated image back
  in a before/after view
- **Video detection** — frame-by-frame processing of MP4/AVI/MOV/MKV into an
  annotated video, with a **real** frame counter (`frame 840 / 1200`) and an
  in-browser player
- **Multiple detections per frame** — every organism gets a bounding box,
  species label and confidence percentage, shown as cards and a table
- **Two original pretrained models** — *Fish & Invertebrates* (15 classes) and
  *MegaFauna* (shark / ray / turtle), selectable or combined (`Auto`)
- **Additional pretrained models** — six verified, opt-in aquatic models
  (brackish/estuarine, aquarium & reef, grayscale underwater fish, 21-species
  Mediterranean, Community Fish, Fishial); `Auto` keeps running only the two
  core models, `Every installed model` runs them all
- **Extensible model registry** — friendly model cards with real status
  labels (installed / not installed); new models (e.g. regional species) are added
  with a JSON entry + a `.pt` file, **no code changes** — see
  [`docs/regional_models.md`](docs/regional_models.md)
- **Indian biodiversity roadmap** — dataset configs and a training pipeline for
  freshwater fish, Gangetic river dolphin, freshwater turtle, gharial and
  coastal species; these are **inactive research targets** (kept in
  `models/custom/inactive_models.json`, never offered in the model selector
  until real weights exist) — the built-in MegaFauna already covers
  shark / ray / turtle at group level — see
  [`docs/indian_biodiversity.md`](docs/indian_biodiversity.md)
- **Configurable confidence threshold** (slider, or one click for each model's
  recommended value)
- **Optional underwater enhancement** — colour correction → CLAHE contrast →
  denoising, switchable on/off (disabled by default; *not* claimed to improve
  accuracy — evaluate it on your own data)
- **Detection statistics** — total objects, unique species, per-class counts,
  average / maximum confidence, frames processed, processing speed
- **JSON reports** for every image/video processed
- **Detection history** — every finished run persisted locally with inputs,
  models, thresholds and outputs (exportable, clearable)
- **Analytics** — real charts computed from that history (top species,
  confidence distribution, models used) — never fabricated numbers
- **Downloadable results** — annotated image/video + JSON from the UI
- **CPU by default, GPU automatically** when CUDA is available
- **Modular architecture** — UI is a thin layer over the detection backend
- **CLI + Web UI + Docker**

## Architecture

```
User
  │
  ▼
BlueEye Web UI (Streamlit)  ──┐
  │                           │
  ├── Image Upload            │   CLI (python -m app.main)
  └── Video Upload            │
          │                   │
          ▼                   ▼
   Input Validation (extension, size, decodability, safe filenames)
          │
   Optional Underwater Enhancement (on/off)
          │
   YOLOv8 Detection  ← MarineDetector ← ModelManager (weights cache,
          │                                device: CUDA/CPU, registry)
          ▼
   Bounding boxes + classes + confidences (confidence filtering)
          │
   Result Generation (statistics, JSON report)
          │
   ┌──────┴──────┐
   ▼             ▼
Annotated     Annotated
Image         Video
   └──────┬──────┘
          ▼
   Results Dashboard / CLI output
```

Code layers (dependency direction: UI → processing → detection → utils):

| Package | Responsibility |
| --- | --- |
| `app/ui/` | Streamlit shell (navigation, theme) + page modules — no model/processing logic |
| `app/main.py` | Command line interface |
| `app/processing/` | Image pipeline, video pipeline, enhancement, drawing |
| `app/detection/` | `MarineDetector` inference wrapper, model registry/download, structured results |
| `app/utils/` | Validation, sanitisation, file I/O helpers, detection history |
| `app/config.py` | Settings (`.env`), paths, logging |

The UI only ever calls the backend's public interface
(`ImageProcessor.process`, `VideoProcessor.process`, `MarineDetector`,
`ModelManager`) — detection behaviour is identical between the CLI and the
web UI.

## Technologies

- **Python** — main language
- **Ultralytics YOLOv8 / PyTorch** — object detection and AI backend
- **OpenCV** — image and video processing
- **imageio-ffmpeg** — bundled ffmpeg used to encode annotated videos as
  **H.264** so they play in the browser preview and in common players
- **NumPy** — array/data handling
- **Pillow** — robust image I/O (EXIF, unicode paths)
- **Streamlit** — web UI
- **pytest** — automated tests
- **Docker / Conda / venv** — environment management

## Installation

### Option A — Python virtual environment (venv)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Option B — Conda

```bash
conda create --name blueeye python=3.10
conda activate blueeye
pip install -r requirements.txt
```

### Option C — Docker

```bash
docker compose up --build          # UI on http://localhost:8501
```

Development/test dependencies:

```bash
pip install -r requirements-dev.txt
```

## Model Setup

BlueEye **never ships or fabricates model weights.** The two pretrained
YOLOv8 weights come from the upstream *marine-detect* release and are
downloaded on first use:

```bash
python scripts/download_models.py
```

or click **"Download model weights"** in the web UI sidebar.

| Model key | Name | Classes | Weights | Recommended conf. |
| --- | --- | --- | --- | --- |
| `fish_inv` | Fish & Invertebrates | 15 (fish families + invertebrates) | `models/fish_inv/FishInv.pt` | 0.523 |
| `megafauna` | MegaFauna | 3 (shark, ray, turtle) | `models/megafauna/MegaFauna.pt` | 0.546 |

> Weights are downloaded from the official links published in the upstream
> marine-detect README and are excluded from version control (`models/*` is
> git-ignored).

### Additional (verified) aquatic models

BlueEye can also run six **genuinely pretrained, publicly downloadable**
aquatic detectors. They are optional and **opt-in** — `auto` still runs only
the two core models. Fetch them (SHA-256-pinned) with:

```bash
python scripts/fetch_additional_models.py
```

| Model key | Classes | Architecture | Licence | Source |
| --- | --- | --- | --- | --- |
| `aquatic_brackish` | crab, fish, jellyfish, shrimp, small_fish, starfish | YOLOv8s | AGPL-3.0 | [dronefreak/brackish-yolov8s](https://huggingface.co/dronefreak/brackish-yolov8s) |
| `aquarium_marine` | fish, jellyfish, penguin, puffin, shark, starfish, stingray | RT-DETR | AGPL-3.0 | [Kanagavel/aquarium-rtdetr](https://huggingface.co/Kanagavel/aquarium-rtdetr) |
| `underwater_fish` | fish | YOLOv8n | US Gov. work (royalty-free) | [akridge/yolo8-fish-detector-grayscale](https://huggingface.co/akridge/yolo8-fish-detector-grayscale) |
| `obsea_mediterranean` | 21 Mediterranean species (groupers, seabreams, wrasse, moray, Myliobatidae) | YOLOv8x | CC-BY-4.0 | [OBSEA / Zenodo 14910365](https://zenodo.org/records/14910365) |
| `community_fish` | fish | YOLOv12x | AGPL-3.0 | [filippovarini/community-fish-detector](https://github.com/filippovarini/community-fish-detector) |
| `fishial_detector` | Fish | YOLO26n | MIT | [fishial/fish-identification](https://github.com/fishial/fish-identification) |

They are **not India-specific**; provenance limits are documented in
[`docs/indian_biodiversity.md`](docs/indian_biodiversity.md). They appear in
the **Models** page under *"Ready and verified"* once installed, and
you can run them with `--model <key>` or the combined `--model all`. A
candidate published only as Axera NPU files (`aquarium_axera`) is preserved in
the inactive registry as **Incompatible**. Verify every model with a real
inference test:

```bash
python -m app.main --verify-models
```


## Indian Biodiversity Models

BlueEye is being extended toward **Indian freshwater and coastal species**.
The architecture already supports any registered model; what is (and is not)
available is documented honestly:

| Target | Habitat | Built-in coverage | Status |
| --- | --- | --- | --- |
| Freshwater fish | Freshwater ponds / rivers / reservoirs | — | **inactive — needs training** (DePondFi / Orange Chromide dataset, CC BY 4.0 — files not publicly downloadable as of Oct 2026) |
| Gangetic river dolphin | Ganga / Brahmaputra / Chambal | — | **inactive — needs data + training** |
| Freshwater turtle | Indian rivers / lakes / ponds | generic `turtle` (MegaFauna) | **inactive — needs data + training** |
| Gharial | clear, fast-flowing rivers | — | **inactive — needs data + training** |
| Whale shark / rays / Olive Ridley | Arabian Sea / Bay of Bengal | generic `shark` / `ray` / `turtle` (MegaFauna) | group-level now; species-level inactive |
| Octopus | Indian coastal waters | — | **inactive — needs training** (NR archive ships images only, no labels) |

> These targets are **not in the active model registry**. Because none has
> verified weights, all of them (plus the incompatible Axera export) live in
> [`models/custom/inactive_models.json`](models/custom/inactive_models.json),
> which is documentation only — they never appear in the model selector or in
> model counts. An entry becomes selectable only after a real, evaluated `.pt`
> is registered in `models/custom/registry.json`.

- **No Indian-species pretrained weights were found.** The public models that
  were found (AquaYOLO, YOLO-Fish, Roboflow Universe projects) did not meet the
  bar of a verifiable licence + known classes + YOLOv8 `.pt` + a trustworthy
  download, so no Indian-species model is claimed. Three **general aquatic**
  pretrained models were integrated instead (above) and are labelled as
  non-Indian — BlueEye does not fabricate a species model.
- **Honest per-model status** on the Models page and in `--list-models`:
  *ready* (weights present **and** an inference test succeeds), *not installed*,
  *needs training* (dataset/config only) or *incompatible* (e.g. NPU-only).
- **Verified, licensed datasets** (DePondFi / Orange Chromide and the
  Underwater Species Dataset NR — both CC BY 4.0) are catalogued with their
  licences **and a downloadability audit** in
  [`docs/indian_biodiversity.md`](docs/indian_biodiversity.md) §3.1. As of
  October 2026 neither can be trained on as-is (DePondFi files are not served
  to anonymous users; the NR archive contains images without label files), so
  the five Indian targets stay **inactive** (needs training) rather than being
  faked.
- **Ready-to-train configs** for each target are in
  [`training/datasets/`](training/README.md). Validate and lay out a downloaded
  dataset with `scripts/prepare_dataset.py` (it refuses unlabelled data), train
  with `scripts/train.py`, evaluate with `scripts/evaluate.py`, then register
  the `.pt`.
- **Freshwater and marine models are kept separate.** A model is only claimed
  to work for the environment it was trained on.

## Running the Application

### Web UI (recommended for demonstrations)

```bash
streamlit run app/ui/streamlit_app.py
```

Open http://localhost:8501 and use the sidebar navigation:

| Page | What it does |
| --- | --- |
| **Dashboard** | Aquatic-biodiversity landing page: branding hero, real *Detect an Image* / *Analyse a Video* actions, an "Explore BlueEye" launcher, "How BlueEye works", capabilities and a responsible-use note |
| **Detect Image** | The detection workflow with the media type set to **Image**: upload → choose model → set confidence → **Detect** → results |
| **Detect Video** | The same workflow with the media type set to **Video**, with a real frame-by-frame progress bar |
| **Species Explorer** | An educational reference library of aquatic species with search and habitat/group filters, a detail view, and an honest per-species note on whether any active model can detect it |
| **Models** | The single source of truth: grouped into *Ready and verified* / *Downloadable but not yet verified* / *Unavailable or incompatible*, with architecture, runtime, classes, licence, image/video verification, limitations and an inference check; download weights |
| **Analytics** | Charts computed from your stored detections (top species, confidence distribution) |
| **History** | Every finished run with inputs, models, thresholds and outputs; export/clear |
| **About** | What BlueEye does, how it works, limitations and attribution |

The sidebar groups the pages (*Overview* / *Detection* / *Explore* / *Activity*
/ *Information*) and shows only a compact readiness indicator — **no individual
model names or per-model badges**; the Models page is the single source of
truth for model status. Hardware/device details live on the Models and About
pages. A single icon set — **Material Symbols Rounded**, bundled locally with
Streamlit — is used across the sidebar, buttons, cards, model tiles and empty
states. The whole interface is one consistent design system
(see [`app/ui/theme.py`](app/ui/theme.py) and [`app/ui/icons.py`](app/ui/icons.py)).

The Detect page shows an original/result side-by-side view, a summary
(total objects, species, average/max confidence), a detections table and
download buttons for the annotated file and the JSON report. Video runs show
a real frame-by-frame progress bar and an in-browser player when finished.
Missing weights produce a friendly **Model unavailable** card with install
instructions instead of an error.

### Image detection (CLI)

```bash
python -m app.main --image assets/images/input_folder/regq.jpg
python -m app.main --image photo.jpg --confidence 0.6 --model fish_inv
python -m app.main --image photo.jpg --enhance      # optional enhancement
```

Example output:

```
Detected aquatic life:
  1. Parrotfish (Scaridae)                87.9%   bbox=[294, 355, 486, 466]   model=fish_inv
  2. Parrotfish (Scaridae)                83.8%   bbox=[184, 282, 301, 332]   model=fish_inv
  3. Sea turtle                           69.5%   bbox=[28, 423, 542, 669]   model=megafauna
  4. Butterfly fish (Chaetodontidae)      54.4%   bbox=[160, 870, 275, 937]   model=fish_inv

Detection summary
  Total objects      : 4
  Unique species     : 3
  Average confidence : 73.9%
  Max confidence     : 87.9%
```

*(Real output of BlueEye on the marine-detect sample image — nothing is
hard-coded.)*

### Video detection (CLI)

```bash
python -m app.main --video dive.mp4
python -m app.main --video dive.mp4 --confidence 0.5 --model megafauna
```

Videos are streamed frame-by-frame (never fully loaded into RAM). Frame
dimensions and FPS are preserved; annotated MP4 + JSON report are written to
`outputs/`.

### Other CLI commands

```bash
python -m app.main --list-models       # registry + status + architecture + classes
python -m app.main --verify-models     # load every model + run a real inference test
python -m app.main --download-models   # fetch missing core weights
python -m app.main --help
```

Exit codes: `0` success · `2` input/validation error · `3` model error ·
`4` processing error · `1` unexpected error.

## Training / Fine-Tuning

Training is optional — BlueEye works with the pretrained weights out of the
box.

First validate and lay out a downloaded, licensed dataset (the helper refuses
a dataset that has images but no YOLO label files, and never invents class
names):

```bash
python scripts/prepare_dataset.py \
    --source "D:/data/my_dataset" \
    --name my_dataset \
    --classes fish crab        # omit if the source ships a data.yaml
```

Then train:

```bash
python scripts/train.py \
    --data data/raw/my_dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 50 --imgsz 640 --batch 16 --device cpu
```

To fine-tune from an existing BlueEye model:

```bash
python scripts/train.py --data data/raw/my_dataset/data.yaml \
    --model models/fish_inv/FishInv.pt --epochs 30
```

For the **Indian-species** models, ready-made dataset configs ship in
[`training/datasets/`](training/README.md):

```bash
python scripts/prepare_dataset.py --source "D:/data/pond_fish" \
    --name freshwater_fish --classes fish
python scripts/train.py \
    --data training/datasets/freshwater_fish.yaml \
    --model models/fish_inv/FishInv.pt --epochs 100 --imgsz 640 --batch 16
```

See [`training/README.md`](training/README.md) for the full per-target workflow
and [`docs/indian_biodiversity.md`](docs/indian_biodiversity.md) for datasets,
licences, the downloadability audit and limitations.

### Evaluation (Precision / Recall / mAP)

```bash
python scripts/evaluate.py \
    --weights runs/detect/blueeye/best.pt \
    --data data/raw/my_dataset/data.yaml \
    --split val --save-report
```

> **Metrics policy:** BlueEye only reports precision, recall, mAP@50 and
> mAP@50:95 when a labelled test/validation set is actually evaluated in
> your environment. The numbers published in the upstream marine-detect
> README were measured by *its* authors on *their* test sets and are
> reference values — they are not re-measured by BlueEye. No metric shown by
> BlueEye is fabricated. The Models page shows an *Evaluation* row **only**
> when a registry entry records real metrics.

## Dataset Format

See [`data/README.md`](data/README.md) for the expected YOLO layout,
`data.yaml` example, directory structure and links to the marine datasets
referenced in the BlueEye report (Shark, Shark Species, Zebra Shark, Fish,
Count-a-Manta, OzFish, …). Ready-to-edit configs for the Indian-species
models are in [`training/datasets/`](training/README.md). Datasets are never
downloaded automatically; use `scripts/prepare_dataset.py` to validate and
lay one out.

## Configuration

Settings come from environment variables or an optional `.env` file
(copy `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `BLUEEYE_MODELS_DIR` | `models` | Where weights live |
| `BLUEEYE_OUTPUTS_DIR` | `outputs` | Annotated files + reports |
| `BLUEEYE_DEFAULT_MODEL` | `auto` | `auto` (core) / `all` / any registered model id |
| `BLUEEYE_DEFAULT_CONFIDENCE` | `0.5` | UI slider default |
| `BLUEEYE_DEVICE` | `auto` | `auto` / `cpu` / `cuda:0` |
| `BLUEEYE_ENHANCEMENT` | `off` | Enhancement default |
| `BLUEEYE_MAX_IMAGE_MB` | `25` | Upload limit |
| `BLUEEYE_MAX_VIDEO_MB` | `400` | Upload limit |
| `BLUEEYE_LOG_LEVEL` | `INFO` | Logging verbosity |

### Model selection

- **Auto** — runs the **two core models** *Fish & Invertebrates* and
  *MegaFauna* whose weights are available and merges the detections. Their
  class sets are **disjoint** (fish/invertebrates vs. shark/ray/turtle), so
  merged results contain no duplicate classes and no cross-model NMS is
  needed. Additional models are **opt-in** and never run under `Auto`.
- **Every installed model** (`all`) — runs every model whose weights are
  available (core + additional + custom), then removes cross-model duplicates
  of the same class at IoU ≥ 0.7, keeping the highest-confidence box. This is
  the slowest option.
- **Fish & Invertebrates** / **MegaFauna** — run only that model.
- **Any registered model** (built-in, additional or a custom / regional one)
  can be selected explicitly: pick it in Detect or pass its id to `--model`.
  If its weights are missing, BlueEye shows a clear error and never disables
  the other working models. Aliases (`fish`, `mega`) keep working.

### Adding a model (registry)

BlueEye's models are described by a registry, not hard-coded:

```
built-in MODEL_REGISTRY  ──┐
                           ├─→ ModelManager.registry() ─→ UI cards, CLI, auto mode
models/custom/registry.json┘
```

To add your own model (for example a regional species model), drop a YOLOv8
`.pt` under `models/` and describe it in `models/custom/registry.json` — see
[`models/custom/README.md`](models/custom/README.md) for the schema,
[`training/README.md`](training/README.md) for the dataset → training →
evaluation workflow, and [`docs/indian_biodiversity.md`](docs/indian_biodiversity.md)
for the Indian-species dataset research and licences. Invalid entries are
logged and skipped, so a broken file never breaks the built-in models.
**BlueEye does not ship or claim a regional / Indian model today** — the
support is ready, the weights are not.

### Confidence threshold

Each detection is kept only if `confidence >= threshold`. With no explicit
value, each model uses the threshold recommended upstream (0.523 / 0.546).

### Underwater enhancement

`Input → colour correction (grey-world white balance) → contrast (CLAHE) →
denoising (bilateral) → detection`. It is **optional and off by default**:
the models were trained on raw imagery, so enhancement is not guaranteed to
help — compare both modes on your data before claiming otherwise.

## Docker

```bash
docker compose up --build                      # start the UI
docker compose run --rm blueeye python scripts/download_models.py   # fetch weights
```

- `./models`, `./data` and `./outputs` are mounted into the container, so
  weights and results persist on the host.
- The image is **CPU-only** by default.

### GPU Support

CUDA is never mandatory — BlueEye uses CUDA automatically when available
(`torch.cuda.is_available()`), otherwise CPU.

For a GPU container:

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
2. Change the Dockerfile base image to a CUDA runtime image, e.g.
   `FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` (then install
   Python), and install a CUDA-enabled PyTorch build from
   https://pytorch.org/get-started/locally/.
3. Run with:
   ```bash
   docker compose run --gpus all blueeye
   ```

On bare metal, a CUDA build of PyTorch is enough:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"   # True = GPU ready
```

Edge devices (NVIDIA Jetson Nano with its JetPack PyTorch build, Raspberry
Pi for lightweight models) can run the same code with `BLUEEYE_DEVICE=cpu`
or the platform's GPU.

## Testing

```bash
pytest                 # unit + integration-less tests (default)
pytest -m integration  # tests using the real downloaded weights
pytest --cov=app       # with coverage
```

- **Unit tests** — validation/sanitisation, statistics/report formatting,
  enhancement, model manager (downloads mocked, no network), detector
  confidence filtering & model selection, image/video pipelines (model mocked),
  the detection history store, the custom-model registry (discovery,
  validation, path containment, fail-safe behaviour), the dataset-preparation
  helper (`scripts/prepare_dataset.py`, including its refusal of unlabelled
  data) and the redesigned navigation / model-card metadata.
- **Integration tests** — real weights on a real image and a real generated
  video; automatically **skipped** when weights are not downloaded and never
  require a GPU.

Run the model self-check (loads each model and runs a real inference test):

```bash
python -m app.main --verify-models
```

## Troubleshooting

| Problem | Solution |
| --- | --- |
| `No model weights found` | `python scripts/download_models.py` (or the UI button) |
| `Could not download … weights` | Check internet/proxy; retry — a partial `.part` file is auto-discarded |
| Upload rejected: *too large* | Raise `BLUEEYE_MAX_IMAGE_MB` / `BLUEEYE_MAX_VIDEO_MB` |
| *Unsupported format* | Images: jpg/jpeg/png/webp · videos: mp4/avi/mov/mkv |
| Slow video processing | CPU inference is slow (~0.5–2 fps); select a single model, raise confidence, or use a GPU |
| `cv2` / `libGL` errors in Docker | The Dockerfile installs `libgl1` + `libglib2.0-0`; rebuild the image |
| CUDA requested but not available | Warning is logged and BlueEye falls back to CPU |
| Corrupt video error | Re-encode the file (`ffmpeg -i in.mov -c:v libx264 out.mp4`) |
| Annotated video won't play in the preview | Install `imageio-ffmpeg` (`pip install imageio-ffmpeg`) — without it BlueEye falls back to the `mp4v` codec, which browsers cannot decode |
| Port already in use | `streamlit run app/ui/streamlit_app.py --server.port 8502` |

Logging is structured (`timestamp | level | module | message`). Run the CLI
with `--verbose` (or `BLUEEYE_LOG_LEVEL=DEBUG`) for debugging detail.

## Project Structure

```
blueeye/
├── app/
│   ├── main.py                     # CLI entry point
│   ├── config.py                   # settings, .env, logging
│   ├── detection/
│   │   ├── detector.py             # MarineDetector (inference API)
│   │   ├── model_manager.py        # registry, download, cached loading, device
│   │   └── results.py              # Detection, stats, JSON reports
│   ├── processing/
│   │   ├── image_processor.py      # image pipeline
│   │   ├── video_processor.py      # frame-by-frame video pipeline
│   │   ├── enhancement.py          # optional underwater enhancement
│   │   └── drawing.py              # bounding-box rendering
│   ├── utils/
│   │   ├── validation.py           # upload validation + sanitisation
│   │   ├── file_utils.py           # safe paths, unique names, image/JSON I/O
│   │   └── history.py              # persistent detection history + aggregates
│   └── ui/
│       ├── streamlit_app.py         # shell: navigation, grouped sidebar, routing
│       ├── theme.py                 # design system (ocean CSS, page ids, nav)
│       ├── icons.py                 # single icon set (Material Symbols Rounded)
│       ├── components.py            # reusable cards, badges, tables, notices
│       ├── state.py                 # session state, upload intake, history
│       └── pages/                   # Dashboard, Detect (Image/Video), Models,
│                                    # Analytics, History, About
├── src/marine_detect/              # upstream reference code (AGPL-3.0, unmodified)
├── scripts/
│   ├── download_models.py          # fetch official weights
│   ├── fetch_additional_models.py  # fetch the additional verified models (SHA-256 pinned)
│   ├── prepare_dataset.py          # validate + lay out a downloaded dataset
│   ├── train.py                    # training / fine-tuning
│   └── evaluate.py                 # precision / recall / mAP evaluation
├── training/
│   ├── README.md                   # per-target training workflow
│   └── datasets/                   # YOLO data.yaml configs (Indian species)
├── docs/
│   ├── regional_models.md          # regional model research + training workflow
│   └── indian_biodiversity.md      # Indian-species datasets, licences, plan
├── models/                         # weights (downloaded, git-ignored)
│   ├── fish_inv/FishInv.pt
│   ├── megafauna/MegaFauna.pt
│   ├── additional/                 # verified opt-in aquatic models
│   └── custom/                     # model extension point (registry.json)
├── data/                           # uploads + datasets (see data/README.md)
├── outputs/
│   ├── images/                     # annotated images
│   ├── videos/                     # annotated videos
│   ├── reports/                    # JSON reports
│   └── history/detections.json     # persisted detection history
├── assets/                         # sample media from marine-detect
├── tests/                          # pytest suite (unit + integration)
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Limitations

- The two **original** pretrained models cover the **upstream species scope**
  (reef fish families, five invertebrates, shark/ray/turtle) — other organisms
  will not be detected until new data is trained in. In particular, no Indian
  freshwater or species-level coastal model is shipped; MegaFauna only gives
  the **group** `shark` / `ray` / `turtle`, and octopus is not covered at all
  (see [`docs/indian_biodiversity.md`](docs/indian_biodiversity.md)).
- The five Indian-species targets remain **NEEDS TRAINING**: as of October 2026
  no licensed, obtainable, labelled dataset was found (DePondFi's files are not
  served to anonymous users; the NR archive contains images without label
  files), and the verification environment has **no CUDA GPU**. BlueEye states
  this rather than fabricating weights.
- CPU inference is comparatively slow for long videos (roughly 0.5–2 fps
  depending on hardware); GPU is recommended for longer footage.
- Underwater enhancement is a heuristic preprocessing step and is **not**
  evaluated here — it may help or hurt depending on the scene.
- mAP/precision/recall metrics require a labelled dataset; none are bundled.
- The web UI stores uploads on the local disk (`data/input/`) and is designed
  for **local/trusted use**, not for public internet exposure.

## Future Scope

Per the BlueEye report: training on larger and more diverse aquatic datasets
to recognise rare/endangered species; real-time deployment on NVIDIA Jetson
Nano / Raspberry Pi with underwater cameras; behaviour analysis, population
tracking and migration patterns; cloud integration for large-scale storage
and sharing; improved handling of low-visibility conditions; mobile/web
applications for wider access.

**Regional models:** the architecture, registry, training scripts and
documentation are ready (`docs/regional_models.md`,
[`docs/indian_biodiversity.md`](docs/indian_biodiversity.md),
`models/custom/README.md`, [`training/`](training/README.md)). The missing
piece is real regional data — collect and label it, train with
`scripts/train.py`, evaluate with `scripts/evaluate.py`, then register the
weights. No regional / Indian model is shipped or claimed before that happens.

## Credits & License

- **BlueEye** — aquatic biodiversity detection application, developed as
  an academic project (Department of CSE, KIT, Tiptur, 2025-26).
- **Pretrained models and reference implementation:** *marine-detect* by
  **Orange Business Services SA / Orange OpenSource**
  <https://github.com/Orange-OpenSource/marine-detect>, licensed under
  **AGPL-3.0-only**. BlueEye reuses and adapts ideas from its inference
  pipeline; the original `src/marine_detect` package is retained
  unmodified with all copyright/SPDX headers intact. Model weights are
  downloaded from the official marine-detect release links and are **not**
  created by BlueEye.
- **YOLOv8 / Ultralytics** — <https://github.com/ultralytics/ultralytics>.
- **Datasets** — the BlueEye report and upstream README reference public
  datasets (Roboflow Universe, OzFish/AIMS, GBIF, …); see
  [`data/README.md`](data/README.md) and the upstream README for citations.

BlueEye is distributed under the **GNU Affero General Public License v3.0
(AGPL-3.0-only)** — see [`LICENSE`](LICENSE). If you reuse code or assets,
preserve the SPDX headers and attribution notices.
