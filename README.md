# BlueEye

**AI-Powered Marine Life Detection** — an end-to-end application that detects
and classifies marine organisms in underwater images and videos using
YOLOv8 deep learning.

> *"Detect and identify marine organisms from underwater images and videos
> using deep learning."*

BlueEye is an academic final-year engineering project (Department of CSE,
KIT, Tiptur — 2025-26). It is built on the open-source
[marine-detect](https://github.com/Orange-OpenSource/marine-detect) project
by Orange OpenSource, which provides the two pretrained YOLOv8 model
weights used here.

---

## Overview

Marine ecosystems are difficult and expensive to monitor manually. Underwater
images suffer from low lighting, noise, colour distortion, blur and low
visibility, and many marine species look very similar. BlueEye automates
species identification so that researchers can analyse large volumes of
underwater imagery for **biodiversity monitoring, conservation, fisheries
management and marine research**.

BlueEye accepts an image or a video, runs YOLOv8 object detection, draws
labelled bounding boxes with confidence scores, and produces a results
dashboard with statistics and a machine-readable JSON report.

## Features

- **Professional multi-page web UI** — Dashboard, Detect, Models, Analytics,
  History and About, with a marine design system, loading states, friendly
  error cards and responsive layout
- **Image detection** — upload JPG/JPEG/PNG/WEBP, get an annotated image back
  in a before/after view
- **Video detection** — frame-by-frame processing of MP4/AVI/MOV/MKV into an
  annotated video, with a **real** frame counter (`frame 840 / 1200`) and an
  in-browser player
- **Multiple detections per frame** — every organism gets a bounding box,
  species label and confidence percentage, shown as cards and a table
- **Two pretrained models** — *Fish & Invertebrates* (15 classes) and
  *MegaFauna* (shark / ray / turtle), selectable or combined (`Auto`)
- **Extensible model registry** — friendly model cards with real status
  (🟢 ready / 🔴 not installed); new models (e.g. regional species) are added
  with a JSON entry + a `.pt` file, **no code changes** — see
  [`docs/regional_models.md`](docs/regional_models.md)
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

## Running the Application

### Web UI (recommended for demonstrations)

```bash
streamlit run app/ui/streamlit_app.py
```

Open http://localhost:8501 and use the sidebar navigation:

| Page | What it does |
| --- | --- |
| 🏠 **Dashboard** | Welcome, quick actions, device/model status, stored statistics |
| 🔍 **Detect** | The 5-step workflow: upload → choose model → set confidence → **Detect Marine Life** → results |
| 🧠 **Models** | Model cards with friendly names, class lists, provenance and 🟢/🔴 status; download weights; add custom/regional models |
| 📊 **Analytics** | Charts computed from your stored detections (top species, confidence distribution) |
| 🕘 **History** | Every finished run with inputs, models, thresholds and outputs; export/clear |
| ℹ️ **About** | What BlueEye does, how it works, limitations and attribution |

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
Detected marine life:
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
python -m app.main --list-models       # registry + availability + classes
python -m app.main --download-models   # fetch missing weights
python -m app.main --help
```

Exit codes: `0` success · `2` input/validation error · `3` model error ·
`4` processing error · `1` unexpected error.

## Training / Fine-Tuning

Training is optional — BlueEye works with the pretrained weights out of the
box.

```bash
python scripts/train.py \
    --data data/processed/my_dataset/data.yaml \
    --model yolov8n.pt \
    --epochs 50 --imgsz 640 --batch 16 --device cpu
```

To fine-tune from an existing BlueEye model:

```bash
python scripts/train.py --data data/processed/my_dataset/data.yaml \
    --model models/fish_inv/FishInv.pt --epochs 30
```

### Evaluation (Precision / Recall / mAP)

```bash
python scripts/evaluate.py \
    --weights runs/detect/blueeye/best.pt \
    --data data/processed/my_dataset/data.yaml \
    --split val --save-report
```

> **Metrics policy:** BlueEye only reports precision, recall, mAP@50 and
> mAP@50:95 when a labelled test/validation set is actually evaluated in
> your environment. The numbers published in the upstream marine-detect
> README were measured by *its* authors on *their* test sets and are
> reference values — they are not re-measured by BlueEye. No metric shown by
> BlueEye is fabricated.

## Dataset Format

See [`data/README.md`](data/README.md) for the expected YOLO layout,
`data.yaml` example, directory structure and links to the marine datasets
referenced in the BlueEye report (Shark, Shark Species, Zebra Shark, Fish,
Count-a-Manta, OzFish, …). Datasets are never downloaded automatically.

## Configuration

Settings come from environment variables or an optional `.env` file
(copy `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `BLUEEYE_MODELS_DIR` | `models` | Where weights live |
| `BLUEEYE_OUTPUTS_DIR` | `outputs` | Annotated files + reports |
| `BLUEEYE_DEFAULT_MODEL` | `auto` | `auto` / `fish_inv` / `megafauna` |
| `BLUEEYE_DEFAULT_CONFIDENCE` | `0.5` | UI slider default |
| `BLUEEYE_DEVICE` | `auto` | `auto` / `cpu` / `cuda:0` |
| `BLUEEYE_ENHANCEMENT` | `off` | Enhancement default |
| `BLUEEYE_MAX_IMAGE_MB` | `25` | Upload limit |
| `BLUEEYE_MAX_VIDEO_MB` | `400` | Upload limit |
| `BLUEEYE_LOG_LEVEL` | `INFO` | Logging verbosity |

### Model selection

- **Auto** — runs *every* model whose weights are available and merges the
  detections. The two built-in models have **disjoint class sets**
  (fish/invertebrates vs. shark/ray/turtle), so merged results contain no
  duplicate classes and no cross-model NMS is needed. Running both models
  roughly doubles inference time — choose a single model for speed-critical
  use.
- **Fish & Invertebrates** / **MegaFauna** — run only that model.

### Adding a model (registry)

BlueEye's models are described by a registry, not hard-coded:

```
built-in MODEL_REGISTRY  ──┐
                           ├─→ ModelManager.registry() ─→ UI cards, CLI, auto mode
models/custom/registry.json┘
```

To add your own model (for example a regional species model), drop a YOLOv8
`.pt` under `models/` and describe it in `models/custom/registry.json` — see
[`models/custom/README.md`](models/custom/README.md) for the schema and
[`docs/regional_models.md`](docs/regional_models.md) for the full dataset →
training → evaluation workflow. Invalid entries are logged and skipped, so a
broken file never breaks the built-in models. **BlueEye does not ship or
claim a regional model today** — the support is ready, the weights are not.

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
  the detection history store, and the custom-model registry (discovery,
  validation, path containment, fail-safe behaviour).
- **Integration tests** — real weights on a real image and a real generated
  video; automatically **skipped** when weights are not downloaded and never
  require a GPU.

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
│       ├── streamlit_app.py         # shell: navigation, sidebar, routing
│       ├── theme.py                 # design system (ocean CSS + page ids)
│       ├── components.py            # reusable cards, badges, tables, notices
│       ├── state.py                 # session state, upload intake, history
│       └── pages/                   # Dashboard, Detect, Models, Analytics,
│                                    # History, About
├── src/marine_detect/              # upstream reference code (AGPL-3.0, unmodified)
├── scripts/
│   ├── download_models.py          # fetch official weights
│   ├── train.py                    # training / fine-tuning
│   └── evaluate.py                 # precision / recall / mAP evaluation
├── docs/
│   └── regional_models.md          # regional model research + training workflow
├── models/                         # weights (downloaded, git-ignored)
│   ├── fish_inv/FishInv.pt
│   ├── megafauna/MegaFauna.pt
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

- The two pretrained models cover the **upstream species scope** (reef fish
  families, five invertebrates, shark/ray/turtle) — other organisms will not
  be detected until new data is trained in.
- CPU inference is comparatively slow for long videos (roughly 0.5–2 fps
  depending on hardware); GPU is recommended for longer footage.
- Underwater enhancement is a heuristic preprocessing step and is **not**
  evaluated here — it may help or hurt depending on the scene.
- mAP/precision/recall metrics require a labelled dataset; none are bundled.
- The web UI stores uploads on the local disk (`data/input/`) and is designed
  for **local/trusted use**, not for public internet exposure.

## Future Scope

Per the BlueEye report: training on larger and more diverse marine datasets
to recognise rare/endangered species; real-time deployment on NVIDIA Jetson
Nano / Raspberry Pi with underwater cameras; behaviour analysis, population
tracking and migration patterns; cloud integration for large-scale storage
and sharing; improved handling of low-visibility conditions; mobile/web
applications for wider access.

**Regional models:** the architecture, registry, training scripts and
documentation are ready (`docs/regional_models.md`,
`models/custom/README.md`). The missing piece is real regional data — collect
and label it, train with `scripts/train.py`, evaluate with
`scripts/evaluate.py`, then register the weights. No regional model is
shipped or claimed before that happens.

## Credits & License

- **BlueEye** — AI-powered marine life detection application, developed as
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
