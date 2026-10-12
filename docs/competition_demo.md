# Blue Eye — competition demonstration guide

A reliable, repeatable demonstration of the **actual** Blue Eye application,
using real models and real detections (no mocked results). Follow the checklist
before you present; the workflow below is exactly what the app does today.

---

## 1. Pre-flight checklist (do this before the event / off the network)

```bash
# 1. Dependencies
pip install -r requirements.txt

# 2. Core model weights (fetched once, then cached under models/)
python scripts/download_models.py          # fish_inv + megafauna

# 3. Optional additional models (verified by SHA-256)
python scripts/fetch_additional_models.py

# 4. Confirm what is actually runnable on this machine
python -m streamlit run app/ui/streamlit_app.py   # → http://localhost:8501
#    Account → Models page shows READY only for weights that load
```

- [ ] The **Models** page lists `fish_inv` and `megafauna` as **READY**.
- [ ] A short **image** and a short **video** are on the device (MP4/H.264
      plays best in the browser).
- [ ] You have logged in at least once so the demo account exists.
- [ ] `pytest -q` passes (see `README.md → Testing`).

> If any weight is missing the app still runs; the Models page tells the truth
> (NOT INSTALLED / NEEDS TRAINING) and detection uses whatever is available.

## 2. The demonstration workflow (≈3 minutes)

### Step 1 — Register / log in
Open <http://localhost:8501>. The account page uses real local authentication
(bcrypt hashing, persistent SQLite at `outputs/auth/blueeye_auth.db`).
Register a demo user or log in with the existing one.

**Say:** *"Authentication is real — passwords are bcrypt-hashed in a local
database, and sessions persist. Each account sees only its own data."*

### Step 2 — Dashboard
The dashboard shows **real** numbers: total detections, detected-species
summary, recent detections and recent activity — all read from this user's
history. There are no hard-coded statistics.

### Step 3 — One Detection section
There is a single **Detection** entry in the sidebar that handles **both**
image and video. Select the input type.

### Step 4 — Image detection
Upload a supported image. Choose the model (e.g. `auto`, `fish_inv`,
`megafauna`, or an additional model if installed).

**Say:** *"We keep the original pretrained weights unchanged; the model
selector runs whichever verified checkpoint is installed."*

### Step 5 — Results
The app displays the detections with **class names and confidence scores**,
and shows the **original image next to the detected image** (side by side).

### Step 6 — Video detection
Switch to video, upload a short clip. The app shows the **processed video**
plus **detection summaries** (totals, per-class counts, confidence).

### Step 7 — History
Open **History**. The stored run appears with its **user id, timestamp, input
type, classes, confidence scores and result location**. Re-open it to view the
original and annotated/processed result again — even after a restart.

**Say:** *"History is per-user and persisted; a stored entry can be re-opened,
and BlueEye refuses to render another user's file or an arbitrary local path."*

### Step 8 — Model information
On the **Models** page, show the selected model's **supported classes** and
metadata (architecture, source, licence, limitations). Class names are the
model's real class list — BlueEye never claims a class a model cannot detect.

### Step 9 — Evaluation (if an experiment has been run)
If a labelled dataset has been evaluated, show
`outputs/evaluation/<timestamp>/summary.md` and `metric_comparison.png`.

**Say honestly:** *"No Indian-region test set has been evaluated yet, so this
comparison is marked PENDING. The pipeline is ready; it never invents a
score."* (See `docs/evaluation.md`.)

### Step 10 — Robustness (deliberately show it)
- Upload an **unsupported file** (e.g. a PDF) → a clear validation error, no
  crash.
- Show that a detection with no objects reports "no detections" gracefully.

## 3. One-line narrative

> *"Blue Eye is an open marine/aquatic detection application: real models,
> real image and video detection, per-user private history, and a reproducible
> evaluation pipeline that reports evidence honestly — including what is still
> pending for Indian waters."*

## 4. Fallbacks

| Risk | Fallback |
|---|---|
| No GPU | Everything runs on CPU: `DEVICE=cpu` (or `set BLUEEYE_DEVICE=cpu` on Windows). |
| Long video | Use a short clip; video H.264 output is browser-playable. |
| Network unavailable | Weights are cached locally after the first fetch; nothing downloads during the demo. |
| A model fails to load | Select another installed model; the Models page shows the truthful status. |
| No labelled dataset | Do not show a metrics table; say the experiment is **PENDING**. |

## 5. Do-not-claim list

- Do **not** claim any model is trained for India; none is.
- Do not quote accuracy numbers unless they come from
  `outputs/evaluation/...` from a real run on a named dataset.
- Do not present published metrics as Blue Eye measurements.
