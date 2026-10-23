# BlueEye deployment guide

> **Live instance (verified 2026-10-23):** <https://blueeye-aquatic.streamlit.app>
> Branch `model-investigation`, entrypoint `app/ui/streamlit_app.py`, Python 3.12,
> Secrets empty. Verified end-to-end on the public URL: health `ok`, account
> creation, `fish_inv` weights download (Azure SAS → "Ready and verified"),
> real inference (3 detections / 2 classes with boxes and confidences),
> observation saved to the account library, Species Explorer rendering.

This guide deploys the complete BlueEye Streamlit app to a public URL on
**Streamlit Community Cloud** (free). It was produced by a 10-phase audit of
the repository; every claim below is either measured from this codebase or
taken from the platform's official documentation. Nothing here disables or
mocks a feature — the same code, models and class mappings run locally and in
the cloud.

## Why Streamlit Community Cloud

| Platform | Verdict | Reason |
| --- | --- | --- |
| **Streamlit Community Cloud** | **chosen (free)** | First-party hosting for Streamlit apps; deploys straight from GitHub; CPU torch wheels install within the free tier; zero container maintenance. |
| Render (paid) | documented upgrade path | Container host with persistent disk. The free instance (512 MB RAM) is too small; the 1 GB/0.5 CPU instance costs ~$7/mo, 2 GB ~$25/mo plus ~$0.25/GB/mo disk. Only needed if you outgrow the Community Cloud limits below. |
| Vercel | rejected | Serverless functions cannot host a long-running Streamlit websocket server, and the 500 MB bundle limit excludes the model stack. |
| Hugging Face Spaces | rejected | Docker-based Spaces require a PRO plan; RAM limits are comparable to Community Cloud. |
| Fly.io | rejected | No free tier; a paid VM is needed for the same resources Render offers. |
| Neon / Supabase | not applicable | BlueEye uses only local SQLite (three databases under `outputs/`) plus JSON history; there is no external database to host. |

## What the platform gives you (verified limits)

- **1 GB RAM** per app. Measured on this codebase (CPU, one request): baseline
  process ~35 MB, one core model loaded ~399 MB, both core models ~580 MB, a
  prediction peak ~777 MB. All nine models loaded at once is ~1.5 GB —
  **the full model registry will exceed the memory limit**; see
  "Memory budget" below.
- **Non-persistent filesystem.** Everything under `outputs/` (SQLite history,
  JSON history, uploaded media) and everything under `models/` (downloaded
  weights) is **lost on every restart, redeploy or hibernation**.
- **App hibernates** after inactivity; the first visit after idle can take
  ~30–60 s while the container cold-starts.
- Region: US only (data residency requirement means you must move hosts).
- GitHub's 100 MB per-file limit is why model weights are **not** committed;
  they are downloaded at runtime (see below).

## Files this deployment adds

| File | Purpose |
| --- | --- |
| `app/ui/requirements.txt` | Deployment-only dependency file. Community Cloud installs the requirements file closest to the entrypoint (`app/ui/streamlit_app.py`), so this file (not the root one) is used in the cloud. It `-r`-includes the root `requirements.txt`, pins `torch==2.13.0+cpu` / `torchvision==0.28.0+cpu` from the official PyTorch CPU index (so the Linux runners never pull multi-GB CUDA wheels; wheels verified for Linux cp311/cp312), and pins `streamlit==1.61.1` — the exact locally tested release. Newer Streamlit releases (e.g. 1.66.x, which `>=1.49` resolves to unpinned) change the widget DOM that `app/ui/theme.py` styles and render the themed sidebar unstyled. |
| `packages.txt` | Apt packages installed on the Debian runners: `libgl1`, `libglib2.0-0` (needed by OpenCV at import). PyTorch bundles its own `libgomp`; `imageio-ffmpeg` ships a self-contained ffmpeg binary. **No `#` comments allowed:** the platform passes every whitespace-separated token straight to `apt-get install`, so a comment line fails the whole build with `E: Unable to locate package #`. Bare package names only. |
| `.streamlit/secrets.toml.example` | Placeholder-only template documenting the optional `BLUEEYE_*` deployment values. The real `.streamlit/secrets.toml` stays git-ignored. |
| `app/config.py::load_streamlit_secrets` | Merges `BLUEEYE_*` keys from `st.secrets` into the environment before settings are built (Community Cloud delivers config via secrets, not env vars). Never overwrites real env vars; no-op when no secrets file exists. Covered by `tests/test_deployment_config.py`. |

No application feature, model, class mapping or weight was changed for the
deployment. The Dockerfile and `docker-compose.yml` are untouched and remain
the local-container path.

## Deploying (one-time, ~15 min)

You need push access to the GitHub repository.

1. **Push the deployment commit.** All four files above (plus
   `tests/test_deployment_config.py` and this guide) must be on the branch you
   deploy from (`model-investigation`).
2. **Open <https://share.streamlit.io>** and sign in with the GitHub account
   that owns the repository.
3. **Create app** →
   - Repository: `Sumatins/BlueEye`
   - Branch: `model-investigation`
   - Main file path: `app/ui/streamlit_app.py`
4. **Advanced settings** →
   - Python version: **3.12** (the version the whole stack is tested on;
     Streamlit's own default may be newer — do not accept it)
   - Leave the Secrets field **empty** for now (see the next section)
5. *(Optional)* set an **App URL** subdomain (e.g. `blueeye-aquatic`) so the
   share link is memorable. If the value from the deploy form does not stick,
   the app is still created (with a generated subdomain) — rename it afterwards
   via the app's `⋮` menu → **Settings** → **App URL** → *Save changes*.
   The rename takes effect immediately and does not redeploy.
6. **Deploy!** The first build installs dependencies (several minutes). Watch
   the build log for `You can now view your Streamlit app in your browser`.

### Optional deployment configuration

BlueEye runs with **no configuration at all** — every `BLUEEYE_*` variable has
a safe default. To override values, paste entries into
**Advanced settings → Secrets** using `.streamlit/secrets.toml.example` as the
template (keys must start with `BLUEEYE_`). Recommended for the free tier:

```toml
BLUEEYE_DEFAULT_MODEL = "fish_inv"
```

`auto`/`all` load every installed model and can exceed the 1 GB limit; a single
model id keeps the app comfortably inside it.

## Model weights (auto-installed on a fresh deployment)

Weights are **not** in the repository (652 MB across nine models; GitHub caps
files at 100 MB and the code loads them by filename), and there is no mock
fallback — a page that needs inference says so honestly. To remove the manual
download step, the app **bootstraps its own core weights**
(`app/detection/bootstrap.py`):

* at process start, a background daemon thread downloads `fish_inv`
  (≈87 MB, Azure SAS link from the project's public asset store) when its
  weights are missing — silent, non-blocking, and it can never crash the app;
* if a user reaches **AI Detection** before the transfer finishes, the page
  shows it with a real progress bar and continues into the normal workflow;
* transfers reuse the Models-page code path (same size / SHA-256 checks), a
  process-wide lock prevents concurrent writers, and any failure degrades to
  the honest manual-install card.

Control it via Secrets: `BLUEEYE_AUTO_DOWNLOAD_WEIGHTS = "off"` disables the
bootstrap; `BLUEEYE_DEFAULT_MODEL = "megafauna"` makes that model the one to
prepare instead. The remaining models (six additional ones from
HF/Zenodo/GitHub/GCS with SHA-256 pinning via
`scripts/fetch_additional_models.py`) install from the **Models** page on
demand. Weights persist **only until the next restart/redeploy** — every
restart simply re-runs the bootstrap. For permanent weights, move to Render
with a persistent disk (below).

## Memory budget (read before a live demo)

Measured CPU peaks for this codebase:

| Scenario | Approx. peak RSS |
| --- | --- |
| App idle, no model loaded | ~35 MB |
| `fish_inv` loaded | ~399 MB |
| `fish_inv` + `megafauna` loaded | ~580 MB |
| One prediction (image) | ~777 MB |
| All nine models loaded | ~1.5 GB → **exceeds 1 GB limit** |

Safe recipe for the free tier: deploy a single core model
(`BLUEEYE_DEFAULT_MODEL="fish_inv"`), avoid the "Every installed model"
option on the Detect page, and keep video clips short (the 400 MB upload cap
in `.streamlit/config.toml` is a *local* convenience; on the free tier a large
video will OOM before it finishes). If you need every model concurrently,
that is the point to move to Render (see below).

## Health checks, logs, verification

- **Health endpoint:** `GET /_stcore/health` returns `ok` when the server is
  up (the Dockerfile already uses this for its container health check). On
  Community Cloud the app runs behind an iframe at `/~/+/`, so the public URL
  is `https://<app>.streamlit.app/~/+/_stcore/health` (the domain root serves
  Streamlit's wrapper page, not the app).
- **Logs:** the deployment dashboard shows build and runtime logs; the app
  logs through Python `logging` at `BLUEEYE_LOG_LEVEL` (default `INFO`).
- **Verify a deployment end-to-end:** open the URL, create an account, open
  Models → download `fish_inv`, then Detect → upload a sample reef image →
  you must see real boxes, class names and confidences (not "no model
  loaded"). The Species page must list 100 species with photographs.

## Updating the deployed app

Push to the deployed branch (`model-investigation`); the platform detects the
commit and rebuilds (dependency changes take a few minutes; rate limit is
~5 pushes/minute). Every redeploy wipes `models/` and `outputs/`.

## Rollback

There is no one-click rollback. To return to a previous state:

```bash
git log --oneline            # find the last good commit
git revert <bad-commit>      # or: git reset --hard <good-commit>
git push origin model-investigation
```

The platform redeploys automatically; then re-download model weights (they
were wiped by the redeploy).

## Troubleshooting

| Symptom | Cause & fix |
| --- | --- |
| Build fails resolving `torch` | The CPU index pin in `app/ui/requirements.txt` is missing or the platform picked up the root `requirements.txt` instead. Ensure the entrypoint is `app/ui/streamlit_app.py` (the file next to it wins). |
| `Error installing requirements` with `E: Unable to locate package #` / `Apt` / `packages` in the log | `packages.txt` contains `#` comment lines — the platform forwards every token to `apt-get`. Keep bare package names only (see the Files table above). |
| App boots but the sidebar renders unstyled (no group headers, no icons, every nav item identical) | Streamlit version drift: the cloud installed a newer Streamlit than the one `app/ui/theme.py` was styled against. Keep the `streamlit==1.61.1` pin in `app/ui/requirements.txt`; the deploy terminal's `Found Streamlit version …` line shows what actually ran. |
| `libGL.so.1: cannot open shared object file` | `packages.txt` missing from the repository root; restore it and redeploy. |
| App starts but AI Detection reports missing weights | The automatic bootstrap is disabled (`BLUEEYE_AUTO_DOWNLOAD_WEIGHTS=off`) or its transfer failed (see the app log for `Weight bootstrap:` lines). Download from the Models page, or fix connectivity and reload. |
| After a push the app crashes with `AttributeError: module 'app.ui.state' has no attribute …` (a name that *does* exist in the pushed commit) | The container absorbed the push as an **in-place "Pulling code changes" update** instead of a fresh build: the entrypoint file is re-read fresh on every run, but modules imported by the still-running process stay cached, so old and new code mix. Fix: manage app ⋮ → **Reboot app** (fresh process re-imports the updated tree), or push another commit to force a full rebuild. Observed 2026-10-24 after the weight-bootstrap push; a fresh container always resolves it. |
| App killed with no Python traceback during inference | Out-of-memory: the 1 GB limit. Use a single model (see Memory budget). |
| Long cold start after idle | Hibernation; the first visit rebuilds the container. |
| Need persistent history or all models | Move to Render with a persistent disk (see below). |

## Upgrade path: Render (paid, persistent)

When the free tier no longer fits (persistent history, all nine models, large
videos):

1. Keep the existing `Dockerfile` — it already installs the apt libs, the
   CPU torch stack, and declares a `/_stcore/health` health check.
2. Create a Render **Web Service** from the repository, instance type
   **2 GB RAM / 0.5–1 CPU** (~$25/mo), attach a **persistent disk** and mount
   it at `/app/outputs` (history) and `/app/models` (weights), and set any
   `BLUEEYE_*` values in the service's Environment panel (Render sets real
   env vars, so the secrets bridge is unnecessary there).
3. Redeploy is still git-driven; weights and history survive because they
   live on the disk, not the container.

## Cost summary

- **Streamlit Community Cloud:** free. Limitations: 1 GB RAM, ephemeral disk,
  hibernation, US-only.
- **Render:** ~$7/mo (1 GB) or ~$25/mo (2 GB) plus ~$0.25/GB/mo disk — only
  if you need persistence or the full model set.

## Local development is unchanged

```bash
pip install -r requirements.txt        # root file, unchanged, for local dev
streamlit run app/ui/streamlit_app.py  # or: python -m app.main
pytest                                 # 433 tests + deployment-config tests
```

The root `requirements.txt` keeps the broad pins (`torch>=2.0`) so a GPU
workstation can install CUDA wheels; the deployment-only file in `app/ui/`
exists precisely so the cloud cannot accidentally pick those up.
