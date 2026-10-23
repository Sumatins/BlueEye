# Deployment checklist (Streamlit Community Cloud)

Short, ordered checklist. Full rationale and troubleshooting:
[`docs/DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md).

**Status: deployed and verified — <https://blueeye-aquatic.streamlit.app>
(2026-10-23).** Every box below was checked against the live public URL.

## Before deploying

- [x] `app/ui/requirements.txt` present (CPU-torch deployment file, mirrors root;
      pins `streamlit==1.61.1` so the cloud matches the locally tested UI stack)
- [x] `packages.txt` present at repository root (`libgl1`, `libglib2.0-0`,
      **bare names only — no `#` comments**, the installer forwards every token
      to `apt-get`)
- [x] `.streamlit/secrets.toml.example` committed; **no** `.streamlit/secrets.toml` in git
- [x] Full test suite green locally: `pytest` (433 baseline + 10 deployment-config tests)
- [x] No secrets, credentials or real passwords in any committed file
- [x] Deployment commit pushed to `model-investigation`

## Deploy

- [x] <https://share.streamlit.io> → Create app
- [x] Repo `Sumatins/BlueEye`, branch `model-investigation`, file `app/ui/streamlit_app.py`
- [x] Advanced settings → Python **3.12**, Secrets left empty (or placeholders only)
- [x] App URL set to `blueeye-aquatic` (via app `⋮` → Settings after deploy)
- [x] Deploy → build log ends with "You can now view your Streamlit app"

## After deploying (verify — do not share the URL yet)

- [x] `GET https://<app-url>/~/+/_stcore/health` returns `ok`
      (Community Cloud serves the app under `/~/+/`; the domain root is
      Streamlit's wrapper page)
- [x] App loads; create an account; sidebar shows all 12 pages with the
      themed groups/icons (identical to local)
- [x] Models page → download `fish_inv` (in-app button) → card flips to
      "Ready and verified" (real load + inference test on the cloud CPU)
- [x] Detect page → real image → real boxes/classes/confidences, no mock data
      (verified: 3 detections, 2 classes, fish_inv attribution)
- [x] Observation saved → visible in My Observations (account library write works)
- [x] Species page lists 100 species with photographs
- [x] Only then share the URL with the team

## Operating notes

- [ ] Recommended on the 1 GB free tier: `BLUEEYE_DEFAULT_MODEL = "fish_inv"` in Secrets
- [x] Weights and history are wiped on every restart/redeploy — re-download after restarts
- [x] Updates: push to `model-investigation` (auto-redeploy); rollback: `git revert` + push
- [ ] Outgrown the free tier (persistence / all 9 models)? Move to Render per the guide

Test account used for verification: `deploy.verify@example.org`
(created through the public sign-up flow; it lives in the app's ephemeral
SQLite database and disappears on the next restart/redeploy).
