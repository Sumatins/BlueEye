# Deployment checklist (Streamlit Community Cloud)

Short, ordered checklist. Full rationale and troubleshooting:
[`docs/DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md).

## Before deploying

- [ ] `app/ui/requirements.txt` present (CPU-torch deployment file, mirrors root)
- [ ] `packages.txt` present at repository root (`libgl1`, `libglib2.0-0`)
- [ ] `.streamlit/secrets.toml.example` committed; **no** `.streamlit/secrets.toml` in git
- [ ] Full test suite green locally: `pytest` (433 baseline + 10 deployment-config tests)
- [ ] No secrets, credentials or real passwords in any committed file
- [ ] Deployment commit pushed to `model-investigation`

## Deploy

- [ ] <https://share.streamlit.io> → Create app
- [ ] Repo `Sumatins/BlueEye`, branch `model-investigation`, file `app/ui/streamlit_app.py`
- [ ] Advanced settings → Python **3.12**, Secrets left empty (or placeholders only)
- [ ] Deploy → build log ends with "You can now view your Streamlit app"

## After deploying (verify — do not share the URL yet)

- [ ] `GET https://<app-url>/_stcore/health` returns `ok`
- [ ] App loads; create an account; sidebar shows all 12 pages
- [ ] Models page → download `fish_inv` (in-app button)
- [ ] Detect page → real image → real boxes/classes/confidences, no mock data
- [ ] Species page lists 100 species with photographs
- [ ] Only then share the URL with the team

## Operating notes

- [ ] Recommended on the 1 GB free tier: `BLUEEYE_DEFAULT_MODEL = "fish_inv"` in Secrets
- [ ] Weights and history are wiped on every restart/redeploy — re-download after restarts
- [ ] Updates: push to `model-investigation` (auto-redeploy); rollback: `git revert` + push
- [ ] Outgrown the free tier (persistence / all 9 models)? Move to Render per the guide
