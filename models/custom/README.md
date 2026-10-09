# BlueEye custom model registry

This folder is BlueEye's **model extension point**. It lets you add new
YOLOv8 models — for example one trained on *regional* species — **without
touching any application code**.

For the Indian-biodiversity roadmap (which models are integrated, which need
training, dataset sources and licences) see
[`docs/indian_biodiversity.md`](../../docs/indian_biodiversity.md); ready-to-edit
dataset configs and the training workflow are in
[`training/`](../../training/README.md).

Nothing in this folder is downloaded automatically. BlueEye never fabricates
weights: you place the `.pt` file yourself and describe it here.

---

## Add a model in 4 steps

1. **Obtain or train** a YOLOv8 `.pt` checkpoint.
   Ready-to-edit dataset configs for Indian species are in
   [`training/datasets/`](../../training/datasets); the full dataset →
   training → evaluation workflow is documented in
   [`training/README.md`](../../training/README.md) and
   [`docs/regional_models.md`](../../docs/regional_models.md).
2. **Copy the weights** anywhere under `models/`, e.g.

   ```
   models/
   └── regional/
       └── karnataka/
           └── BlueEyeRegional.pt
   ```

3. **Describe the model** in `models/custom/registry.json`.
   Start from the template:

   ```
   copy models\custom\registry.json.example models\custom\registry.json    # Windows
   cp models/custom/registry.json.example models/custom/registry.json      # macOS/Linux
   ```

4. **Restart BlueEye.** The model now appears on the *Models* page, in the
   *Detect* model selector and in the CLI (`--model <id>`). It joins `auto`
   mode unless its entry sets `"auto": false`.

## Schema

`registry.json` is a JSON object with a `models` array:

| Field | Required | Type | Meaning |
|---|---|---|---|
| `id` | ✅ | string | Unique key, `1–64` chars of `[A-Za-z0-9_-]`. Must not collide with `fish_inv` / `megafauna`. Used by the CLI and the API. |
| `name` | ✅ | string | Friendly display name (shown on model cards). |
| `weights` | ✅ | string | Path of the `.pt` file **relative to `models/`**. Absolute paths and `..` are rejected. |
| `classes` | | array of string | Class names, in training order. |
| `type` | | string | `yolov8` (default) or `rtdetr` (opened with Ultralytics `RTDETR`). Any other type is rejected. |
| `recommended_confidence` | | number | Threshold in `[0, 1]` used by `auto` mode (default `0.5`). |
| `description` / `summary` | | string | Short text shown on cards and under the selector. |
| `version` | | string | Your model's version, e.g. `"0.1"`. |
| `source` | | string | Provenance, e.g. `"Trained in-house on a local dataset"`. |
| `license` | | string | Licence of *your* weights. |
| `category` | | string | Free-form grouping (`regional`, `custom`, …). |
| `icon` | | string | Icon name for the card (emoji or Material Symbols name). |
| `homepage` | | string | Link to a dataset/paper/repository. |
| `architecture` | | string | Informational label shown on the card (defaults from `type`). |
| `auto` | | boolean | Include in the `auto` selection (default `true`). Set `false` for optional / overlapping models. |
| `habitat` | | string | e.g. `"Freshwater"`, `"Brackish / estuarine"`, `"Marine"`. |
| `readiness` | | string | Force a status when no usable weights exist: `needs_training` or `incompatible`. |
| `status_note` | | string | One-line honest note about provenance / limitations, shown on the card. |

Example:

```json
{
  "models": [
    {
      "id": "regional_karnataka",
      "name": "Karnataka Coastal Species",
      "weights": "regional/karnataka/BlueEyeRegional.pt",
      "classes": ["tuna", "mackerel", "shark"],
      "type": "yolov8",
      "version": "0.1",
      "recommended_confidence": 0.5,
      "source": "Trained in-house on a local dataset",
      "license": "CC-BY-4.0 (dataset), model by us",
      "category": "regional",
      "icon": "🌊",
      "summary": "Species recorded along the Karnataka coast."
    }
  ]
}
```

## Behaviour and safety rules

- **Fail-safe**: an invalid entry is *logged and skipped* — a broken file
  never prevents the built-in models from working.
- **Path containment**: `weights` must resolve inside `models/`.
- **No auto-download**: custom models have no official URL, so
  `scripts/download_models.py` never fetches them.
- **Live reload**: the file is re-read whenever it changes (restart not
  strictly required for the CLI; the web UI re-reads it on the next run).
- **`auto` mode** includes every installed custom model whose `auto` is not
  `false` (the default). The curated additions (`models/custom/registry.json`)
  set `"auto": false`, so `auto` still runs only the two core models; use the
  **`all`** selection — or pick the model explicitly — to run them.

## Testing your model

```bash
# CLI
python -m app.main --image path/to/image.jpg --model regional_karnataka

# Web UI
python -m streamlit run app/ui/streamlit_app.py
```

> **Indian-biodiversity templates.** `registry.json.example` also contains
> ready-to-edit entries for the Indian models (`freshwater_fish`,
> `gangetic_dolphin`, `freshwater_turtle`, `gharial`, `indian_marine`). See
> [`docs/indian_biodiversity.md`](../../docs/indian_biodiversity.md) for what
> each one needs before it can run.

## Honesty policy

BlueEye will **not** advertise a model that does not exist. Until real weights
are registered here, the *Models* page shows:

> No regional model installed — support is ready for a custom model
> integration.

See [`docs/regional_models.md`](../../docs/regional_models.md) for dataset
research, licensing notes, the training workflow and how to evaluate a new
model before registering it.
