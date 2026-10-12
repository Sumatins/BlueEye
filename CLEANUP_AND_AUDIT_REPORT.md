# BlueEye — Cleanup & Auth Audit Report

Scope: remove the Map section, fix the dashboard visual artefacts, redesign the
Account page, audit authentication / per-user isolation, confirm detection is
unaffected, and prove it all with tests.

SPDX-License-Identifier: AGPL-3.0-only

---

## 1. Map section removed

The Map was a self-contained discovery surface. It has been removed completely
and no dead navigation or unreachable map-only code remains.

| Change | File |
| --- | --- |
| Removed the `map` entry from the sidebar (`PAGES`) | `app/ui/theme.py` |
| Removed `map` from the "Explore" nav group (`NAV_GROUPS`) | `app/ui/theme.py` |
| Deleted all map-only CSS (`.be-mapgrid`, `.be-map-species`, `.be-map-area`, `.be-obs-src`, `.be-map-legend`) and the responsive `.be-mapgrid` rule | `app/ui/theme.py` |
| Removed the `map_page` import and the `"map": map_page` route entry | `app/ui/streamlit_app.py` |
| Deleted the page module | `app/ui/pages/map_page.py` (removed) |
| Deleted the region dataset, which was used **only** by the map | `app/content/regions.py` (removed) |
| Removed the now-unused `pydeck` dependency line | `requirements.txt` |
| Updated the page list in the docs | `README.md`, `AUTH_COMPLETION_REPORT.md` |
| Deleted map-only tests | `tests/test_map_page.py`, `tests/test_global_map.py` (removed) |

Species **location metadata that is used elsewhere is untouched**: the
`distribution` text and geographic ranges in `app/content/species.py` (Species
Explorer, dashboard featured card, JSON reports) are unchanged. `regions.py` was
a separate map-only lookup table with no other importer.

## Task 2 — Dashboard artefacts fixed at the source

**Root cause (verified, not guessed):** two of the dashboard's decorative icons
used names that are **not real Material Symbols ligatures**. When a ligature
name does not exist, the icon font renders the literal word instead of a glyph,
and because these sit inside small fixed-size `.be-ic` tiles the text overflowed
the card — the stray oversized **"TURTLE"** and **"OSYSTEM"** words.

* `app/ui/pages/dashboard.py` — featured green-sea-turtle card used
  `span("turtle")` → **`turtle` is not a Material Symbols icon**.
* `_MINI_ICONS["dugong"] = "ecosystem"` → **`ecosystem` is not a Material
  Symbols icon**.

Fixes (valid ligature names, no CSS masking):

* Featured card now uses the real icon `water_drop` (same icon as the olive
  ridley turtle card), keyed by species id.
* `_MINI_ICONS` now maps `green_sea_turtle → water_drop`, `olive_ridley_turtle →
  water_drop`, `dugong → water`, `whale_shark → waves`, `spotted_seahorse →
  pets`.

**Every icon name in the app was audited** against the glyph names of the
Material Symbols font bundled with the installed Streamlit (AST scan of
`span(...)`, `icon=...`, `:material/...:` and any `*ICONS` dict). After the fix
the only remaining hit is a false positive (`span(name)` in the helper itself).
No other invalid names exist.

## Task 3 — Professional Account page

`app/ui/auth_ui.py` (with supporting CSS in `app/ui/theme.py`) was reworked:

* **Identity banner** — avatar tile, the signed-in full name and email, and a
  live **Active** status badge.
* **Aligned summary cards** — Account status, Member since, Last sign-in, in a
  responsive grid (3 → 2 → 1 columns).
* **Edit profile** — loads the current name from the database, trims and
  server-validates it, persists per user, and refreshes immediately with a
  confirmation notice.
* **Change password** — current + new + confirm, validated server-side
  (length/complexity, current password verified before anything changes); on
  success the other devices are signed out while this one stays in.
* **Active sessions** — real list with "This device" / "Other device",
  sign-in and expiry times, and per-session **Revoke** plus "Sign out everywhere
  else". Revocation only reports success when the store actually revoked a row.
* **Removed** the caption that printed the raw account id and the internal
  database path — no storage paths, ids, hashes or tokens are rendered anywhere
  on the page.

Role: the schema has no role column (single-tier accounts), so no role is
claimed or fabricated. Status is derived from a real, server-validated account.

## Task 4 — Authentication / data-isolation audit

Audited registration, login, logout, profile, password and sessions end to end.
The implementation (from the previous round) already satisfies the requirements;

- **Passwords** — bcrypt (cost 12); only hashes are stored (`$2b$…`), plaintext
  never persisted/logged/returned.
- **Registration** — email normalised, duplicates rejected by a unique index +
  explicit `DuplicateEmailError`.
- **Login** — unknown email and wrong password both return the same generic
  error, and a dummy bcrypt verification runs on unknown emails, so there is no
  email enumeration or timing oracle.
- **Sessions** — opaque 256-bit tokens; only the SHA-256 digest is stored;
  expiry and revocation are re-checked against the database on **every** run;
  the `blueeye_session` cookie is `SameSite=Lax` (`Secure` on HTTPS) and holds
  no user data.
- **Logout** — revokes the session in the database first, then clears local
  state and the cookie; a genuinely failed revocation is reported as such.
- **Protected routes** — Dashboard, AI Detection, Detection History, Analytics
  and Account require a server-validated session; species/models/about/help stay
  public.
- **No cross-user access** — every profile/history/file operation derives the
  user id from the validated token, never from the browser; per-user directories
  and ownership checks are covered by tests.
- **Legacy ownerless records** — kept on disk, never attributed to a user.

No code defect was found in this area; no behavioural change was required.

## Task 5 — AI detection unchanged

Detection code, model registry and processing were not touched. Verified by
running the **real-model** integration tests (YOLOv8 image + video) — see below.

## Task 6 — Tests actually executed

| Command | Result |
| --- | --- |
| `python -m pytest -q` | **295 passed, 3 deselected** |
| `python -m pytest -m integration -q` (real models) | **3 passed** |
| `python -m compileall app` | ok |
| Icon audit (AST + bundled font glyph names) | no invalid icon names (one false positive) |
| Browser E2E, headless Chrome (`e2e_ui_task.py`) | **31/31 checks passed** |

New/updated automated tests:

* `tests/test_ui_redesign.py` — `test_map_section_is_removed` (nav, routing,
  source files, shell import) and
  `test_dashboard_biodiversity_icons_are_valid_material_symbols` (the two bad
  names are gone and every species icon is a real name); Map removed from the
  required-links and public-pages assertions.
* `tests/test_auth_ui.py` — `test_account_page_hides_internal_implementation_details`
  (no DB path, account id, temp path or bcrypt hash on the page);
  `test_every_sidebar_section_renders_for_a_signed_in_user` (every remaining
  nav entry renders and Map is not offered); Map removed from the guest-pages
  parametrisation.

Browser E2E checks included: sidebar has no Map; every other section label is
present and renders; dashboard sets no `turtle`/`ecosystem` icon text; the
featured card uses `water_drop` and four mini cards render; the account page
hides the database path and account ids while showing status/member-since and
the profile, password and session controls.

The application is running at <http://localhost:8501>.
