# Blue Eye — Authentication: Completion Report

Scope: add **real** authentication (accounts, sessions, protected routes,
per-account data) to the Blue Eye Streamlit app — backed by a real SQLite
database and real bcrypt hashing, with no frontend-only or fake auth.

Status: **complete and verified.** All automated tests pass and a real browser
end-to-end run against the running server passes. Nothing has been pushed.

---

## 1. What was built

A self-contained auth backend in `app/auth/` that the Streamlit UI calls
directly (there is no separate API server):

| File | Responsibility |
| --- | --- |
| `app/auth/models.py` | `User`, `SessionInfo` value objects (never carry password hashes to the UI) |
| `app/auth/errors.py` | Typed errors (`InvalidCredentialsError`, `DuplicateEmailError`, `AuthValidationError`, …) |
| `app/auth/security.py` | bcrypt hashing (rounds = 12, 72-byte cap), email normalisation, password policy, opaque token minting/`sha256` digesting |
| `app/auth/store.py` | SQLite persistence (`users`, `sessions`); `executescript` schema creation; per-operation connections |
| `app/auth/service.py` | Business logic: register, login, authenticate, logout, profile, change password, list/revoke sessions |

UI layer:

- `app/ui/auth_ui.py` — gate (sign in / create account), server-side session
  bootstrap, sign-out, sidebar account block, and the Account page internals.
- `app/ui/pages/account.py` — Account page (profile, password change, device
  sessions).
- `app/config.py` — auth settings (`BLUEEYE_AUTH_DB`, `BLUEEYE_SESSION_TTL_DAYS`,
  `BLUEEYE_PASSWORD_MIN_LENGTH`) read from the environment.
- `app/ui/streamlit_app.py` — app-wide gate with a private/public page allowlist
  and the sidebar account block.

## 2. Behaviour

- **Registration** — full name, email (normalised + validated), password
  (min length + letter + number). Duplicate emails rejected case-insensitively.
  Password hashed with bcrypt; plaintext never stored or logged. Bad input and
  DB errors are surfaced honestly.
- **Login** — verified against the stored hash; a single generic message and
  constant-ish timing mean the form never reveals whether an email exists.
- **Sessions** — 256-bit opaque token; only its SHA-256 digest is stored.
  Fixed server-side expiry (default 7 days) and immediate server-side
  revocation. The token is re-validated against the database on **every** run.
- **Refresh survival** — Streamlit session state alone is lost on a full page
  reload, so the token is also mirrored into a `SameSite=Lax` cookie (with
  `Secure` on HTTPS) written from the page and read back on load via
  `st.context.cookies`, then validated exactly like a session-state token.
- **Sign-out** — revokes the session in the database first, then clears local
  state and the cookie. If revocation genuinely fails, the UI says so instead
  of falsely claiming success.
- **Protected routes** — `dashboard`, `detect`, `history`, `analytics`,
  `account` require a signed-in user. `species`, `models`, `about`,
  `help` stay public with a guest sign-in prompt. Anonymous visitors land on the
  private dashboard, so the gate is the landing screen.
- **Per-account data** — history (`outputs/history/users/<user_id>/detections.json`),
  uploads (`data/input/<user_id>/`), annotated outputs and JSON reports live in
  per-user directories. The user id is server-generated; ownership is checked
  before any file is shown or downloaded. Pre-auth history is preserved on disk
  but hidden from per-user views.
- **Cross-account safety** — signing in as another account clears private
  session state and caches, so one account never sees another's results.

## 3. Tests actually executed

Command: `python -m pytest -q`

```
320 passed, 3 deselected in 58.99s
```

(The 3 deselected are integration tests excluded by `pyproject.toml` `addopts`.)

| Suite | Tests | What it covers |
| --- | --- | --- |
| `tests/test_auth_service.py` | 20 | Real backend: hashing, duplicate email, bad credentials, session expiry/revocation, password change, session listing/revocation |
| `tests/test_auth_ui.py` | 27 | `AppTest` against the real `streamlit_app.py`: gate on every private page, public pages open, register/login/logout, refresh keeps the session, revoked token rejected, cookie emitted on sign-in / cleared on sign-out |
| `tests/test_user_scoping.py` | 8 | Per-user history paths, ownership checks, cross-user isolation |
| `tests/test_ui_redesign.py` | 15 | Navigation, Account page presence, every page render accepts `user`, private/public gating, auth UI uses the real backend |

`compileall` over `app/` and `tests/` is clean.

### Browser end-to-end (real server, real database)

Playwright (system Chrome) driving the real Streamlit server on `:8501` with a
throwaway SQLite database and temp data/output dirs. **21/21 checks passed**:

```
[PASS] gate shown to anonymous visitor
[PASS] anonymous is not signed in
[PASS] account A registered and signed in
[PASS] account A exists in the real database
[PASS] session cookie set after sign-in
[PASS] session survives browser refresh
[PASS] real detection completed in the browser
[PASS] account A has a per-account history file
[PASS] account A history recorded 1 run
[PASS] history entry carries account A's id
[PASS] logout returns to the gate
[PASS] session cookie cleared after logout
[PASS] refresh after logout keeps the gate
[PASS] reopening the app after logout shows the gate
[PASS] account A sessions revoked in the database
[PASS] account B registered with a distinct id
[PASS] account B has no history of its own yet
[PASS] account B page is authenticated
[PASS] account A can sign back in
[PASS] account A history still present on disk
[PASS] account B never received account A's run
```

This used two independent accounts, a genuine image detection through the real
models, on-disk history inspection, and a direct SQLite query to confirm
server-side revocation.

## 4. Configuration and secrets

| Variable | Default | Purpose |
| --- | --- | --- |
| `BLUEEYE_AUTH_DB` | `<outputs>/auth/blueeye_auth.db` | Accounts + sessions (git-ignored, under the outputs volume) |
| `BLUEEYE_SESSION_TTL_DAYS` | `7` | Server-side session lifetime |
| `BLUEEYE_PASSWORD_MIN_LENGTH` | `8` | Minimum password length |

- Documented in `.env.example` and the README; no credentials are committed.
- `.env` and `outputs/` are git-ignored, so the database (emails, bcrypt hashes,
  session digests) is never tracked.
- bcrypt was added to `requirements.txt`.

## 5. Limitations (honest)

- **No email verification / password reset / SSO.** Accounts are active
  immediately; there is no SMTP flow. Appropriate for self-hosted use.
- **Single SQLite database.** Fine for one instance; multi-instance deployments
  need shared storage or an external identity provider.
- **The session cookie is not `HttpOnly`.** A Streamlit app cannot set response
  headers, so the cookie is written from the page. It holds only a random,
  revocable token (no personal data) and is useless once revoked; serve over
  HTTPS and treat the browser as trusted.
- **Tabs share the cookie** (standard web behaviour): signing in once signs in
  the same browser's other tabs.
- **Pre-auth history** that predates accounts is preserved on disk but is not
  visible in per-user views.
- **No linter run:** `ruff`/`flake8` are not installed in this environment, so
  only `compileall` and the test suite were used.
- Authentication does **not** change detection: models, thresholds, enhancement
  and unrelated features are untouched.
