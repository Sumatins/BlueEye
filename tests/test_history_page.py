"""AppTest coverage for the detection History page.

The page is driven exactly as a browser drives it - through the real
``app/ui/streamlit_app.py`` and against a throwaway account database. A run
is then seeded into the account's *own* history store, and we assert the page
re-opens it faithfully while never printing an internal file path.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from app.config import Settings, reset_settings_cache
from app.utils.history import HistoryEntry, add_entry, history_path

APP = str(Path(__file__).resolve().parents[1] / "app" / "ui" / "streamlit_app.py")


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture()
def app_env(tmp_path: Path, monkeypatch):
    """Point the app at temporary directories and a throwaway auth DB."""
    monkeypatch.setenv("BLUEEYE_AUTH_DB", str(tmp_path / "auth" / "blueeye_auth.db"))
    monkeypatch.setenv("BLUEEYE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BLUEEYE_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("BLUEEYE_MODELS_DIR", str(tmp_path / "models"))
    reset_settings_cache()
    st.cache_resource.clear()
    st.cache_data.clear()
    yield tmp_path
    reset_settings_cache()
    st.cache_resource.clear()
    st.cache_data.clear()
    shutil.rmtree(tmp_path, ignore_errors=True)


def _settings_for(app_env: Path) -> Settings:
    return Settings(
        outputs_dir=app_env / "outputs",
        data_dir=app_env / "data",
        auth_db=app_env / "auth" / "blueeye_auth.db",
    )


def _by_key(elements, key):
    for element in elements:
        if getattr(element, "key", None) == key:
            return element
    available = [getattr(e, "key", None) for e in elements]
    raise AssertionError(f"no {key!r} widget (available: {available})")


def _app(page: str = "dashboard") -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["nav_sel"] = page
    return at


def _register(at: AppTest, name: str, email: str, password: str) -> AppTest:
    at.run()
    _by_key(at.radio, "auth_mode").set_value("Create account")
    at.run()
    _by_key(at.text_input, "auth_register_name").set_value(name)
    _by_key(at.text_input, "auth_register_email").set_value(email)
    _by_key(at.text_input, "auth_register_password").set_value(password)
    _by_key(at.text_input, "auth_register_confirm").set_value(password)
    _by_key(at.button, "auth_register_submit").click()
    return at.run()


def _goto(at: AppTest, page: str) -> AppTest:
    _by_key(at.radio, "nav_sel").set_value(page)
    return at.run()


def _rendered_text(at: AppTest) -> str:
    parts: list[str] = []
    for attr in ("markdown", "caption", "text", "info", "warning", "success", "error", "subheader"):
        for element in getattr(at, attr, []) or []:
            parts.append(str(getattr(element, "value", "") or ""))
    return "\n".join(parts)


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (6, 6), (9, 40, 90)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _seed_image_run(app_env: Path, user_id: str, output_exists: bool = True) -> None:
    settings = _settings_for(app_env)

    original = settings.input_dir / user_id / "reef_photo.jpg"
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(_jpeg_bytes())

    output = settings.images_output_dir / user_id / "detection_1.jpg"
    if output_exists:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_jpeg_bytes())

    add_entry(
        HistoryEntry(
            timestamp="2026-10-08T19:42:00",
            kind="image",
            filename="reef_photo.jpg",
            models=["fish_inv"],
            total_detections=3,
            unique_classes=2,
            average_confidence=0.74,
            max_confidence=0.88,
            confidence_threshold=0.5,
            enhanced=False,
            output=str(output),
            report=None,
            elapsed_seconds=1.4,
            detections_per_class={"scaridae": 2, "turtle": 1},
            user_id=user_id,
        ),
        path=history_path(settings, user_id=user_id),
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_history_page_reopens_a_saved_run(app_env) -> None:
    at = _register(_app("dashboard"), "Ada", "ada@example.org", "secreT123")
    user_id = at.session_state["be_auth_user_id"]
    _seed_image_run(app_env, user_id)

    at = _goto(at, "history")

    assert not at.exception, at.exception
    _by_key(at.selectbox, "history_select")

    blob = _rendered_text(at)
    assert "reef_photo.jpg" in blob
    assert "Scaridae" in blob or "scaridae" in blob


def test_history_page_never_prints_internal_paths(app_env) -> None:
    at = _register(_app("dashboard"), "Ada", "ada@example.org", "secreT123")
    user_id = at.session_state["be_auth_user_id"]
    _seed_image_run(app_env, user_id)

    at = _goto(at, "history")

    assert not at.exception, at.exception
    blob = _rendered_text(at)
    assert "detections.json" not in blob
    assert "blueeye_auth.db" not in blob
    assert str(app_env) not in blob


def test_history_page_survives_a_deleted_output(app_env) -> None:
    at = _register(_app("dashboard"), "Ada", "ada@example.org", "secreT123")
    user_id = at.session_state["be_auth_user_id"]
    _seed_image_run(app_env, user_id, output_exists=False)

    at = _goto(at, "history")

    assert not at.exception, at.exception
    blob = _rendered_text(at)
    assert "no longer available" in blob


def test_history_is_empty_for_a_brand_new_account(app_env) -> None:
    at = _register(_app("dashboard"), "Ada", "ada@example.org", "secreT123")

    at = _goto(at, "history")

    assert not at.exception, at.exception
    blob = _rendered_text(at)
    assert "No detections recorded yet" in blob
