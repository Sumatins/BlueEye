"""AppTest coverage for the "My Observations" workspace.

The page is driven exactly as a browser drives it, through the real
``app/ui/streamlit_app.py`` shell and a throwaway account database. A saved
observation is seeded the same way the app stores one (via the real builder and
store), then the UI is asserted to reopen it faithfully - without ever printing
an internal file path.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import streamlit as st

from app.config import Settings, reset_settings_cache
from app.observations.builder import build_analysis
from app.observations.store import ObservationStore

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
        models_dir=app_env / "models",
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


def _seed_observation(app_env: Path, user_id: str, filename: str = "reef_photo.jpg") -> str:
    """Store a real observation for ``user_id`` and return its analysis id."""
    settings = _settings_for(app_env)
    settings.ensure_dirs()
    original = settings.input_dir / user_id / filename
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(_jpeg_bytes())
    annotated = settings.images_output_dir / user_id / "detection_1.jpg"
    annotated.parent.mkdir(parents=True, exist_ok=True)
    annotated.write_bytes(_jpeg_bytes())
    report_path = settings.reports_output_dir / user_id / "reef.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{}", encoding="utf-8")

    report = {
        "input": filename,
        "timestamp": "2026-10-08T19:42:00",
        "models": ["fish_inv"],
        "confidence_threshold": 0.5,
        "statistics": {
            "total_detections": 2,
            "unique_classes": 2,
            "average_confidence": 0.6,
            "max_confidence": 0.8,
        },
        "detections": [
            {"class_name": "fish", "confidence": 0.8, "bbox": [1, 2, 30, 40], "model": "fish_inv"},
            {"class_name": "serranidae", "confidence": 0.4, "bbox": [5, 6, 70, 80], "model": "fish_inv"},
        ],
    }
    analysis = build_analysis(
        user_id=user_id,
        kind="image",
        report=report,
        original_path=original,
        annotated_path=annotated,
        report_path=report_path,
        project="Reef survey 2026",
    )
    ObservationStore(settings.observations_db_path).save_analysis(analysis)
    return analysis.id


def _register_user(app_env: Path) -> AppTest:
    return _register(_app("dashboard"), "Ada", "ada@example.org", "secreT123")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_observations_page_is_empty_for_a_new_account(app_env) -> None:
    at = _register_user(app_env)
    at = _goto(at, "observations")

    assert not at.exception, at.exception
    assert "No observations saved yet" in _rendered_text(at)


def test_saved_observation_appears_in_the_library(app_env) -> None:
    at = _register_user(app_env)
    user_id = at.session_state["be_auth_user_id"]
    _seed_observation(app_env, user_id)

    at = _goto(at, "observations")

    assert not at.exception, at.exception
    blob = _rendered_text(at)
    assert "reef_photo.jpg" in blob
    assert "Reef survey 2026" in blob


def test_observation_detail_never_prints_internal_paths(app_env) -> None:
    at = _register_user(app_env)
    user_id = at.session_state["be_auth_user_id"]
    analysis_id = _seed_observation(app_env, user_id)

    at.session_state["observation_open"] = analysis_id
    at = _goto(at, "observations")

    assert not at.exception, at.exception
    blob = _rendered_text(at)
    assert "reef_photo.jpg" in blob
    assert "blueeye_observations.db" not in blob
    assert str(app_env) not in blob
    assert "detections.json" not in blob


def test_another_accounts_observation_is_not_openable(app_env) -> None:
    at = _register_user(app_env)
    other_id = _seed_observation(app_env, "someone-else", filename="other.jpg")

    at.session_state["observation_open"] = other_id
    at = _goto(at, "observations")

    assert not at.exception, at.exception
    assert "Observation not found" in _rendered_text(at)


def test_correction_through_the_ui_keeps_the_original_and_updates_status(app_env) -> None:
    at = _register_user(app_env)
    user_id = at.session_state["be_auth_user_id"]
    analysis_id = _seed_observation(app_env, user_id)

    at.session_state["observation_open"] = analysis_id
    at = _goto(at, "observations")

    _by_key(at.text_input, f"correct_label_{analysis_id}").set_value("Grouper (Serranidae)")
    at = at.run()
    _by_key(at.button, f"correct_save_{analysis_id}").click()
    at = at.run()

    assert not at.exception, at.exception
    assert "Correction history" in _rendered_text(at)
    stored = ObservationStore(_settings_for(app_env).observations_db_path).get_analysis(
        analysis_id, at.session_state["be_auth_user_id"]
    )
    assert stored.review_status == "corrected"
    assert any(det.class_name == "fish" for det in stored.detections)  # original kept


def test_mark_needs_review_updates_the_record(app_env) -> None:
    at = _register_user(app_env)
    analysis_id = _seed_observation(app_env, at.session_state["be_auth_user_id"])

    at.session_state["observation_open"] = analysis_id
    at = _goto(at, "observations")
    _by_key(at.button, f"review_needs_{analysis_id}").click()
    at = at.run()

    assert not at.exception, at.exception
    stored = ObservationStore(_settings_for(app_env).observations_db_path).get_analysis(
        analysis_id, at.session_state["be_auth_user_id"]
    )
    assert stored.review_status == "needs_review"


def test_delete_removes_the_observation(app_env) -> None:
    at = _register_user(app_env)
    user_id = at.session_state["be_auth_user_id"]
    analysis_id = _seed_observation(app_env, user_id)

    at.session_state["observation_open"] = analysis_id
    at = _goto(at, "observations")

    _by_key(at.checkbox, f"delete_confirm_{analysis_id}").set_value(True)
    at = at.run()
    _by_key(at.button, f"delete_{analysis_id}").click()
    at = at.run()

    assert not at.exception, at.exception
    store = ObservationStore(_settings_for(app_env).observations_db_path)
    assert store.get_analysis(analysis_id, user_id) is None


def test_dashboard_shows_real_saved_record_metrics(app_env) -> None:
    at = _register_user(app_env)
    _seed_observation(app_env, at.session_state["be_auth_user_id"])

    at = _goto(at, "dashboard")

    assert not at.exception, at.exception
    blob = _rendered_text(at)
    assert "Saved research records" in blob
    assert "Saved observations" in blob
    assert "reef_photo.jpg" in blob


def test_dashboard_empty_state_for_new_account(app_env) -> None:
    at = _register_user(app_env)
    at = _goto(at, "dashboard")

    assert not at.exception, at.exception
    assert "No saved observations yet" in _rendered_text(at)


def test_save_observation_helper_dedupes_and_keeps_the_original(app_env) -> None:
    """The exact seam the results-screen "Save observation" button calls."""
    from app.ui import state

    settings = _settings_for(app_env)
    settings.ensure_dirs()
    original = settings.input_dir / "u9" / "photo.jpg"
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(_jpeg_bytes())
    annotated = settings.images_output_dir / "u9" / "detection_1.jpg"
    annotated.parent.mkdir(parents=True, exist_ok=True)
    annotated.write_bytes(_jpeg_bytes())

    report = {
        "input": "photo.jpg",
        "timestamp": "2026-10-08T19:42:00",
        "models": ["fish_inv"],
        "statistics": {"total_detections": 1, "unique_classes": 1},
        "detections": [
            {"class_name": "fish", "confidence": 0.7, "bbox": [1, 2, 3, 4], "model": "fish_inv"}
        ],
    }

    first, created = state.save_observation(
        user_id="u9", kind="image", report=report,
        original_path=original, annotated_path=annotated, report_path=None,
    )
    assert created is True
    again, created_again = state.save_observation(
        user_id="u9", kind="image", report=report,
        original_path=original, annotated_path=annotated, report_path=None,
    )
    assert created_again is False
    assert again.id == first.id
