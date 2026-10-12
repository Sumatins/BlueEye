"""AppTest coverage for the Research pages.

Both consoles are driven through the real application shell and a throwaway
account database. With no datasets and no weights in the temporary workspace,
the pages must show their honest empty/pending states - never invented content
and never an exception.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.config import get_settings, reset_settings_cache

APP = str(Path(__file__).resolve().parents[1] / "app" / "ui" / "streamlit_app.py")


@pytest.fixture()
def app_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BLUEEYE_AUTH_DB", str(tmp_path / "auth" / "blueeye_auth.db"))
    monkeypatch.setenv("BLUEEYE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BLUEEYE_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("BLUEEYE_MODELS_DIR", str(tmp_path / "models"))
    # Keep the continuous worker out of the UI tests: scheduling is exercised
    # directly in tests/test_continuous_evaluation.py.
    monkeypatch.setenv("BLUEEYE_SCHEDULER", "off")
    reset_settings_cache()
    st.cache_resource.clear()
    yield tmp_path
    reset_settings_cache()
    st.cache_resource.clear()


def _by_key(elements, key):
    for element in elements:
        if getattr(element, "key", None) == key:
            return element
    raise AssertionError(f"no widget with key {key!r}")


def _app(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["nav_sel"] = page
    return at


def _register(at: AppTest) -> AppTest:
    at.run()
    _by_key(at.radio, "auth_mode").set_value("Create account")
    at.run()
    _by_key(at.text_input, "auth_register_name").set_value("Ada")
    _by_key(at.text_input, "auth_register_email").set_value("ada@example.org")
    _by_key(at.text_input, "auth_register_password").set_value("secreT123")
    _by_key(at.text_input, "auth_register_confirm").set_value("secreT123")
    _by_key(at.button, "auth_register_submit").click()
    return at.run()


def _goto(at: AppTest, page: str) -> AppTest:
    _by_key(at.radio, "nav_sel").set_value(page)
    return at.run()


def _rendered_text(at: AppTest) -> str:
    parts: list[str] = []
    for attr in (
        "markdown",
        "caption",
        "text",
        "info",
        "warning",
        "success",
        "error",
        "subheader",
        "dataframe",
    ):
        for element in getattr(at, attr, []) or []:
            value = getattr(element, "value", "")
            parts.append("" if value is None else str(value))
    return "\n".join(parts)


def test_evaluation_center_is_pending_without_a_dataset(app_env) -> None:
    at = _register(_app("evaluation"))
    at = _goto(at, "evaluation")

    assert not at.exception, at.exception
    blob = _rendered_text(at)
    assert "pending" in blob.lower()
    # The page explains how to configure a dataset instead of inventing a score.
    assert "no labelled dataset" in blob.lower()
    assert "convert / register an existing yolo export" in blob.lower()
    assert "never reports a metric it did not measure" in blob


def test_comparison_page_is_honest_when_no_weights_are_installed(app_env) -> None:
    at = _register(_app("compare"))
    at = _goto(at, "compare")

    assert not at.exception, at.exception
    blob = _rendered_text(at)
    assert "No model weights installed" in blob
    # The scope caveat is still shown before any result.
    assert "How to read this" in blob


def test_all_evaluation_sections_render(app_env) -> None:
    at = _register(_app("evaluation"))
    user_id = at.session_state["be_auth_user_id"]
    _seed_stored_run(user_id, "Owned Model", "Owned Model")
    at = _goto(at, "evaluation")

    for section in ("Run", "Experiments", "Schedules", "Datasets", "Compare", "Export"):
        _by_key(at.radio, "eval_section").set_value(section)
        at = at.run()
        assert not at.exception, (section, at.exception)
        if section == "Experiments":
            assert "Owned Model" in _rendered_text(at)


def test_compare_section_offers_a_comparison_export(app_env) -> None:
    at = _register(_app("evaluation"))
    user_id = at.session_state["be_auth_user_id"]
    _seed_stored_run(user_id, "Run A", "Run A")
    _seed_stored_run(user_id, "Run B", "Run B")
    at = _goto(at, "evaluation")
    _by_key(at.radio, "eval_section").set_value("Compare")
    at = at.run()

    assert not at.exception, at.exception
    download_labels = [
        getattr(element, "label", "") for element in at.get("download_button")
    ]
    assert any("comparison" in label.lower() for label in download_labels)


def test_schedules_section_lists_a_saved_schedule(app_env) -> None:
    at = _register(_app("evaluation"))
    user_id = at.session_state["be_auth_user_id"]
    _seed_schedule(user_id, "Reef watch")
    at = _goto(at, "evaluation")
    _by_key(at.radio, "eval_section").set_value("Schedules")
    at = at.run()

    assert not at.exception, at.exception
    assert "Reef watch" in _rendered_text(at)


def test_export_section_builds_a_publication_bundle(app_env) -> None:
    at = _register(_app("evaluation"))
    user_id = at.session_state["be_auth_user_id"]
    _seed_stored_run(user_id, "Paper run", "Paper run")
    at = _goto(at, "evaluation")
    _by_key(at.radio, "eval_section").set_value("Export")
    at = at.run()

    assert not at.exception, at.exception
    assert "Paper run" in _rendered_text(at)

    _by_key(at.button, "export_prepare").click()
    at = at.run()

    assert not at.exception, at.exception
    download_labels = [
        getattr(element, "label", "") for element in at.get("download_button")
    ]
    assert any("publication_metrics.csv" in label for label in download_labels)
    assert any("publication_report.html" in label for label in download_labels)


def _seed_schedule(user_id: str, name: str) -> str:
    """Save one schedule row for the account (no run is started)."""
    from app.evaluation.schedules import ScheduleStore

    store = ScheduleStore(get_settings().evaluation_db_path)
    return store.create(
        user_id,
        name,
        "tiny/data.yaml",
        dataset_name="tiny",
        split="test",
        class_names=("scaridae",),
        model_keys=("fish_inv",),
        trigger_mode="on_change",
        interval_minutes=60,
    ).id


# --------------------------------------------------------------------------- #
# Stored evaluation runs (per-account, read back from the persistent store)
# --------------------------------------------------------------------------- #
def _seed_stored_run(user_id: str, name: str, model: str) -> str:
    """Create a completed experiment with one result and one artifact."""
    from app.evaluation.compat import compare_classes
    from app.evaluation.experiments import EvaluationStore
    from app.evaluation.metrics import DetectionBox, evaluate
    from app.evaluation.runner import ModelEvaluation

    settings = get_settings()
    store = EvaluationStore(settings.evaluation_db_path)
    experiment = store.create_experiment(
        user_id,
        name,
        dataset_name="tiny",
        data_yaml="tiny/data.yaml",
        split="test",
        class_names=("scaridae",),
        status="completed",
    )
    metrics = evaluate(
        {"a.jpg": [DetectionBox(0, 0, 0, 10, 10, 1.0)]},
        {"a.jpg": []},
        class_names={0: "scaridae"},
    )
    evaluation = ModelEvaluation(
        model=model,
        dataset="tiny",
        split="test",
        model_classes=("scaridae",),
        compat=compare_classes(("scaridae",), ("scaridae",)),
        metrics=metrics,
        num_images=1,
        images_evaluated=1,
        num_predictions=0,
        inference_seconds=0.8,
        status="complete",
        notes=("Ground truth is empty; metrics are honestly zero.",),
    )
    store.add_model_result(experiment.id, user_id, evaluation)

    run_dir = get_settings().outputs_dir / "evaluation" / user_id / experiment.id
    run_dir.mkdir(parents=True, exist_ok=True)
    results = run_dir / "results.json"
    results.write_text(
        json.dumps({"config": {"split": "test"}, "summary": []}), encoding="utf-8"
    )
    store.add_artifact(experiment.id, user_id, results, name="results.json", role="report")
    return experiment.id


def test_stored_evaluation_run_is_rendered_for_its_owner(app_env) -> None:
    at = _register(_app("evaluation"))
    user_id = at.session_state["be_auth_user_id"]
    _seed_stored_run(user_id, "Owned Model", "Owned Model")
    # Another account's run must never surface in this account's console.
    _seed_stored_run("b" * 32, "Someone Elses Model", "Someone Elses Model")

    at = _goto(at, "evaluation")

    assert not at.exception, at.exception
    blob = _rendered_text(at)
    assert "Owned Model" in blob
    assert "mAP@0.5" in blob
    assert "Someone Elses Model" not in blob
    # Downloads for the stored run are offered.
    download_labels = [
        getattr(element, "label", "") for element in at.get("download_button")
    ]
    assert any(label.startswith("Results (JSON)") for label in download_labels)

