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
    assert "Prepare a licensed dataset" in blob
    assert "never reports a metric it did not measure" in blob


def test_comparison_page_is_honest_when_no_weights_are_installed(app_env) -> None:
    at = _register(_app("compare"))
    at = _goto(at, "compare")

    assert not at.exception, at.exception
    blob = _rendered_text(at)
    assert "No model weights installed" in blob
    # The scope caveat is still shown before any result.
    assert "How to read this" in blob


# --------------------------------------------------------------------------- #
# Stored evaluation runs (per-account, read back from disk)
# --------------------------------------------------------------------------- #
def _stored_payload(model: str) -> dict:
    return {
        "created": "2026-10-12T10:00:00",
        "config": {"split": "test", "device": "cpu"},
        "environment": {
            "platform": "test",
            "hardware": {"gpu": None, "cuda_available": False},
            "packages": {},
        },
        "evaluations": [
            {
                "model": model,
                "dataset": "tiny",
                "split": "test",
                "status": "complete",
                "model_classes": ["scaridae"],
                "comparison": {
                    "supported": ["scaridae"],
                    "unsupported_dataset_classes": ["shark"],
                    "extra_model_classes": ["fish"],
                },
                "metrics": {
                    "mAP50": 0.0,
                    "mAP50_95": 0.0,
                    "classes": [
                        {
                            "class_id": 0,
                            "name": "scaridae",
                            "gt": 0,
                            "predictions": 2,
                            "precision": 0.0,
                            "recall": 0.0,
                            "AP50": 0.0,
                            "AP50_95": 0.0,
                        }
                    ],
                    "notes": [],
                },
                "timing": {
                    "images": 1,
                    "predictions": 2,
                    "inference_seconds": 0.8,
                    "seconds_per_image": 0.8,
                },
                "notes": ["Ground truth is empty; metrics are honestly zero."],
            }
        ],
        "summary": [
            {
                "model": model,
                "dataset": "tiny",
                "split": "test",
                "status": "complete",
                "images": 1,
                "objects": 0,
                "mAP50": 0.0,
                "mAP50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "seconds_per_image": 0.8,
            }
        ],
    }


def _seed_stored_run(user_id: str, name: str, model: str) -> None:
    run_dir = get_settings().outputs_dir / "evaluation" / user_id / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps(_stored_payload(model)), encoding="utf-8"
    )
    (run_dir / "metrics.csv").write_text("model,mAP50\n%s,0.0\n" % model, encoding="utf-8")


def test_stored_evaluation_run_is_rendered_for_its_owner(app_env) -> None:
    at = _register(_app("evaluation"))
    user_id = at.session_state["be_auth_user_id"]
    _seed_stored_run(user_id, "20261012-100000", "Owned Model")
    # Another account's run must never surface in this account's console.
    _seed_stored_run("b" * 32, "20261012-090000", "Someone Elses Model")

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

