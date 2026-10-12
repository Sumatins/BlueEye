"""Integration test: the Research Evaluation page runs a real evaluation.

Skipped unless the official marine-detect weights and the bundled sample image
are present (weights are never fabricated for testing). The test prepares a
one-image labelled dataset on disk, drives the page through the real Streamlit
shell with an actual signed-in account, clicks **Run evaluation**, and checks
that the report landed under the account's own evaluation directory.

Ground truth is intentionally empty, so the metric is honestly zero - this test
verifies the *UI path* (discovery -> run -> write -> read back), not accuracy.

Run explicitly with::

    pytest -m integration

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.config import PROJECT_ROOT, get_settings, reset_settings_cache
from app.detection.detector import MarineDetector
from app.detection.model_manager import ModelManager
from app.evaluation import model_class_names
from app.evaluation.experiments import TERMINAL_STATUSES, EvaluationStore

APP = str(Path(__file__).resolve().parents[1] / "app" / "ui" / "streamlit_app.py")
SAMPLE_IMAGE = PROJECT_ROOT / "assets" / "images" / "input_folder" / "regq.jpg"


def _weights_present() -> bool:
    if not SAMPLE_IMAGE.is_file():
        return False
    manager = ModelManager()
    return manager.is_available("fish_inv") and manager.is_available("megafauna")


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _weights_present(),
        reason="core weights / sample image unavailable (run scripts/download_models.py)",
    ),
]


@pytest.fixture()
def app_env(tmp_path: Path, monkeypatch):
    """Temp workspace, but the *real* model weights (models dir untouched)."""
    monkeypatch.setenv("BLUEEYE_AUTH_DB", str(tmp_path / "auth" / "blueeye_auth.db"))
    monkeypatch.setenv("BLUEEYE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BLUEEYE_OUTPUTS_DIR", str(tmp_path / "outputs"))
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


def _build_dataset(data_dir: Path) -> Path:
    """A one-image YOLO dataset using the real sample image, no fake labels."""
    root = data_dir / "raw" / "tiny"
    (root / "images" / "test").mkdir(parents=True)
    (root / "labels" / "test").mkdir(parents=True)
    shutil.copyfile(SAMPLE_IMAGE, root / "images" / "test" / "regq.jpg")
    (root / "labels" / "test" / "regq.txt").write_text("", encoding="utf-8")

    detector = MarineDetector(settings=get_settings())
    classes = model_class_names(detector, "fish_inv")
    lines = [
        "name: tiny",
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines += [f"  {index}: {name}" for index, name in enumerate(classes)]
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_evaluation_page_runs_and_stores_a_real_report(app_env) -> None:
    _build_dataset(app_env / "data")

    at = AppTest.from_file(APP, default_timeout=600)
    at.session_state["nav_sel"] = "evaluation"
    at = _register(at)
    _by_key(at.radio, "nav_sel").set_value("evaluation")
    at = at.run()

    # One model, one image, no charts - keep the integration run quick.
    _by_key(at.multiselect, "eval_models").set_value(["fish_inv"])
    _by_key(at.number_input, "eval_limit").set_value(1)
    _by_key(at.checkbox, "eval_plots").set_value(False)
    at.run()

    _by_key(at.button, "eval_run").click()
    at = at.run()

    user_id = at.session_state["be_auth_user_id"]
    settings = get_settings()
    store = EvaluationStore(settings.evaluation_db_path)

    # The run happens on a background thread; poll the persistent store until it
    # reaches a terminal status (or the deadline passes).
    record = None
    deadline = time.time() + 300
    while time.time() < deadline:
        finished = [
            item
            for item in store.list_experiments(user_id)
            if item.status in TERMINAL_STATUSES
        ]
        if finished:
            record = finished[0]
            break
        time.sleep(1)
        at = at.run()

    assert record is not None, "the evaluation never reached a terminal status"
    assert record.status in {"completed", "partially_completed"}, record.status

    # Re-render the page now the run is stored; the report download appears.
    at = at.run()
    assert not at.exception, at.exception

    reports = list((settings.outputs_dir / "evaluation" / user_id).glob("*/results.json"))
    assert len(reports) == 1, reports
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["config"]["split"] == "test"
    assert payload["summary"][0]["images"] == 1
    assert payload["summary"][0]["status"] in {"complete", "not_comparable"}

    # The stored run is read back and its exports are offered.
    labels = [getattr(element, "label", "") for element in at.get("download_button")]
    assert any(label.startswith("Results (JSON)") for label in labels)
