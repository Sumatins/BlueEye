"""Tests for deployment-time configuration helpers.

Covers the Streamlit-secrets bridge in :mod:`app.config` (used by Streamlit
Community Cloud, where deployment values arrive via ``st.secrets`` rather
than process environment variables) and a few consistency checks on the
deployment files themselves.

SPDX-License-Identifier: AGPL-3.0-only
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import streamlit as st

import app.config as config
from app.config import load_streamlit_secrets, reset_settings_cache

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _FakeSecrets(dict):
    """Minimal stand-in for ``st.secrets``: a mapping with ``keys()``."""


class _BoomSecrets:
    """A secrets object that fails loudly if it is ever touched."""

    def keys(self):
        raise AssertionError("st.secrets must not be read without a secrets file")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure no BLUEEYE_* variable leaks between tests."""
    for key in list(os.environ):
        if key.startswith("BLUEEYE_"):
            monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


# --------------------------------------------------------------------------- #
# load_streamlit_secrets
# --------------------------------------------------------------------------- #
def test_secrets_noop_without_secrets_file(tmp_path, monkeypatch):
    """Without a secrets file nothing is imported or changed."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(st, "secrets", _BoomSecrets(), raising=False)
    load_streamlit_secrets()
    assert not any(key.startswith("BLUEEYE_") for key in os.environ)


def test_secrets_merges_blueeye_keys(tmp_path, monkeypatch):
    secrets_file = tmp_path / ".streamlit" / "secrets.toml"
    secrets_file.parent.mkdir(parents=True)
    secrets_file.write_text('BLUEEYE_LOG_LEVEL = "DEBUG"\n', encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(st, "secrets", _FakeSecrets({"BLUEEYE_LOG_LEVEL": "DEBUG"}), raising=False)

    load_streamlit_secrets()

    assert os.environ.get("BLUEEYE_LOG_LEVEL") == "DEBUG"


def test_secrets_never_overwrite_existing_env(tmp_path, monkeypatch):
    secrets_file = tmp_path / ".streamlit" / "secrets.toml"
    secrets_file.parent.mkdir(parents=True)
    secrets_file.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("BLUEEYE_LOG_LEVEL", "INFO")
    monkeypatch.setattr(st, "secrets", _FakeSecrets({"BLUEEYE_LOG_LEVEL": "DEBUG"}), raising=False)

    load_streamlit_secrets()

    assert os.environ["BLUEEYE_LOG_LEVEL"] == "INFO"


def test_secrets_ignore_non_blueeye_and_non_string(tmp_path, caplog, monkeypatch):
    secrets_file = tmp_path / ".streamlit" / "secrets.toml"
    secrets_file.parent.mkdir(parents=True)
    secrets_file.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        st,
        "secrets",
        _FakeSecrets({"SOME_API_TOKEN": "s3cret", "BLUEEYE_PORT": 8501}),
        raising=False,
    )

    load_streamlit_secrets()

    assert "SOME_API_TOKEN" not in os.environ
    assert "BLUEEYE_PORT" not in os.environ
    assert any("BLUEEYE_PORT" in record.message for record in caplog.records)


def test_secrets_survive_streamlit_errors(tmp_path, monkeypatch):
    """A broken secrets source must never break application startup."""

    class _Raiser:
        def keys(self):
            raise RuntimeError("bad toml")

    secrets_file = tmp_path / ".streamlit" / "secrets.toml"
    secrets_file.parent.mkdir(parents=True)
    secrets_file.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(st, "secrets", _Raiser(), raising=False)

    load_streamlit_secrets()  # must not raise


def test_get_settings_reads_secrets(tmp_path, monkeypatch):
    """Settings built after the bridge see secret-provided values."""
    secrets_file = tmp_path / ".streamlit" / "secrets.toml"
    secrets_file.parent.mkdir(parents=True)
    secrets_file.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(st, "secrets", _FakeSecrets({"BLUEEYE_LOG_LEVEL": "WARNING"}), raising=False)

    settings = config.get_settings()

    assert settings.log_level == "WARNING"


# --------------------------------------------------------------------------- #
# Deployment file consistency
# --------------------------------------------------------------------------- #
def test_deployment_requirements_include_root_and_cpu_pins():
    """The Community Cloud requirements file must mirror the root file and
    force CPU-only torch wheels (the Linux runners must not pull CUDA)."""
    deployment = PROJECT_ROOT / "app" / "ui" / "requirements.txt"
    root = PROJECT_ROOT / "requirements.txt"
    assert deployment.is_file(), "app/ui/requirements.txt is required for Community Cloud"
    assert root.is_file()
    text = deployment.read_text(encoding="utf-8")
    assert "-r ../../requirements.txt" in text
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in text
    assert "torch==2.13.0+cpu" in text
    assert "torchvision==0.28.0+cpu" in text


def test_packages_txt_installs_opencv_system_libs():
    packages = PROJECT_ROOT / "packages.txt"
    assert packages.is_file(), "packages.txt is required for Community Cloud apt deps"
    text = packages.read_text(encoding="utf-8")
    assert "libgl1" in text
    assert "libglib2.0-0" in text


def test_secrets_template_has_placeholders_only():
    """The committed template must not contain any real-looking secret."""
    template = PROJECT_ROOT / ".streamlit" / "secrets.toml.example"
    assert template.is_file()
    for line in template.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Every non-comment line in the template is a commented-out key; if
        # this ever fails, someone uncommented a placeholder.
        pytest.fail(f"secrets template must stay fully commented out, found: {stripped!r}")


def test_gitignore_allows_secrets_template_but_not_real_secrets():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!.streamlit/secrets.toml.example" in gitignore
    assert ".streamlit/*" in gitignore
