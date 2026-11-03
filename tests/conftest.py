from __future__ import annotations

from pathlib import Path

import os

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "songs"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """别让开发机上真实的 .env / 环境变量影响测试。"""
    monkeypatch.setenv("DUBBING_NO_DOTENV", "1")
    for name in (
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "DUBBING_CLIENT",
        "DUBBING_MODEL",
        "DUBBING_BASE_URL",
        "DUBBING_TEMPERATURE",
        "DUBBING_TEMPERATURE_A",
        "DUBBING_TEMPERATURE_B",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def data_root() -> Path:
    return DATA_ROOT


@pytest.fixture
def root(tmp_path: Path) -> Path:
    d = tmp_path / "songs"
    d.mkdir()
    return d
