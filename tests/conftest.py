from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "songs"


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
