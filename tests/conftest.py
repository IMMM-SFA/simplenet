"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from simplenet.case import PowerCase
from simplenet.io.matpower import load_m


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def case9(fixtures_dir: Path) -> PowerCase:
    """The canonical 9-bus MATPOWER test case."""

    return load_m(fixtures_dir / "case9.m")
