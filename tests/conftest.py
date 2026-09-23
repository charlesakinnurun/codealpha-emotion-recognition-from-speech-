"""Shared fixtures backed by the synthetic RAVDESS tree in ``tests.helpers``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import make_ravdess_tree


@pytest.fixture
def ravdess_tree(tmp_path: Path) -> Path:
    """A clean synthetic RAVDESS tree. Returns the data directory."""
    return make_ravdess_tree(tmp_path, with_anomalies=False)


@pytest.fixture
def ravdess_tree_with_anomalies(tmp_path: Path) -> Path:
    """Synthetic RAVDESS tree plus corrupt, song-channel, and duplicate files."""
    return make_ravdess_tree(tmp_path, with_anomalies=True)
