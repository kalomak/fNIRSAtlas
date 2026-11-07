"""Test path resolution functionality."""

import pytest
from pathlib import Path
from fnirsatlas.config import load_config
from fnirsatlas.utils import resolve_path


def test_resolve_path_relative():
    """Test that relative paths use the current directory."""
    assert resolve_path("./results").resolve() == Path.cwd() / "results"


def test_resolve_path_absolute():
    """Test resolving absolute paths."""
    absolute_path = resolve_path("/tmp/test")
    expected_path = Path("/tmp/test")
    assert absolute_path == expected_path


def test_resolve_path_expands_user_home():
    """Test resolving paths that use the home directory shortcut."""
    path = resolve_path("~/mne_data/fnirsatlas/datasets")
    expected_path = Path.home() / "mne_data" / "fnirsatlas" / "datasets"
    assert path == expected_path


def test_load_config_applies_full_path_from_env(tmp_path, monkeypatch):
    """Test that FNIRSATLAS_CONFIG accepts a full path outside the package."""
    override = tmp_path / "my_config.yaml"
    override.write_text('paths:\n  results_cache: "/tmp/my_results"\n')
    monkeypatch.setenv("FNIRSATLAS_CONFIG", str(override))
    assert load_config()["paths"]["results_cache"] == "/tmp/my_results"


def test_load_config_raises_for_missing_explicit_file(tmp_path, monkeypatch):
    """Test that a requested override file must exist."""
    monkeypatch.setenv("FNIRSATLAS_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(FileNotFoundError):
        load_config()
