#!/usr/bin/env python3
"""
Test for compare_models.py script.
"""
import pytest
import subprocess
import numpy as np
import json
from pathlib import Path


def test_compare_models_help():
    """Test that the script runs with help argument."""
    cmd = [
        "python",
        "scripts/compare_models.py",
        "--help",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Compare multiple model results" in result.stdout


def test_compare_models_basic():
    """Test basic functionality with available results."""
    # Test with available results - this will work if results exist
    cmd = [
        "python",
        "scripts/compare_models.py",
        "--models",
        "lda",
        "dummy",
        "--datasets",
        "shin2019__ma",
        "--metric",
        "cross_acc",
        "--seeds",
        "42",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    # The script should run without error, even if no results are found
    # It will log warnings but should not crash
    assert result.returncode == 0 or "No results found" in result.stderr


def test_compare_models_with_output():
    """Test comparison with different parameters."""
    # Test with different alpha level
    cmd = [
        "python",
        "scripts/compare_models.py",
        "--models",
        "lda",
        "dummy",
        "--datasets",
        "shin2019__ma",
        "--metric",
        "cross_acc",
        "--alpha",
        "0.01",
        "--seeds",
        "42",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    # The script should run without error, even if no results are found
    assert result.returncode == 0 or "No results found" in result.stderr


if __name__ == "__main__":
    pytest.main([__file__])
