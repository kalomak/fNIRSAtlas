"""
Test compare_models with mixed input types (file paths and dictionaries).
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

from fnirsatlas.compare import compare_models


def test_compare_models_mixed_inputs():
    """Test compare_models with dictionary inputs."""

    # Create mock results
    mock_results1 = {
        "dataset1": {
            "task1": {"cross_acc": [0.8, 0.85, 0.82]},
            "task2": {"cross_acc": [0.7, 0.75, 0.72]},
        }
    }

    mock_results2 = {
        "dataset1": {
            "task1": {"cross_acc": [0.75, 0.78, 0.76]},
            "task2": {"cross_acc": [0.65, 0.68, 0.66]},
        }
    }

    # Add model names
    mock_results1["model_name"] = "Model 1"
    mock_results2["model_name"] = "Model 2"

    # Test with dictionary inputs
    comparisons = compare_models(
        [mock_results1, mock_results2], metric="cross_acc", alpha=0.05
    )

    # Check that comparison was performed
    assert len(comparisons) > 0

    # Check that model names are generated correctly
    comparison_key = list(comparisons.keys())[0]
    assert (
        "Model 1_vs_Model 2" in comparison_key or "Model 2_vs_Model 1" in comparison_key
    )

    # Check that results contain expected data
    for task_results in comparisons.values():
        for task, results in task_results.items():
            assert "p_value" in results
            assert "p_value_corrected" in results
            assert "significant" in results
            assert "mean_diff" in results


def test_compare_models_all_dictionaries():
    """Test compare_models with all dictionary inputs."""

    # Create mock results
    mock_results1 = {
        "dataset1": {
            "task1": {"cross_acc": [0.8, 0.85, 0.82]},
        }
    }

    mock_results2 = {
        "dataset1": {
            "task1": {"cross_acc": [0.75, 0.78, 0.76]},
        }
    }

    # Add model names
    mock_results1["model_name"] = "Model 1"
    mock_results2["model_name"] = "Model 2"

    # Test with all dictionary inputs
    comparisons = compare_models(
        [mock_results1, mock_results2], metric="cross_acc", alpha=0.05
    )

    # Check that comparison was performed
    assert len(comparisons) > 0

    # Check that model names are generated correctly
    comparison_key = list(comparisons.keys())[0]
    assert "Model 1_vs_Model 2" in comparison_key

    # Check that results contain expected data
    for task_results in comparisons.values():
        for task, results in task_results.items():
            assert "p_value" in results
            assert "p_value_corrected" in results
            assert "significant" in results
            assert "mean_diff" in results


if __name__ == "__main__":
    pytest.main([__file__])
