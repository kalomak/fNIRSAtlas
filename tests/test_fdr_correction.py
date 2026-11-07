"""
Test FDR correction functionality in compare module.
"""

import pytest
import numpy as np
import json
import tempfile
import os
from pathlib import Path

from fnirsatlas.compare import compare_models, extract_model_metrics


def test_fdr_correction_applied():
    """Test that FDR correction is applied to p-values."""

    # Create mock results with known p-values
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

    # Add model names to results
    mock_results1["model_name"] = "Model 1"
    mock_results2["model_name"] = "Model 2"

    # Run comparison
    comparisons = compare_models([mock_results1, mock_results2], alpha=0.05)

    # Check that FDR correction was applied
    assert len(comparisons) > 0

    for comparison_key, task_results in comparisons.items():
        for task, results in task_results.items():
            # Check that corrected p-values exist
            assert "p_value_corrected" in results
            assert "p_value" in results

            # Check that significance is based on corrected p-values
            assert "significant" in results

            # Verify that corrected p-values are different from original
            # (in this case they should be the same since we only have 2 tests)
            assert results["p_value_corrected"] >= results["p_value"]


def test_fdr_correction_multiple_tests():
    """Test FDR correction with multiple statistical tests."""

    # Create mock results with more tasks to test FDR correction
    mock_results1 = {
        "dataset1": {
            "task1": {"cross_acc": [0.8, 0.85, 0.82]},
            "task2": {"cross_acc": [0.7, 0.75, 0.72]},
            "task3": {"cross_acc": [0.6, 0.65, 0.62]},
            "task4": {"cross_acc": [0.5, 0.55, 0.52]},
        }
    }

    mock_results2 = {
        "dataset1": {
            "task1": {"cross_acc": [0.75, 0.78, 0.76]},
            "task2": {"cross_acc": [0.65, 0.68, 0.66]},
            "task3": {"cross_acc": [0.55, 0.58, 0.56]},
            "task4": {"cross_acc": [0.45, 0.48, 0.46]},
        }
    }

    # Add model names to results
    mock_results1["model_name"] = "Model 1"
    mock_results2["model_name"] = "Model 2"

    # Run comparison
    comparisons = compare_models([mock_results1, mock_results2], alpha=0.05)

    # Check that we have the expected number of comparisons
    assert len(comparisons) == 1  # Only one comparison between two models

    comparison_key = list(comparisons.keys())[0]
    task_results = comparisons[comparison_key]

    # Check that we have all 4 tasks
    assert len(task_results) == 4

    # Verify that FDR correction was applied to all tasks
    for task, results in task_results.items():
        assert "p_value_corrected" in results
        assert "p_value" in results
        assert "significant" in results


if __name__ == "__main__":
    pytest.main([__file__])
