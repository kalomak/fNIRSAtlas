"""Tests for the format module."""

import numpy as np
import pytest
from fnirsatlas.format import (
    format_dataset_name,
    format_score,
    _extract_metric,
    _format_metric_value,
    format_benchmark_table,
    format_model_comparison_table,
    format_paper_model_name,
)


def test_format_dataset_name():
    """Test dataset name formatting."""
    assert format_dataset_name("bak2019") == "Bak2019"
    assert format_dataset_name("shin2019__motor") == "Shin2019 (motor)"


def test_format_score():
    """Test score formatting."""
    assert format_score(0.5, 8) == "    50.0"  # Right-aligned with width 8
    assert format_score(np.nan, 8) == "nan"


def test_format_classical_model_names():
    assert format_paper_model_name("logreg") == "LR"
    assert format_paper_model_name("svm") == "SVM"
    assert format_paper_model_name("rf") == "RF"
    assert format_paper_model_name("knn") == "kNN"


def test_extract_metric():
    """Test metric extraction."""
    # Test list format
    task_results = {"cross_acc": [0.5, 0.6, 0.7]}
    value, std = _extract_metric(task_results, "cross_acc")
    assert np.isclose(value, 60.0)
    assert std > 0

    # Test dict format
    task_results = {"cross_acc": {"mean": 0.5, "std": 0.1}}
    value, std = _extract_metric(task_results, "cross_acc")
    assert np.isclose(value, 50.0)
    assert np.isclose(std, 10.0)

    # Test single value
    task_results = {"cross_acc": 0.5}
    value, std = _extract_metric(task_results, "cross_acc")
    assert np.isclose(value, 50.0)
    assert std == 0

    # Test missing metric
    value, std = _extract_metric(task_results, "missing")
    assert np.isnan(value)
    assert np.isnan(std)


def test_format_metric_value():
    """Test metric value formatting."""
    assert _format_metric_value(50.0, 5.0, True) == "50.0±5.0"
    assert _format_metric_value(50.0, 0.0, False) == "50.0"
    assert _format_metric_value(np.nan, 0.0, False) == "nan"


def test_format_benchmark_table(capsys):
    """Test benchmark table formatting."""
    results_dict = {
        "test_dataset": {
            "test_task": {
                "cross_acc": [0.5, 0.6],
                "cross_kappa": [0.3, 0.4],
                "within_acc": [0.7, 0.8],
                "within_kappa": [0.5, 0.6],
                "n_classes": 2,
                "n_epochs": 100,
            }
        }
    }

    format_benchmark_table(results_dict)
    captured = capsys.readouterr()

    # Check that output contains expected elements
    assert "Summary:" in captured.out
    assert "Test_dataset" in captured.out  # Dataset name gets formatted
    assert "test_task" in captured.out
    assert "MEAN" in captured.out


def test_format_model_comparison_table_latex_adds_midrule_before_mean(capsys):
    """LaTeX output should separate the summary row with a midrule."""
    results_dicts = [
        {
            "model_name": "lda",
            "test_dataset": {
                "test_task": {
                    "cross_acc": [0.5, 0.6],
                    "n_classes": 2,
                    "n_epochs": 100,
                }
            },
        },
        {
            "model_name": "eegnet",
            "test_dataset": {
                "test_task": {
                    "cross_acc": [0.7, 0.8],
                    "n_classes": 2,
                    "n_epochs": 100,
                }
            },
        },
    ]

    format_model_comparison_table(results_dicts, metric="cross_acc", latex=True)
    captured = capsys.readouterr()

    assert r"\midrule" in captured.out
    assert captured.out.index(r"\midrule") < captured.out.index("Mean &")

    format_model_comparison_table(
        results_dicts, metric="cross_acc", latex=True, columns=("dataset", "classes")
    )
    rows = capsys.readouterr().out.splitlines()
    assert rows[0].startswith("Test_dataset & 2 & ")
    assert rows[-1].startswith("Mean &  & ")
    assert all(row.count("&") == 3 for row in (rows[0], rows[-1]))
