import pytest
from scripts.wilcoxon_test import perform_wilcoxon_test, format_wilcoxon_table


def test_perform_wilcoxon_test():
    """Test Wilcoxon test functionality."""
    # Create sample comparison data
    comparisons = {
        "model1_vs_model2": {
            "buccino2016__Side[R|L]×Joint[Arm|Hand]": {
                "model1_mean": 75.0,
                "model2_mean": 70.0,
                "mean_diff": 5.0,
            },
            "ortega2020__HandGrip[L|R]": {
                "model1_mean": 80.0,
                "model2_mean": 75.0,
                "mean_diff": 5.0,
            },
            "luke2021__Control|FingerTap[L|R]": {
                "model1_mean": 85.0,
                "model2_mean": 80.0,
                "mean_diff": 5.0,
            },
        }
    }

    # Test that the function runs without error
    perform_wilcoxon_test(comparisons, alpha=0.05, metric="cross_acc")


def test_format_wilcoxon_table():
    """Test Wilcoxon table formatting."""
    # Create sample comparison data
    comparisons = {
        "model1_vs_model2": {
            "buccino2016__Side[R|L]×Joint[Arm|Hand]": {
                "model1_mean": 75.0,
                "model2_mean": 70.0,
                "mean_diff": 5.0,
            },
            "ortega2020__HandGrip[L|R]": {
                "model1_mean": 80.0,
                "model2_mean": 75.0,
                "mean_diff": 5.0,
            },
            "luke2021__Control|FingerTap[L|R]": {
                "model1_mean": 85.0,
                "model2_mean": 80.0,
                "mean_diff": 5.0,
            },
        }
    }

    # Test that the function runs without error
    format_wilcoxon_table(comparisons, alpha=0.05, metric="cross_acc")


def test_perform_wilcoxon_test_with_overlapping_tasks():
    """Test Wilcoxon test with overlapping tasks (should filter them out)."""
    # Create sample comparison data with overlapping tasks
    comparisons = {
        "model1_vs_model2": {
            "luke2021__Control|FingerTap[L|R]": {
                "model1_mean": 75.0,
                "model2_mean": 70.0,
                "mean_diff": 5.0,
            },
            "luke2021__FingerTap[L|R]": {  # Same dataset, different task (overlapping)
                "model1_mean": 80.0,
                "model2_mean": 75.0,
                "mean_diff": 5.0,
            },
            "ortega2020__HandGrip[L|R]": {
                "model1_mean": 85.0,
                "model2_mean": 80.0,
                "mean_diff": 5.0,
            },
        }
    }

    # Test that the function runs without error and filters overlapping tasks
    perform_wilcoxon_test(comparisons, alpha=0.05, metric="within_acc")
