import pytest
import numpy as np
from fnirsatlas.stats import (
    bootstrap_ci,
    bootstrap_se,
    nemenyi_critical_difference,
    nemenyi_significant_pairs,
)


def test_bootstrap_ci():
    """Test bootstrap confidence interval calculation."""
    # Create sample data
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Calculate CI
    ci = bootstrap_ci(data, confidence_level=0.95)

    # Check that CI is a tuple with 2 values
    assert isinstance(ci, tuple)
    assert len(ci) == 2

    # Check that lower bound < upper bound
    assert ci[0] < ci[1]

    # Check that mean is within CI
    mean = np.mean(data)
    assert ci[0] <= mean <= ci[1]


def test_bootstrap_ci_consistency():
    """Test that bootstrap CI is consistent with SE."""
    # Create sample data
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Calculate both CI and SE
    ci = bootstrap_ci(data, confidence_level=0.95)
    se = bootstrap_se(data)
    mean = np.mean(data)

    # CI should be roughly mean ± 2*SE for 95% confidence
    expected_lower = mean - 2 * se
    expected_upper = mean + 2 * se

    # Allow some tolerance for bootstrap variability
    assert abs(ci[0] - expected_lower) < 1.0
    assert abs(ci[1] - expected_upper) < 1.0


def test_nemenyi_critical_difference_decisions():
    # Demšar (2006) Table 5: q_0.05 = 2.343 for 3 models
    cd = nemenyi_critical_difference(n_models=3, n_blocks=20)
    assert abs(cd - 2.343 * np.sqrt(3 * 4 / (6 * 20))) < 1e-3
    pairs = nemenyi_significant_pairs(["a", "b", "c"], np.array([1.0, 1.0 + cd, 1.01 + cd]), cd)
    assert pairs == {("a", "c")}
