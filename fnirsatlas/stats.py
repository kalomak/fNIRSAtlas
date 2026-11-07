# Adapted from https://github.com/scikit-learn/scikit-learn (BSD-3-Clause).

"""
corrected_t_test implementation from:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html#comparing-two-models-frequentist-approach
"""

import numpy as np
from scipy.stats import bootstrap, studentized_range, t


def bootstrap_se(values, **kwargs):
    """Calculate bootstrap standard error of the mean using scipy."""
    values = np.array(values)
    if values.std() == 0:
        return 0
    bootstrap_result = bootstrap((values,), np.mean, **kwargs)
    return bootstrap_result.standard_error


def bootstrap_ci(values, **kwargs):
    """Calculate bootstrap confidence interval of the mean using scipy."""
    values = np.array(values)
    bootstrap_result = bootstrap((values,), np.mean, **kwargs)
    return bootstrap_result.confidence_interval


def cohens_d(a, b, epsilon=1e-8):
    """Compute Cohen's d_z for paired data.

    d_z = mean(a - b) / (std(a - b) + epsilon)
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return np.mean(diff) / (np.std(diff, ddof=1) + epsilon)


def corrected_variance(x, n_train, n_test):
    """
    Corrected repeated cross-validation variance from:
    Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366
    """
    j = len(x)  # seeds * folds per seed
    return np.var(x, ddof=1) * (1 / j + n_test / n_train)


def corrected_t_test(x, n_train, n_test, two_tailed=False):
    """
    Corrected repeated cross-validation t-test from:
    Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366
    """
    mu = np.mean(x)
    var = corrected_variance(x, n_train, n_test)
    std = np.max([np.sqrt(var), 1e-9])
    t_stat = mu / std
    p = t.sf(np.abs(t_stat), len(x) - 1)
    p = 2 * p if two_tailed else p
    return t_stat, p


def nemenyi_critical_difference(n_models: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Two-sided Nemenyi critical difference of mean ranks (Demšar 2006)."""
    q_alpha = studentized_range.ppf(1 - alpha, n_models, np.inf) / np.sqrt(2)
    return float(q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_blocks)))


def nemenyi_significant_pairs(
    model_names: list[str], mean_ranks: np.ndarray, cd: float
) -> set[tuple[str, str]]:
    """Return sorted model-name pairs whose mean-rank gap exceeds ``cd``."""
    return {
        tuple(sorted((model_names[i], model_names[j])))
        for i in range(len(model_names))
        for j in range(i + 1, len(model_names))
        if abs(mean_ranks[i] - mean_ranks[j]) > cd
    }
