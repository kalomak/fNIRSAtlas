"""Task informativeness metrics for benchmark comparison."""

import numpy as np
from scipy.stats import rankdata
import logging

logger = logging.getLogger(__name__)

def paired_residual_spread(fold_scores_matrix):
    """Residual SD after removing model and fold main effects.
    Args:
        fold_scores_matrix: shape (n_folds, n_models).
    """
    grand_mean = fold_scores_matrix.mean()
    model_means = fold_scores_matrix.mean(axis=0, keepdims=True)
    fold_means = fold_scores_matrix.mean(axis=1, keepdims=True)
    residuals = fold_scores_matrix - model_means - fold_means + grand_mean
    return float(np.std(residuals, ddof=1))


def bootstrap_rank_stability(fold_scores_matrix, n_boot=2000, rng=None):
    """Rank stability via paired bootstrap over fold indices for one task.

    Resamples fold indices, recomputes model means and ranks,
    returns mean Spearman rho with original ranking.

    Args:
        fold_scores_matrix: shape (n_folds, n_models).
        n_boot: number of bootstrap resamples.
        rng: numpy Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_folds, n_models = fold_scores_matrix.shape
    if n_models < 2:
        return 0.0
    original_means = fold_scores_matrix.mean(axis=0)
    # If all models tied, ranking is arbitrary -> stability undefined
    if np.std(original_means) < 1e-12:
        return 0.0
    original_ranks = rankdata(-original_means)
    idx = rng.integers(0, n_folds, size=(n_boot, n_folds))
    boot_means = fold_scores_matrix[idx].mean(axis=1)
    boot_ranks = rankdata(-boot_means, axis=1)
    centered_original = original_ranks - original_ranks.mean()
    centered_boot = boot_ranks - boot_ranks.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(centered_original) * np.linalg.norm(
        centered_boot, axis=1
    )
    rhos = np.divide(
        centered_boot @ centered_original,
        denominator,
        out=np.zeros(n_boot),
        where=denominator > 0,
    )
    return float(np.mean(rhos))


def validate_result_cube(
    scores_by_seed,
    expected_seeds,
    expected_tasks,
    expected_models,
    expected_folds,
):
    """Validate a complete seed-task-model-fold result cube."""
    expected_seeds = set(expected_seeds)
    if set(scores_by_seed) != expected_seeds:
        raise ValueError(
            f"Incomplete seeds: expected {sorted(expected_seeds)}, "
            f"found {sorted(scores_by_seed)}"
        )
    expected_tasks = set(expected_tasks)
    expected_models = set(expected_models)
    for seed in sorted(expected_seeds):
        tasks = scores_by_seed[seed]
        if set(tasks) != expected_tasks:
            raise ValueError(
                f"Incomplete tasks for seed {seed}: expected {sorted(expected_tasks)}, "
                f"found {sorted(tasks)}"
            )
        for task in sorted(expected_tasks):
            models = tasks[task]
            if set(models) != expected_models:
                raise ValueError(
                    f"Incomplete models for seed {seed}, task {task}: expected "
                    f"{sorted(expected_models)}, found {sorted(models)}"
                )
            for model, scores in models.items():
                values = np.asarray(scores, dtype=float)
                if values.ndim != 1 or len(values) != expected_folds:
                    raise ValueError(
                        f"Incomplete folds for seed {seed}, task {task}, model "
                        f"{model}: expected {expected_folds}, found {values.shape}"
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Non-finite scores for seed {seed}, task {task}, model {model}"
                    )


def combine_result_cube(scores_by_seed):
    """Concatenate validated seed-specific folds into task-model arrays."""
    seeds = sorted(scores_by_seed)
    first_seed = scores_by_seed[seeds[0]]
    return {
        task: {
            model: np.concatenate(
                [np.asarray(scores_by_seed[seed][task][model]) for seed in seeds]
            )
            for model in first_seed[task]
        }
        for task in first_seed
    }


def _minmax_normalize(x, epsilon=1e-8):
    """Min-max normalize to [0, 1]."""
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + epsilon)


def informativeness_score(rank_stability, sprr, margin):
    """Equal-weight composite informativeness score per task."""
    return (
        _minmax_normalize(rank_stability)
        + _minmax_normalize(sprr)
        + _minmax_normalize(margin)
    ) / 3.0


def _top_k_task_mean(score_mat, top_k_models=None):
    """Compute per-task mean over the top-k model means."""
    if top_k_models is None:
        top_k_models = score_mat.shape[1]
    if top_k_models < 1:
        raise ValueError("top_k_models must be at least 1")
    top_k_models = min(top_k_models, score_mat.shape[1])
    top_scores = np.sort(score_mat, axis=1)[:, -top_k_models:]
    return top_scores.mean(axis=1)


def compute_task_informativeness(
    fold_scores_by_task,
    n_boot=9999,
    rng_seed=42,
    top_k_models=None,
):
    """Compute kappa-based informativeness metrics for each task."""
    task_ids = sorted(fold_scores_by_task.keys())
    model_names = sorted(next(iter(fold_scores_by_task.values())).keys())
    n_tasks = len(task_ids)
    rng = np.random.default_rng(rng_seed)
    # Build score matrix and per-task arrays
    score_mat = np.zeros((n_tasks, len(model_names)))
    residual_sd = np.zeros(n_tasks)
    rank_stabs = np.zeros(n_tasks)
    for i, tid in enumerate(task_ids):
        fs = fold_scores_by_task[tid]
        lengths = {m: len(scores) for m, scores in fs.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Mismatched fold counts for {tid}: {lengths}")
        for j, m in enumerate(model_names):
            score_mat[i, j] = np.mean(fs[m])
        fold_mat = np.column_stack([np.asarray(fs[m]) for m in model_names])
        residual_sd[i] = paired_residual_spread(fold_mat)
        rank_stabs[i] = bootstrap_rank_stability(fold_mat, n_boot, rng)
    task_means = _top_k_task_mean(score_mat, top_k_models=top_k_models)
    sd_between = np.std(score_mat, axis=1, ddof=1)
    sprr = sd_between / (residual_sd + 1e-8)
    scores = informativeness_score(rank_stabs, sprr, task_means)
    out = {}
    for i, tid in enumerate(task_ids):
        out[tid] = {
            "mean_perf": float(task_means[i]),
            "paired_residual_sd": float(residual_sd[i]),
            "sprr": float(sprr[i]),
            "rank_stability": float(rank_stabs[i]),
            "informativeness": float(scores[i]),
        }
    return out
