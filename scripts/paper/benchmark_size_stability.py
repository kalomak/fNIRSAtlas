#!/usr/bin/env python3
"""Model-ranking stability and Nemenyi power versus the number of benchmark tasks.

For each subset size k, draws task subsets from the independent cross-subject
tasks (all subsets if at most ``--n-draws``, otherwise ``--n-draws`` random ones)
and compares the subset ranking with the full ranking. Significant pairs are
counted per subset as pairs whose mean-rank gap exceeds the Nemenyi critical
difference for k tasks.

Example:
    uv run python scripts/paper/benchmark_size_stability.py
"""

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from fnirsatlas.config import (
    get_cv_config,
    get_paper_models,
    get_results_cache_dir,
    get_results_dir,
)
from fnirsatlas.informativeness import combine_result_cube
from fnirsatlas.parsers import get_available_datasets
from fnirsatlas.stats import nemenyi_critical_difference, nemenyi_significant_pairs

if __package__:
    from scripts.paper.task_informativeness_group_holdout import pairwise_agreement
    from scripts.paper.task_informativeness_lomo import (
        SEEDS,
        load_result_cube,
        model_ranking,
    )
else:
    from task_informativeness_group_holdout import pairwise_agreement
    from task_informativeness_lomo import SEEDS, load_result_cube, model_ranking

N_DRAWS = 1000
RNG_SEED = 42
ALPHA = 0.05
METRIC = "cross_acc"
# BenchNIRS datasets (Benerradi et al. 2023) mapped to our task definitions.
BENCHNIRS_TASKS = [
    "bak2019__FingerTap[L|R]",
    "herff2014__1-back|2-back|3-back",
    "shin2016__ma__BL|MA",
    "shin2018__nback__0-back|2-back|3-back",
    "shin2018__wg__BL|WG",
]


def task_subsets(tasks, size, n_draws, rng):
    """All size-k subsets if there are at most ``n_draws``, else ``n_draws`` random ones."""
    if comb(len(tasks), size) <= n_draws:
        return combinations(tasks, size)
    return (
        tuple(sorted(rng.choice(tasks, size=size, replace=False).tolist()))
        for _ in range(n_draws)
    )


def compare_subset(fold_scores, full, tasks):
    """Agreement with the full ranking and Nemenyi significant pairs for one subset."""
    reduced = model_ranking(fold_scores, list(tasks))
    models = sorted(full["rank"])
    cd = nemenyi_critical_difference(len(models), len(tasks), ALPHA)
    mean_ranks = np.array([reduced["mean_rank"][model] for model in models])
    return {
        "ranking": reduced["ranking"],
        "ranks": [reduced["rank"][model] for model in models],
        "spearman": float(spearmanr(
            [full["rank"][model] for model in models],
            [reduced["rank"][model] for model in models],
        ).statistic),
        "pairwise_agreement": pairwise_agreement(full, reduced, models),
        "critical_difference": cd,
        "significant_pairs": len(nemenyi_significant_pairs(models, mean_ranks, cd)),
    }


def _summary(values, prefix):
    q025, median, q975 = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        f"{prefix}_q025": float(q025),
        f"{prefix}_median": float(median),
        f"{prefix}_q975": float(q975),
    }


def intersubset_stability(rank_vectors):
    """Spearman and pairwise order agreement between all pairs of subset rankings."""
    ranks = np.asarray(rank_vectors, dtype=float)
    if len(ranks) == 1:
        return np.array([1.0]), np.array([1.0])
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    correlations = centered @ centered.T / np.outer(norms, norms)
    pairs = np.array(list(combinations(range(ranks.shape[1]), 2)))
    signs = np.sign(ranks[:, pairs[:, 0]] - ranks[:, pairs[:, 1]])
    agreements = (signs @ signs.T / len(pairs) + 1) / 2
    upper = np.triu_indices(len(ranks), k=1)
    return correlations[upper], agreements[upper]


def evaluate_subset_sizes(fold_scores, full, n_draws=N_DRAWS, rng_seed=RNG_SEED):
    """Summarize subset-versus-full agreement and significant pairs per subset size."""
    rng = np.random.default_rng(rng_seed)
    tasks = full["tasks"]
    rows = []
    for size in range(2, len(tasks) + 1):
        results = [
            compare_subset(fold_scores, full, subset)
            for subset in task_subsets(tasks, size, n_draws, rng)
        ]
        inter_rho, inter_pairwise = intersubset_stability([r["ranks"] for r in results])
        rows.append({
            "subset_size": size,
            "draws": len(results),
            "critical_difference": results[0]["critical_difference"],
            **{
                key: value
                for metric in ("spearman", "pairwise_agreement", "significant_pairs")
                for key, value in _summary([r[metric] for r in results], metric).items()
            },
            **_summary(inter_rho, "intersubset_spearman"),
            **_summary(inter_pairwise, "intersubset_pairwise_agreement"),
        })
    return rows


def plot(rows, benchnirs, pair_count, output_path):
    """(a) Spearman vs full ranking, (b) Nemenyi significant pairs, against subset size."""
    sizes = [row["subset_size"] for row in rows]
    fig, (ax_rho, ax_pairs) = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    ink, accent = "0.15", "tab:orange"
    _band(ax_rho, sizes, rows, "spearman", ink)
    ax_rho.plot(benchnirs["subset_size"], benchnirs["spearman"], "D", color=accent, ms=6, label="BenchNIRS datasets")
    ax_rho.set(xlabel="Number of tasks", ylabel="Spearman ρ vs full ranking", ylim=(None, 1.02))
    ax_rho.legend(loc="lower right", frameon=False)
    _band(ax_pairs, sizes, rows, "significant_pairs", ink)
    ax_pairs.plot(benchnirs["subset_size"], benchnirs["significant_pairs"], "D", color=accent, ms=6)
    ax_pairs.set(xlabel="Number of tasks", ylabel=f"Significant pairs (of {pair_count})")
    ax_pairs.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Panel labels left of the top y tick, at the height of rho = 1, so they add no vertical space.
    for ax, label, offset in ((ax_rho, "a)", -27), (ax_pairs, "b)", -36)):
        ax.annotate(
            label, xy=(0, 1.0), xycoords=(ax.transAxes, ax_rho.transData),
            xytext=(offset, 0), textcoords="offset points",
            ha="right", va="center", fontweight="bold",
        )
    for ax in (ax_rho, ax_pairs):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3, lw=0.5)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _band(ax, sizes, rows, metric, color):
    ax.fill_between(
        sizes,
        [row[f"{metric}_q025"] for row in rows],
        [row[f"{metric}_q975"] for row in rows],
        color=color, alpha=0.15, lw=0, label="95% percentile interval",
    )
    ax.plot(sizes, [row[f"{metric}_median"] for row in rows], color=color, lw=1.5, label="Median")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=get_results_cache_dir("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=get_results_dir() / "benchmark_size")
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    return parser.parse_args()


def main():
    """Run the benchmark-size analysis and save the figure and its data."""
    args = parse_args()
    models = get_paper_models()
    cube = load_result_cube(
        models,
        get_available_datasets(),
        SEEDS,
        get_cv_config()["cross"]["kwargs"]["n_splits"],
        args.results_dir.expanduser(),
        metric=METRIC,
    )
    scores = combine_result_cube(cube)
    full = model_ranking(scores, list(scores))
    missing = sorted(set(BENCHNIRS_TASKS) - set(full["tasks"]))
    if missing:
        raise ValueError(f"BenchNIRS tasks absent from independent tasks: {missing}")
    benchnirs = {"subset_size": len(BENCHNIRS_TASKS), **compare_subset(scores, full, BENCHNIRS_TASKS)}
    rows = evaluate_subset_sizes(scores, full, args.n_draws, args.rng_seed)
    pair_count = len(models) * (len(models) - 1) // 2
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "benchmark_size_stability.png"
    plot(rows, benchnirs, pair_count, figure_path)
    data_path = output_dir / "benchmark_size_stability.json"
    payload = {
        "metadata": {
            "models": models,
            "seeds": SEEDS,
            "metric": METRIC,
            "alpha": ALPHA,
            "independent_tasks": full["tasks"],
            "excluded_tasks": full["dropped_tasks"],
            "pair_count": pair_count,
            "draws_per_size_max": args.n_draws,
            "rng_seed": args.rng_seed,
            "full_ranking": full["ranking"],
            "interval": "Empirical 2.5th and 97.5th percentiles across subsets.",
            "intersubset_comparison": "All unique pairs of subset rankings of the same size. Pairs are not independent, so percentiles are descriptive.",
            "significant_pairs": "Per subset: pairs whose mean-rank gap exceeds the Nemenyi CD for that subset size.",
            "benchnirs_caveat": "Our task definitions; Bak 2019 is binary rather than BenchNIRS's three-class task.",
        },
        "results": rows,
        "benchnirs_subset": {"tasks": BENCHNIRS_TASKS, **benchnirs},
    }
    data_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"FIGURE: {figure_path}\nDATA: {data_path}")


if __name__ == "__main__":
    main()
