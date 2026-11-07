#!/usr/bin/env python3
"""
Script for comparing 3+ models with a Friedman test.

Uses one non-overlapping task mean per dataset as the independent block.
Post-hoc Nemenyi test is reported by default.
"""

import argparse
import logging

import numpy as np
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, rankdata

from fnirsatlas.compare import extract_model_metrics
from fnirsatlas.config import get_paper_models
from fnirsatlas.format import format_dataset_name
from fnirsatlas.logging import setup_logging
from fnirsatlas.results import Results
from fnirsatlas.utils import get_datasets_from_args, filter_independent_task_ids


setup_logging()
logger = logging.getLogger(__name__)


SETTING_DEFAULT_STRATEGY = {"cross": "nested", "within": "block"}
MODELS = get_paper_models()


def format_block_label(task_id, max_task_len=28):
    """Format a task id as dataset and task labels."""
    dataset_id, task_desc = task_id.rsplit("__", 1)
    dataset = format_dataset_name(dataset_id)
    if len(task_desc) > max_task_len:
        task_desc = task_desc[: max_task_len - 3] + "..."
    return dataset, task_desc


def significance_stars(p_value, alpha=0.05):
    """Return significance stars for a p-value."""
    if p_value < alpha / 10:
        return "***"
    if p_value < alpha / 5:
        return "**"
    if p_value < alpha:
        return "*"
    return ""


def build_score_matrix(results_dicts, metric="cross_acc"):
    """Build a block-by-model score matrix from result dictionaries."""
    model_names = [
        results.get("model_name", f"model_{i}")
        for i, results in enumerate(results_dicts)
    ]
    all_metrics = {
        model_name: extract_model_metrics(results, metric)[0]
        for model_name, results in zip(model_names, results_dicts)
    }
    common_task_ids = set.intersection(
        *[set(metrics) for metrics in all_metrics.values()]
    )
    if not common_task_ids:
        raise ValueError("No common tasks found across models")

    task_ids, dropped_ids = filter_independent_task_ids(common_task_ids)
    if len(task_ids) < 2:
        raise ValueError("Need at least 2 independent tasks for Friedman test")

    score_matrix = np.array(
        [
            [
                np.mean(all_metrics[model_name][task_id]) * 100
                for model_name in model_names
            ]
            for task_id in task_ids
        ]
    )
    return model_names, task_ids, score_matrix, dropped_ids


def compute_model_summary(model_names, score_matrix):
    """Compute mean performance and mean rank per model."""
    block_ranks = np.array([rankdata(-row, method="average") for row in score_matrix])
    summary_rows = []
    for idx, model_name in enumerate(model_names):
        summary_rows.append(
            {
                "model": model_name,
                "mean_perf": float(np.mean(score_matrix[:, idx])),
                "mean_rank": float(np.mean(block_ranks[:, idx])),
            }
        )
    summary_rows.sort(key=lambda row: (row["mean_rank"], -row["mean_perf"]))
    return summary_rows


def run_friedman_test(score_matrix):
    """Run the omnibus Friedman test."""
    statistic, p_value = friedmanchisquare(
        *[score_matrix[:, idx] for idx in range(score_matrix.shape[1])]
    )
    return float(statistic), float(p_value)


def run_posthoc_nemenyi(score_matrix, model_names, alpha=0.05):
    """Run Nemenyi post-hoc test via scikit-posthocs."""
    p_matrix = sp.posthoc_nemenyi_friedman(score_matrix)
    model_order = sorted(
        range(len(model_names)),
        key=lambda idx: (-float(np.mean(score_matrix[:, idx])), model_names[idx]),
    )
    rows = []
    for pos1, idx1 in enumerate(model_order):
        for idx2 in reversed(model_order[pos1 + 1 :]):
            p_value = float(p_matrix.iloc[idx1, idx2])
            mean_diff = float(np.mean(score_matrix[:, idx1] - score_matrix[:, idx2]))
            rows.append(
                {
                    "model1": model_names[idx1],
                    "model2": model_names[idx2],
                    "mean_diff": mean_diff,
                    "p_value": p_value,
                    "significant": bool(p_value < alpha),
                    "significance": significance_stars(p_value, alpha),
                }
            )
    return rows


def print_model_summary(summary_rows):
    """Print model-level summary stats."""
    print("\nModel summary")
    print("=" * 52)
    print(f"{'Model':<20} {'Mean Perf':>12} {'Mean Rank':>12}")
    print("-" * 52)
    for row in summary_rows:
        print(
            f"{row['model']:<20} {row['mean_perf']:>11.1f}% {row['mean_rank']:>12.2f}"
        )


def print_posthoc_rows(rows):
    """Print Nemenyi post-hoc comparison rows."""
    print("\nPost-hoc Nemenyi")
    print("=" * 68)
    print(f"{'Model 1':<18} {'Model 2':<18} {'Diff':>8} {'P-Value':>10} {'Sig':>4}")
    print("-" * 68)
    for row in rows:
        print(
            f"{row['model1']:<18} {row['model2']:<18} {row['mean_diff']:>8.1f} "
            f"{row['p_value']:>10.3f} {row['significance']:>4}"
        )


def print_block_rows(task_ids):
    """Print independent blocks used in the test."""
    print("\nBlocks used")
    print("=" * 52)
    for task_id in task_ids:
        dataset, task = format_block_label(task_id)
        print(f"- {dataset}: {task}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform a Friedman test across 3+ models."
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        default=MODELS,
        help="Models to compare (default: config paper comparison_models)",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        help="Specific datasets to compare. If not provided, all available datasets will be used.",
    )
    parser.add_argument(
        "-s",
        "--suite",
        type=str,
        help="Suite name to compare (e.g., motor, MA, MI, nback, Emotion, MISC)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cross_acc",
        help="Metric to analyze (default: cross_acc)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(5)),
        help="Random seeds used for results (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--no-post-hoc",
        action="store_true",
        help="Skip pairwise post-hoc Nemenyi test",
    )
    parser.add_argument(
        "--show-blocks",
        action="store_true",
        help="Print the independent dataset-task blocks used in the test",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.models) < 3:
        raise ValueError("Provide at least 3 models for Friedman test")
    setting = Results._infer_classification_settings(args.metric)
    search_strategy = SETTING_DEFAULT_STRATEGY.get(setting, "none")

    datasets = get_datasets_from_args(args.datasets, args.suite)
    if not datasets:
        logger.error("No datasets specified or found")
        return

    results_dicts = Results.gather_results(
        models=args.models,
        datasets=datasets,
        metric=args.metric,
        seeds=args.seeds,
        search_strategy=search_strategy,
        classification_setting=setting,
    )
    if len(results_dicts) < 3:
        logger.error("Need results for at least 3 models")
        return

    model_names, task_ids, score_matrix, dropped_ids = build_score_matrix(
        results_dicts, metric=args.metric
    )
    statistic, p_value = run_friedman_test(score_matrix)
    summary_rows = compute_model_summary(model_names, score_matrix)

    print("\n" + "=" * 88)
    print(f"Friedman test on {args.metric} task means (alpha = {args.alpha})")
    print("=" * 88)
    print(f"Models: {', '.join(model_names)}")
    print(f"Independent blocks: {len(task_ids)}")
    if dropped_ids:
        print(f"Dropped overlapping tasks: {len(dropped_ids)}")
    print_model_summary(summary_rows)
    print(
        f"\nOmnibus result: chi2={statistic:.3f}, p={p_value:.3f} {significance_stars(p_value, args.alpha)}"
    )

    if not args.no_post_hoc:
        posthoc_rows = run_posthoc_nemenyi(score_matrix, model_names, alpha=args.alpha)
        print_posthoc_rows(posthoc_rows)

    if args.show_blocks:
        print_block_rows(task_ids)


if __name__ == "__main__":
    main()
