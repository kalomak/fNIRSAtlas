#!/usr/bin/env python3
"""
Script for performing Wilcoxon tests on model comparison results.

Uses a single classification task per dataset to avoid overlapping data (dependent samples).

Usage:
    python scripts/wilcoxon_test.py --models lda dummy --datasets shin2019 --metric cross_acc --alpha 0.05
    python scripts/wilcoxon_test.py --models lda eegnet --suite motor --metric within_acc --seeds 1 2 3
    python scripts/wilcoxon_test.py --models lda eegnet --format latex --output './../paper/nirsatlas/tex/results/wilcoxon_table.tex'
"""

import logging
import argparse
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon, rankdata
from fnirsatlas.utils import get_non_overlapping_clf_tasks, get_datasets_from_args
from fnirsatlas.compare import compare_models
from fnirsatlas.format import format_dataset_name
from fnirsatlas.results import Results
from fnirsatlas.logging import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


SETTING_DEFAULT_STRATEGY = {"cross": "nested", "within": "block"}


def create_wilcoxon_table(comparisons, alpha=0.05, metric="cross_acc"):
    """Create pandas DataFrame for Wilcoxon test data."""
    if not comparisons:
        return None
    rows = []
    for comparison_key, task_results in comparisons.items():
        # Extract model names from comparison key
        if "_vs_" in comparison_key:
            model1, model2 = comparison_key.split("_vs_")
        else:
            model1, model2 = "Model1", "Model2"

        # Calculate differences and prepare for ranking
        differences = []
        task_info = []

        for task, results in task_results.items():
            # Format task name
            parts = task.split("__")
            if len(parts) >= 2:
                dataset_id = "__".join(parts[:-1])
                clf_task_desc = parts[-1]
                dataset = format_dataset_name(dataset_id)
            else:
                dataset = task
                clf_task_desc = ""

            # Truncate task description to fit in 30 characters
            if len(clf_task_desc) > 28:
                clf_task_desc = clf_task_desc[:25] + "..."

            diff = results["mean_diff"]
            differences.append(abs(diff))
            task_info.append((dataset, clf_task_desc, diff))

        # Calculate ranks (scipy.stats.rankdata gives 1-based ranks)
        ranks = rankdata(differences)

        # Create list of rows with rank info for sorting
        for i, (dataset, clf_task_desc, diff) in enumerate(task_info):
            rank = int(ranks[i])
            signed_rank = rank if diff > 0 else -rank
            signed_rank_str = f"+{rank}" if diff > 0 else f"-{rank}"

            rows.append(
                {
                    "Comparison": f"{model1} vs {model2}",
                    "Dataset": dataset,
                    "Task": clf_task_desc,
                    "Diff": diff,
                    "Signed Rank": signed_rank,
                    "Signed Rank Str": signed_rank_str,
                }
            )

    df = pd.DataFrame(rows)

    # Sort within each comparison group by signed rank (descending)
    df = df.sort_values(["Comparison", "Signed Rank"], ascending=[True, False])

    return df


def format_wilcoxon_table(
    comparisons,
    alpha=0.05,
    metric="cross_acc",
    output_format="console",
    output_path=None,
):
    """Format and print Wilcoxon test data summary."""
    df = create_wilcoxon_table(comparisons, alpha, metric)

    if df is None or df.empty:
        logger.info("No comparisons to display")
        return

    if output_format == "latex":
        format_wilcoxon_table_latex(df, output_path)
    else:
        format_wilcoxon_table_console(df)


def format_wilcoxon_table_console(df):
    """Format Wilcoxon table for console output."""
    # Group by comparison and print each table
    for comparison_key, group in df.groupby("Comparison"):
        print(f"\n{comparison_key}")
        print("=" * 100)
        print(f"{'Dataset':<16} {'Task':<30} {'Diff':>12} {'Signed Rank':>12}")
        print("-" * 100)

        for _, row in group.iterrows():
            print(
                f"{row['Dataset']:<16} {row['Task']:<30} {row['Diff']:>12.1f} {row['Signed Rank Str']:>12}"
            )


def format_wilcoxon_table_latex(df, output_path=None):
    """Format Wilcoxon table as LaTeX."""
    # Pivot the data to get the desired format
    pivot_df = df.pivot_table(
        index=["Dataset", "Task"],
        columns="Comparison",
        values="Signed Rank Str",
        aggfunc="first",
    ).reset_index()

    # Set multi-index for proper LaTeX formatting
    pivot_df = pivot_df.set_index(["Dataset", "Task"])

    # Generate LaTeX output
    latex_output = pivot_df.style.to_latex()
    latex_output = latex_output.replace(r"[c]", "[t]")

    if output_path:
        Path(output_path).write_text(latex_output)
        logger.info(f"LaTeX table saved to {output_path}")
    else:
        print(latex_output)


def perform_wilcoxon_test(comparisons, alpha=0.05, metric="cross_acc"):
    """Perform Wilcoxon test on task means for each comparison."""
    print(f"\nWilcoxon test on task means ({metric}):")

    for comparison_key, task_results in comparisons.items():
        # Extract model names from comparison key
        if "_vs_" in comparison_key:
            model1, model2 = comparison_key.split("_vs_")
        else:
            model1, model2 = "Model1", "Model2"

        # Extract means for Wilcoxon test
        model1_means = [results["model1_mean"] for results in task_results.values()]
        model2_means = [results["model2_mean"] for results in task_results.values()]

        # Perform Wilcoxon test
        w_stat, p_value = wilcoxon(model1_means, model2_means, alternative="two-sided")

        # Determine significance level
        sig = (
            "***"
            if p_value < alpha / 10
            else "**"
            if p_value < alpha / 5
            else "*"
            if p_value < alpha
            else ""
        )

        print(
            f"{model1} vs {model2}: W={w_stat:.1f}, p={p_value:.3f} {sig} n={len(model1_means)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Perform Wilcoxon test on model comparison results."
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        required=True,
        help="Models to compare (e.g., lda dummy eegnet)",
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
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cross_acc",
        help="Metric to analyze (default: cross_acc)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(5)),
        help="Random seeds used for results (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["console", "latex"],
        default="console",
        help="Output format (default: console)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for LaTeX table (optional, prints to stdout if not specified)",
    )
    args = parser.parse_args()

    assert len(args.models) == 2, "Provide 2 models for comparison"
    setting = Results._infer_classification_settings(args.metric)
    search_strategy = SETTING_DEFAULT_STRATEGY.get(setting, "none")

    # Get datasets to compare
    datasets = get_datasets_from_args(args.datasets, args.suite)
    if not datasets:
        logger.error("No datasets specified or found")
        return

    print("\n" + "=" * 100)
    print(f"Wilcoxon test on {args.metric} task means (α = {args.alpha})")
    print("=" * 100 + "\n")

    # Load results using the Results class
    results_dicts = Results.gather_results(
        models=args.models,
        datasets=datasets,
        metric=args.metric,
        seeds=args.seeds,
        search_strategy=search_strategy,
        classification_setting=setting,
    )

    if not results_dicts:
        logger.error("No results found for any model")
        return

    # Generate comparisons
    comparisons = compare_models(results_dicts, metric=args.metric, alpha=args.alpha)

    # Filter comparisons for overlapping partitions
    for task_results in comparisons.values():
        dataset_ids = [k.rsplit("__", 1)[0] for k in task_results]
        if "shin2019__ma" in dataset_ids and "shin2019__mi" in dataset_ids:
            # remove shin2019__mi entries
            shin2019_mi_keys = [k for k in task_results if k.startswith("shin2019__mi")]
            for key in shin2019_mi_keys:
                task_results.pop(key, None)

    # Filter comparisons for overlapping tasks
    filtered_comparisons = {}
    for comparison_key, task_results in comparisons.items():
        non_overlapping_task_ids = get_non_overlapping_clf_tasks(task_results.keys())
        if len(non_overlapping_task_ids) != len(task_results):
            filtered = set(task_results.keys()) - set(non_overlapping_task_ids)
            print(
                f"Filtering out the following tasks due to (possibly) overlapping data with other tasks:"
            )
            for task in filtered:
                print(f"  - {task}")
        filtered_comparisons[comparison_key] = {
            task: results
            for task, results in task_results.items()
            if task in non_overlapping_task_ids
        }

    # Print summary table
    format_wilcoxon_table(
        filtered_comparisons,
        alpha=args.alpha,
        metric=args.metric,
        output_format=args.format,
        output_path=args.output,
    )

    # Perform Wilcoxon test
    perform_wilcoxon_test(filtered_comparisons, alpha=args.alpha, metric=args.metric)


if __name__ == "__main__":
    main()
