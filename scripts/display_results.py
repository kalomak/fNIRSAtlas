#!/usr/bin/env python3
"""
Script to display classification results from the pipeline.

This script collects results from processed datasets and formats them
into a benchmark table.

Usage:
    python scripts/display_results.py -d luke2021 -m lda --seeds 0
    python scripts/display_results.py -d luke2021 -m lda fnirsnet23 --metric cross_acc
"""

import argparse
from fnirsatlas.format import format_model_comparison_table
from fnirsatlas.utils import get_datasets_from_args
from fnirsatlas.results import Results


def display_results(datasets, metric, models, seeds, setting, search_strategy):
    """Display classification results."""
    results_dicts = Results.gather_results(
        models=models,
        datasets=datasets,
        metric=metric,
        seeds=seeds,
        search_strategy=search_strategy,
        classification_setting=setting,
    )
    if results_dicts:
        format_model_comparison_table(results_dicts, metric)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Display classification results for specified datasets"
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        help="Specific datasets to display results for",
    )
    parser.add_argument(
        "-s",
        "--suite",
        type=str,
        help="Suite name to process (e.g., motor, MA, MI, nback, Emotion, MISC)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cross_acc",
        help="Metric to display: cross_acc, cross_kappa, within_acc, within_kappa (default: cross_acc)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds for cross-validation (default: 42)",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=["lda"],
        help="Model(s) used for classification (default: lda)",
    )
    parser.add_argument(
        "-c",
        "--classification_setting",
        type=str,
        default="cross",
        help="Classification setting: cross or within",
    )
    parser.add_argument(
        "--search-strategy",
        type=str,
        default="none",
        choices=["none", "nested", "single", "block"],
        help="Parameter search strategy",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    datasets = get_datasets_from_args(args.datasets, args.suite)
    display_results(
        datasets,
        args.metric,
        args.models,
        args.seeds,
        args.classification_setting,
        args.search_strategy,
    )


if __name__ == "__main__":
    main()
