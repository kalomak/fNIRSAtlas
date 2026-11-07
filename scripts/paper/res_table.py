#!/usr/bin/env python3
"""
Display one metric across all models for paper comparison.

Usage:
    python scripts/paper/res_table.py -d luke2021
    python scripts/paper/res_table.py -s motor --metric cross_acc --seeds 0 1 2
    python scripts/paper/res_table.py -c within --latex --columns dataset classes
"""

import argparse
from fnirsatlas.config import get_paper_models
from fnirsatlas.format import LATEX_INFO_COLUMNS, format_model_comparison_table
from fnirsatlas.utils import get_datasets_from_args
from fnirsatlas.results import Results

MODELS = get_paper_models()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare all models on a single metric"
    )
    parser.add_argument("-d", "--datasets", nargs="+")
    parser.add_argument("-s", "--suite", type=str)
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metric: cross_acc, cross_kappa, within_acc, within_kappa (default: {setting}_acc)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument(
        "-c",
        "--classification_setting",
        type=str,
        default=None,
        help="Classification setting: cross or within (default: inferred from metric)",
    )
    parser.add_argument("--search_strategy", type=str, default=None)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument(
        "--columns",
        nargs="+",
        choices=list(LATEX_INFO_COLUMNS),
        default=list(LATEX_INFO_COLUMNS),
        help="Info columns before the model columns in LaTeX rows (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help=f"Models to compare (default: {' '.join(MODELS)})",
    )
    return parser.parse_args()


SETTING_DEFAULT_STRATEGY = {"cross": "nested", "within": "block"}


def main():
    args = parse_arguments()
    setting = args.classification_setting or Results._infer_classification_settings(
        args.metric or "cross_acc"
    )
    metric = args.metric or f"{setting}_acc"
    search_strategy = args.search_strategy or SETTING_DEFAULT_STRATEGY.get(
        setting, "none"
    )
    datasets = get_datasets_from_args(args.datasets, args.suite)
    results_dicts = Results.gather_results(
        args.models,
        datasets,
        metric,
        args.seeds,
        search_strategy,
        setting,
    )
    if results_dicts:
        format_model_comparison_table(
            results_dicts, metric, latex=args.latex, columns=args.columns
        )


if __name__ == "__main__":
    main()
