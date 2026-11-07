"""
Script for comparing multiple model results using statistical testing.

Usage:
python scripts/compare_models.py --models lda eegnet --output './../paper/nirsatlas/tex/results/comparison_table.tex'
python scripts/compare_models.py --models lda dummy --datasets shin2019 --metric cross_acc --alpha 0.05
python scripts/compare_models.py --models lda eegnet --suite motor --metric within_acc --seeds 1 2 3
"""

import logging
import argparse
from fnirsatlas.compare import compare_models, format_comparison_table
from fnirsatlas.results import Results
from fnirsatlas.utils import get_datasets_from_args
from fnirsatlas.logging import setup_logging


setup_logging()
logger = logging.getLogger(__name__)

SETTING_DEFAULT_STRATEGY = {"cross": "nested", "within": "block"}
STRATEGY_NONE_MODELS = {"dummy"}


def get_search_strategy(model, setting):
    """Get search strategy for a model: 'none' for dummy, setting default otherwise."""
    if model in STRATEGY_NONE_MODELS:
        return "none"
    return SETTING_DEFAULT_STRATEGY.get(setting, "none")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple model results.")
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
        "-c",
        "--classification-setting",
        type=str,
        help="Classification setting: cross or within (default: inferred from metric)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cross_acc",
        help="Metric to compare (default: cross_acc)",
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
        "--output",
        type=str,
        help="Output path for LaTeX table (optional, prints to stdout if not specified)",
    )
    parser.add_argument(
        "--one-sided",
        action="store_true",
        help="One-sided test: model1 > model2",
    )
    args = parser.parse_args()

    assert len(args.models) == 2, "Provide 2 models for comparison"

    setting = args.classification_setting or Results._infer_classification_settings(
        args.metric
    )
    datasets = get_datasets_from_args(args.datasets, args.suite)
    if not datasets:
        logger.error("No datasets specified or found")
        return

    # Load results per model (each may have a different search strategy)
    results_dicts = []
    for model in args.models:
        strategy = get_search_strategy(model, setting)
        model_results = Results.gather_results(
            models=[model],
            datasets=datasets,
            metric=args.metric,
            seeds=args.seeds,
            search_strategy=strategy,
            classification_setting=setting,
        )
        results_dicts.extend(model_results)

    if len(results_dicts) < 2:
        logger.error("Need results for both models to compare")
        return

    comparisons = compare_models(
        results_dicts, metric=args.metric, alpha=args.alpha, one_sided=args.one_sided
    )

    for comparison_key, task_results in comparisons.items():
        format_comparison_table(
            comparison_key,
            task_results,
            alpha=args.alpha,
            metric=args.metric,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
