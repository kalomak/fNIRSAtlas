#!/usr/bin/env python3
"""
Short classification script for all datasets using new classification module.

Usage:
    python scripts/classify_dataset.py -d luke2021 --model lda --seed 0
"""

import logging
import argparse
import json
from fnirsatlas.classification import resolve_search_strategy, run_benchmark
from fnirsatlas.utils import dict_to_json, build_output_path
from fnirsatlas.logging import setup_logging
from fnirsatlas.config import get_param_grid


setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Classify all processed datasets.")
    parser.add_argument(
        "-d", "--datasets", nargs="+", help="Specific datasets to classify (optional)"
    )
    parser.add_argument(
        "-s",
        "--suite",
        type=str,
        help="Suite name to classify (e.g. motor, MI, MA, etc.)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model name (e.g. lda, logreg, svm, rf, knn)",
        default="lda",
    )
    parser.add_argument(
        "-j",
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs for parallel execution",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for cross-validation (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for saved results (default: results_{suite}_{model}.json)",
    )
    parser.add_argument(
        "-c",
        "--classification_setting",
        type=str,
        help="Classification type: cross or within",
        default="cross",
    )
    parser.add_argument(
        "--search-strategy",
        type=str,
        choices=["none", "nested", "single", "block"],
        default=None,
        help="Parameter search strategy (loads param grid from config automatically)",
    )
    parser.add_argument(
        "--no_table",
        action="store_true",
        help="Do not show the benchmark table",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "--split-ix",
        type=int,
        default=None,
        help="Run only a specific split index (0-based). If given, only runs splits[split_ix].",
    )
    args = parser.parse_args()
    param_grid = None
    if args.search_strategy and args.search_strategy != "none":
        param_grid = get_param_grid(args.model)
        if param_grid:
            logger.info(f"Using parameter grid for {args.model}: {param_grid}")
        else:
            logger.warning(f"No parameter grid found for model {args.model}")

    args.search_strategy = resolve_search_strategy(
        args.classification_setting,
        args.search_strategy,
        param_grid,
    )
    output_path = build_output_path(args)
    if output_path.exists() and not args.override:
        logger.info(
            f"Output file already exists: {output_path} (use --override to overwrite)"
        )
        return 0

    results = run_benchmark(
        suite=args.suite,
        datasets=args.datasets,
        n_jobs=args.n_jobs,
        seed=args.seed,
        model_name=args.model,
        classification_setting=args.classification_setting,
        search_strategy=args.search_strategy,
        param_grid=param_grid,
        show_table=not args.no_table,
        split_ix=args.split_ix,
    )

    # Save results
    if not results:
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_results = dict_to_json(results)
    json_results["model_name"] = args.model
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
