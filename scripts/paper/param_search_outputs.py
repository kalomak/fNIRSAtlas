#!/usr/bin/env python3
"""Save parameter search heatmaps and tables for all paper models.

Example:
    uv run python scripts/paper/param_search_outputs.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fnirsatlas.config import get_paper_models, get_results_dir
from fnirsatlas.format import PAPER_MODEL_LABELS
from fnirsatlas.utils import format_seeds, get_datasets_from_args

from scripts.display_search_results import display_search_results


MODELS = get_paper_models()
SETTING_DEFAULT_STRATEGY = {"cross": "nested", "within": "block"}
SETTING_LABELS = {"cross": "cross-subject", "within": "within-subject"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Save parameter search heatmaps and tables for paper models"
    )
    parser.add_argument("-d", "--datasets", nargs="+")
    parser.add_argument("-s", "--suite", type=str)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument(
        "--metric",
        type=str,
        default="kappa",
        help="Search metric to summarize (default: kappa)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help=f"Models to include (default: {' '.join(MODELS)})",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=get_results_dir() / "paper/param_search/figures",
        help="Directory for saved heatmaps",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=get_results_dir() / "paper/param_search/tables",
        help="Directory for saved tables",
    )
    parser.add_argument("--sort", action="store_true")
    return parser.parse_args()


def build_scope_tag(args: argparse.Namespace) -> str:
    """Build a filename-safe tag for the selected dataset scope."""
    if args.datasets:
        return "datasets_" + "-".join(args.datasets)
    if args.suite:
        return f"suite_{args.suite}"
    return "all_datasets"


def save_outputs_for_setting(
    datasets: list[str],
    models: list[str],
    seeds: list[int],
    setting: str,
    metric: str,
    scope_tag: str,
    figure_dir: Path,
    table_dir: Path,
    sort: bool,
) -> list[str]:
    """Save outputs for one classification setting and return missing models."""
    missing = []
    seed_tag = format_seeds(seeds)
    strategy = SETTING_DEFAULT_STRATEGY[setting]
    for model in models:
        stem = f"{scope_tag}_{setting}_{metric}_{model}_seeds_{seed_tag}"
        mean_scores = display_search_results(
            datasets=datasets,
            model=model,
            seeds=seeds,
            classification_setting=setting,
            search_strategy=strategy,
            metric=metric,
            sort=sort,
            lm=True,
            plot=True,
            heatmap_output=figure_dir / f"{stem}.png",
            lm_latex_output=table_dir / f"{stem}.tex",
            lm_caption=(
                f"Coefficient summary for the {PAPER_MODEL_LABELS[model]} "
                f"hyperparameter analysis ({SETTING_LABELS[setting]})."
            ),
            lm_label=f"tab:hparam_lm_{setting}_{model}",
            show_plot=False,
        )
        if mean_scores is None:
            missing.append(f"{setting}/{model}")
    return missing


def main() -> int:
    """Save parameter-search artifacts for cross and within paper runs."""
    args = parse_args()
    datasets = get_datasets_from_args(args.datasets, args.suite)
    scope_tag = build_scope_tag(args)
    missing = []
    for setting in ["cross", "within"]:
        missing.extend(
            save_outputs_for_setting(
                datasets=datasets,
                models=args.models,
                seeds=args.seeds,
                setting=setting,
                metric=args.metric,
                scope_tag=scope_tag,
                figure_dir=args.figure_dir,
                table_dir=args.table_dir,
                sort=args.sort,
            )
        )

    if missing:
        raise ValueError(f"Missing search results for: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
