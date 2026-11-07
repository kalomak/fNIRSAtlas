"""LaTeX table of within-subject accuracy per dataset under shuffled and blocked CV.

Shuffled results come from the main results cache. Blocked results come from the
cache of the `fnirsatlas/configs/blocked_cv_gap0.yaml` overlay (full within-subject grid run).

Examples:
    uv run python scripts/checks/blocked_cv_table.py
    uv run python scripts/checks/blocked_cv_table.py --output results/blocked_cv/dataset_table.tex
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fnirsatlas.config import get_paper_models, get_results_cache_dir
from fnirsatlas.dataset_catalog import get_available_datasets
from fnirsatlas.format import format_dataset_name
from fnirsatlas.logging import setup_logging
from fnirsatlas.utils import CONFIG_DIR, metric_to_subject_scores

setup_logging()
logger = logging.getLogger(__name__)

BLOCKED_CONFIG = str(CONFIG_DIR / "blocked_cv_gap0.yaml")
# Class balance differs strongly between recording sessions (one class per session, see parsers).
SESSION_DEPENDENT = {"luke2024b", "cockx2023"}
# SD across subjects with ddof=0, as in fnirsatlas.format._extract_metric.
STATS = {"mean": np.mean, "sd": np.std}


def subject_accuracy(cache_dir, model, dataset, seed):
    """Return within-subject accuracy per subject, averaged over tasks and folds."""
    path = cache_dir / "within" / "block" / model / dataset / str(seed) / "results.json"
    if not path.exists():
        raise FileNotFoundError(path)
    tasks = json.loads(path.read_text())[dataset]
    return np.mean([metric_to_subject_scores(t["within_acc"]) for t in tasks.values()], axis=0)


def dataset_subject_accuracy(cache_dir, models, dataset, seeds):
    """Return accuracy per subject × 100, averaged over models and seeds."""
    return 100 * np.mean(
        [subject_accuracy(cache_dir, m, dataset, s) for m in models for s in seeds], axis=0
    )


def dataset_row(dataset, models, seeds):
    """Return mean and SD across subjects of shuffled, blocked and their paired difference."""
    shuffled = dataset_subject_accuracy(get_results_cache_dir(), models, dataset, seeds)
    blocked = dataset_subject_accuracy(
        get_results_cache_dir(BLOCKED_CONFIG), models, dataset, seeds
    )
    columns = {"Shuffled": shuffled, "Blocked": blocked, "Difference": blocked - shuffled}
    return {(c, stat): f(v) for c, v in columns.items() for stat, f in STATS.items()}


def build_table(seeds):
    """Return per-dataset mean and SD, sorted by mean difference, with a Mean row of means."""
    models, datasets = get_paper_models(), get_available_datasets()
    table = pd.DataFrame({d: dataset_row(d, models, seeds) for d in datasets}).T
    table = table.sort_values(("Difference", "mean"))
    table.index = [
        format_dataset_name(d) + ("$^\\dagger$" if d in SESSION_DEPENDENT else "") for d in table.index
    ]
    table.loc["Mean"] = table.xs("mean", axis=1, level=1, drop_level=False).mean()
    return table


def format_value(value):
    """Format one decimal, with a math minus sign for negative values."""
    return f"${value:.1f}$" if value < 0 else f"{value:.1f}"


def format_cell(mean, sd):
    """Format mean ± SD, or the mean alone when SD is missing."""
    return format_value(mean) if np.isnan(sd) else f"{format_value(mean)} $\\pm$ {sd:.1f}"


def to_latex(table):
    """Return the table as LaTeX with a rule above the Mean row."""
    cells = pd.DataFrame({
        c: [format_cell(m, s) for m, s in zip(table[(c, "mean")], table[(c, "sd")])]
        for c in table.columns.get_level_values(0).unique()
    }, index=pd.Index(table.index, name="Dataset"))
    latex = cells.reset_index().style.hide(axis="index").to_latex(
        hrules=True, column_format="lrrr"
    )
    return latex.replace("\nMean &", "\n\\midrule\nMean &")


def main():
    """Print or save the table."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(5)), help="Seeds (default: 0-4)"
    )
    parser.add_argument("--output", type=Path, help="Output .tex path (default: print)")
    args = parser.parse_args()
    latex = to_latex(build_table(args.seeds))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(latex)
        logger.info(f"LaTeX table saved to {args.output}")
    else:
        print(latex)


if __name__ == "__main__":
    main()
