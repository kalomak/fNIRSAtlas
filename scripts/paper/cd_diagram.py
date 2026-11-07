#!/usr/bin/env python3
"""Create an aeon critical difference diagram for the paper models.

Example:
    uv run --extra aeon python scripts/paper/cd_diagram.py --metric cross_acc
"""

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fnirsatlas.compare import extract_model_metrics
from fnirsatlas.config import get_paper_models, get_results_dir
from fnirsatlas.format import format_paper_model_name
from fnirsatlas.results import Results
from fnirsatlas.utils import filter_independent_task_ids, get_datasets_from_args


SETTING_DEFAULT_STRATEGY = {"cross": "nested", "within": "block"}
MODELS = get_paper_models()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create an aeon critical difference diagram for paper models"
    )
    parser.add_argument("-d", "--datasets", nargs="+")
    parser.add_argument("-s", "--suite", type=str)
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metric: cross_acc, cross_kappa, within_acc, within_kappa (default: {setting}_acc)",
    )
    parser.add_argument(
        "-c",
        "--classification_setting",
        type=str,
        default=None,
        help="Classification setting: cross or within (default: inferred from metric)",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help=f"Models to compare (default: {' '.join(MODELS)})",
    )
    parser.add_argument(
        "--test",
        choices=["nemenyi", "wilcoxon"],
        default="nemenyi",
        help="Clique-forming test used by aeon (default: nemenyi)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: results/paper/cd_diagram_<metric>.png)",
    )
    return parser.parse_args()


def build_score_matrix(results_dicts, metric: str) -> tuple[list[str], np.ndarray, list[str]]:
    """Build a block-by-model score matrix from result dictionaries."""
    model_names = [
        results.get("model_name", f"model_{i}")
        for i, results in enumerate(results_dicts)
    ]
    all_metrics = {
        model_name: extract_model_metrics(results, metric)[0]
        for model_name, results in zip(model_names, results_dicts)
    }
    common_task_ids = set.intersection(*[set(metrics) for metrics in all_metrics.values()])
    if not common_task_ids:
        raise ValueError("No common tasks found across models")
    task_ids, _ = filter_independent_task_ids(common_task_ids)
    if len(task_ids) < 2:
        raise ValueError("Need at least 2 independent tasks for a CD diagram")
    score_matrix = np.array(
        [
            [np.mean(all_metrics[model_name][task_id]) * 100 for model_name in model_names]
            for task_id in task_ids
        ]
    )
    return model_names, score_matrix, task_ids


def load_score_matrix(
    setting: str, metric: str, seeds, models=MODELS, datasets=None
) -> tuple[list[str], np.ndarray, list[str]]:
    """Load paper results for a setting and build the task-by-model score matrix."""
    results_dicts = Results.gather_results(
        models=models,
        datasets=datasets or get_datasets_from_args(None, None),
        metric=metric,
        seeds=list(seeds),
        search_strategy=SETTING_DEFAULT_STRATEGY.get(setting, "none"),
        classification_setting=setting,
    )
    if len(results_dicts) < 3:
        raise ValueError("Need results for at least 3 models")
    return build_score_matrix(results_dicts, metric)


def plot_cd(score_matrix: np.ndarray, model_names: list[str], alpha=0.05, test="nemenyi"):
    """Plot an aeon CD diagram with paper model names. Returns ``(fig, ax)``."""
    try:
        from aeon.visualisation import plot_critical_difference
    except ImportError as exc:
        raise ImportError("Install aeon or run with `uv run --extra aeon`.") from exc
    return plot_critical_difference(
        score_matrix,
        [format_paper_model_name(name) for name in model_names],
        alpha=alpha,
        test=test,
        width=5,
        textspace=1,
    )


def cd_image(setting: str, seeds) -> np.ndarray:
    """Render the paper-model CD diagram of ``{setting}_acc`` to an RGBA image array."""
    model_names, score_matrix, _ = load_score_matrix(setting, f"{setting}_acc", seeds)
    fig, _ = plot_cd(score_matrix, model_names)
    buf = BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def save_side_by_side(images: dict[str, np.ndarray], output_path: Path) -> None:
    """Save labelled CD diagram images side by side, separated by a thin line."""
    height = max(img.shape[0] for img in images.values())
    widths = [img.shape[1] for img in images.values()]
    fig, axes = plt.subplots(
        1,
        len(images),
        figsize=(sum(widths) / height * 2, 2),
        gridspec_kw={"width_ratios": widths, "wspace": 0},
    )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for i, (ax, (label, img)) in enumerate(zip(axes, images.items())):
        ax.imshow(img)
        ax.axis("off")
        ax.text(0, 1, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        if i > 0:
            ax.axvline(0, color="0.5", lw=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def default_output_path(metric: str) -> Path:
    """Return the default figure path."""
    return get_results_dir() / "paper" / f"cd_diagram_{metric}.png"


def main() -> int:
    """Load paper results and save an aeon CD diagram."""
    args = parse_args()
    setting = args.classification_setting or Results._infer_classification_settings(
        args.metric or "cross_acc"
    )
    metric = args.metric or f"{setting}_acc"
    datasets = get_datasets_from_args(args.datasets, args.suite)
    model_names, score_matrix, task_ids = load_score_matrix(
        setting, metric, args.seeds, args.models, datasets
    )
    fig, ax = plot_cd(score_matrix, model_names, args.alpha, args.test)
    ax.set_title(f"Paper models CD diagram ({metric}, n={len(task_ids)} tasks)")

    output_path = args.output or default_output_path(metric)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
