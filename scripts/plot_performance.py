#!/usr/bin/env python3
"""
Script for creating bar plots of classification performance.

Usage:
    python scripts/plot_performance.py -m lda eegnet --metric cross_acc --output './../paper/nirsatlas/tex/results/comparison_bar_plot.png'

"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from fnirsatlas.parsers import SUITE_MAP
from fnirsatlas.results import Results
from fnirsatlas.utils import get_datasets_from_args, order_tasks_by_suite
from fnirsatlas.format import get_task_labels
from fnirsatlas.logging import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

COLORBLIND_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


def get_plot_labels(task_keys, task_key_to_label, dataset_only=False):
    labels = []
    for task_key in task_keys:
        label = task_key_to_label.get(task_key, task_key)
        if dataset_only:
            labels.append(label.split(" (")[0])
            continue
        parts = task_key.split("__")
        if len(parts) >= 2:
            task_desc = parts[-1]
            dataset_name = label.split(" (")[0]
            plot_label = (
                f"{dataset_name}\n({task_desc[:20]})"
                if len(task_desc) > 20
                else f"{dataset_name}\n({task_desc})"
            )
        else:
            plot_label = task_key
        labels.append(plot_label)
    return labels


def get_colors(n_colors, palette_name):
    if palette_name == "colorblind":
        return COLORBLIND_PALETTE[:n_colors]
    return plt.cm.tab10(np.linspace(0, 1, n_colors))


def create_bar_plot(args, output_path=None):
    """Create a bar plot of performance across datasets for one or more models."""
    datasets = get_datasets_from_args(args.datasets, args.suite)
    models = args.model if isinstance(args.model, list) else [args.model]

    # Collect all results across datasets and models
    all_results = {}
    all_task_labels = []

    for model in models:
        all_results[model] = {}

        for dataset in datasets:
            try:
                results = Results(
                    dataset=dataset,
                    metric=args.metric,
                    model=model,
                    seeds=args.seeds,
                )
                results.load()
                stats = results.compute_stats(return_ci=True)

                # Collect task labels from all datasets (only once per dataset)
                if model == models[0]:
                    dataset_task_labels = get_task_labels(dataset, results.task_dict)
                    # Check if we already have labels for this dataset
                    existing_dataset_keys = {
                        key.split("__")[0] for key, _ in all_task_labels
                    }
                    if dataset not in existing_dataset_keys:
                        all_task_labels.extend(dataset_task_labels)

                # Store stats with percentage conversion
                for task, stat in stats.items():
                    full_key = f"{dataset}__{task}"
                    all_results[model][full_key] = {
                        "mean": stat["mean"] * 100,
                        "ci_lower": stat["ci_lower"] * 100,
                        "ci_upper": stat["ci_upper"] * 100,
                    }
            except FileNotFoundError:
                logger.warning(f"No results found for {dataset} {model} {args.metric}")
                continue

    if not all_task_labels:
        logger.error(f"No {args.metric} data found")
        return

    # Order task labels by suite
    task_keys = [task_key for task_key, _ in all_task_labels]
    ordered_task_keys, suite_to_tasks = order_tasks_by_suite(task_keys)

    # Create mapping from task_key to label
    task_key_to_label = dict(all_task_labels)

    # Prepare data for plotting
    model_data = {
        model: {"means": [], "ci_lower": [], "ci_upper": []} for model in models
    }

    for task_key in ordered_task_keys:
        for model in models:
            if task_key in all_results[model]:
                model_data[model]["means"].append(all_results[model][task_key]["mean"])
                model_data[model]["ci_lower"].append(
                    all_results[model][task_key]["ci_lower"]
                )
                model_data[model]["ci_upper"].append(
                    all_results[model][task_key]["ci_upper"]
                )
            else:
                model_data[model]["means"].append(0)
                model_data[model]["ci_lower"].append(0)
                model_data[model]["ci_upper"].append(0)

    # Calculate suite boundaries
    suite_boundaries = []
    current_pos = 0
    for suite in list(SUITE_MAP.keys()) + [None]:
        if suite_to_tasks[suite]:
            suite_boundaries.append(current_pos)
            current_pos += len(suite_to_tasks[suite])

    labels = get_plot_labels(
        ordered_task_keys,
        task_key_to_label,
        dataset_only=args.dataset_only_labels,
    )

    figsize = tuple(args.figsize) if args.figsize else (max(12, len(labels) * 0.75), 8)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    width = 0.8 / len(models)

    colors = get_colors(len(models), args.palette)
    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        means = np.array(model_data[model]["means"])
        ci_lower = np.array(model_data[model]["ci_lower"])
        ci_upper = np.array(model_data[model]["ci_upper"])

        # Calculate error bars as differences from mean
        lower_errors = means - ci_lower
        upper_errors = ci_upper - means

        bars = ax.bar(
            x + offset,
            means,
            width,
            label=model.upper(),
            yerr=[lower_errors, upper_errors],
            capsize=5,
            alpha=0.8,
            color=colors[i],
        )

    ax.set_xlabel("Dataset" if args.dataset_only_labels else "Dataset (Task)")
    ax.set_ylabel(f"{args.metric.replace('_', ' ').title()} (%)")
    if not args.no_title:
        title = (
            f"Performance: {args.metric.replace('_', ' ').title()} (95% Bootstrap CI)"
        )
        if len(models) > 1:
            title = f"Model Comparison: {title}"
        ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=args.font_size)
    ax.tick_params(axis="y", labelsize=args.font_size)
    ax.xaxis.label.set_size(args.font_size)
    ax.yaxis.label.set_size(args.font_size)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.legend(fontsize=args.font_size)
    ax.grid(True, alpha=0.3, axis="y")
    for boundary in suite_boundaries[1:]:
        ax.axvline(
            x=boundary - 0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5
        )
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.28)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Create bar plot of classification performance"
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        help="Specific datasets to plot. If not provided, all available datasets will be plotted.",
    )
    parser.add_argument(
        "-s",
        "--suite",
        type=str,
        help="Suite name to plot (e.g., motor, MA, MI, nback, Emotion, MISC)",
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        default=["lda"],
        help="Model(s) to plot (default: lda). Can specify multiple models.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds used for results (default: 42)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cross_acc",
        help="Metric to plot (default: cross_acc). Classification setting is inferred from metric name.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for the plot (optional, shows plot if not specified)",
    )
    parser.add_argument(
        "--palette",
        choices=["tab10", "colorblind"],
        default="tab10",
        help="Color palette for model bars (default: tab10)",
    )
    parser.add_argument(
        "--dataset-only-labels",
        action="store_true",
        help="Use dataset names only on the x-axis",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the plot title",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=10,
        help="Base font size for axis ticks and legend (default: 10)",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches, e.g. --figsize 18 8",
    )
    args = parser.parse_args()
    create_bar_plot(args, args.output)


if __name__ == "__main__":
    main()
