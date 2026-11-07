#!/usr/bin/env python3
"""Plot subject-count learning-curve results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
import numpy as np
import pandas as pd
from scipy import stats

from fnirsatlas.config import get_paper_models, get_results_cache_dir, get_results_dir
from fnirsatlas.format import format_paper_model_name

METRIC_LABELS = {
    "kappa": r"Cohen's $\kappa$",
    "balanced_accuracy_adjusted": "Adjusted balanced accuracy",
    "accuracy": "Accuracy",
}
COLORS = dict(zip(get_paper_models(), plt.get_cmap("tab10").colors))


def load(root, metric):
    """Load per-run learning-curve rows from ``root/<dataset>/<model>/*.json``."""
    rows = []
    for path in root.glob("*/*/seed-*.json"):
        rows.extend(json.loads(path.read_text()))
    if not rows:
        raise ValueError(f"No results in {root}")
    frame = pd.DataFrame(rows)
    if metric not in frame or frame[metric].isna().any():
        raise ValueError(f"Some results in {root} have no {metric}")
    return frame


def per_dataset(frame, metric):
    """Average runs per dataset, model and size, as ``(dataset, model) x n_train``."""
    return frame.pivot_table(index=["dataset", "model"], columns="n_train",
                             values=metric, aggfunc="mean")


def summarize(values):
    """Mean and 95% t-interval half-width across datasets, as ``model x n_train``."""
    grouped = values.groupby(level="model")
    n = grouped.size()
    if n.nunique() != 1:
        raise ValueError(f"Models have different dataset counts: {n.to_dict()}")
    t = stats.t.ppf(0.975, n.iloc[0] - 1)
    return grouped.mean(), grouped.sem() * t


def model_order(wide):
    """Models sorted by their value at the largest training size, best first."""
    return wide[wide.columns.max()].sort_values(ascending=False).index


def style_size_axis(axis, sizes):
    """Log2 x-axis with one tick per training size and a light grid."""
    axis.set_xscale("log", base=2)
    axis.set_xticks(sizes, [str(size) for size in sizes])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.margins(x=0.03)
    axis.grid(which="both", color="0.9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlabel("Training subjects")


def save(fig, path):
    """Save and close a figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_summary(wide, metric, path):
    """Plot one line per model against log2 training subjects."""
    fig, axis = plt.subplots(figsize=(6, 6))
    models = model_order(wide)
    for rank, model in enumerate(models):
        # Better models draw on top, while the legend keeps best-first order below it.
        axis.plot(wide.columns, wide.loc[model], marker="D", markersize=3.5, linewidth=1.5,
                  color=COLORS[model], label=format_paper_model_name(model),
                  zorder=2 + (len(models) - rank) / len(models))
    style_size_axis(axis, wide.columns)
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_locator(MultipleLocator(0.05))
    axis.set_ylabel(f"Mean {METRIC_LABELS[metric]}")
    axis.legend(loc="upper left", framealpha=1, edgecolor="0.6")
    save(fig, path)


def plot_difference(difference, interval, metric, path):
    """Forest plot of each model's difference from the mean of models.

    One row per model, sorted by the largest training size, with one dot and
    error bar per training size shaded from light (smallest) to dark (largest).
    """
    sizes = difference.columns
    models = model_order(difference)
    rows = np.arange(len(models))[::-1]
    shades = plt.get_cmap("Blues")(np.linspace(0.35, 1, len(sizes)))
    offsets = np.linspace(0.3, -0.3, len(sizes))
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.axvline(0, color="0.6", linewidth=0.8)
    for size, shade, offset in zip(sizes, shades, offsets):
        axis.errorbar(difference.loc[models, size], rows + offset,
                      xerr=interval.loc[models, size], fmt="o", markersize=4,
                      linewidth=1, capsize=2, color=shade, label=str(size))
    for row in rows[:-1]:
        axis.axhline(row - 0.5, color="0.6", linewidth=0.8)
    axis.set_yticks(rows, [format_paper_model_name(model) for model in models])
    axis.spines[["top", "right"]].set_visible(False)
    axis.xaxis.set_major_locator(MultipleLocator(0.02))
    axis.xaxis.set_minor_locator(MultipleLocator(0.01))
    axis.grid(axis="x", which="both", color="0.9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_xlabel(f"{METRIC_LABELS[metric]} − mean of models")
    axis.legend(title="Training subjects", loc="upper left", framealpha=1, edgecolor="0.6")
    save(fig, path)


def write_table(mean, path):
    """Write mean ``x 100`` per model (rows, best first) and training size as LaTeX.

    Cells after the smallest size add the gain over the smallest size in
    parentheses. The best model per training size is bold.
    """
    table = mean.loc[model_order(mean)] * 100
    first = table.columns.min()
    best = table.round(1) == table.round(1).max()
    cells = table.map("{:.1f}".format).where(~best, "\\textbf{" + table.map("{:.1f}".format) + "}")
    gain = table.sub(table[first], axis=0).map("({:+.1f})".format)
    cells.loc[:, table.columns != first] += " " + gain.loc[:, table.columns != first]
    cells.index = [format_paper_model_name(model) for model in cells.index]
    cells.columns = pd.Index([str(size) for size in cells.columns], name="Training subjects")
    path.write_text(cells.style.to_latex(hrules=True, column_format="l" + "r" * len(table.columns)))


def plot(root, output_dir, metric):
    """Export learning curves of ``metric``.

    Writes ``{metric}.csv``, the mean over datasets as a plot (``{metric}.png``)
    and a LaTeX table (``{metric}_table.tex``), and each model's difference from
    the mean of models on the same dataset (``{metric}_difference.png``) with 95%
    t-intervals across datasets.
    """
    values = per_dataset(load(root, metric), metric)
    difference = values - values.groupby(level="dataset").transform("mean")
    mean, _ = summarize(values)
    difference_mean, interval = summarize(difference)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat({"mean": mean, "difference": difference_mean,
               "difference_ci95": interval}, axis=1).to_csv(output_dir / f"{metric}.csv")
    plot_summary(mean, metric, output_dir / f"{metric}.png")
    write_table(mean, output_dir / f"{metric}_table.tex")
    plot_difference(difference_mean, interval, metric,
                    output_dir / f"{metric}_difference.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=METRIC_LABELS, default="kappa")
    parser.add_argument(
        "--root", type=Path, default=get_results_cache_dir() / "learning_curve"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=get_results_dir() / "learning_curve"
    )
    args = parser.parse_args()
    plot(args.root, args.output_dir, args.metric)


if __name__ == "__main__":
    main()
