#!/usr/bin/env python3
"""
Script to display hyperparameter search results from classification.

Usage:
    python scripts/display_search_results.py -d bnci --model lda --seeds 0
"""

import argparse
import json
import re
import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import seaborn as sns
from pathlib import Path
from fnirsatlas.format import format_dataset_name
from fnirsatlas.utils import escape_latex, get_datasets_from_args
from fnirsatlas.logging import setup_logging
from fnirsatlas.model_param_counts import param_key
from fnirsatlas.search_results import (
    dataset_mean_scores,
    get_param_cols,
    load_search_data,
    normalize_param_value,
)
import statsmodels.formula.api as smf

setup_logging()
logger = logging.getLogger(__name__)
PARAM_COUNT_FILE = Path(__file__).resolve().parent / "paper" / "model_param_counts.json"
PARAM_LABELS = {
    "n_params": "#Params",
    "lr": "LR",
    "kernel_size": "Kernel Size",
    "kernel_size_1": "Kernel Size",
    "n_filters": "N Filters",
    "p_dropout": "Dropout",
    "hidden_size": "Hidden Size",
    "num_layers": "Num Layers",
    "activation": "Activation",
    "hidden_layer_sizes": "Hidden Size",
    "learning_rate_init": "Init LR",
    "n_windows": "Windows",
    "features": "Features",
    "F1": "F1",
    "F2": "F2",
    "D1": "D1",
    "n_groups": "N Groups",
    "n_kernels": "N Kernels",
    "depth": "Depth",
    "dim": "Dim",
    "dropout": "Dropout",
    "spatial_filters": "Spatial Filters",
    "temporal_filters": "Temporal Filters",
    "penalty": "Penalty",
    "C": "C",
    "n_estimators": "N Trees",
    "min_samples_leaf": "Min Leaf",
    "n_neighbors": "N Neighbors",
    "weights": "Weights",
}
LM_PARAM_LABELS = {"lr": "Learning Rate"}


def to_safe_str(value):
    """Convert values to formula-safe strings."""
    s = str(value)
    for ch in " ,()'\"-":
        s = s.replace(ch, "_")
    return s.strip("_")


def order_levels(values):
    """Sort levels numerically when all are numbers, else by formula-safe text."""
    values = list(values)
    normalized = [normalize_param_value(v) for v in values]
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in normalized):
        return [v for _, v in sorted(zip(normalized, values), key=lambda pair: pair[0])]
    return sorted(values, key=to_safe_str)


def prepare_lm_data(combined_data):
    """
    Prepare categorical parameter data for statsmodels formulas.

    Levels are ordered by `order_levels`, so the first (reference) level of a
    numeric parameter is its smallest value.
    """
    param_cols = [c for c in combined_data.columns if c.startswith("param_")]
    df = combined_data[param_cols + ["mean_score"]].copy()
    value_labels = {}
    for col in param_cols:
        levels = order_levels(df[col].drop_duplicates())
        value_labels[col] = {
            to_safe_str(v): format_param_value(normalize_param_value(v)) for v in levels
        }
        df[col] = pd.Categorical(
            df[col].map(to_safe_str), categories=list(value_labels[col])
        )
    safe = {c: c.replace(".", "_").replace("-", "_") for c in param_cols}
    df = df.rename(columns=safe)
    return df, param_cols, safe, value_labels


def fit_lm(combined_data):
    """Fit OLS with all params as categorical predictors."""
    df, _, safe, value_labels = prepare_lm_data(combined_data)
    formula = "mean_score ~ " + " + ".join(f"C({v})" for v in safe.values())
    result = smf.ols(formula, data=df).fit()
    print("\nLinear model summary:")
    print(result.summary())
    return result, safe, value_labels


def load_param_counts(model, settings):
    """
    Return reference parameter counts and storage bytes for each setting.

    Counts come from PARAM_COUNT_FILE, written by scripts/paper/model_param_counts.py.

    Parameters
    ----------
    model : str
    settings : pd.MultiIndex
        Parameter combinations, level names are `param_*` columns.

    Returns
    -------
    pd.DataFrame
        Columns `n_params` and `n_bytes`, indexed by `settings`.
    """
    count_data = json.loads(PARAM_COUNT_FILE.read_text())["models"][model]
    lookup = {param_key(s["params"]): (s["n_params"], s["n_bytes"]) for s in count_data}
    rows = [lookup[param_key(dict(zip(settings.names, values)))] for values in settings]
    return pd.DataFrame(rows, index=settings, columns=["n_params", "n_bytes"])


def format_param_value(value):
    """Format a parameter value for display, e.g. [20] -> 20, [mean, slope] -> mean+slope."""
    if isinstance(value, (list, tuple)):
        return "+".join(map(str, value))
    return str(value)


def format_count(value, unit=""):
    """Format counts with k/M suffix and 3 significant figures, e.g. 2.35MB."""
    for divisor, suffix in ((1e6, "M"), (1e3, "k")):
        if value >= divisor:
            return f"{value / divisor:.3g}{suffix}{unit}"
    return f"{value:.3g}{unit}"


def format_param_label(param_col):
    """Format parameter column names for display."""
    key = param_col.removeprefix("param_").replace("extractor__", "").replace(
        "clf__", ""
    )
    return PARAM_LABELS.get(key, key.replace("_", " ").title())


def parse_lm_term(term, safe_cols):
    """Return `(param_col, safe_value)` of a statsmodels categorical term."""
    safe_col, safe_value = re.fullmatch(r"C\((.+)\)\[T\.(.+)\]", term).groups()
    orig_col = next(col for col, safe in safe_cols.items() if safe == safe_col)
    return orig_col, safe_value


def format_lm_level(param_col, safe_value, value_labels):
    """Format one parameter level, e.g. `Features = mean+slope`."""
    key = param_col.removeprefix("param_").replace("extractor__", "").replace(
        "clf__", ""
    )
    label = LM_PARAM_LABELS.get(key, format_param_label(param_col))
    return f"{label} = {value_labels[param_col][safe_value]}"


def lm_reference_levels(terms, safe_cols, value_labels):
    """Return the formatted reference (dropped) level of each parameter."""
    used = {parse_lm_term(term, safe_cols) for term in terms if term != "Intercept"}
    references = []
    for col, levels in value_labels.items():
        (reference,) = [v for v in levels if (col, v) not in used]
        references.append(format_lm_level(col, reference, value_labels))
    return references


def build_lm_summary_table(result, safe_cols, value_labels):
    """
    Build a Term / Coef. table from a fitted linear model.

    Coefficients are kappa x100: the intercept is the mean of the reference
    setting, each other term the difference from its parameter's reference level.
    """
    terms = result.params.index
    references = lm_reference_levels(terms, safe_cols, value_labels)
    labels = [
        f"Intercept ({', '.join(references)})"
        if term == "Intercept"
        else format_lm_level(*parse_lm_term(term, safe_cols), value_labels)
        for term in terms
    ]
    return pd.DataFrame({"Term": labels, "Coef.": result.params.values * 100})


def format_lm_latex_table(summary_df, caption, label):
    """Format a Term / Coef. summary as a LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{escape_latex(caption)}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\begin{tabular}{l c}",
        r"\toprule",
        r"Term & Coef. \\",
        r"\midrule",
    ]
    for term, coef in summary_df.itertuples(index=False):
        # Adding 0.0 turns -0.0 into 0.0.
        lines.append(f"{escape_latex(term)} & {round(coef, 1) + 0.0:.1f} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def save_lm_latex_table(summary_df, output_path, caption, label):
    """Save a linear-model coefficient summary as LaTeX."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_lm_latex_table(summary_df, caption, label) + "\n")


def add_multiindex_labels(ax, columns, level_labels):
    """Add MultiIndex-style parameter labels above the heatmap."""
    n_cols = len(columns)
    axes = []
    for level in range(columns.nlevels):
        top_ax = ax.twiny()
        top_ax.set_xlim(ax.get_xlim())
        top_ax.xaxis.set_ticks_position("top")
        offset = 22 * (columns.nlevels - level - 1)
        top_ax.spines["top"].set_position(("outward", offset))
        for spine in ["bottom", "left", "right"]:
            top_ax.spines[spine].set_visible(False)

        ticks = []
        labels = []
        start = 0
        while start < n_cols:
            value = columns[start][level]
            end = start + 1
            while end < n_cols and columns[end][level] == value:
                end += 1
            ticks.append((start + end - 1) / 2 + 0.5)
            labels.append(format_param_value(value))
            start = end

        top_ax.set_xticks(ticks)
        top_ax.set_xticklabels(labels, rotation=0)
        top_ax.tick_params(axis="x", length=0, pad=2)
        text_transform = mtransforms.offset_copy(
            ax.transAxes,
            fig=ax.figure,
            x=14,
            y=offset + 8,
            units="points",
        )
        ax.text(
            -0.03,
            1.0,
            level_labels[level],
            transform=text_transform,
            ha="right",
            va="center",
            fontweight="bold",
        )
        axes.append(top_ax)

    return axes


def default_heatmap_path(model, classification_setting, metric):
    """Build the default heatmap output path."""
    from fnirsatlas.config import get_results_dir

    return get_results_dir() / f"param_heatmap_{classification_setting}_{metric}_{model}.png"


def add_param_count_level(columns, counts):
    """Prepend `#Params` over `(storage)`, e.g. `588k` over `(2.35MB)`, as the first column level."""
    return pd.MultiIndex.from_tuples(
        [
            (f"{format_count(n_params)}\n({format_count(n_bytes, 'B')})", *column)
            for column, (n_params, n_bytes) in zip(columns, counts.itertuples(index=False))
        ]
    )


def plot_heatmap(
    combined_data,
    model,
    param_cols,
    classification_setting,
    metric,
    sort=False,
    output_path=None,
    show=True,
):
    """Plot dataset x parameter-combination mean scores (x100) and #Params."""
    dataset_scores = dataset_mean_scores(combined_data, param_cols)
    dataset_scores["dataset"] = dataset_scores["dataset"].map(format_dataset_name)
    heatmap_data = dataset_scores.pivot(
        index="dataset", columns=param_cols, values="mean_score"
    ) * 100
    if sort:
        order = heatmap_data.mean(axis=0).sort_values(ascending=False).index
    else:
        order = pd.MultiIndex.from_frame(dataset_scores[param_cols].drop_duplicates())
    heatmap_data = heatmap_data.reindex(columns=order)
    heatmap_data.loc["Mean"] = heatmap_data.mean(axis=0)
    counts = load_param_counts(model, heatmap_data.columns)

    row_min = heatmap_data.min(axis=1)
    row_range = heatmap_data.max(axis=1) - row_min
    color_data = heatmap_data.sub(row_min, axis=0).div(row_range.replace(0, 1), axis=0)

    plt.figure(
        figsize=(
            max(8, len(color_data.columns) * 0.8),
            max(4, len(color_data.index) * 0.6),
        )
    )
    ax = sns.heatmap(
        color_data,
        annot=heatmap_data,
        fmt=".1f",
        cmap="RdYlGn",
        cbar=False,
    )
    # Separate the Mean row from the dataset rows.
    ax.axhline(len(color_data.index) - 1, color="black", linewidth=1.5)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    add_multiindex_labels(
        ax,
        add_param_count_level(color_data.columns, counts),
        ["#Params", *map(format_param_label, param_cols)],
    )
    plt.tight_layout()
    output_path = output_path or default_heatmap_path(
        model, classification_setting, metric
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def format_mean_scores_table(mean_scores, metric):
    """Format mean scores as a text table."""
    return (
        f"Hyperparameter search results ({metric}, averaged across datasets):\n"
        f"{mean_scores.to_string()}\n"
    )


def save_mean_scores_table(mean_scores, metric, output_path):
    """Save mean scores as CSV or text based on suffix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".csv":
        mean_scores.to_csv(output_path)
        return
    output_path.write_text(format_mean_scores_table(mean_scores, metric))


def display_search_results(
    datasets,
    model,
    seeds,
    classification_setting,
    search_strategy,
    metric="kappa",
    sort=False,
    lm=False,
    plot=False,
    heatmap_output=None,
    table_output=None,
    lm_latex_output=None,
    lm_caption=None,
    lm_label=None,
    show_plot=True,
):
    """Display hyperparameter search results as a table."""
    combined_data = load_search_data(
        datasets, model, seeds, classification_setting, search_strategy, metric
    )
    if combined_data is None:
        print("No search results found")
        return None

    param_cols = get_param_cols(combined_data)
    mean_scores = combined_data.groupby(param_cols)["mean_score"].mean().reset_index()
    mean_scores = mean_scores.set_index(param_cols).round(4)
    counts = load_param_counts(model, mean_scores.index)
    mean_scores["n_params"] = counts["n_params"].map(format_count)
    mean_scores["n_bytes"] = counts["n_bytes"].map(lambda n: format_count(n, "B"))
    if sort:
        mean_scores = mean_scores.sort_values("mean_score", ascending=False)

    print()
    print(format_mean_scores_table(mean_scores, metric), end="")
    if table_output:
        save_mean_scores_table(mean_scores, metric, Path(table_output))
    if lm:
        lm_result, safe_cols, value_labels = fit_lm(combined_data)
        if lm_latex_output:
            summary_df = build_lm_summary_table(lm_result, safe_cols, value_labels)
            save_lm_latex_table(
                summary_df,
                Path(lm_latex_output),
                lm_caption or f"Coefficient summary for the {model} hyperparameter analysis.",
                lm_label or f"tab:{model}hparamcoeffs",
            )
    if plot:
        plot_heatmap(
            combined_data,
            model,
            param_cols,
            classification_setting,
            metric,
            sort=sort,
            output_path=Path(heatmap_output) if heatmap_output else None,
            show=show_plot,
        )
    return mean_scores


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Display hyperparameter search results"
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
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(5)),
        help="Random seeds (default: 0-4)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="lda",
        help="Model used for classification (default: lda)",
    )
    parser.add_argument(
        "-c",
        "--classification_setting",
        type=str,
        default="cross",
        help="Classification setting: cross or within (default: cross)",
    )
    parser.add_argument(
        "--search-strategy",
        type=str,
        default=None,
        choices=["nested", "block"],
        help="Search strategy (default: nested for cross, block for within)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="kappa",
        help="Metric to display (default: kappa)",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort results by score descending",
    )
    parser.add_argument(
        "--lm",
        action="store_true",
        help="Fit a linear model with all params as categorical predictors",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot a heatmap of mean scores by dataset and parameter combination",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=None,
        help="Output path for the heatmap image",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=None,
        help="Output path for the saved table (.csv or .txt)",
    )
    parser.add_argument(
        "--lm-latex-output",
        type=Path,
        default=None,
        help="Output path for the linear-model coefficient LaTeX table",
    )
    parser.add_argument(
        "--lm-caption",
        type=str,
        default=None,
        help="Caption for the linear-model coefficient LaTeX table",
    )
    parser.add_argument(
        "--lm-label",
        type=str,
        default=None,
        help="Label for the linear-model coefficient LaTeX table",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the heatmap window",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    datasets = get_datasets_from_args(args.datasets, args.suite)
    search_strategy = args.search_strategy or (
        "block" if args.classification_setting == "within" else "nested"
    )
    display_search_results(
        datasets,
        args.model,
        args.seeds,
        args.classification_setting,
        search_strategy,
        metric=args.metric,
        sort=args.sort,
        lm=args.lm,
        plot=args.plot,
        heatmap_output=args.heatmap_output,
        table_output=args.table_output,
        lm_latex_output=args.lm_latex_output,
        lm_caption=args.lm_caption,
        lm_label=args.lm_label,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
