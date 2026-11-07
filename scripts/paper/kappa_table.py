"""
Script for creating LaTeX tables from computed classification results.

Generates a table with both cross and within kappa in the same table.

Example usage:
    python scripts/paper/kappa_table.py --output './../paper/nirsatlas/tex/results/kappa_table.tex'

"""

import argparse
import pandas as pd
from pathlib import Path
from fnirsatlas.parsers import get_suite_for_dataset, SUITE_MAP
from fnirsatlas.results import Results
from fnirsatlas.utils import get_datasets_from_args
from fnirsatlas.format import format_dataset_name, format_metric
from fnirsatlas.logging import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


def format_value(mean_val, se, precision=1):
    """Format value with standard error for LaTeX."""
    if mean_val == 0 and se == 0:
        return "---"
    mean_str = f"{mean_val:.{precision}f}"
    se_str = f"{se:.{precision}f}"
    return f"{mean_str} $\\pm$ {se_str}"


SETTING_DEFAULT_STRATEGY = {"cross": "nested", "within": "block"}
SUITE_DISPLAY = {"motor": "Motor", "nback": "N-back"}


def load_metric_stats(metric, model, seeds, search_strategy=None):
    """Load stats for a single metric across all datasets."""
    setting = Results._infer_classification_settings(metric)
    strategy = search_strategy or SETTING_DEFAULT_STRATEGY.get(setting, "none")
    datasets = get_datasets_from_args(None, None)
    rows = []
    for dataset in datasets:
        results = Results(
            dataset=dataset, metric=metric, model=model,
            seeds=seeds, search_strategy=strategy,
            classification_setting=setting,
        ).load()
        stats = results.compute_stats()
        raw_suite = get_suite_for_dataset(dataset)
        suite = SUITE_DISPLAY.get(raw_suite, raw_suite)
        for task, s in stats.items():
            rows.append([suite, format_dataset_name(dataset), task,
                         s["mean"] * 100, s["se"] * 100])
    return rows


def create_kappa_table(model="lda", seeds=None, output_path=None, search_strategy=None):
    metrics = ["cross_kappa", "within_kappa"]
    dfs = {}
    for metric in metrics:
        rows = load_metric_stats(metric, model, seeds, search_strategy)
        col = format_metric(metric)
        df = pd.DataFrame(rows, columns=["Suite", "Dataset", "Task", col, f"{col}_se"])
        df[col + "_fmt"] = df.apply(
            lambda x, c=col: f"{x[c]:3.1f} ± {x[c + '_se']:3.1f}", axis=1
        )
        dfs[col] = df
    # Merge cross and within on shared keys
    keys = ["Suite", "Dataset", "Task"]
    cross_col = format_metric("cross_kappa")
    within_col = format_metric("within_kappa")
    merged = dfs[cross_col][keys + [cross_col, cross_col + "_fmt"]].merge(
        dfs[within_col][keys + [within_col, within_col + "_fmt"]],
        on=keys, how="outer",
    )
    # Preserve SUITE_MAP order
    suite_order = [SUITE_DISPLAY.get(s, s) for s in SUITE_MAP]
    merged["Suite"] = pd.Categorical(merged["Suite"], categories=suite_order, ordered=True)
    merged = merged.sort_values(["Suite", "Dataset"])
    # Suite means (numeric, before formatting)
    for col in [cross_col, within_col]:
        g = merged.groupby("Suite", observed=True)[col].mean().to_dict()
        merged[f"{col} mean"] = merged["Suite"].map(g).apply(lambda x: f"{x:3.1f}")
    # Replace numeric with formatted
    merged[cross_col] = merged[cross_col + "_fmt"]
    merged[within_col] = merged[within_col + "_fmt"]
    merged = merged.drop(columns=[cross_col + "_fmt", within_col + "_fmt"])
    merged = merged.set_index(["Suite", f"{cross_col} mean", f"{within_col} mean", "Dataset"])
    latex_output = merged.style.to_latex()
    latex_output = latex_output.replace(r"[c]", "[t]")
    if output_path:
        Path(output_path).write_text(latex_output)
        logger.info(f"LaTeX table saved to {output_path}")
    else:
        print(latex_output)


def main():
    parser = argparse.ArgumentParser(
        description="Create LaTeX table from classification results"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--search_strategy", type=str, default=None)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    create_kappa_table(seeds=args.seeds, output_path=args.output, search_strategy=args.search_strategy)


if __name__ == "__main__":
    main()
