"""
Task difficulty analysis using linear mixed-effects models.

Tests which dataset/task properties predict cross-subject kappa,
with suite as a random intercept. Runs single-factor models first.

Example usage:
    python scripts/paper/task_difficulty.py
    python scripts/paper/task_difficulty.py --seeds 0 1 2 3 4
"""

import argparse
import logging
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from fnirsatlas.io import fNIRSAtlasDataset
from fnirsatlas.results import Results
from fnirsatlas.parsers import get_suite_for_dataset
from fnirsatlas.utils import get_datasets_from_args
from fnirsatlas.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

SETTING = "cross"
METRIC = "cross_kappa"
STRATEGY = "nested"
PREDICTORS = [
    "n_samples", "n_channels", "epoch_length", "n_classes", "n_subjects",
]


def collect_task_data(datasets, seeds):
    """Collect per-task metadata and kappa scores."""
    rows = []
    for dataset in datasets:
        try:
            ds = fNIRSAtlasDataset(dataset)
        except FileNotFoundError:
            logger.warning("Processed file not found for %s, skipping", dataset)
            continue
        try:
            results = Results(
                dataset=dataset, metric=METRIC, model="lda",
                seeds=seeds, search_strategy=STRATEGY,
                classification_setting=SETTING,
            ).load()
        except FileNotFoundError:
            logger.warning("No results for %s, skipping", dataset)
            continue
        tasks = ds.get_tasks()
        for task in tasks:
            desc = task["description"]
            if desc not in results.task2metric:
                continue
            ds.set_task(task)
            kappa = float(np.mean(results.task2metric[desc]))
            rows.append({
                "dataset": dataset,
                "task": desc,
                "suite": get_suite_for_dataset(dataset),
                "kappa": kappa,
                "n_samples": ds.X.shape[0],
                "n_channels": ds.X.shape[1],
                "epoch_length": ds.info["epoch_duration"],
                "n_classes": len(np.unique(ds.y)),
                "n_subjects": len(np.unique(ds.groups)),
            })
    return pd.DataFrame(rows)


def run_single_factor_models(df):
    """Fit one mixed-effects model per predictor (suite as random intercept)."""
    print(f"\n{'='*60}")
    print(f"Single-factor mixed-effects models (N={len(df)})")
    print(f"DV: cross kappa | Random intercept: suite")
    print(f"{'='*60}")
    results = []
    for pred in PREDICTORS:
        formula = f"kappa ~ {pred}"
        try:
            model = smf.mixedlm(formula, df, groups=df["suite"])
            fit = model.fit(reml=True)
            coef = fit.fe_params[pred]
            pval = fit.pvalues[pred]
            results.append({
                "predictor": pred,
                "coef": coef,
                "se": fit.bse_fe[pred],
                "p": pval,
                "sig": _sig_stars(pval),
            })
            print(f"\n--- {pred} ---")
            print(f"  coef = {coef:.4f}, SE = {fit.bse_fe[pred]:.4f}, p = {pval:.4f} {_sig_stars(pval)}")
        except Exception as e:
            logger.warning("Failed to fit model for %s: %s", pred, e)
    print(f"\n{'='*60}")
    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        print("\nSummary:")
        print(summary_df.to_string(index=False))
    return summary_df


def _sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Task difficulty analysis")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    datasets = get_datasets_from_args(None, None)
    df = collect_task_data(datasets, args.seeds)
    if df.empty:
        logger.error("No data collected")
        return
    print("\nTask data:")
    print(df.to_string(index=False))
    run_single_factor_models(df)


if __name__ == "__main__":
    main()
