"""
LDA vs chance (dummy model) comparison table for paper.

Combines cross-subject and within-subject results in a single table.

Usage:
    python scripts/paper/chance_table.py
    python scripts/paper/chance_table.py --output './../paper/nirsatlas/tex/results/chance_table.tex'
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
from fnirsatlas.compare import compare_models
from fnirsatlas.format import format_dataset_name, format_metric
from fnirsatlas.logging import setup_logging
from fnirsatlas.parsers import get_suite_for_dataset, SUITE_MAP
from fnirsatlas.results import Results
from fnirsatlas.utils import get_datasets_from_args

setup_logging()
logger = logging.getLogger(__name__)

SETTING_DEFAULT_STRATEGY = {"cross": "nested", "within": "block"}
SUITE_DISPLAY = {"motor": "Motor", "nback": "N-back"}


def get_search_strategy(model, setting):
    if model == "dummy":
        return "none"
    return SETTING_DEFAULT_STRATEGY.get(setting, "none")


def load_comparison(metric, seeds, alpha=0.05):
    """Run one-sided LDA > dummy comparison for a single metric."""
    setting = Results._infer_classification_settings(metric)
    datasets = get_datasets_from_args(None, None)
    results_dicts = []
    for model in ["lda", "dummy"]:
        strategy = get_search_strategy(model, setting)
        dicts = Results.gather_results(
            models=[model], datasets=datasets, metric=metric,
            seeds=seeds, search_strategy=strategy,
            classification_setting=setting,
        )
        results_dicts.extend(dicts)
    if len(results_dicts) < 2:
        logger.error(f"Missing results for {metric}")
        return pd.DataFrame()
    comparisons = compare_models(
        results_dicts, metric=metric, alpha=alpha, one_sided=True
    )
    rows = []
    for _, task_results in comparisons.items():
        for task_key, r in task_results.items():
            parts = task_key.split("__")
            dataset_id = "__".join(parts[:-1])
            task_desc = parts[-1] if len(parts) >= 2 else ""
            raw_suite = get_suite_for_dataset(dataset_id)
            suite = SUITE_DISPLAY.get(raw_suite, raw_suite)
            sig = _significance_marker(r, alpha)
            rows.append({
                "Suite": suite,
                "Dataset": format_dataset_name(dataset_id),
                "Task": task_desc,
                "dataset_id": dataset_id,
                "lda_mean": r["model1_mean"],
                "lda_se": r["model1_se"],
                "p_fdr": r.get("p_value_corrected", r["p_value"]),
                "sig": sig,
            })
    return pd.DataFrame(rows)


def _significance_marker(r, alpha):
    p = r.get("p_value_corrected", r["p_value"])
    if not r["significant"]:
        return ""
    if p < alpha / 10:
        return "***"
    if p < alpha / 5:
        return "**"
    return "*"


def _fmt(mean, se):
    return f"{mean:3.1f} $\\pm$ {se:3.1f}"


def create_chance_table(seeds=None, alpha=0.05, output_path=None):
    dfs = {}
    for metric in ["cross_acc", "within_acc"]:
        dfs[metric] = load_comparison(metric, seeds, alpha)
    keys = ["Suite", "Dataset", "Task", "dataset_id"]
    cross = dfs["cross_acc"]
    within = dfs["within_acc"]
    cross_label = format_metric("cross_acc")
    within_label = format_metric("within_acc")
    # Format display columns
    cross["cross_fmt"] = cross.apply(
        lambda x: _fmt(x["lda_mean"], x["lda_se"]), axis=1
    )
    cross["cross_sig"] = cross["sig"]
    cross["cross_p"] = cross["p_fdr"]
    within["within_fmt"] = within.apply(
        lambda x: _fmt(x["lda_mean"], x["lda_se"]), axis=1
    )
    within["within_sig"] = within["sig"]
    within["within_p"] = within["p_fdr"]
    merged = cross[keys + ["cross_fmt", "cross_sig", "cross_p"]].merge(
        within[keys + ["within_fmt", "within_sig", "within_p"]],
        on=keys, how="outer",
    )
    # Sort by suite order then dataset
    suite_order = [SUITE_DISPLAY.get(s, s) for s in SUITE_MAP]
    merged["Suite"] = pd.Categorical(
        merged["Suite"], categories=suite_order, ordered=True
    )
    merged = merged.sort_values(["Suite", "Dataset"])
    # Build final table
    rows = []
    for _, row in merged.iterrows():
        cross_val = row.get("cross_fmt", "---")
        within_val = row.get("within_fmt", "---")
        cross_sig = row.get("cross_sig", "")
        within_sig = row.get("within_sig", "")
        rows.append({
            "Suite": row["Suite"],
            "Dataset": row["Dataset"],
            "Task": row["Task"],
            cross_label: f"{cross_val}{cross_sig}" if pd.notna(cross_val) else "---",
            within_label: f"{within_val}{within_sig}" if pd.notna(within_val) else "---",
        })
    df = pd.DataFrame(rows)
    df = df.set_index(["Suite", "Dataset"])
    latex_output = df.style.to_latex()
    latex_output = latex_output.replace(r"[c]", "[t]")
    if output_path:
        Path(output_path).write_text(latex_output)
        logger.info(f"LaTeX table saved to {output_path}")
    else:
        print(latex_output)


def main():
    parser = argparse.ArgumentParser(
        description="LDA vs chance comparison table"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    create_chance_table(
        seeds=args.seeds, alpha=args.alpha, output_path=args.output
    )


if __name__ == "__main__":
    main()
