#!/usr/bin/env python3
"""Compare task-level and collection-level Friedman/Nemenyi results.

Task level: independent task blocks (as in ``friedman_test.py``).
Collection level: one non-overlapping task per dataset configuration, averaged
within each data collection.

Example:
    uv run python -m scripts.paper.dependence_sensitivity
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fnirsatlas.compare import extract_model_metrics
from fnirsatlas.config import get_paper_models, get_results_dir
from fnirsatlas.dataset_catalog import get_available_datasets
from fnirsatlas.results import Results
from fnirsatlas.stats import nemenyi_critical_difference, nemenyi_significant_pairs
from fnirsatlas.utils import get_non_overlapping_clf_tasks
from scripts.friedman_test import (
    SETTING_DEFAULT_STRATEGY,
    build_score_matrix,
    compute_model_summary,
    run_friedman_test,
    run_posthoc_nemenyi,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=get_paper_models())
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output", type=Path, default=get_results_dir() / "paper" / "dependence_sensitivity.json"
    )
    return parser.parse_args()


def collection_for_task(task_id: str) -> str:
    """Return the data collection of a ``<dataset>__<task>`` id, e.g. ``shin2016__ma__X`` -> ``shin2016``."""
    return task_id.split("__", 1)[0]


def aggregate_collection_scores(task_ids: list[str], scores: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Average task-by-model scores within each collection. Returns sorted collection ids and scores."""
    grouped = defaultdict(list)
    for task_id, row in zip(task_ids, scores, strict=True):
        grouped[collection_for_task(task_id)].append(row)
    collection_ids = sorted(grouped)
    return collection_ids, np.array([np.mean(grouped[key], axis=0) for key in collection_ids])


def build_configuration_scores(results_dicts: list[dict], metric: str):
    """Build one score row per dataset configuration (non-overlapping task)."""
    model_names = [results["model_name"] for results in results_dicts]
    metrics = {
        name: extract_model_metrics(results, metric)[0]
        for name, results in zip(model_names, results_dicts)
    }
    common_ids = set.intersection(*(set(values) for values in metrics.values()))
    task_ids = sorted(get_non_overlapping_clf_tasks(common_ids))
    scores = np.array(
        [[np.mean(metrics[name][task_id]) * 100 for name in model_names] for task_id in task_ids]
    )
    return model_names, task_ids, scores


def analyze_level(model_names: list[str], block_ids: list[str], scores: np.ndarray, alpha: float) -> dict:
    """Friedman test, mean ranks, Nemenyi CD, significant pairs and top group."""
    statistic, p_value = run_friedman_test(scores)
    summary = compute_model_summary(model_names, scores)
    ranks = {row["model"]: row["mean_rank"] for row in summary}
    cd = nemenyi_critical_difference(len(model_names), len(block_ids), alpha)
    significant = nemenyi_significant_pairs(model_names, np.array([ranks[m] for m in model_names]), cd)
    p_significant = {
        tuple(sorted((row["model1"], row["model2"])))
        for row in run_posthoc_nemenyi(scores, model_names, alpha)
        if row["significant"]
    }
    if significant != p_significant:
        raise ValueError("Nemenyi critical-difference and p-value decisions disagree")
    leader = summary[0]["model"]
    top_group = [row["model"] for row in summary if row["mean_rank"] - ranks[leader] <= cd]
    return {
        "block_ids": block_ids,
        "statistic": statistic,
        "p_value": p_value,
        "df": len(model_names) - 1,
        "critical_difference": cd,
        "summary": summary,
        "leader": leader,
        "top_group": top_group,
        "significant_pairs": sorted(significant),
    }


def analyze_setting(results_dicts: list[dict], metric: str, alpha: float) -> dict:
    """Compare task- and collection-level results for one metric."""
    model_names, task_ids, task_scores, _ = build_score_matrix(results_dicts, metric)
    config_names, config_ids, config_scores = build_configuration_scores(results_dicts, metric)
    if config_names != model_names:
        raise ValueError("Model order differs between task and configuration scores")
    collection_ids, collection_scores = aggregate_collection_scores(config_ids, config_scores)
    task = analyze_level(model_names, task_ids, task_scores, alpha)
    collection = analyze_level(model_names, collection_ids, collection_scores, alpha)
    task_pairs, collection_pairs = set(task["significant_pairs"]), set(collection["significant_pairs"])
    return {
        "metric": metric,
        "n_configurations": len(config_ids),
        "levels": {"task": task, "collection": collection},
        "gained": sorted(collection_pairs - task_pairs),
        "lost": sorted(task_pairs - collection_pairs),
    }


def print_analysis(analysis: dict) -> None:
    """Print the collection-level result and its differences to the task level."""
    task, collection = analysis["levels"]["task"], analysis["levels"]["collection"]
    print(f"{analysis['metric']}: {len(task['block_ids'])} tasks, "
          f"{analysis['n_configurations']} configurations, {len(collection['block_ids'])} collections")
    for name, level in analysis["levels"].items():
        print(f"  {name}: chi2({level['df']})={level['statistic']:.3f}, p={level['p_value']:.3g}, "
              f"CD={level['critical_difference']:.3f}, leader={level['leader']}, "
              f"top group={','.join(level['top_group'])}")
    print(f"  gained={analysis['gained']} lost={analysis['lost']}")


def main() -> int:
    args = parse_args()
    analyses = []
    for setting, strategy in SETTING_DEFAULT_STRATEGY.items():
        metric = f"{setting}_acc"
        results = Results.gather_results(
            args.models, get_available_datasets(), metric=metric, seeds=args.seeds,
            search_strategy=strategy, classification_setting=setting, accept_partial=False,
        )
        if len(results) != len(args.models):
            raise ValueError(f"Expected results for {len(args.models)} models, found {len(results)}")
        analyses.append(analyze_setting(results, metric, args.alpha))
        print_analysis(analyses[-1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seeds": args.seeds, "alpha": args.alpha, "analyses": analyses}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
