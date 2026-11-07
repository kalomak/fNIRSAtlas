#!/usr/bin/env python3
"""Test task selection on every held-out model pair."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from fnirsatlas.config import (
    get_cv_config,
    get_paper_models,
    get_results_cache_dir,
    get_results_dir,
)
from fnirsatlas.informativeness import combine_result_cube
from fnirsatlas.parsers import get_available_datasets
from fnirsatlas.utils import escape_latex, order_tasks_by_suite

if __package__:
    from scripts.paper.task_informativeness_lomo import (
        N_BOOT,
        RNG_SEED,
        SEEDS,
        TOP_K,
        load_result_cube,
        model_ranking,
        score_model_holdout,
    )
else:
    from task_informativeness_lomo import (
        N_BOOT,
        RNG_SEED,
        SEEDS,
        TOP_K,
        load_result_cube,
        model_ranking,
        score_model_holdout,
    )

RANDOM_DRAWS = 1000


def held_out_groups(models, holdout_size):
    """Return every held-out model group deterministically."""
    if holdout_size < 2 or holdout_size >= len(models):
        raise ValueError("holdout_size must be between 2 and n_models - 1")
    return list(combinations(sorted(models), holdout_size))


def suite_counts(tasks):
    """Count tasks per benchmark suite."""
    _, suites = order_tasks_by_suite(list(tasks))
    return {suite: len(suite_tasks) for suite, suite_tasks in suites.items()}


def random_control_subsets(tasks, selected, subset_name, n_draws, rng):
    """Draw controls matching the constraints of one selection method."""
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    if not set(selected).issubset(tasks):
        raise ValueError("selected tasks must belong to the task universe")
    if subset_name == "HighInfo":
        return [
            sorted(rng.choice(tasks, size=len(selected), replace=False).tolist())
            for _ in range(n_draws)
        ]
    if subset_name != "Balanced":
        raise ValueError(f"Unknown subset: {subset_name}")
    _, universe = order_tasks_by_suite(list(tasks))
    counts = suite_counts(selected)
    if counts["MISC"] != len(universe["MISC"]):
        raise ValueError("Balanced selection must retain all MISC tasks")
    draws = []
    for _ in range(n_draws):
        draw = list(universe["MISC"])
        for suite, count in counts.items():
            if suite != "MISC" and count:
                draw.extend(rng.choice(universe[suite], count, replace=False).tolist())
        draws.append(sorted(draw))
    return draws


def _held_out_scores(fold_scores, group):
    return {
        task: {model: model_scores[model] for model in group}
        for task, model_scores in fold_scores.items()
    }


def pairwise_agreement(full, reduced, models):
    """Return the fraction of model-pair orderings preserved."""
    return float(np.mean([
        (full["rank"][left] < full["rank"][right])
        == (reduced["rank"][left] < reduced["rank"][right])
        for left, right in combinations(models, 2)
    ]))


def _control_agreements(scores, full, models, subsets):
    """Evaluate pair-order agreement for all controls in one operation."""
    tasks = full["tasks"]
    task_index = {task: index for index, task in enumerate(tasks)}
    selection = np.zeros((len(subsets), len(tasks)))
    for row, subset in enumerate(subsets):
        selection[row, [task_index[task] for task in subset]] = 1
    means = np.array([[np.mean(scores[task][model]) for model in models] for task in tasks])
    mean_ranks = selection @ rankdata(-means, axis=1) / selection.sum(axis=1)[:, None]
    mean_kappas = selection @ means / selection.sum(axis=1)[:, None]
    order = np.lexsort((-mean_kappas, mean_ranks), axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(len(subsets))[:, None], order] = np.arange(1, len(models) + 1)
    full_ranks = np.array([full["rank"][model] for model in models])
    agreements = [
        (full_ranks[left] < full_ranks[right])
        == (ranks[:, left] < ranks[:, right])
        for left, right in combinations(range(len(models)), 2)
    ]
    return np.mean(agreements, axis=0).tolist()


def _evaluation_row(group, subset_name, tasks, full, reduced, controls):
    agreement = pairwise_agreement(full, reduced, group)
    random_rate = float(np.mean(controls))
    return {
        "held_out_models": list(group),
        "subset": subset_name,
        "selected_task_count": len(tasks),
        "selected_suite_counts": suite_counts(tasks),
        "selected_tasks": tasks,
        "evaluation_task_count": len(reduced["tasks"]),
        "evaluation_suite_counts": suite_counts(reduced["tasks"]),
        "evaluation_tasks": reduced["tasks"],
        "full_ranking": full["ranking"],
        "reduced_ranking": reduced["ranking"],
        "pairwise_agreement": agreement,
        "random_control": "size-matched" if subset_name == "HighInfo" else "suite-matched",
        "random_agreement_rate": random_rate,
        "advantage_over_random": float(agreement - random_rate),
        "random_agreements": controls,
    }


def run_group_holdout(
    fold_scores,
    models,
    holdout_size=2,
    n_boot=N_BOOT,
    rng_seed=RNG_SEED,
    top_k=TOP_K,
    random_draws=RANDOM_DRAWS,
):
    """Select tasks without each model group and evaluate that group only."""
    rng = np.random.default_rng(rng_seed)
    task_metrics = {}
    rows = []
    for group in held_out_groups(models, holdout_size):
        metrics, subsets = score_model_holdout(fold_scores, group, n_boot, rng_seed, top_k)
        task_metrics["|".join(group)] = metrics
        scores = _held_out_scores(fold_scores, group)
        full = model_ranking(scores, list(scores))
        for subset_name, tasks in subsets.items():
            reduced = model_ranking(scores, tasks)
            random_subsets = random_control_subsets(
                full["tasks"], reduced["tasks"], subset_name, random_draws, rng
            )
            controls = _control_agreements(scores, full, list(group), random_subsets)
            rows.append(
                _evaluation_row(group, subset_name, tasks, full, reduced, controls)
            )
    return task_metrics, rows


def aggregate_rows(rows):
    """Aggregate selected and random-control agreement by subset."""
    aggregates = {}
    for subset in ["Balanced", "HighInfo"]:
        selected = [row for row in rows if row["subset"] == subset]
        agreement = float(np.mean([row["pairwise_agreement"] for row in selected]))
        random_rate = float(np.mean([row["random_agreement_rate"] for row in selected]))
        counts = [row["selected_task_count"] for row in selected]
        aggregates[subset] = {
            "pairwise_agreement": agreement,
            "random_agreement_rate": random_rate,
            "advantage_over_random": agreement - random_rate,
            "selected_task_count_median": float(np.median(counts)),
            "selected_task_count_range": [min(counts), max(counts)],
        }
    return aggregates


def _write_csv(path, rows):
    rows = [{key: value for key, value in row.items() if key != "random_agreements"} for row in rows]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def _write_latex(path, aggregates):
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Subset & Selected & Random control & Difference \\",
        r"\midrule",
    ]
    for subset, values in aggregates.items():
        lines.append(
            f"{escape_latex(subset)} & {values['pairwise_agreement']:.3f} & "
            f"{values['random_agreement_rate']:.3f} & "
            f"{values['advantage_over_random']:.3f} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_outputs(output_dir, task_metrics, rows, metadata):
    """Save pair-holdout CSV, JSON, and LaTeX outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        kind: output_dir / f"task_informativeness_group_holdout_k{metadata['holdout_size']}.{extension}"
        for kind, extension in [("csv", "csv"), ("json", "json"), ("latex", "tex")]
    }
    aggregates = aggregate_rows(rows)
    _write_csv(paths["csv"], rows)
    paths["json"].write_text(json.dumps({
        "metadata": metadata,
        "group_task_metrics": task_metrics,
        "held_out_evaluations": rows,
        "aggregates": aggregates,
    }, indent=2) + "\n")
    _write_latex(paths["latex"], aggregates)
    return paths


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_results_dir() / "task_informativeness_group_holdout",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=get_results_cache_dir("config.yaml")
    )
    parser.add_argument("--random-draws", type=int, default=RANDOM_DRAWS)
    parser.add_argument("--holdout-size", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    return parser.parse_args()


def main():
    """Run pair holdout and save paper outputs."""
    args = parse_args()
    models = get_paper_models()
    cube = load_result_cube(
        models,
        get_available_datasets(),
        SEEDS,
        get_cv_config()["cross"]["kwargs"]["n_splits"],
        args.results_dir.expanduser(),
    )
    fold_scores = combine_result_cube(cube)
    task_metrics, rows = run_group_holdout(
        fold_scores,
        models,
        args.holdout_size,
        args.n_boot,
        args.rng_seed,
        TOP_K,
        args.random_draws,
    )
    metadata = {
        "models": models,
        "holdout_size": args.holdout_size,
        "group_count": len(held_out_groups(models, args.holdout_size)),
        "seeds": SEEDS,
        "task_count_informativeness": len(fold_scores),
        "n_boot": args.n_boot,
        "rng_seed": args.rng_seed,
        "top_k": TOP_K,
        "random_draws_per_pair_subset": args.random_draws,
        "primary_evaluation": "Held-out pair ordering on reduced versus full independent-task rankings.",
        "random_controls": {
            "HighInfo": "Uniform subsets matched by independent-task count.",
            "Balanced": "Subsets matched by suite counts with all MISC tasks retained.",
        },
    }
    paths = save_outputs(args.output_dir.resolve(), task_metrics, rows, metadata)
    for output_type, path in paths.items():
        print(f"{output_type.upper()}: {path}")


if __name__ == "__main__":
    main()
