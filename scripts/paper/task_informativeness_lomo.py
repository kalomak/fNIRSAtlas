#!/usr/bin/env python3
"""Evaluate nine-model task selection on each held-out model."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from fnirsatlas.compare import extract_model_metrics
from fnirsatlas.config import (
    get_cv_config,
    get_paper_models,
    get_results_cache_dir,
    get_results_dir,
)
from fnirsatlas.informativeness import (
    combine_result_cube,
    compute_task_informativeness,
    validate_result_cube,
)
from fnirsatlas.parsers import get_available_datasets
from fnirsatlas.results import Results
from fnirsatlas.utils import escape_latex, filter_independent_task_ids

if __package__:
    from scripts.paper.task_informativeness import _high_info_flags, _suite_balance_flags
else:
    from task_informativeness import _high_info_flags, _suite_balance_flags

N_BOOT = 9999
RNG_SEED = 42
SEEDS = list(range(5))
TOP_K = 3
METRIC = "cross_kappa"
SEARCH_STRATEGY = "nested"


def load_result_cube(models, datasets, seeds, expected_folds, results_dir, metric=METRIC):
    """Load every requested seed separately and reject incomplete results."""
    cube = {seed: {} for seed in seeds}
    for seed in seeds:
        for model in models:
            for dataset in datasets:
                loader = Results(
                    dataset, metric, model, [seed], SEARCH_STRATEGY, "cross", False
                )
                path = (
                    results_dir
                    / "cross"
                    / SEARCH_STRATEGY
                    / model
                    / dataset
                    / str(seed)
                    / "results.json"
                )
                task_dict, _ = loader.load_single_seed(path)
                metrics, _ = extract_model_metrics({dataset: task_dict}, metric=metric)
                if len(metrics) != 1:
                    raise ValueError(
                        f"Expected one task for {dataset}, {model}, seed {seed}; "
                        f"found {sorted(metrics)}"
                    )
                task, scores = next(iter(metrics.items()))
                cube[seed].setdefault(task, {})[model] = np.asarray(scores)
    tasks = sorted(cube[seeds[0]])
    if len(tasks) != len(datasets):
        raise ValueError(f"Expected {len(datasets)} tasks, found {len(tasks)}")
    validate_result_cube(cube, seeds, tasks, models, expected_folds)
    return cube


def _score_subsets(fold_scores, n_boot, rng_seed, top_k):
    results = compute_task_informativeness(
        fold_scores, n_boot=n_boot, rng_seed=rng_seed, top_k_models=top_k
    )
    selection_flags = {
        "Balanced": _suite_balance_flags(results),
        "HighInfo": _high_info_flags(results),
    }
    subsets = {
        name: sorted(task for task, selected in flags.items() if selected)
        for name, flags in selection_flags.items()
    }
    return results, subsets


def derive_lomo_subsets(fold_scores, held_out, n_boot, rng_seed, top_k):
    """Derive subsets without exposing any held-out-model scores."""
    return derive_model_holdout_subsets(
        fold_scores, {held_out}, n_boot, rng_seed, top_k
    )


def derive_model_holdout_subsets(fold_scores, held_out, n_boot, rng_seed, top_k):
    """Derive subsets without scores from any held-out model."""
    return score_model_holdout(fold_scores, held_out, n_boot, rng_seed, top_k)[1]


def _score_lomo(fold_scores, held_out, n_boot, rng_seed, top_k):
    return score_model_holdout(
        fold_scores, {held_out}, n_boot, rng_seed, top_k
    )


def score_model_holdout(fold_scores, held_out, n_boot, rng_seed, top_k):
    """Score task subsets after excluding a set of models."""
    held_out = set(held_out)
    models = set(next(iter(fold_scores.values())))
    if not held_out or not held_out < models:
        raise ValueError("held_out must be a non-empty proper subset of models")
    training_scores = {
        task: {
            model: scores
            for model, scores in model_scores.items()
            if model not in held_out
        }
        for task, model_scores in fold_scores.items()
    }
    return _score_subsets(training_scores, n_boot, rng_seed, top_k)


def model_ranking(fold_scores, task_ids):
    """Rank all models by mean task-wise rank on independent tasks."""
    independent, dropped = filter_independent_task_ids(task_ids)
    if not independent:
        raise ValueError("Subset has no independent tasks")
    models = sorted(next(iter(fold_scores.values())))
    means = np.array(
        [[np.mean(fold_scores[task][model]) for model in models] for task in independent]
    )
    mean_ranks = np.mean([rankdata(-row) for row in means], axis=0)
    order = sorted(range(len(models)), key=lambda i: (mean_ranks[i], -means[:, i].mean()))
    ranking = [models[i] for i in order]
    return {
        "ranking": ranking,
        "rank": {model: ranking.index(model) + 1 for model in models},
        "mean_kappa": {model: float(means[:, i].mean()) for i, model in enumerate(models)},
        "mean_rank": {model: float(mean_ranks[i]) for i, model in enumerate(models)},
        "tasks": independent,
        "dropped_tasks": dropped,
    }


def _jaccard(left, right):
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union)


def _result_row(held_out, subset_name, tasks, original, reduced, full, rho):
    """Build one held-out evaluation row with full-to-reduced changes."""
    kappa_full = full["mean_kappa"][held_out]
    kappa_reduced = reduced["mean_kappa"][held_out]
    rank_full = full["rank"][held_out]
    rank_reduced = reduced["rank"][held_out]
    return {
        "held_out_model": held_out,
        "subset": subset_name,
        "held_out_mean_kappa_full": kappa_full,
        "held_out_mean_kappa_reduced": kappa_reduced,
        "held_out_mean_kappa_delta": kappa_reduced - kappa_full,
        "held_out_rank_full": rank_full,
        "held_out_rank_reduced": rank_reduced,
        "held_out_rank_delta": rank_reduced - rank_full,
        "selected_task_count": len(tasks),
        "selected_tasks": tasks,
        "evaluation_tasks": reduced["tasks"],
        "jaccard_original": _jaccard(tasks, original[subset_name]),
        "ranking_spearman": float(rho),
        "all_model_ranking": reduced["ranking"],
        "full_benchmark_ranking": full["ranking"],
    }


def run_lomo(fold_scores, models, n_boot=N_BOOT, rng_seed=RNG_SEED, top_k=TOP_K):
    """Select tasks with nine models and evaluate the held-out model."""
    original_metrics, original = _score_subsets(fold_scores, n_boot, rng_seed, top_k)
    full = model_ranking(fold_scores, list(fold_scores))
    rows = []
    lomo_metrics = {}
    for held_out in models:
        lomo_metrics[held_out], subsets = _score_lomo(
            fold_scores, held_out, n_boot=n_boot, rng_seed=rng_seed, top_k=top_k
        )
        for subset_name, tasks in subsets.items():
            reduced = model_ranking(fold_scores, tasks)
            rho = spearmanr(
                [reduced["rank"][model] for model in models],
                [full["rank"][model] for model in models],
            ).statistic
            rows.append(
                _result_row(held_out, subset_name, tasks, original, reduced, full, rho)
            )
    return original_metrics, original, full, lomo_metrics, rows


def aggregate_rows(rows):
    """Summarize primary and diagnostic outputs by subset."""
    fields = [
        "held_out_mean_kappa_full",
        "held_out_mean_kappa_reduced",
        "held_out_mean_kappa_delta",
        "held_out_rank_full",
        "held_out_rank_reduced",
        "held_out_rank_delta",
        "selected_task_count",
        "jaccard_original",
        "ranking_spearman",
    ]
    aggregates = {}
    for subset in ["Balanced", "HighInfo"]:
        subset_rows = [row for row in rows if row["subset"] == subset]
        aggregates[subset] = {}
        for field in fields:
            values = [row[field] for row in subset_rows]
            aggregates[subset][field] = {
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }
    return aggregates


def _write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def _summary_cell(summary, digits=3):
    return (
        f"{summary['median']:.{digits}f} "
        f"[{summary['min']:.{digits}f}, {summary['max']:.{digits}f}]"
    )


def _write_latex(path, aggregates):
    end = r"\\"
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule"]
    lines.append(
        r"Subset & $\kappa$ full & $\kappa$ red. & $\Delta\kappa$ & "
        r"Rank full & Rank red. & $\Delta$rank & Tasks " + end
    )
    lines.append(r"\midrule")
    for subset, values in aggregates.items():
        lines.append(
            f"{escape_latex(subset)} & "
            f"{_summary_cell(values['held_out_mean_kappa_full'])} & "
            f"{_summary_cell(values['held_out_mean_kappa_reduced'])} & "
            f"{_summary_cell(values['held_out_mean_kappa_delta'])} & "
            f"{_summary_cell(values['held_out_rank_full'], 1)} & "
            f"{_summary_cell(values['held_out_rank_reduced'], 1)} & "
            f"{_summary_cell(values['held_out_rank_delta'], 1)} & "
            f"{_summary_cell(values['selected_task_count'], 1)} " + end
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def save_outputs(output_dir, original_metrics, original, full, lomo_metrics, rows, metadata):
    """Save detailed CSV/JSON and a held-out-focused LaTeX table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": output_dir / "task_informativeness_lomo.csv",
        "json": output_dir / "task_informativeness_lomo.json",
        "latex": output_dir / "task_informativeness_lomo.tex",
    }
    aggregates = aggregate_rows(rows)
    _write_csv(paths["csv"], rows)
    paths["json"].write_text(
        json.dumps(
            {
                "metadata": metadata,
                "original_task_metrics": original_metrics,
                "original_subsets": original,
                "full_ranking": full,
                "lomo_task_metrics": lomo_metrics,
                "held_out_evaluations": rows,
                "aggregates": aggregates,
            },
            indent=2,
        )
        + "\n"
    )
    _write_latex(paths["latex"], aggregates)
    return paths


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=get_results_dir() / "task_informativeness_lomo",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=get_results_cache_dir("config.yaml")
    )
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    return parser.parse_args()


def main():
    """Run the paper analysis and save its outputs."""
    args = parse_args()
    models = get_paper_models()
    datasets = get_available_datasets()
    cube = load_result_cube(
        models,
        datasets,
        SEEDS,
        get_cv_config()["cross"]["kwargs"]["n_splits"],
        args.results_dir.expanduser(),
    )
    fold_scores = combine_result_cube(cube)
    original_metrics, original, full, lomo_metrics, rows = run_lomo(
        fold_scores, models, args.n_boot, args.rng_seed, TOP_K
    )
    metadata = {
        "models": models,
        "seeds": SEEDS,
        "task_count_informativeness": len(fold_scores),
        "task_count_full_ranking": len(full["tasks"]),
        "ranking_excluded_tasks": full["dropped_tasks"],
        "n_boot": args.n_boot,
        "bootstrap_rng_seed": args.rng_seed,
        "top_k": TOP_K,
        "primary_evaluation": "Held-out model full-versus-reduced mean kappa and rank, including reduced-minus-full deltas.",
        "diagnostic_evaluation": "Subset Jaccard overlap and all-model rank Spearman correlation are sensitivity diagnostics, not evidence of held-out generalization.",
        "interpretation": "Comparable held-out rank supports out-of-sample utility. Higher reduced-set kappa alone is not proof of intrinsic informativeness because selection favors decodable tasks.",
        "ranking_method": "Mean task-wise model rank using Cohen's kappa after reintroducing the held-out model.",
        "task_universe_convention": "Informativeness uses all 23 tasks; rankings apply the manuscript independence filter.",
    }
    paths = save_outputs(
        args.output_dir.resolve(),
        original_metrics,
        original,
        full,
        lomo_metrics,
        rows,
        metadata,
    )
    for output_type, path in paths.items():
        print(f"{output_type.upper()}: {path}")


if __name__ == "__main__":
    main()
