#!/usr/bin/env python3
"""Task informativeness analysis for benchmark comparison."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from fnirsatlas.config import get_paper_models
from fnirsatlas.compare import extract_model_metrics
from fnirsatlas.format import format_dataset_name
from fnirsatlas.informativeness import compute_task_informativeness
from fnirsatlas.results import Results
from fnirsatlas.utils import escape_latex, get_datasets_from_args, order_tasks_by_suite

MODELS = get_paper_models()
SEARCH_STRATEGY = "nested"
METRIC = "cross_kappa"


def load_fold_scores(models, datasets, seeds):
    """Load fold-level kappa scores for all model-task pairs."""
    results_dicts = Results.gather_results(
        models,
        datasets,
        METRIC,
        seeds,
        SEARCH_STRATEGY,
        METRIC.split("_")[0],
    )
    if not results_dicts:
        raise ValueError("No results loaded")
    model_metrics = {}
    for rd in results_dicts:
        name = rd["model_name"]
        metrics, _ = extract_model_metrics(rd, METRIC)
        model_metrics[name] = metrics
    common = set.intersection(*[set(m) for m in model_metrics.values()])
    if not common:
        raise ValueError("No common tasks across models")
    return {
        task_key: {
            name: np.array(model_metrics[name][task_key]) for name in model_metrics
        }
        for task_key in common
    }


def _ordered_results(results):
    _, suite_to_tasks = order_tasks_by_suite(list(results))
    ordered = []
    for tasks in suite_to_tasks.values():
        ordered.extend(
            sorted(
                tasks,
                key=lambda task_id: results[task_id]["informativeness"],
                reverse=True,
            )
        )
    return ordered



def _suite_balance_flags(results):
    _, suite_to_tasks = order_tasks_by_suite(list(results))
    balance_flags = {}
    for suite, tasks in suite_to_tasks.items():
        if not tasks:
            continue
        if suite == "MISC":
            for task_id in tasks:
                balance_flags[task_id] = True
            continue
        suite_median = np.median(
            [results[task_id]["informativeness"] for task_id in tasks]
        )
        cutoff = suite_median / 2.0
        for task_id in tasks:
            balance_flags[task_id] = results[task_id]["informativeness"] > cutoff
    return balance_flags


def _high_info_flags(results):
    cutoff = np.median([result["informativeness"] for result in results.values()]) / 2.0
    return {
        task_id: result["informativeness"] > cutoff
        for task_id, result in results.items()
    }


def _task_to_suite(results):
    _, suite_to_tasks = order_tasks_by_suite(list(results))
    return {
        task_id: suite for suite, tasks in suite_to_tasks.items() for task_id in tasks
    }


def _format_task_label(task_id):
    parts = task_id.split("__")
    ds_id = "__".join(parts[:-1])
    task_desc = parts[-1] if len(parts) >= 2 else ""
    return format_dataset_name(ds_id), task_desc


def _format_latex_bool(value):
    return r"\ding{51}" if value else r"\ding{55}"


def _format_percent(value):
    return f"{value * 100:0.1f}"


def _format_msr(value):
    return f"{value:0.2f}"


def _plot_labels(ordered):
    labels = []
    for tid in ordered:
        ds_name, task_desc = _format_task_label(tid)
        labels.append(f"{ds_name} | {task_desc}" if task_desc else ds_name)
    return labels


def _plot_metrics(results, ordered):
    metric_specs = [
        ("RS%", lambda r: r["rank_stability"] * 100),
        (r"M$\kappa$%", lambda r: r["mean_perf"] * 100),
        ("MSR", lambda r: r["sprr"]),
        (r"U$_d$%", lambda r: r["informativeness"] * 100),
    ]
    values = np.array(
        [[extractor(results[tid]) for _, extractor in metric_specs] for tid in ordered],
        dtype=float,
    )
    return [label for label, _ in metric_specs], values


def _normalize_plot_values(values):
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    spans = np.where(maxs > mins, maxs - mins, 1.0)
    return (values - mins) / spans, mins, maxs


def _dataset_colors(ordered):
    dataset_ids = []
    for tid in ordered:
        dataset_id = tid.rsplit("__", 1)[0]
        if dataset_id not in dataset_ids:
            dataset_ids.append(dataset_id)
    cmap = plt.get_cmap("tab20", len(dataset_ids))
    color_map = {dataset_id: cmap(i) for i, dataset_id in enumerate(dataset_ids)}
    return [color_map[tid.rsplit("__", 1)[0]] for tid in ordered]


def _format_console_table(results):
    ordered = list(_ordered_results(results))
    balance_flags = _suite_balance_flags(results)
    high_info_flags = _high_info_flags(results)
    print(f"\n{'=' * 112}")
    print(f"Task informativeness ({len(ordered)} tasks)")
    print("Bal = Balanced partition (suite-specific median U_d / 2 cutoff)")
    print("HI = HighInfo partition (global median U_d / 2 cutoff)")
    print(f"{'=' * 112}")
    print(
        f"{'Dataset':<16} {'Task':<28} {'RS%':>6} {'M\\kappa%':>9} {'MSR':>6} {'U_d%':>6} {'Bal':>5} {'HI':>5}"
    )
    print("-" * 112)
    for tid in ordered:
        ds_name, task_desc = _format_task_label(tid)
        r = results[tid]
        print(
            f"{ds_name:<16} {task_desc:<28} "
            f"{_format_percent(r['rank_stability']):>6} "
            f"{_format_percent(r['mean_perf']):>9} "
            f"{_format_msr(r['sprr']):>6} "
            f"{_format_percent(r['informativeness']):>6} "
            f"{str(balance_flags[tid]):>5} "
            f"{str(high_info_flags[tid]):>5}"
        )


def _format_latex_table(results):
    row_end = chr(92) * 2
    balance_flags = _suite_balance_flags(results)
    high_info_flags = _high_info_flags(results)
    task_to_suite = _task_to_suite(results)
    ordered = _ordered_results(results)
    lines = [r"\begin{tabular}{llrrrrcc}"]
    lines.append(r"\toprule")
    lines.append(r"Dataset & Task & RS\% & M$\kappa$\% & MSR & U$_d$\% & Bal & HI " + row_end)
    lines.append(r"\midrule")
    prev_suite = None
    for tid in ordered:
        suite = task_to_suite[tid]
        if prev_suite is not None and suite != prev_suite:
            lines.append(r"\midrule")
        ds_name, task_desc = _format_task_label(tid)
        r = results[tid]
        lines.append(
            f"{escape_latex(ds_name)} & {escape_latex(task_desc)} & "
            f"{r['rank_stability'] * 100:0.1f} & {r['mean_perf'] * 100:0.1f} & "
            f"{r['sprr']:0.2f} & {r['informativeness'] * 100:0.1f} & "
            f"{_format_latex_bool(balance_flags[tid])} & "
            f"{_format_latex_bool(high_info_flags[tid])} " + row_end
        )
        prev_suite = suite
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def plot_task_informativeness(results, output_path=None):
    ordered = _ordered_results(results)
    metric_labels, values = _plot_metrics(results, ordered)
    normalized, mins, maxs = _normalize_plot_values(values)
    colors = _dataset_colors(ordered)
    x = np.arange(len(metric_labels))
    fig, ax = plt.subplots(figsize=(9, max(6, len(ordered) * 0.22)))
    for i, tid in enumerate(ordered):
        label = _plot_labels([tid])[0]
        ax.plot(x, normalized[i], color=colors[i], alpha=0.6, linewidth=1.4)
        ax.scatter(x, normalized[i], color=colors[i], s=18)
        ax.text(x[-1] + 0.03, normalized[i, -1], label, fontsize=7, va="center")
    for j, x_pos in enumerate(x):
        ax.axvline(x_pos, color="0.85", linewidth=1, zorder=0)
        ax.text(x_pos, -0.08, f"min {mins[j]:0.1f}", ha="center", va="top", fontsize=8)
        ax.text(x_pos, 1.03, f"max {maxs[j]:0.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, metric_labels)
    ax.set_xlim(x[0] - 0.1, x[-1] + 0.8)
    ax.set_ylim(-0.12, 1.08)
    ax.set_ylabel("Normalized value")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Task Informativeness Across Metrics")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return
    plt.show()


def print_table(results, latex=False):
    """Print informativeness table sorted by composite score."""
    if latex:
        print(_format_latex_table(results))
        return
    _format_console_table(results)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help=f"Models to compare (default: {' '.join(MODELS)})",
    )
    parser.add_argument("-d", "--datasets", nargs="+")
    parser.add_argument("-s", "--suite", type=str)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--top-k-models", type=int)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = get_datasets_from_args(args.datasets, args.suite)
    fold_scores = load_fold_scores(args.models, datasets, args.seeds)
    results = compute_task_informativeness(
        fold_scores,
        n_boot=args.n_boot,
        top_k_models=args.top_k_models,
    )
    if args.plot:
        plot_task_informativeness(results, output_path=args.output)
        return
    print_table(results, latex=args.latex)


if __name__ == "__main__":
    main()
