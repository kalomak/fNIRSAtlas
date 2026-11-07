#!/usr/bin/env python3
"""Nemenyi-significant model pairs of the paper models vs number of seeds.

For each k, the count is averaged over all k-seed subsets of the available seeds.

Example:
    uv run python scripts/paper/seed_significant_pairs.py
"""

from itertools import combinations

import numpy as np
from scipy.stats import rankdata

from cd_diagram import MODELS, load_score_matrix
from fnirsatlas.config import get_results_dir
from fnirsatlas.stats import nemenyi_critical_difference

SEEDS = [0, 1, 2, 3, 4]
SETTINGS = {"cross": "Cross", "within": "Within"}
ALPHA = 0.05


def significant_pairs(setting: str, seeds) -> tuple[int, int]:
    """Number of model pairs whose average-rank gap exceeds the Nemenyi CD, and the number of tasks."""
    _, scores, _ = load_score_matrix(setting, f"{setting}_acc", seeds)
    ranks = np.apply_along_axis(lambda r: rankdata(-r), 1, scores).mean(0)
    gaps = np.abs(ranks[:, None] - ranks[None, :])[np.triu_indices(len(ranks), 1)]
    return int((gaps > nemenyi_critical_difference(len(ranks), scores.shape[0], ALPHA)).sum()), scores.shape[0]


def summary_cell(setting: str, k: int) -> tuple[str, int]:
    """Mean (min–max) significant pairs over all k-seed subsets."""
    results = [significant_pairs(setting, s) for s in combinations(SEEDS, k)]
    counts = np.array([c for c, _ in results])
    cell = f"{counts.mean():.1f} ({counts.min()}–{counts.max()})" if len(counts) > 1 else f"{counts[0]}"
    return cell, results[0][1]


def main() -> int:
    rows, n_tasks = {}, set()
    for k in range(1, len(SEEDS) + 1):
        for setting, name in SETTINGS.items():
            cell, n = summary_cell(setting, k)
            rows.setdefault(k, {})[name] = cell
            n_tasks.add(n)
    if len(n_tasks) != 1:
        raise ValueError(f"Settings differ in task count: {n_tasks}")
    n_pairs = len(MODELS) * (len(MODELS) - 1) // 2
    names = list(SETTINGS.values())
    lines = [
        f"Nemenyi-significant pairs (α = {ALPHA}) out of {n_pairs}, {len(MODELS)} paper models, "
        f"{n_tasks.pop()} tasks. Mean (min–max) over all k-seed subsets of seeds {SEEDS[0]}–{SEEDS[-1]}.",
        "",
        "| Seeds | " + " | ".join(names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    lines += [f"| {k} | " + " | ".join(row[n] for n in names) + " |" for k, row in rows.items()]
    text = "\n".join(lines) + "\n"
    output_path = get_results_dir() / "paper" / "seed_significant_pairs.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    print(text)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
