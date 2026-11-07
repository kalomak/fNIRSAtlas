from itertools import combinations

import numpy as np

from scripts.paper.task_informativeness_group_holdout import (
    held_out_groups,
    pairwise_agreement,
    random_control_subsets,
    run_group_holdout,
    suite_counts,
)
from scripts.paper.task_informativeness_lomo import derive_model_holdout_subsets


def _fold_scores():
    tasks = [
        "bak2019__t1",
        "buccino2016__t1",
        "bnci__t1",
        "shin2016__ma__t1",
        "shader2021__t1",
        "luke2024a__t1",
    ]
    return {
        task: {
            model: np.array([task_ix + model_ix, task_ix + model_ix + 0.5])
            for model_ix, model in enumerate(["m1", "m2", "m3", "m4", "m5"])
        }
        for task_ix, task in enumerate(tasks)
    }


def test_group_subset_derivation_does_not_use_held_out_models():
    scores = _fold_scores()
    changed = {
        task: {model: values.copy() for model, values in models.items()}
        for task, models in scores.items()
    }
    for task in changed:
        changed[task]["m3"][:] = [-1000, 1000]
        changed[task]["m4"][:] = [1000, -1000]

    subsets = derive_model_holdout_subsets(scores, {"m3", "m4"}, 20, 7, 2)
    changed_subsets = derive_model_holdout_subsets(
        changed, {"m3", "m4"}, 20, 7, 2
    )

    assert subsets == changed_subsets


def test_held_out_groups_are_exhaustive_and_deterministic():
    groups = held_out_groups(["m4", "m2", "m1", "m3"], 3)

    assert groups == list(combinations(["m1", "m2", "m3", "m4"], 3))


def test_pairwise_agreement_counts_each_pair():
    full = {"rank": {"m1": 1, "m2": 2, "m3": 3}}
    reduced = {"rank": {"m1": 2, "m2": 1, "m3": 3}}

    assert pairwise_agreement(full, reduced, ["m1", "m2", "m3"]) == 2 / 3


def test_highinfo_controls_match_only_task_count_and_seed():
    tasks = list(_fold_scores())
    selected = ["bak2019__t1", "buccino2016__t1"]

    first = random_control_subsets(
        tasks, selected, "HighInfo", 30, np.random.default_rng(4)
    )
    second = random_control_subsets(
        tasks, selected, "HighInfo", 30, np.random.default_rng(4)
    )

    assert first == second
    assert all(len(subset) == len(selected) for subset in first)
    assert any(suite_counts(subset) != suite_counts(selected) for subset in first)


def test_balanced_controls_match_suites_and_retain_misc():
    tasks = list(_fold_scores())
    selected = [
        "bak2019__t1",
        "bnci__t1",
        "shader2021__t1",
        "luke2024a__t1",
    ]

    controls = random_control_subsets(
        tasks, selected, "Balanced", 30, np.random.default_rng(4)
    )

    assert all(suite_counts(subset) == suite_counts(selected) for subset in controls)
    assert all({"shader2021__t1", "luke2024a__t1"} <= set(subset) for subset in controls)


def test_group_holdout_evaluates_only_held_out_pairs():
    _, rows = run_group_holdout(
        _fold_scores(), ["m1", "m2", "m3", "m4", "m5"], n_boot=5, random_draws=2
    )

    assert len(rows) == 2 * len(list(combinations(range(5), 2)))
    for row in rows:
        held_out = set(row["held_out_models"])
        assert set(row["full_ranking"]) == held_out
        assert set(row["reduced_ranking"]) == held_out
        assert row["pairwise_agreement"] in {0.0, 1.0}


def test_group_holdout_supports_three_models():
    _, rows = run_group_holdout(
        _fold_scores(),
        ["m1", "m2", "m3", "m4", "m5"],
        holdout_size=3,
        n_boot=5,
        random_draws=2,
    )

    assert len(rows) == 2 * len(list(combinations(range(5), 3)))
    assert all(len(row["full_ranking"]) == 3 for row in rows)
