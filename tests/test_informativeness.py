import numpy as np
import pytest
from fnirsatlas.informativeness import (
    bootstrap_rank_stability,
    combine_result_cube,
    compute_task_informativeness,
    informativeness_score,
    paired_residual_spread,
    validate_result_cube,
)
from scripts.paper.task_informativeness import _ordered_results
from scripts.paper.task_informativeness_lomo import _result_row, derive_lomo_subsets


def test_paired_residual_spread_shared_fold_effects():
    fold_mat = np.array(
        [
            [0.60, 0.70],
            [0.80, 0.90],
            [0.70, 0.80],
        ]
    )
    spread = paired_residual_spread(fold_mat)
    assert spread < 1e-10


def test_paired_residual_spread_single_model():
    fold_mat = np.array([[0.5], [0.6]])
    assert paired_residual_spread(fold_mat) == 0.0


def test_paired_residual_spread_variable_scores():
    rng = np.random.default_rng(0)
    fold_mat = np.column_stack(
        [
            rng.normal(0.7, 0.05, 25),
            rng.normal(0.5, 0.05, 25),
        ]
    )
    spread = paired_residual_spread(fold_mat)
    assert spread > 0.0


def test_bootstrap_rank_stability_clear_order():
    # Models are well separated -> stability near 1.0
    rng = np.random.default_rng(0)
    n_folds = 25
    fold_mat = np.column_stack(
        [
            rng.normal(0.9, 0.01, n_folds),
            rng.normal(0.7, 0.01, n_folds),
            rng.normal(0.5, 0.01, n_folds),
        ]
    )
    stab = bootstrap_rank_stability(fold_mat, n_boot=500, rng=rng)
    assert stab > 0.9


def test_bootstrap_rank_stability_tied():
    # All models identical -> stability should be low
    fold_mat = np.ones((25, 3)) * 0.5
    stab = bootstrap_rank_stability(fold_mat)
    assert stab == 0.0


def test_informativeness_score_range():
    rs = np.array([0.9, 0.5, 0.1])
    sprr = np.array([3.0, 1.0, 0.1])
    margin = np.array([0.2, 0.1, 0.0])
    scores = informativeness_score(rs, sprr, margin)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_informativeness_score_single_weight():
    rs = np.array([0.9, 0.5, 0.1])
    sprr = np.array([1.0, 1.0, 1.0])
    margin = np.array([0.0, 0.0, 0.0])
    scores = informativeness_score(rs, sprr, margin)
    assert scores[0] > scores[1] > scores[2]


def test_compute_task_informativeness_integration():
    rng = np.random.default_rng(0)
    fold_scores_by_task = {
        "ds1__task1": {
            "lda": rng.normal(0.7, 0.05, 25),
            "eegnet": rng.normal(0.8, 0.05, 25),
        },
        "ds2__task2": {
            "lda": rng.normal(0.5, 0.05, 25),
            "eegnet": rng.normal(0.5, 0.05, 25),
        },
    }
    result = compute_task_informativeness(fold_scores_by_task, n_boot=200)
    assert set(result.keys()) == {"ds1__task1", "ds2__task2"}
    expected_keys = {
        "mean_perf",
        "paired_residual_sd",
        "sprr",
        "rank_stability",
        "informativeness",
    }
    for metrics in result.values():
        assert set(metrics.keys()) == expected_keys
    # Task 1 (models differ) should be more informative than task 2 (same)
    assert (
        result["ds1__task1"]["informativeness"]
        > result["ds2__task2"]["informativeness"]
    )


def test_compute_task_informativeness_rejects_mismatched_fold_counts():
    with pytest.raises(ValueError, match="Mismatched fold counts"):
        compute_task_informativeness(
            {
                "ds1__task1": {
                    "lda": np.array([0.2, 0.3, 0.4]),
                    "eegnet": np.array([0.3, 0.4]),
                }
            }
        )


def test_compute_task_informativeness_uses_top_k_model_mean():
    fold_scores_by_task = {
        "ds1__task1": {
            "m1": np.array([0.1, 0.1]),
            "m2": np.array([0.5, 0.5]),
            "m3": np.array([0.9, 0.9]),
        }
    }

    result_all = compute_task_informativeness(fold_scores_by_task, n_boot=10)
    result_top2 = compute_task_informativeness(
        fold_scores_by_task,
        n_boot=10,
        top_k_models=2,
    )

    assert result_all["ds1__task1"]["mean_perf"] == pytest.approx(0.5)
    assert result_top2["ds1__task1"]["mean_perf"] == pytest.approx(0.7)


def test_task_informativeness_ordered_by_suite_then_score():
    results = {
        "herff2014__task1": {"informativeness": 0.99},
        "bak2019__task1": {"informativeness": 0.50},
        "cockx2023__task1": {"informativeness": 0.80},
        "bnci__task1": {"informativeness": 0.95},
    }

    assert _ordered_results(results) == [
        "cockx2023__task1",
        "bak2019__task1",
        "bnci__task1",
        "herff2014__task1",
    ]


def _complete_cube():
    return {
        seed: {
            task: {
                model: np.array([seed + task_ix + model_ix, seed + 0.5])
                for model_ix, model in enumerate(["m1", "m2", "m3", "m4"])
            }
            for task_ix, task in enumerate(["bak2019__t1", "bnci__t2"])
        }
        for seed in [0, 1]
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cube: cube.pop(1), "seeds"),
        (lambda cube: cube[1].pop("bnci__t2"), "tasks for seed 1"),
        (lambda cube: cube[1]["bnci__t2"].pop("m4"), "models for seed 1, task bnci__t2"),
        (
            lambda cube: cube[1]["bnci__t2"].__setitem__("m4", np.array([1.0])),
            "folds for seed 1, task bnci__t2, model m4",
        ),
    ],
)
def test_validate_result_cube_rejects_incomplete_inputs(mutation, message):
    cube = _complete_cube()
    mutation(cube)
    with pytest.raises(ValueError, match=message):
        validate_result_cube(
            cube,
            expected_seeds=[0, 1],
            expected_tasks=["bak2019__t1", "bnci__t2"],
            expected_models=["m1", "m2", "m3", "m4"],
            expected_folds=2,
        )


def test_lomo_subset_derivation_does_not_use_held_out_model():
    fold_scores = combine_result_cube(_complete_cube())
    changed = {
        task: {model: scores.copy() for model, scores in model_scores.items()}
        for task, model_scores in fold_scores.items()
    }
    for task in changed:
        changed[task]["m4"][:] = np.array([-1000.0, 1000.0] * 2)

    subsets = derive_lomo_subsets(fold_scores, "m4", 20, 7, 3)
    changed_subsets = derive_lomo_subsets(changed, "m4", 20, 7, 3)

    assert subsets == changed_subsets


def test_lomo_result_reports_held_out_deltas():
    reduced = {
        "tasks": ["task1"],
        "ranking": ["m2", "m1"],
        "rank": {"m1": 2},
        "mean_kappa": {"m1": 0.4},
    }
    full = {
        "ranking": ["m1", "m2"],
        "rank": {"m1": 1},
        "mean_kappa": {"m1": 0.3},
    }

    row = _result_row(
        "m1", "HighInfo", ["task1"], {"HighInfo": ["task1"]}, reduced, full, 0.5
    )

    assert row["held_out_mean_kappa_delta"] == pytest.approx(0.1)
    assert row["held_out_rank_delta"] == 1
