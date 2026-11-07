#!/usr/bin/env python3
"""Tests for the Friedman test script."""

import subprocess
import sys

from scripts.friedman_test import (
    build_score_matrix,
    compute_model_summary,
    run_friedman_test,
    run_posthoc_nemenyi,
)
from fnirsatlas.utils import filter_independent_task_ids
from fnirsatlas.results import Results


def make_results(model_name, block_scores):
    """Create a result dictionary for one model."""
    return {
        "model_name": model_name,
        "buccino2016": {
            "Side[R|L]xJoint[Arm|Hand]": {"cross_acc": block_scores[0]},
        },
        "ortega2020": {
            "HandGrip[L|R]": {"cross_acc": block_scores[1]},
        },
        "luke2021": {
            "Control|FingerTap[L|R]": {"cross_acc": block_scores[2]},
        },
        "herff2014": {
            "Arithmetic|Baseline": {"cross_acc": block_scores[3]},
        },
    }


def sample_results_dicts():
    """Create 4 models with clearly different scores."""
    return [
        make_results("lda", [[0.90, 0.89], [0.88, 0.87], [0.91, 0.90], [0.89, 0.88]]),
        make_results(
            "eegnet", [[0.84, 0.83], [0.82, 0.81], [0.85, 0.84], [0.83, 0.82]]
        ),
        make_results("dummy", [[0.62, 0.61], [0.60, 0.59], [0.63, 0.62], [0.61, 0.60]]),
        make_results("svm", [[0.78, 0.77], [0.76, 0.75], [0.79, 0.78], [0.77, 0.76]]),
    ]


def sample_within_results_dicts():
    """Create 4 models with within_acc scores."""
    results_dicts = sample_results_dicts()
    for results in results_dicts:
        for dataset, tasks in results.items():
            if dataset == "model_name":
                continue
            for task in tasks.values():
                task["within_acc"] = task.pop("cross_acc")
    return results_dicts


def test_friedman_test_help():
    """Test that the script runs with help argument."""
    result = subprocess.run(
        [sys.executable, "scripts/friedman_test.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Perform a Friedman test" in result.stdout


def test_build_score_matrix_and_summary():
    """Test score matrix construction and summary stats."""
    model_names, task_ids, score_matrix, dropped_ids = build_score_matrix(
        sample_results_dicts()
    )
    assert model_names == ["lda", "eegnet", "dummy", "svm"]
    assert len(task_ids) == 4
    assert dropped_ids == []
    assert score_matrix.shape == (4, 4)

    summary_rows = compute_model_summary(model_names, score_matrix)
    assert summary_rows[0]["model"] == "lda"
    assert summary_rows[-1]["model"] == "dummy"


def test_friedman_and_posthoc_results():
    """Test omnibus Friedman result and pairwise post-hoc comparisons."""
    model_names, _, score_matrix, _ = build_score_matrix(sample_results_dicts())
    statistic, p_value = run_friedman_test(score_matrix)
    posthoc_rows = run_posthoc_nemenyi(score_matrix, model_names)

    assert statistic > 0
    assert p_value < 0.05
    assert len(posthoc_rows) == 6
    assert all("p_value" in row for row in posthoc_rows)
    assert all(row["mean_diff"] >= 0 for row in posthoc_rows)
    mean_scores = {
        name: float(score_matrix[:, idx].mean()) for idx, name in enumerate(model_names)
    }
    assert [mean_scores[row["model1"]] for row in posthoc_rows] == sorted(
        [mean_scores[row["model1"]] for row in posthoc_rows], reverse=True
    )
    assert all(
        mean_scores[row["model1"]] >= mean_scores[row["model2"]] for row in posthoc_rows
    )
    first_block = [row["model2"] for row in posthoc_rows if row["model1"] == "lda"]
    assert [mean_scores[name] for name in first_block] == sorted(
        [mean_scores[name] for name in first_block]
    )


def test_filter_independent_task_ids_uses_config_drop(monkeypatch):
    """Test config-driven overlap drop before independent-task filtering."""

    monkeypatch.setattr(
        "fnirsatlas.config.load_config",
        lambda: {
            "task_selection": {"drop_if_present": {"shin2019__ma": ["shin2019__mi__"]}}
        },
    )
    task_ids = {
        "shin2019__ma__TaskA",
        "shin2019__mi__TaskB",
        "luke2021__TaskC",
    }

    filtered_ids, dropped_ids = filter_independent_task_ids(task_ids)

    assert "shin2019__ma__TaskA" in filtered_ids
    assert "luke2021__TaskC" in filtered_ids
    assert "shin2019__mi__TaskB" in dropped_ids


def test_friedman_main_with_mocked_results(monkeypatch, capsys):
    """Test main flow with mocked result loading."""
    from scripts import friedman_test

    captured = {}

    def fake_gather_results(models, datasets, metric="cross_acc", seeds=None, **kwargs):
        captured.update(kwargs)
        return sample_results_dicts()

    monkeypatch.setattr(Results, "gather_results", staticmethod(fake_gather_results))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "friedman_test.py",
            "--models",
            "lda",
            "eegnet",
            "dummy",
            "svm",
            "--datasets",
            "buccino2016",
            "ortega2020",
            "luke2021",
            "herff2014",
            "--show-blocks",
        ],
    )

    friedman_test.main()
    output = capsys.readouterr().out
    assert "Friedman test on cross_acc task means" in output
    assert "Post-hoc Nemenyi" in output
    assert "Blocks used" in output
    assert captured["search_strategy"] == "nested"
    assert captured["classification_setting"] == "cross"


def test_friedman_main_uses_config_models_by_default(monkeypatch, capsys):
    """Test that the script falls back to config paper models."""
    from scripts import friedman_test

    captured = {}

    def fake_gather_results(models, datasets, metric="cross_acc", seeds=None, **kwargs):
        captured["models"] = models
        captured.update(kwargs)
        return sample_results_dicts()

    monkeypatch.setattr(friedman_test, "MODELS", ["lda", "eegnet", "dummy", "svm"])
    monkeypatch.setattr(Results, "gather_results", staticmethod(fake_gather_results))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "friedman_test.py",
            "--datasets",
            "buccino2016",
            "ortega2020",
            "luke2021",
            "herff2014",
            "--no-post-hoc",
        ],
    )

    friedman_test.main()
    capsys.readouterr()
    assert captured["models"] == ["lda", "eegnet", "dummy", "svm"]


def test_friedman_within_metric_infers_block_strategy(monkeypatch, capsys):
    """Test within metrics use within/block result paths."""
    from scripts import friedman_test

    captured = {}

    def fake_gather_results(models, datasets, metric="cross_acc", seeds=None, **kwargs):
        captured.update(kwargs)
        return sample_within_results_dicts()

    monkeypatch.setattr(Results, "gather_results", staticmethod(fake_gather_results))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "friedman_test.py",
            "--models",
            "lda",
            "eegnet",
            "dummy",
            "svm",
            "--datasets",
            "buccino2016",
            "ortega2020",
            "luke2021",
            "herff2014",
            "--metric",
            "within_acc",
        ],
    )

    friedman_test.main()
    output = capsys.readouterr().out
    assert "Friedman test on within_acc task means" in output
    assert captured["search_strategy"] == "block"
    assert captured["classification_setting"] == "within"
