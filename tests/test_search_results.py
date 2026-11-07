import numpy as np
import pytest
from pathlib import Path
from fnirsatlas.search_results import GridSearchResults, load_search_results


def test_grid_search_results_loading(tmp_path):
    cv_results = {
        "params": [
            {"n_estimators": 10, "max_depth": 1},
            {"n_estimators": 10, "max_depth": 2},
            {"n_estimators": 20, "max_depth": 1},
            {"n_estimators": 20, "max_depth": 2},
        ],
        "mean_test_accuracy": np.array([0.8, 0.85, 0.82, 0.88]),
        "std_test_accuracy": np.array([0.1, 0.1, 0.1, 0.1]),
        "rank_test_accuracy": np.array([4, 2, 3, 1]),
        "split0_test_accuracy": np.array([0.7, 0.8, 0.75, 0.85]),
        "split1_test_accuracy": np.array([0.9, 0.9, 0.89, 0.91]),
        "split2_test_accuracy": np.array([0.8, 0.85, 0.82, 0.88]),
        "mean_test_kappa": np.array([0.6, 0.7, 0.64, 0.76]),
        "std_test_kappa": np.array([0.1, 0.1, 0.1, 0.1]),
    }
    
    npz_path = tmp_path / "test_0.npz"
    np.savez(npz_path, **cv_results)
    
    gs = GridSearchResults([npz_path], metric="accuracy")
    
    assert not gs.data.empty
    assert "param_n_estimators" in gs.data.columns
    assert "param_max_depth" in gs.data.columns
    assert "mean_score" in gs.data.columns
    assert "rank" in gs.data.columns


def test_grid_search_results_with_list_params(tmp_path):
    cv_results = {
        "params": [
            {"features": ["mean"], "n_windows": 3},
            {"features": ["mean", "slope"], "n_windows": 3},
            {"features": ["mean"], "n_windows": 6},
            {"features": ["mean", "slope"], "n_windows": 6},
        ],
        "mean_test_accuracy": np.array([0.8, 0.85, 0.82, 0.88]),
        "std_test_accuracy": np.array([0.1, 0.1, 0.1, 0.1]),
        "rank_test_accuracy": np.array([4, 2, 3, 1]),
        "split0_test_accuracy": np.array([0.7, 0.8, 0.75, 0.85]),
        "split1_test_accuracy": np.array([0.9, 0.9, 0.89, 0.91]),
    }
    
    npz_path = tmp_path / "test_0.npz"
    np.savez(npz_path, **cv_results)
    
    gs = GridSearchResults([npz_path], metric="accuracy")
    
    assert not gs.data.empty
    assert "param_features" in gs.data.columns
    assert "param_n_windows" in gs.data.columns
    assert gs.data["param_features"].iloc[0] == ("mean",)


def test_grid_search_results_pivot_table(tmp_path):
    cv_results = {
        "params": [
            {"n_estimators": 10, "max_depth": 1},
            {"n_estimators": 10, "max_depth": 2},
            {"n_estimators": 20, "max_depth": 1},
            {"n_estimators": 20, "max_depth": 2},
        ],
        "mean_test_accuracy": np.array([0.8, 0.85, 0.82, 0.88]),
        "std_test_accuracy": np.array([0.1, 0.1, 0.1, 0.1]),
        "rank_test_accuracy": np.array([4, 2, 3, 1]),
        "split0_test_accuracy": np.array([0.7, 0.8, 0.75, 0.85]),
        "split1_test_accuracy": np.array([0.9, 0.9, 0.89, 0.91]),
    }
    
    npz_path = tmp_path / "test_0.npz"
    np.savez(npz_path, **cv_results)
    
    gs = GridSearchResults([npz_path], metric="accuracy")
    gs.data["dataset"] = "test_ds"
    
    pivot = gs.get_pivot_table()
    
    assert not pivot.empty
    assert pivot.shape[0] == 4


def test_grid_search_results_best_params(tmp_path):
    cv_results = {
        "params": [
            {"n_estimators": 10, "max_depth": 1},
            {"n_estimators": 10, "max_depth": 2},
            {"n_estimators": 20, "max_depth": 1},
            {"n_estimators": 20, "max_depth": 2},
        ],
        "mean_test_accuracy": np.array([0.8, 0.85, 0.82, 0.88]),
        "std_test_accuracy": np.array([0.1, 0.1, 0.1, 0.1]),
        "rank_test_accuracy": np.array([4, 2, 3, 1]),
        "split0_test_accuracy": np.array([0.7, 0.8, 0.75, 0.85]),
        "split1_test_accuracy": np.array([0.9, 0.9, 0.89, 0.91]),
    }
    
    npz_path = tmp_path / "test_0.npz"
    np.savez(npz_path, **cv_results)
    
    gs = GridSearchResults([npz_path], metric="accuracy")
    gs.data["dataset"] = "test_ds"
    
    best = gs.get_best_params()
    
    assert "test_ds" in best
    assert best["test_ds"]["n_estimators"] == 20
    assert best["test_ds"]["max_depth"] == 2


def test_load_search_results(tmp_path):
    cv_results = {
        "params": [{"n_estimators": 10}],
        "mean_test_accuracy": np.array([0.8]),
        "std_test_accuracy": np.array([0.1]),
        "rank_test_accuracy": np.array([1]),
        "split0_test_accuracy": np.array([0.8]),
    }
    
    np.savez(tmp_path / "test_0.npz", **cv_results)
    
    gs = load_search_results(tmp_path, "test_*.npz", metric="accuracy")
    
    assert not gs.data.empty
