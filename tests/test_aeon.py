import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

import fnirsatlas.classification as classification
from fnirsatlas.classification import classify_cross, get_baseline_model
from fnirsatlas.config import get_param_grid


pytest.importorskip("aeon")


def _make_dummy_data(n_samples=12, n_chans=3, n_times=32, n_subjects=4):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, n_chans, n_times)).astype(np.float32)
    y = np.tile([0, 1], n_samples // 2)
    groups = np.repeat(np.arange(n_subjects), n_samples // n_subjects)
    info = {"sfreq": 5.0, "ch_names": [f"ch{i}" for i in range(n_chans)]}
    return X, y, groups, info


@pytest.mark.parametrize(
    ("model_name", "model_kwargs"),
    [
        ("aeon_minirocket", {"n_kernels": 8, "n_jobs": 1}),
        ("aeon_multirocket", {"n_kernels": 84, "n_jobs": 1}),
        ("aeon_multirockethydra", {"n_kernels": 8, "n_jobs": 1}),
    ],
)
def test_aeon_model_fit_predict(model_name, model_kwargs):
    X, y, _, info = _make_dummy_data()
    model = get_baseline_model(model_name, X=X, y=y, info=info, **model_kwargs)
    assert isinstance(model, Pipeline)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_aeon_multirockethydra_param_grid_forward_passes():
    grid = get_param_grid("aeon_multirockethydra")
    assert grid, "No param grid found for aeon_multirockethydra"
    X, y, _, info = _make_dummy_data()
    base_model = get_baseline_model(
        "aeon_multirockethydra", X=X, y=y, info=info, n_jobs=1
    )

    for params in ParameterGrid(grid):
        model = clone(base_model)
        model.set_params(**params)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert model.get_params()["clf__n_kernels"] == params["clf__n_kernels"]
        assert model.get_params()["clf__n_groups"] == params["clf__n_groups"]


def test_aeon_multirockethydra_grid_search_applies_params(monkeypatch):
    X, y, groups, info = _make_dummy_data()
    seen_params = []

    class FakeGridSearchCV:
        def __init__(self, estimator, param_grid, scoring, cv, n_jobs, refit, verbose):
            self.estimator = estimator
            self.param_grid = param_grid

        def fit(self, X, y, groups=None):
            for params in ParameterGrid(self.param_grid):
                candidate = clone(self.estimator)
                candidate.set_params(**params)
                candidate.fit(X, y)
                preds = candidate.predict(X)
                assert preds.shape == (len(X),)
                seen_params.append(
                    {
                        "n_kernels": candidate.get_params()["clf__n_kernels"],
                        "n_groups": candidate.get_params()["clf__n_groups"],
                    }
                )
            self.best_estimator_ = candidate
            self.cv_results_ = {"params": list(ParameterGrid(self.param_grid))}
            return self

        def predict(self, X):
            return self.best_estimator_.predict(X)

    monkeypatch.setattr(classification, "GridSearchCV", FakeGridSearchCV)

    classify_cross(
        X,
        y,
        groups,
        get_clf=lambda **kwargs: get_baseline_model("aeon_multirockethydra", **kwargs),
        seed=0,
        clf_kwargs={"X": X, "y": y, "info": info, "n_jobs": 1},
        param_grid={"clf__n_kernels": [8, 16], "clf__n_groups": [32, 64]},
    )

    expected = [
        {"n_kernels": 8, "n_groups": 32},
        {"n_kernels": 8, "n_groups": 64},
        {"n_kernels": 16, "n_groups": 32},
        {"n_kernels": 16, "n_groups": 64},
    ]
    assert len(seen_params) == len(expected) * 4
    for params in expected:
        assert seen_params.count(params) == 4
