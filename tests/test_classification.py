import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import fnirsatlas.classification as classification
from fnirsatlas.classification import (
    classify_cross,
    classify_within,
    classify_task,
    classify_dataset,
    get_baseline_model,
)
from fnirsatlas.config import get_param_grid


class FeatureClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_idx=0):
        self.feature_idx = feature_idx

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (X[:, self.feature_idx] > 0).astype(int)


def get_feature_clf(**kwargs):
    return FeatureClassifier(feature_idx=kwargs.get("feature_idx", 0))


def make_within_data(n_subjects=4):
    X, y, groups = [], [], []
    for subj in range(n_subjects):
        for _ in range(4):
            X.extend([[1.0, -1.0], [-1.0, 1.0]])
            y.extend([1, 0])
            groups.extend([subj, subj])
    return np.array(X), np.array(y), np.array(groups)


@pytest.mark.slow
def test_classify_task():
    """Test the classify_task function with task object."""
    # Test with shin2019__ma dataset and a specific task
    task = {
        "description": "MA vs Rest",
        "mapping": {"Idle State": 0, "Mental Arithmetic": 1},
    }

    # Test cross-subject classification
    results = classify_task(
        "shin2019__ma",
        task,
        classification_setting="cross",
        model_name="lda",
    )

    assert "cross_acc" in results
    assert "cross_kappa" in results
    assert "cross_f1_macro" in results
    assert "cross_precision" in results
    assert "cross_recall" in results
    assert "n_classes" in results
    assert "n_epochs" in results
    assert "mapping" in results
    assert results["n_classes"] == 2
    assert len(results["cross_acc"]) > 0
    assert len(results["cross_kappa"]) > 0
    assert len(results["cross_f1_macro"]) > 0
    assert len(results["cross_precision"]) > 0
    assert len(results["cross_recall"]) > 0


@pytest.mark.slow
def test_classify_task_custom_classifier():
    """Test classify_task with custom classifier function."""
    task = {
        "description": "MA vs Rest",
        "mapping": {"Idle State": 0, "Mental Arithmetic": 1},
    }

    results = classify_task(
        "shin2019__ma",
        task,
        classification_setting="cross",
        model_name="lda",
        get_clf=get_baseline_model,
    )

    assert "cross_acc" in results
    assert "cross_kappa" in results
    assert "cross_f1_macro" in results
    assert "cross_precision" in results
    assert "cross_recall" in results
    assert "n_classes" in results
    assert "n_epochs" in results
    assert "mapping" in results
    assert results["n_classes"] == 2
    assert len(results["cross_acc"]) > 0
    assert len(results["cross_kappa"]) > 0
    assert len(results["cross_f1_macro"]) > 0
    assert len(results["cross_precision"]) > 0
    assert len(results["cross_recall"]) > 0


@pytest.mark.slow
def test_classify_dataset():
    """Test the classify_dataset function which handles task preparation."""
    # Test with shin2019__ma dataset which has defined tasks
    results = classify_dataset(
        "shin2019__ma", classification_setting="cross", model_name="lda"
    )

    # Check that results are returned for each task
    assert isinstance(results, dict)
    assert len(results) > 0

    # Check structure of results
    for task_desc, task_results in results.items():
        assert "n_classes" in task_results
        assert "n_epochs" in task_results
        assert "mapping" in task_results
        assert "cross_acc" in task_results
        assert "cross_kappa" in task_results
        assert "cross_f1_macro" in task_results
        assert "cross_precision" in task_results
        assert "cross_recall" in task_results


def test_get_baseline_model_ann():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 6, 20))
    y = np.array([0, 1] * 6)

    model = get_baseline_model("ann", random_state=7)

    assert isinstance(model, Pipeline)
    assert model.named_steps["extractor"].n_windows == 1
    assert model.named_steps["extractor"].features == ["mean", "slope"]
    assert model.named_steps["clf"].hidden_layer_sizes == (20,)
    assert model.named_steps["clf"].random_state == 7

    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


@pytest.mark.parametrize(
    ("model_name", "classifier_type"),
    [
        ("logreg", LogisticRegression),
        ("svm", SVC),
        ("rf", RandomForestClassifier),
        ("knn", KNeighborsClassifier),
    ],
)
def test_get_classical_baseline_model(model_name, classifier_type):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(18, 4, 20))
    y = np.tile([0, 1, 2], 6)

    model = get_baseline_model(model_name, random_state=7)

    assert isinstance(model, Pipeline)
    assert model.named_steps["extractor"].features == ["mean"]
    assert isinstance(model.named_steps["clf"], classifier_type)
    if model_name != "knn":
        assert model.named_steps["clf"].random_state == 7
    if model_name == "logreg":
        assert model.named_steps["clf"].max_iter == 100_000
        assert model.named_steps["clf"].solver == "saga"
    elif model_name == "svm":
        assert model.named_steps["clf"].max_iter == -1
        assert model.named_steps["clf"].kernel == "rbf"
        assert model.named_steps["clf"].gamma == "scale"
    elif model_name == "rf":
        assert model.named_steps["clf"].n_jobs == 1
    elif model_name == "knn":
        assert model.named_steps["clf"].n_jobs == 1

    grid = get_param_grid(model_name)
    if model_name == "logreg":
        assert grid["clf__penalty"] == ["l1", "l2"]
        assert grid["clf__C"] == [1e-5, 1e-2, 1, 1e2]
    elif model_name == "rf":
        assert grid["clf__n_estimators"] == [10, 50, 100, 1000]
        assert grid["clf__min_samples_leaf"] == [1, 2]
    elif model_name == "knn":
        assert grid["clf__n_neighbors"] == [1, 3, 5, 7]
        assert grid["clf__weights"] == ["uniform", "distance"]
    candidates = list(ParameterGrid(grid))
    assert len(candidates) == 16
    for params in candidates:
        model.set_params(**params)

    model.fit(X, y)
    assert set(model.named_steps["clf"].classes_) == {0, 1, 2}
    assert set(model.predict(X)).issubset({0, 1, 2})


def test_classify_cross_uses_group_aware_inner_cv(monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 6))
    y = np.tile([0, 1, 0, 1], 4)
    groups = np.repeat(np.array(["s1", "s2", "s3", "s4"]), 4)
    calls = []

    class FakeGridSearchCV:
        def __init__(self, estimator, param_grid, scoring, cv, n_jobs, refit, verbose):
            self.estimator = estimator
            self.cv = cv

        def fit(self, X, y, groups=None):
            calls.append(
                {
                    "cv": self.cv,
                    "groups": None if groups is None else np.array(groups, copy=True),
                }
            )
            self.estimator.fit(X, y)
            self.cv_results_ = {}
            return self

        def predict(self, X):
            return self.estimator.predict(X)

    def get_clf(**kwargs):
        return LogisticRegression(random_state=kwargs["random_state"], max_iter=1000)

    monkeypatch.setattr(classification, "GridSearchCV", FakeGridSearchCV)

    classification.classify_cross(
        X,
        y,
        groups,
        get_clf=get_clf,
        seed=0,
        clf_kwargs={"param_grid": {"C": [0.1, 1.0]}},
    )

    assert calls
    assert all(isinstance(call["cv"], GroupKFold) for call in calls)
    assert all(call["groups"] is not None for call in calls)
    assert {len(np.unique(call["groups"])) for call in calls} == {3}


def test_classify_cross_single_uses_group_shuffle_split(monkeypatch):
    rng = np.random.default_rng(0)
    n_subjects = 5
    X = rng.normal(size=(n_subjects * 4, 6))
    y = np.tile([0, 1, 0, 1], n_subjects)
    groups = np.repeat(np.arange(n_subjects), 4)
    calls = []

    class FakeGridSearchCV:
        def __init__(self, estimator, param_grid, scoring, cv, n_jobs, refit, verbose):
            self.estimator = estimator
            self.cv = cv

        def fit(self, X, y, groups=None):
            calls.append({"cv": self.cv, "groups": groups})
            self.estimator.fit(X, y)
            self.cv_results_ = {}
            return self

        def predict(self, X):
            return self.estimator.predict(X)

    def get_clf(**kwargs):
        return LogisticRegression(random_state=kwargs["random_state"], max_iter=1000)

    monkeypatch.setattr(classification, "GridSearchCV", FakeGridSearchCV)

    classification.classify_cross(
        X,
        y,
        groups,
        get_clf=get_clf,
        seed=0,
        search_strategy="single",
        param_grid={"C": [0.1, 1.0]},
    )

    assert calls
    assert all(isinstance(call["cv"], GroupShuffleSplit) for call in calls)
    assert all(call["groups"] is not None for call in calls)


def test_classify_within_preserves_split_scores():
    X, y, groups = make_within_data()

    results = classify_within(X, y, groups, get_clf=get_feature_clf, seed=0)

    assert len(results["within_acc"]) == len(np.unique(groups))
    assert all(isinstance(scores, list) for scores in results["within_acc"])
    assert all(len(scores) == 4 for scores in results["within_acc"])


def test_classify_default_metrics_on_synthetic_data():
    X, y, groups = make_within_data()

    cross_results = classify_cross(X, y, groups, get_clf=get_feature_clf, seed=0)
    within_results = classify_within(X, y, groups, get_clf=get_feature_clf, seed=0)

    for metric in ["cross_f1_macro", "cross_precision", "cross_recall"]:
        assert metric in cross_results
        assert len(cross_results[metric]) > 0

    for metric in ["within_f1_macro", "within_precision", "within_recall"]:
        assert metric in within_results
        assert len(within_results[metric]) == len(np.unique(groups))


def test_classify_within_block_selects_from_other_subjects():
    X, y, groups = make_within_data()

    results = classify_within(
        X,
        y,
        groups,
        get_clf=get_feature_clf,
        seed=0,
        clf_kwargs={
            "param_grid": {"feature_idx": [0, 1]},
            "search_strategy": "block",
        },
    )

    assert all(
        all(score == 1.0 for score in scores) for scores in results["within_acc"]
    )
    assert results["search"]["selected_param_ix_by_subject"] == [0, 0, 0, 0]
    assert len(results["search"]["candidate_accuracy"]) == 4
    assert len(results["search"]["candidate_accuracy"][0]) == 2
    assert len(results["search"]["candidate_accuracy"][0][0]) == 4


def test_within_cv_disables_tqdm(monkeypatch):
    X, y, groups = make_within_data()

    def fail_tqdm(*args, **kwargs):
        raise AssertionError("tqdm should not be used for within-subject CV")

    monkeypatch.setattr(classification, "tqdm", fail_tqdm)
    monkeypatch.setattr(classification.logger, "isEnabledFor", lambda level: True)

    classify_within(X, y, groups, get_clf=get_feature_clf, seed=0)


def test_cross_block_search_rejected():
    X, y, groups = make_within_data()

    with pytest.raises(ValueError, match="not supported"):
        classify_cross(
            X,
            y,
            groups,
            get_clf=get_feature_clf,
            seed=0,
            clf_kwargs={
                "param_grid": {"feature_idx": [0, 1]},
                "search_strategy": "block",
            },
        )
