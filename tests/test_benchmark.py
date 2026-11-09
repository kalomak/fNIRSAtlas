from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold

from fnirsatlas import benchmark


class _FakeParadigm:
    def __init__(self, **kwargs):
        self.datasets = [SimpleNamespace(code="FakeFnirs")]
        self.scorer = kwargs.get("scorer")


def _config(name, estimator, **extra):
    return {
        "name": name,
        "paradigms": ["FNIRSClassification"],
        "pipeline": estimator,
        **extra,
    }


@pytest.fixture
def fake_benchmark(monkeypatch, tmp_path):
    """Run ``fnirs_benchmark`` against a stand-in paradigm and evaluation.

    ``run`` forwards to ``fnirs_benchmark``, ``calls`` collects the keyword
    arguments each evaluation was constructed with, and ``on_process`` is an
    optional hook called with the pipelines handed to every evaluation.
    """
    runner = SimpleNamespace(calls=[], on_process=None)

    class FakeEvaluation:
        def __init__(self, **kwargs):
            runner.calls.append(kwargs)

        def process(self, pipelines, param_grid):
            if runner.on_process is not None:
                runner.on_process(pipelines, param_grid)
            return pd.DataFrame(
                [
                    {"dataset": "FakeFnirs", "pipeline": name, "score": 0.5}
                    for name in pipelines
                ]
            )

    monkeypatch.setattr(benchmark, "PARADIGMS", {"FNIRSClassification": _FakeParadigm})
    monkeypatch.setattr(benchmark, "analyze", lambda *args, **kwargs: None)

    def run(pipelines, evaluations=("WithinSession",), **kwargs):
        monkeypatch.setattr(
            benchmark, "EVALUATIONS", dict.fromkeys(evaluations, FakeEvaluation)
        )
        return benchmark.fnirs_benchmark(
            pipelines=pipelines,
            evaluations=list(evaluations),
            results=tmp_path / "results",
            output=tmp_path / "output",
            **kwargs,
        )

    runner.run = run
    return runner


def test_aggregate_seed_results_averages_matching_rows():
    first = pd.DataFrame(
        [
            {
                "dataset": "Fake",
                "subject": "1",
                "session": "0",
                "pipeline": "LDA",
                "paradigm": "MotorImagery",
                "evaluation": "WithinSession",
                "score": 0.6,
                "time": 2.0,
            }
        ]
    )
    second = first.assign(score=0.8, time=4.0)

    result = benchmark.aggregate_seed_results([first, second])

    assert result.loc[0, "score"] == 0.7
    assert result.loc[0, "time"] == 3.0


def test_fnirs_benchmark_forwards_random_state(fake_benchmark):
    fake_benchmark.run([_config("Dummy", DummyClassifier())], random_state=7)
    (call,) = fake_benchmark.calls
    assert call["random_state"] == 7


def _group_folds(cv_kwargs):
    subjects = np.repeat(np.arange(12), 4)
    splitter = GroupKFold(**{**cv_kwargs, "n_splits": 4})
    return [
        tuple(np.unique(subjects[test]))
        for _, test in splitter.split(subjects, groups=subjects)
    ]


def _stratified_folds(cv_kwargs):
    labels = np.tile([0, 1], 24)
    splitter = StratifiedKFold(**{**cv_kwargs, "n_splits": 4})
    return [tuple(test) for _, test in splitter.split(labels, labels)]


@pytest.mark.parametrize(
    ("evaluation", "split"),
    [("CrossSubject", _group_folds), ("WithinSession", _stratified_folds)],
)
def test_benchmark_seed_shuffles_folds(evaluation, split, fake_benchmark):
    def folds(random_state):
        fake_benchmark.run(
            [_config("Dummy", DummyClassifier())],
            evaluations=(evaluation,),
            random_state=random_state,
        )
        return split(fake_benchmark.calls[-1]["cv_kwargs"])

    assert folds(7) == folds(7)
    assert folds(7) != folds(8)


def test_benchmark_seed_reaches_pipeline_estimators(fake_benchmark):
    fitted = []
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = np.tile([0, 1], 20)

    def fit(pipelines, param_grid):
        fitted.extend(
            tuple(pipeline.fit(X, y).predict(X)) for pipeline in pipelines.values()
        )

    fake_benchmark.on_process = fit

    def predictions(random_state):
        fitted.clear()
        fake_benchmark.run(
            [_config("Forest", RandomForestClassifier(n_estimators=5))],
            random_state=random_state,
        )
        return fitted[0]

    assert predictions(7) == predictions(7)
    assert predictions(7) != predictions(8)


def test_no_grid_suppresses_declared_param_grids(fake_benchmark):
    """``no_grid`` hands evaluations an empty grid, so pipelines fit as declared."""
    seen = []
    fake_benchmark.on_process = lambda pipelines, param_grid: seen.append(param_grid)

    fake_benchmark.run(
        [_config("Dummy", DummyClassifier(), param_grid={"strategy": ["prior"]})],
        no_grid=True,
    )

    assert seen == [{}]


@pytest.fixture
def selection():
    """Paradigm with defaults ``A`` and ``B`` that accepts MI datasets only."""
    a, b, c = (SimpleNamespace(code=code, paradigm="mi") for code in "ABC")
    other = SimpleNamespace(code="Other", paradigm="me")
    paradigm = SimpleNamespace(
        datasets=[a, b], is_valid=lambda dataset: dataset.paradigm == "mi"
    )
    return SimpleNamespace(paradigm=paradigm, a=a, b=b, c=c, other=other)


def test_included_datasets_outside_the_defaults_are_selected(selection):
    selected = benchmark.select_datasets(selection.paradigm, [selection.c])
    assert selected == [selection.c]


def test_included_datasets_of_another_paradigm_are_skipped(selection):
    assert benchmark.select_datasets(selection.paradigm, [selection.other]) == []


def test_default_selection_leaves_out_excluded_datasets(selection):
    selected = benchmark.select_datasets(selection.paradigm, exclude=["A"])
    assert selected == [selection.b]
