import warnings

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from fnirsatlas.model_selection import BlockedKFold, create_splitter


def test_create_splitter_imports_configured_class_and_applies_overrides():
    spec = {
        "class": "sklearn.model_selection.StratifiedKFold",
        "kwargs": {"n_splits": 5, "shuffle": True, "random_state": None},
    }

    splitter = create_splitter(spec, n_splits=3, random_state=7)

    assert isinstance(splitter, StratifiedKFold)
    assert splitter.n_splits == 3
    assert splitter.random_state == 7


def test_blocked_kfold_uses_contiguous_test_blocks():
    X = np.zeros((11, 2))
    splits = list(BlockedKFold(n_splits=3).split(X))
    expected_tests = [np.arange(0, 4), np.arange(4, 8), np.arange(8, 11)]
    assert len(splits) == 3
    for (train, test), expected_test in zip(splits, expected_tests):
        np.testing.assert_array_equal(test, expected_test)
        np.testing.assert_array_equal(train, np.delete(np.arange(11), expected_test))


def test_blocked_kfold_excludes_gap_around_test_blocks():
    splits = list(BlockedKFold(n_splits=3, gap=1).split(np.zeros((9, 1))))
    expected_trains = [np.arange(4, 9), np.array([0, 1, 7, 8]), np.arange(5)]
    for (train, _), expected_train in zip(splits, expected_trains):
        np.testing.assert_array_equal(train, expected_train)


def test_blocked_kfold_warns_when_test_fold_misses_class():
    y = np.array([0, 0, 0, 1, 1, 1])
    cv = BlockedKFold(n_splits=3)
    with pytest.warns(UserWarning, match=r"class distribution.*TV="):
        list(cv.split(np.zeros((len(y), 1)), y))


def test_blocked_kfold_warns_when_train_fold_misses_class():
    y = np.array([0, 0, 1, 1])
    with pytest.warns(UserWarning) as warnings:
        list(BlockedKFold(n_splits=2).split(np.zeros((len(y), 1)), y))

    assert any(str(w.message).startswith("Train fold") for w in warnings)


@pytest.mark.parametrize("min_test_size", [0, 1.5])
def test_blocked_kfold_rejects_invalid_min_test_size(min_test_size):
    with pytest.raises(ValueError):
        BlockedKFold(min_test_size=min_test_size)


def test_blocked_kfold_enforces_min_test_size():
    cv = BlockedKFold(n_splits=3, min_test_size=3)

    with pytest.raises(ValueError, match="min_test_size=3"):
        list(cv.split(np.zeros((8, 1))))


@pytest.mark.parametrize(
    "kwargs, message",
    [({"shuffle": True}, "shuffle"), ({"random_state": 7}, "random_state")],
)
def test_blocked_kfold_warns_and_keeps_contiguous_folds_for_unsupported_args(
    kwargs, message
):
    with pytest.warns(UserWarning, match=message):
        cv = BlockedKFold(n_splits=3, **kwargs)

    tests = [test for _, test in cv.split(np.zeros((9, 1)))]
    for test, expected in zip(tests, np.split(np.arange(9), 3)):
        np.testing.assert_array_equal(test, expected)
