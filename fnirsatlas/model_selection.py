import warnings
from collections.abc import Iterator
from importlib import import_module
from numbers import Integral, Real
from typing import Any
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import _num_samples, indexable


def create_splitter(spec: dict[str, Any], **overrides: Any) -> BaseCrossValidator:
    """Create a cross-validator from a fully qualified class and keyword arguments."""
    module_name, class_name = spec["class"].rsplit(".", 1)
    splitter_class = getattr(import_module(module_name), class_name)
    kwargs = {**spec.get("kwargs", {}), **overrides}
    return splitter_class(**kwargs)


class BlockedKFold(BaseCrossValidator):
    """K-fold cross-validator with contiguous test blocks.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=False
        Randomize fold boundaries. Not implemented: warns and keeps fixed
        contiguous folds.
    random_state : int or None, default=None
        Seed for randomized fold boundaries. Unused: warns when set.
    gap : int, default=0
        Samples excluded from training on each side of a test block.
    min_test_size : int, default=1
        Minimum samples required in each test block.
    class_balance_warning_threshold : float, default=0.2
        Warn when a train or test fold misses a class and its total variation
        distance from the full class distribution reaches this threshold.
    past_only : bool, default=False
        Train only on samples preceding the test block. Not yet implemented.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
        gap: int = 0,
        min_test_size: int = 1,
        class_balance_warning_threshold: float = 0.2,
        past_only: bool = False,
    ) -> None:
        if not isinstance(n_splits, Integral):
            raise ValueError(
                f"n_splits must be an integer, got {type(n_splits).__name__}"
            )
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        if not isinstance(min_test_size, Integral) or min_test_size < 1:
            raise ValueError(
                f"min_test_size must be a positive integer, got {min_test_size}"
            )
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        if gap < 0:
            raise ValueError(f"gap must be non-negative, got {gap}")
        self.gap = gap
        self.min_test_size = int(min_test_size)
        if not isinstance(class_balance_warning_threshold, Real) or isinstance(
            class_balance_warning_threshold, bool
        ):
            raise ValueError("class_balance_warning_threshold must be a number")
        if not 0 <= class_balance_warning_threshold <= 1:
            raise ValueError("class_balance_warning_threshold must be between 0 and 1")
        self.class_balance_warning_threshold = float(class_balance_warning_threshold)
        self.past_only = past_only
        if shuffle:
            warnings.warn(
                "BlockedKFold shuffle is not implemented; using fixed contiguous folds",
                UserWarning,
                stacklevel=2,
            )
        if random_state is not None:
            warnings.warn(
                f"BlockedKFold ignores random_state={random_state}; folds are fixed",
                UserWarning,
                stacklevel=2,
            )
        if past_only:
            raise NotImplementedError("past-only training is not implemented")

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return the number of folds."""
        return self.n_splits

    def split(
        self, X: Any, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        """Generate train and contiguous test indices."""
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                f"n_splits={self.n_splits} cannot exceed n_samples={n_samples}"
            )
        if n_samples // self.n_splits < self.min_test_size:
            raise ValueError(
                f"min_test_size={self.min_test_size} cannot be met with "
                f"n_samples={n_samples} and n_splits={self.n_splits}"
            )
        # pre-compute once for class-balance checks
        classes = np.array([])
        full_proportions = np.array([])
        if y is not None:
            y = np.asarray(y)
            classes, full_counts = np.unique(y, return_counts=True)
            full_proportions = full_counts / len(y)
        indices = np.arange(n_samples)
        for fold, (start, stop) in enumerate(
            self._iter_fold_boundaries(n_samples, y), start=1
        ):
            train = self._get_train_indices(indices, start, stop)
            test = indices[start:stop]
            self._warn_class_imbalance(y, train, test, fold, classes, full_proportions)
            yield train, test

    def _iter_fold_boundaries(
        self, n_samples: int, y: Any = None
    ) -> Iterator[tuple[int, int]]:
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        start = 0
        for fold_size in fold_sizes:
            stop = start + int(fold_size)
            yield start, stop
            start = stop

    # Kept separate to support gaps and past-only training later.
    def _get_train_indices(
        self, indices: NDArray[np.int_], test_start: int, test_stop: int
    ) -> NDArray[np.int_]:
        before_stop = max(0, test_start - self.gap)
        after_start = min(len(indices), test_stop + self.gap)
        return np.concatenate((indices[:before_stop], indices[after_start:]))

    def _check_class_imbalance(
        self,
        y: NDArray[Any] | None,
        subset: NDArray[np.int_],
        classes: NDArray[Any],
        full_proportions: NDArray[np.float64],
    ) -> tuple[NDArray[Any], float] | None:
        counts = np.array([np.count_nonzero(y[subset] == cls) for cls in classes])
        missing = classes[counts == 0]
        if not missing.size:
            return None
        proportions = counts / len(subset)
        tv = np.abs(proportions - full_proportions).sum() / 2
        if tv < self.class_balance_warning_threshold:
            return None
        return proportions, float(tv)

    def _warn_class_imbalance(
        self,
        y: NDArray[Any],
        train: NDArray[np.int_],
        test: NDArray[np.int_],
        fold: int,
        classes: NDArray[Any],
        full_proportions: NDArray[np.float64],
    ) -> None:
        if y is None:
            return
        for name, subset in (("Train", train), ("Test", test)):
            imbalance = self._check_class_imbalance(
                y, subset, classes, full_proportions
            )
            if imbalance is not None:
                proportions, tv = imbalance
                distribution = dict(
                    zip(classes.tolist(), proportions.round(3).tolist())
                )
                warnings.warn(
                    f"{name} fold {fold} class distribution {distribution}; TV={tv:.3f}",
                    UserWarning,
                    stacklevel=3,
                )
