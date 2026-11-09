"""Adapter making aeon classifiers usable inside MOABB evaluations."""

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted


class AeonClassifier(ClassifierMixin, BaseEstimator):
    """Delegate classification to an aeon estimator.

    aeon estimators raise ``NotImplementedError`` from ``get_metadata_routing``,
    which scikit-learn calls for every pipeline step while MOABB fits with
    metadata routing enabled. Wrapping restores the scikit-learn default.

    Parameters
    ----------
    estimator : aeon classifier
        Unfitted estimator. A clone is fitted, leaving the input untouched.

    Notes
    -----
    Input must be shaped ``(trials, channels, timepoints)``.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        """Fit a clone of the wrapped estimator."""
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        """Return predicted labels."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        """Return class probabilities."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    def __sklearn_tags__(self):
        """Declare the three-dimensional input expected by aeon estimators."""
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        tags.input_tags.three_d_array = True
        return tags
