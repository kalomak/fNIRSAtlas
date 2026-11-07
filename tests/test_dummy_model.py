#!/usr/bin/env python3
"""
Test for dummy model functionality.
"""

import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from fnirsatlas.feature_extraction import WindowedFeatureExtractor
from scripts.dummy_model_benchmark import get_dummy_model


def test_dummy_classifier():
    """Test the dummy classifier functionality."""
    # Create dummy classifier
    clf = DummyClassifier(random_state=42)

    # Create dummy data
    X = np.random.randn(10, 5)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Fit the classifier
    clf.fit(X, y)

    # Test predictions
    predictions = clf.predict(X)
    probas = clf.predict_proba(X)

    # Check that predictions are valid
    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)

    # Check that probabilities sum to 1
    assert probas.shape == (10, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_dummy_model_pipeline():
    """Test the dummy model pipeline."""
    # Create dummy data
    X = np.random.randn(10, 6, 100)  # (n_samples, n_channels, n_timepoints)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Create dummy model
    model = get_dummy_model(random_state=42, X=X, y=y)

    # Test that it's a pipeline
    assert isinstance(model, Pipeline)

    # Test that it can fit and predict
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)


def test_dummy_model_with_kwargs():
    """Test that dummy model accepts kwargs properly."""
    # Test with minimal kwargs
    model = get_dummy_model(random_state=42)
    assert isinstance(model, Pipeline)

    # Test that random_state is passed through
    clf = model.named_steps["clf"]
    assert clf.random_state == 42
    assert clf.strategy == "prior"  # Default strategy


if __name__ == "__main__":
    pytest.main([__file__])
