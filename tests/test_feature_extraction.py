import numpy as np
import pytest
from fnirsatlas.feature_extraction import WindowedFeatureExtractor


def test_windowed_feature_extractor_single_feature():
    """Test WindowedFeatureExtractor with a single feature."""
    # Create test data: 2 samples, 3 channels, 12 time points
    X = np.random.randn(2, 3, 12)

    extractor = WindowedFeatureExtractor(n_windows=3, features=["mean"])
    X_transformed = extractor.transform(X)

    # Should have 2 samples, 9 features (3 windows * 3 channels * 1 feature)
    assert X_transformed.shape == (2, 9)


def test_windowed_feature_extractor_multiple_features():
    """Test WindowedFeatureExtractor with multiple features."""
    # Create test data: 2 samples, 3 channels, 12 time points
    X = np.random.randn(2, 3, 12)

    extractor = WindowedFeatureExtractor(n_windows=3, features=["mean", "std", "slope"])
    X_transformed = extractor.transform(X)

    # Should have 2 samples, 27 features (3 windows * 3 channels * 3 features)
    assert X_transformed.shape == (2, 27)


def test_windowed_feature_extractor_slope_calculation():
    """Test that slope calculation works correctly."""
    # Create test data with known linear trend
    n_samples, n_ch, n_time = 1, 1, 6
    X = np.zeros((n_samples, n_ch, n_time))
    # Linear increase from 0 to 5
    X[0, 0, :] = np.arange(n_time)

    extractor = WindowedFeatureExtractor(n_windows=2, features=["slope"])
    X_transformed = extractor.transform(X)

    # Should have 2 features (2 windows * 1 channel * 1 feature)
    assert X_transformed.shape == (1, 2)
    # Both windows should have slope of 1 (linear increase)
    np.testing.assert_allclose(X_transformed[0], [1.0, 1.0], rtol=1e-5)


def test_windowed_feature_extractor_edge_case_single_timepoint():
    """Test when window has only one time point (slope should be 0)."""
    # Create test data: 1 sample, 1 channel, 2 time points
    X = np.random.randn(1, 1, 2)

    # 2 windows with 1 time point each
    extractor = WindowedFeatureExtractor(n_windows=2, features=["slope"])
    X_transformed = extractor.transform(X)

    # Slope should be 0 when there's only 1 time point per window
    assert X_transformed.shape == (1, 2)
    np.testing.assert_allclose(X_transformed[0], [0.0, 0.0])


def test_windowed_feature_extractor_feature_names():
    """Test feature name generation."""
    extractor = WindowedFeatureExtractor(n_windows=2, features=["mean", "std"])
    feature_names = extractor.get_feature_names(n_ch=3)

    expected_names = [
        "win1_mean_ch1",
        "win1_mean_ch2",
        "win1_mean_ch3",
        "win1_std_ch1",
        "win1_std_ch2",
        "win1_std_ch3",
        "win2_mean_ch1",
        "win2_mean_ch2",
        "win2_mean_ch3",
        "win2_std_ch1",
        "win2_std_ch2",
        "win2_std_ch3",
    ]

    assert feature_names == expected_names


def test_windowed_feature_extractor_fit_transform():
    """Test that fit and transform work correctly together."""
    X = np.random.randn(5, 4, 20)

    extractor = WindowedFeatureExtractor(n_windows=4, features=["mean", "std", "slope"])

    # Fit should return self
    fitted = extractor.fit(X)
    assert fitted is extractor

    # Transform should work
    X_transformed = extractor.transform(X)
    expected_features = 4 * 4 * 3  # n_windows * n_ch * n_features
    assert X_transformed.shape == (5, expected_features)


def test_windowed_feature_extractor_consistency():
    """Test that multiple transforms give same result."""
    X = np.random.randn(3, 2, 8)

    extractor = WindowedFeatureExtractor(n_windows=2, features=["mean", "std", "slope"])

    X_transformed1 = extractor.transform(X)
    X_transformed2 = extractor.transform(X)

    np.testing.assert_array_equal(X_transformed1, X_transformed2)


def test_windowed_feature_extractor_empty_windows_warn_and_zero_fill():
    """Empty windows should warn and contribute zero-valued features."""
    X = np.arange(10, dtype=float).reshape(1, 1, 10)

    extractor = WindowedFeatureExtractor(
        n_windows=12, features=["mean", "std", "slope"]
    )

    with pytest.warns(
        RuntimeWarning,
        match="likely because n_windows=12 exceeds the available timepoints=10",
    ):
        X_transformed = extractor.transform(X)

    assert X_transformed.shape == (1, 36)
    np.testing.assert_array_equal(X_transformed[0, -6:], np.zeros(6))
    assert not np.isnan(X_transformed).any()
