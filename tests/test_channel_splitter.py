import numpy as np
import pytest
from fnirsatlas.feature_extraction import ChannelSplitter


class TestChannelSplitter:
    """Test cases for ChannelSplitter class."""

    def test_channel_names_match_positive_case(self):
        """Test that ChannelSplitter works when channel names match across types."""
        # Create test data with matching channel names
        ch_names = [
            "S1_D1 hbo",
            "S1_D1 hbr",
            "S1_D2 hbo",
            "S1_D2 hbr",
            "S2_D1 hbo",
            "S2_D1 hbr",
            "S2_D2 hbo",
            "S2_D2 hbr",
        ]

        # Create dummy data: (n_samples, n_channels, n_timepoints)
        X = np.random.randn(10, 8, 100)

        splitter = ChannelSplitter(ch_names=ch_names)
        splitter.fit(X)

        # Should not raise an assertion error
        result = splitter.transform(X)

        # Check output shape: (n_samples, n_ch_types, n_channels_per_type, n_timepoints)
        assert result.shape == (10, 2, 4, 100)

        # Check that hbo channels are in the first dimension
        assert np.array_equal(result[:, 0, 0, :], X[:, 0, :])  # S1_D1 hbo
        assert np.array_equal(result[:, 0, 1, :], X[:, 2, :])  # S1_D2 hbo
        assert np.array_equal(result[:, 0, 2, :], X[:, 4, :])  # S2_D1 hbo
        assert np.array_equal(result[:, 0, 3, :], X[:, 6, :])  # S2_D2 hbo

        # Check that hbr channels are in the second dimension
        assert np.array_equal(result[:, 1, 0, :], X[:, 1, :])  # S1_D1 hbr
        assert np.array_equal(result[:, 1, 1, :], X[:, 3, :])  # S1_D2 hbr
        assert np.array_equal(result[:, 1, 2, :], X[:, 5, :])  # S2_D1 hbr
        assert np.array_equal(result[:, 1, 3, :], X[:, 7, :])  # S2_D2 hbr

    def test_channel_names_match_negative_case(self):
        """Test that ChannelSplitter raises an assertion error when channel names don't match."""
        # Create test data with mismatched channel names
        ch_names = [
            "S1_D1 hbo",
            "S1_D1 hbr",
            "S1_D2 hbo",
            "S2_D1 hbr",  # Mismatch: S1_D2 vs S2_D1
            "S2_D1 hbo",
            "S2_D2 hbr",
            "S2_D2 hbo",
            "S1_D2 hbr",
        ]

        # Create dummy data: (n_samples, n_channels, n_timepoints)
        X = np.random.randn(10, 8, 100)

        splitter = ChannelSplitter(ch_names=ch_names)

        # Should raise an assertion error during fit
        with pytest.raises(
            AssertionError, match="Channel names must match across all types"
        ):
            splitter.fit(X)

    def test_channel_names_match_negative_case_different_order(self):
        """Test that ChannelSplitter raises an assertion error when channel names are in different order."""
        # Create test data with same names but different order
        ch_names = [
            "S1_D1 hbo",
            "S1_D1 hbr",
            "S1_D2 hbo",
            "S1_D2 hbr",
            "S2_D1 hbo",
            "S2_D2 hbr",  # Different order: S2_D1 vs S2_D2
            "S2_D2 hbo",
            "S2_D1 hbr",
        ]

        # Create dummy data: (n_samples, n_channels, n_timepoints)
        X = np.random.randn(10, 8, 100)

        splitter = ChannelSplitter(ch_names=ch_names)

        # Should raise an assertion error during fit
        with pytest.raises(
            AssertionError, match="Channel names must match across all types"
        ):
            splitter.fit(X)

    def test_channel_names_match_negative_case_missing_channel(self):
        """Test that ChannelSplitter raises an assertion error when one channel type has fewer channels."""
        # Create test data with missing channel in one type
        ch_names = [
            "S1_D1 hbo",
            "S1_D1 hbr",
            "S1_D2 hbo",
            "S1_D2 hbr",
            "S2_D1 hbo",
            "S2_D1 hbr",
            "S2_D2 hbo",  # Missing S2_D2 hbr
        ]

        # Create dummy data: (n_samples, n_channels, n_timepoints)
        X = np.random.randn(10, 7, 100)

        splitter = ChannelSplitter(ch_names=ch_names)

        # Should raise an assertion error during fit
        with pytest.raises(
            AssertionError, match="Channel names must match across all types"
        ):
            splitter.fit(X)
