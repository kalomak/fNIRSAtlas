import mne
import numpy as np
import pytest

from fnirsatlas.pipelines import (
    ChannelMerger,
    ChannelSplitter,
    StandardScalerND,
    UnsqueezeLayer,
    WindowedFeatureExtractor,
)


def test_empty_windows_warn_and_zero_fill():
    X = np.arange(10, dtype=float).reshape(1, 1, 1, 10)
    extractor = WindowedFeatureExtractor(
        n_windows=12, features=["mean", "std", "slope"]
    )

    with pytest.warns(RuntimeWarning, match="n_windows=12 exceeds"):
        transformed = extractor.transform(X)

    assert transformed.shape == (1, 36)
    np.testing.assert_array_equal(transformed[0, -6:], np.zeros(6))
    assert not np.isnan(transformed).any()


def test_feature_names_follow_output_order():
    extractor = WindowedFeatureExtractor(n_windows=2, features=["mean", "std"])

    assert extractor.get_feature_names(2) == [
        "win1_mean_ch1",
        "win1_mean_ch2",
        "win1_std_ch1",
        "win1_std_ch2",
        "win2_mean_ch1",
        "win2_mean_ch2",
        "win2_std_ch1",
        "win2_std_ch2",
    ]


def test_standard_scaler_nd_scales_selected_axis():
    X = np.arange(24, dtype=float).reshape(2, 3, 4)

    transformed = StandardScalerND(dim=0).fit_transform(X)

    np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(transformed.std(axis=0), 1)


def test_channel_splitter_groups_channel_types():
    X = np.arange(16).reshape(1, 4, 4)
    splitter = ChannelSplitter(
        ch_names=["S1_D1 hbo", "S1_D1 hbr", "S1_D2 hbo", "S1_D2 hbr"]
    )

    transformed = splitter.fit_transform(X)

    assert transformed.shape == (1, 2, 2, 4)
    np.testing.assert_array_equal(transformed[:, 0], X[:, [0, 2]])
    np.testing.assert_array_equal(transformed[:, 1], X[:, [1, 3]])


def test_channel_splitter_uses_epochs_channel_names():
    ch_names = ["S1_D1 hbo", "S1_D1 hbr", "S1_D2 hbo", "S1_D2 hbr"]
    data = np.arange(32).reshape(2, 4, 4)
    epochs = mne.EpochsArray(data, mne.create_info(ch_names, 1.0, "misc"))

    splitter = ChannelSplitter()
    transformed = splitter.fit_transform(epochs)

    assert transformed.shape == (2, 2, 2, 4)
    np.testing.assert_array_equal(transformed[:, 0], data[:, [0, 2]])
    np.testing.assert_array_equal(transformed[:, 1], data[:, [1, 3]])


def test_channel_splitter_rejects_changed_epochs_channels():
    ch_names = ["S1_D1 hbo", "S1_D1 hbr", "S1_D2 hbo", "S1_D2 hbr"]
    info = mne.create_info(ch_names, 1.0, "misc")
    splitter = ChannelSplitter().fit(mne.EpochsArray(np.ones((1, 4, 4)), info))
    reordered = mne.EpochsArray(np.ones((1, 4, 4)), info).reorder_channels(
        ch_names[::-1]
    )

    with pytest.raises(ValueError, match="channel names and order"):
        splitter.transform(reordered)


def test_unsqueeze_layer_inserts_dimension():
    X = np.ones((2, 3, 4))

    transformed = UnsqueezeLayer(dim=1).fit_transform(X)

    assert transformed.shape == (2, 1, 3, 4)
    np.testing.assert_array_equal(transformed[:, 0], X)


def test_channel_merger_flattens_the_chromophore_axis():
    """Merging keeps each chromophore's channels contiguous."""
    grouped = np.random.default_rng(0).normal(size=(4, 2, 3, 20))

    merged = ChannelMerger().fit_transform(grouped)

    assert merged.shape == (4, 6, 20)
    np.testing.assert_allclose(merged[:, :3], grouped[:, 0])
    np.testing.assert_allclose(merged[:, 3:], grouped[:, 1])


def test_windowed_features_reject_a_flat_channel_axis():
    with pytest.raises(ValueError, match="4-dimensional"):
        WindowedFeatureExtractor().transform(np.zeros((4, 6, 20)))
