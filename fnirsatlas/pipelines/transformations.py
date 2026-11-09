"""Transformations for fNIRS pipeline data."""

import warnings

import numpy as np
from mne import BaseEpochs
from sklearn.base import BaseEstimator, TransformerMixin


class WindowedFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract per-window features from chromophore epoch arrays.

    Splits each epoch into ``n_windows`` temporal segments and computes
    summary statistics per channel per window. The result is a 2D array
    suitable for classifiers such as LDA.

    Parameters
    ----------
    n_windows : int
        Number of equal-length temporal windows (default 6).
    features : list of str
        Features to compute per window. Supported values are ``"mean"``,
        ``"std"``, and ``"slope"``.

    Notes
    -----
    Input must have shape ``(trials, chromophores, channels, timepoints)``. The
    chromophore and channel dimensions are merged, so features are computed per
    channel of each chromophore.
    """

    def __init__(self, n_windows=6, features=None):
        self.n_windows = n_windows
        self.features = features if features is not None else ["mean"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = ChannelMerger().transform(X)
        n_samples, n_ch, n_time = X.shape
        windows = np.array_split(np.arange(n_time), self.n_windows)
        feats = []
        n_empty = sum(len(idx) == 0 for idx in windows)
        if n_empty:
            warnings.warn(
                f"WindowedFeatureExtractor produced {n_empty} empty window(s), "
                f"likely because n_windows={self.n_windows} exceeds the available "
                f"timepoints={n_time}. Zero-valued features will be used for those "
                "windows.",
                RuntimeWarning,
                stacklevel=2,
            )
        for idx in windows:
            seg = X[:, :, idx]
            if seg.shape[-1] == 0:
                feats.extend(
                    np.zeros((n_samples, n_ch), dtype=X.dtype) for _ in self.features
                )
                continue
            if "mean" in self.features:
                feats.append(seg.mean(-1))
            if "std" in self.features:
                feats.append(seg.std(-1))
            if "slope" in self.features:
                feats.append(self._slope(seg))
        return np.concatenate(feats, axis=1)

    @staticmethod
    def _slope(seg):
        if seg.shape[-1] <= 1:
            return np.zeros(seg.shape[:2], dtype=seg.dtype)
        t = np.arange(seg.shape[-1], dtype=seg.dtype)
        t -= t.mean()
        denom = (t**2).sum()
        return np.einsum("ijk,k->ij", seg, t) / denom

    def get_feature_names(self, n_ch):
        """Return names matching the columns produced by ``transform``."""
        return [
            f"win{window + 1}_{feature}_ch{channel + 1}"
            for window in range(self.n_windows)
            for feature in self.features
            for channel in range(n_ch)
        ]


class StandardScalerND(TransformerMixin, BaseEstimator):
    """Standardize an N-dimensional array along selected axes.

    Parameters
    ----------
    dim : int, tuple of int, or None
        Axes used to calculate the mean and standard deviation. The default
        uses the trial axis ``0``.

    Notes
    -----
    For input shaped ``(trials, chromophores, channels, timepoints)``, use
    ``dim=(0, 2, 3)`` to scale each chromophore independently.
    """

    def __init__(self, dim=None):
        self.dim = dim

    def fit(self, X, y=None):
        """Compute means and standard deviations."""
        axis = self.dim if self.dim is not None else 0
        self.means_ = np.mean(X, axis=axis, keepdims=True)
        self.stds_ = np.std(X, axis=axis, keepdims=True)
        return self

    def transform(self, X):
        """Standardize ``X`` using the fitted statistics."""
        if not hasattr(self, "means_"):
            raise RuntimeError("The scaler has not been fitted yet.")
        return (X - self.means_) / (self.stds_ + 1e-8)


class ChannelSplitter(TransformerMixin, BaseEstimator):
    """Split fNIRS channels into a channel-type dimension.

    Channels are grouped by the type suffix in their names. For example,
    ``S1_D1 hbo`` and ``S1_D1 hbr`` become the same channel in separate
    chromophore groups. Every group must contain the same channels in the same
    order.

    Parameters
    ----------
    ch_names : sequence of str or None
        Channel names for NumPy input. Names are read from MNE Epochs when
        omitted.

    Notes
    -----
    Input must be MNE Epochs or a NumPy array shaped ``(trials, channels,
    timepoints)``. Output is a NumPy array shaped ``(trials, chromophores,
    channels, timepoints)``.
    """

    def __init__(self, ch_names=None):
        self.ch_names = ch_names

    def fit(self, X, y=None):
        """Validate and group channel names by type."""
        if isinstance(X, BaseEpochs):
            if self.ch_names is not None and list(self.ch_names) != X.ch_names:
                raise ValueError("ch_names must match Epochs channel names and order")
            ch_names = X.ch_names
        elif self.ch_names is None:
            raise ValueError("ch_names must be provided during fit")
        else:
            ch_names = self.ch_names
        n_channels = len(X.ch_names) if isinstance(X, BaseEpochs) else X.shape[1]
        if len(ch_names) != n_channels:
            raise ValueError("Number of channel names must match X")
        self.ch_names_ = np.asarray(ch_names)
        self.ch_types_ = np.unique([name.split()[1] for name in self.ch_names_])
        self.ch_masks_ = {
            kind: np.array([name.split()[1] == kind for name in self.ch_names_])
            for kind in self.ch_types_
        }
        base_names = self._channel_names(self.ch_types_[0])
        for kind in self.ch_types_[1:]:
            if self._channel_names(kind) != base_names:
                raise AssertionError("Channel names must match across all types")
        return self

    def transform(self, X):
        """Return data shaped as samples, types, channels, timepoints."""
        if not hasattr(self, "ch_names_"):
            raise RuntimeError("The transformer has not been fitted yet.")
        if isinstance(X, BaseEpochs):
            if X.ch_names != self.ch_names_.tolist():
                raise ValueError("Epochs channel names and order differ from fit")
            X = X.get_data(copy=True)
        return np.stack(
            [X[:, self.ch_masks_[kind], :] for kind in self.ch_types_], axis=1
        )

    def _channel_names(self, kind):
        return [name.split()[0] for name in self.ch_names_[self.ch_masks_[kind]]]


class ChannelMerger(TransformerMixin, BaseEstimator):
    """Merge a channel-type dimension back into the channel dimension.

    The inverse of :class:`ChannelSplitter`, for estimators that expect a flat
    channel axis. Channel names are not needed, so this is a plain reshape.

    Notes
    -----
    Input shaped ``(trials, chromophores, channels, timepoints)`` becomes
    ``(trials, chromophores * channels, timepoints)``.
    """

    def fit(self, X, y=None):
        """Return the stateless fitted transformer."""
        return self

    def transform(self, X):
        """Merge the chromophore dimension into the channel dimension."""
        if X.ndim != 4:
            raise ValueError(f"X must be 4-dimensional, got {X.ndim}")
        trials, chromophores, channels, times = X.shape
        return X.reshape(trials, chromophores * channels, times)


class UnsqueezeLayer(TransformerMixin, BaseEstimator):
    """Insert a singleton dimension into an array.

    Parameters
    ----------
    dim : int
        Position of the new dimension. With the default ``dim=1``, input
        shaped ``(trials, channels, timepoints)`` becomes ``(trials, 1,
        channels, timepoints)``.
    """

    def __init__(self, dim=1):
        self.dim = dim

    def fit(self, X, y=None):
        """Return the stateless fitted transformer."""
        return self

    def transform(self, X):
        """Insert the configured dimension."""
        return np.expand_dims(X, axis=self.dim)
