import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WindowedFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_windows=6, features=["mean"]):
        self.n_windows = n_windows
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_samples, n_ch, n_time = X.shape
        feats = []
        window_indices = np.array_split(np.arange(n_time), self.n_windows)
        # Too many windows for a short epoch leaves trailing empty splits.
        n_empty_windows = sum(len(idx) == 0 for idx in window_indices)
        if n_empty_windows:
            warnings.warn(
                "WindowedFeatureExtractor produced "
                f"{n_empty_windows} empty window(s), likely because n_windows="
                f"{self.n_windows} exceeds the available timepoints={n_time}; "
                "zero-valued features will be used for those windows.",
                RuntimeWarning,
                stacklevel=2,
            )
        for idx in window_indices:
            seg = X[:, :, idx]
            # Replace empty windows with zeros
            if seg.shape[-1] == 0:
                # One zero per feature
                for _ in self.features:
                    feats.append(np.zeros((n_samples, n_ch), dtype=X.dtype))
                continue
            if "mean" in self.features:
                feats.append(seg.mean(-1))
            if "std" in self.features:
                feats.append(seg.std(-1))
            if "slope" in self.features:
                if seg.shape[-1] > 1:
                    t = np.arange(seg.shape[-1])
                    slope = np.zeros(seg.shape[:2], dtype=X.dtype)
                    for sample in range(seg.shape[0]):
                        for ch in range(seg.shape[1]):
                            slope[sample, ch] = np.polyfit(t, seg[sample, ch, :], 1)[0]
                else:
                    slope = np.zeros(seg.shape[:2], dtype=X.dtype)
                feats.append(slope)
        return np.concatenate(feats, axis=1)

    def get_feature_names(self, n_ch):
        names = []
        for win in range(self.n_windows):
            for feat in self.features:
                for ch in range(n_ch):
                    names.append(f"win{win + 1}_{feat}_ch{ch + 1}")
        return names


class StandardScalerND(TransformerMixin, BaseEstimator):
    """
    Standardizes multi-dimensional arrays along specified dimensions.

    This transformer extends the functionality of the standard StandardScaler to
    handle N-dimensional arrays by allowing the user to specify the axis along
    which to calculate the mean and standard deviation for normalization.
    E.g. for CNN input (b, c, h, w) and dim=(0, 2, 3) the mean and std will have
    shape (c) and every channel will be normalized separately.
    If dim=0, the mean and std will have shape (1, c, h, w), meaning that each
    individual feature will be normalized separately.


    Parameters
    ----------
    dim : int, optional (default=None)
        The axis along which to calculate the mean and standard deviation.
        If None, normalization is performed along the first axis (axis 0), which
        is equivalent to the behavior of the standard `StandardScaler`.

    Attributes
    ----------
    means_ : ndarray of shape (n_dims, ...)
        The mean of the data along the specified axis.  The number of
        dimensions of `means_` will be the same as the input data `X`, except
        for the dimension specified by `dim`, which will be 1.

    stds_ : ndarray of shape (n_dims, ...)
        The standard deviation of the data along the specified axis.  The
        number of dimensions of `stds_` will be the same as the input data
        `X`, except for the dimension specified by `dim`, which will be 1.

    epsilon_ : float (default=1e-8)
        A small value added to the standard deviation to prevent division by zero.
    """

    def __init__(self, dim=None):
        self.dim = dim
        self.means = None
        self.stds = None
        self.epsilon = 1e-8

    def fit(self, X, y=None):
        axis = self.dim if self.dim is not None else 0
        self.means = np.mean(X, axis=axis, keepdims=True)
        self.stds = np.std(X, axis=axis, keepdims=True)
        return self

    def transform(self, X, y=None):
        if self.means is None or self.stds is None:
            raise RuntimeError("The scaler has not been fitted yet.")
        X_normalized = (X - self.means) / (self.stds + self.epsilon)
        return X_normalized

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class ChannelSplitter(TransformerMixin, BaseEstimator):
    """
    Transformer that splits channels by type for fNIRS data.

    This transformer takes fNIRS data with shape (n_samples, n_channels, n_timepoints)
    and channel names, and splits it into separate channels by type (e.g., hbo, hbr).
    The output has shape (n_samples, n_ch_types, n_channels_per_type, n_timepoints).

    Parameters
    ----------
    ch_names : list of str, optional
        Channel names. If None, will be set during fit.

    Attributes
    ----------
    ch_names_ : list of str
        Channel names used for splitting.
    ch_types_ : list of str
        Unique channel types found in the data.
    """

    def __init__(self, ch_names=None):
        self.ch_names = ch_names
        self.ch_names_ = None
        self.ch_types_ = None

    def fit(self, X, y=None):
        """Fit the transformer."""
        if self.ch_names is None:
            raise ValueError(
                "ch_names must be provided during fit if not set in __init__"
            )
        self.ch_names_ = np.array(self.ch_names)
        self.ch_types_ = np.unique([ch_name.split()[1] for ch_name in self.ch_names_])

        self.ch_masks_ = {}
        for ch_type in self.ch_types_:
            self.ch_masks_[ch_type] = [
                ch_name.split()[1] == ch_type for ch_name in self.ch_names_
            ]

        base_mask = self.ch_masks_[self.ch_types_[0]]
        base_names = [ch_name.split()[0] for ch_name in self.ch_names_[base_mask]]
        for ch_type in self.ch_types_[1:]:
            type_mask = self.ch_masks_[ch_type]
            ch_names = [ch_name.split()[0] for ch_name in self.ch_names_[type_mask]]
            assert len(ch_names) == len(base_names) and ch_names == base_names, (
                f"Channel names must match across all types"
            )
        return self

    def transform(self, X, y=None):
        """Transform the data by splitting channels by type."""
        if self.ch_names_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")
        n, c, t = X.shape
        X_ = np.zeros((n, len(self.ch_types_), c // len(self.ch_types_), t))
        for ch_type_ix, ch_type in enumerate(self.ch_types_):
            X_[:, ch_type_ix, :, :] = X[:, self.ch_masks_[ch_type], :]
        return X_


class UnsqueezeLayer(TransformerMixin, BaseEstimator):
    def __init__(self, dim=1):
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.expand_dims(X, axis=self.dim)

    def fit_transform(self, X, y=None):
        return self.transform(X)
