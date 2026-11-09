"""Conversion from MNE epochs to model inputs."""

from copy import deepcopy

import numpy as np


def epochs_to_array(epochs, unit_factor=1.0):
    """Return model data, labels, metadata, and information from MNE epochs.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Processed epochs to convert.
    unit_factor : float
        Scaling applied to the data, matching MOABB's array conversion.

    Returns
    -------
    X : numpy.ndarray
        Epoch data shaped as trials, channels, and timepoints.
    y : numpy.ndarray
        Event names corresponding to each epoch.
    metadata : pandas.DataFrame or None
        A copy of the epoch metadata.
    info : dict
        Serializable acquisition and channel information.
    """
    event_names = {value: name for name, value in epochs.event_id.items()}
    y = np.asarray([event_names[event] for event in epochs.events[:, 2]])
    metadata = None if epochs.metadata is None else epochs.metadata.copy()
    montage = epochs.get_montage()
    info = {
        "sfreq": float(epochs.info["sfreq"]),
        "ch_names": list(epochs.ch_names),
        "cond2label": dict(epochs.event_id),
        "epoch_duration": float(epochs.tmax - epochs.tmin),
        "tmin": float(epochs.tmin),
        "tmax": float(epochs.tmax),
        "chs": deepcopy(epochs.info["chs"]),
        "montage_positions": None if montage is None else montage.get_positions(),
    }
    return epochs.get_data(copy=True) * unit_factor, y, metadata, info
