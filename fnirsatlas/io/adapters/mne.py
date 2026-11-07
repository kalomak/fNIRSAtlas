from collections import defaultdict

import numpy as np


def _group_indices(groups: np.ndarray) -> dict[str, list[int]]:
    subj2ix = defaultdict(list)
    for i, subj in enumerate(groups):
        subj2ix[str(subj)].append(i)
    return subj2ix


def _create_raw_from_subject_data(X_subj, info, mne):
    from fnirsatlas.channels import COIL_TYPE_TO_CH_TYPE, ch_constants_to_mne_format

    n_epochs, n_channels, n_times = X_subj.shape
    ch_names = info["ch_names"]
    sfreq = info["sfreq"]
    chs = info.get("chs")
    ch_types = None
    if chs:
        mne_chs = ch_constants_to_mne_format([dict(ch) for ch in chs])
        ch_types = [COIL_TYPE_TO_CH_TYPE.get(ch["coil_type"], "misc") for ch in mne_chs]
    if ch_types is None:
        ch_types = ["misc"] * len(ch_names)
    info_obj = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, verbose=False)
    raw_data = X_subj.transpose(1, 0, 2).reshape(n_channels, n_epochs * n_times)
    raw = mne.io.RawArray(raw_data, info_obj, verbose=False)
    return raw, n_epochs, n_times, sfreq


def _build_events(y_subj: np.ndarray, n_epochs: int, n_times: int) -> np.ndarray:
    return np.c_[np.arange(n_epochs) * n_times, np.zeros(n_epochs, dtype=int), y_subj.astype(int)]


def to_mne_epochs(X, y, groups, info):
    try:
        import mne
    except ModuleNotFoundError as e:
        raise ImportError("MNE not available. Install mne to use to_mne().") from e

    subj2ix = _group_indices(groups)
    epochs_dict = {}
    for subj, idxs in subj2ix.items():
        X_subj = X[idxs]
        y_subj = y[idxs]
        raw, n_epochs, n_times, sfreq = _create_raw_from_subject_data(X_subj, info, mne)
        montage_positions = info.get("montage_positions")
        if montage_positions is not None:
            try:
                montage = mne.channels.make_dig_montage(**montage_positions)
                raw.set_montage(montage, verbose=False)
            except Exception:
                pass
        event_id = info.get("cond2label") or {str(int(lbl)): int(lbl) for lbl in np.unique(y_subj)}
        events = _build_events(y_subj, n_epochs, n_times)
        epochs_dict[subj] = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=0,
            tmax=(n_times - 1) / sfreq,
            baseline=None,
            preload=True,
            verbose=False,
        )
    return epochs_dict
