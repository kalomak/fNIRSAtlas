from typing import Dict

import numpy as np

from fnirsatlas.io.dataset import fNIRSAtlasDataset
from fnirsatlas.io.processed_dataset import (
    load_processed_dataset,
    is_processed_dataset,
    get_available_processed_datasets,
)
from fnirsatlas.utils import check_duplicates, to_plain


def to_dataset(epochs_dict: Dict[str, object], dataset_name: str = None):
    del dataset_name
    X_list, y_list, groups_list = [], [], []
    cond2label = {}
    for subj, epochs in epochs_dict.items():
        data = epochs.crop(tmin=0).get_data()
        y = np.array(epochs.events[:, 2])
        X_list.append(data)
        y_list.append(y)
        groups_list.append(np.array([subj] * len(y), dtype=str))
        cond2label.update({str(k): int(v) for k, v in epochs.event_id.items()})

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(groups_list, axis=0)
    dup_count = check_duplicates(X)
    if dup_count > 0:
        raise ValueError(f"Found {dup_count} duplicate samples.")

    first_epochs = list(epochs_dict.values())[0]
    montage_positions = None
    montage = first_epochs.get_montage()
    if montage is not None:
        montage_positions = montage.get_positions()
    info = to_plain(
        {
            "sfreq": first_epochs.info["sfreq"],
            "ch_names": first_epochs.ch_names,
            "cond2label": cond2label,
            "epoch_duration": first_epochs.tmax - first_epochs.tmin,
            "tmin": first_epochs.tmin,
            "tmax": first_epochs.tmax,
            "chs": first_epochs.info["chs"],
            "montage_positions": montage_positions,
        }
    )
    return X, y, groups, info


def from_dataset(X, y, groups, info):
    return fNIRSAtlasDataset.from_arrays(X, y, groups, info).to_mne()


__all__ = [
    "to_dataset",
    "from_dataset",
    "load_processed_dataset",
    "is_processed_dataset",
    "get_available_processed_datasets",
]
