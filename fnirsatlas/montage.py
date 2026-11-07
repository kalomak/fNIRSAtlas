import json
import numpy as np
import mne
from pathlib import Path

from fnirsatlas.utils import MONTAGES_DIR


def montage_to_dict(montage):
    """JSON serializable representation of a montage."""
    mtg_dict = montage.get_positions()
    for k, v in mtg_dict.items():
        if isinstance(v, np.ndarray):
            mtg_dict[k] = v.tolist()
        elif isinstance(v, dict):
            mtg_dict[k] = {k2: v2.tolist() for k2, v2 in v.items()}
    return mtg_dict


def load_montage_from_file(dataset_id: str, montages_dir: str = None):
    """
    Load a montage for a dataset from the fnirsatlas/montages/ directory.
    Returns an mne.channels.DigMontage or None if not found.
    """
    if montages_dir is None:
        montages_dir = MONTAGES_DIR
    montage_path = Path(montages_dir) / f"{dataset_id}_montage_3d.json"
    if not montage_path.exists():
        return None
    with open(montage_path, "r") as f:
        info = json.load(f)
    montage = mne.channels.make_dig_montage(**info)
    return montage


def has_saved_montage(dataset_id: str, montages_dir: str = None):
    """Check if a saved montage exists for a dataset."""
    if montages_dir is None:
        montages_dir = MONTAGES_DIR
    montage_path = Path(montages_dir) / f"{dataset_id}_montage_3d.json"
    return montage_path.exists()
