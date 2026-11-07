from pathlib import Path
import importlib
import sys

import pytest


@pytest.mark.slow
def test_load_processed_dataset_without_mne_dependency():
    # Ensure importing fnirsatlas.io does not import mne
    if "mne" in sys.modules:
        del sys.modules["mne"]

    io = importlib.import_module("fnirsatlas.io")
    assert "mne" not in sys.modules

    X, y, groups, info = io.load_processed_dataset("bnci")
    _ = io.fNIRSAtlasDataset("bnci")

    # Basic sanity checks
    assert X.ndim == 3
    assert y.ndim == 1
    assert groups.ndim == 1
    assert isinstance(info, dict)
    # Required keys present
    for key in [
        "sfreq",
        "ch_names",
        "cond2label",
        "epoch_duration",
        "tmin",
        "tmax",
        "chs",
        "montage_positions",
    ]:
        assert key in info
