"""Tests for the Gao2023FnirsSqueezing dataset."""

import mne
import numpy as np

from fnirsatlas.datasets.gao2023_fnirs_squeezing import Gao2023FnirsSqueezing
from fnirsatlas.datasets.openneuro import download_openneuro_dataset_files


def test_dataset_uses_ball_squeezing_session():
    dataset = Gao2023FnirsSqueezing()

    assert dataset.event_id == {"Left": 1, "Right": 2}
    assert dataset.doi == "10.1117/1.NPh.10.2.025007"


def test_openneuro_cache_uses_dataset_code(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "fnirsatlas.datasets.bids_base.get_dataset_path",
        lambda code, path: tmp_path,
    )

    root = download_openneuro_dataset_files(
        "ds007420", "Gao2023FnirsSqueezing", None, []
    )

    assert root == tmp_path / "MNE-gao2023fnirssqueezing-data"


def test_numeric_annotations_are_mapped_to_hand_labels(monkeypatch):
    raw = mne.io.RawArray(np.zeros((1, 10)), mne.create_info(["stim"], 1, "stim"))
    raw.set_annotations(mne.Annotations([1, 2], [1, 1], ["1", "2"]))
    monkeypatch.setattr(
        "mne.preprocessing.nirs.temporal_derivative_distribution_repair",
        lambda raw: raw,
    )

    Gao2023FnirsSqueezing()._standardize_raw(raw)

    assert raw.annotations.description.tolist() == ["Left", "Right"]
