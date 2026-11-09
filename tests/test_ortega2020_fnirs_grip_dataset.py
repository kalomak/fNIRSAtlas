"""Behavioral tests for the Ortega2020 fNIRS dataset."""

from pathlib import Path

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.ortega2020_fnirs_grip import (
    Ortega2020FnirsGrip,
    _interleave_wavelengths,
)
from fnirsatlas.paradigms import FNIRSClassification

LOCAL_FILE = Path.home() / "mne_data/fnirsatlas/datasets/Ortega2020/hygrip.h5"
SUBJECTS = list(range(1, 15))


def _local_data_path(
    self, subject, path=None, force_update=False, update_path=None, verbose=None
):
    del self, subject, path, force_update, update_path, verbose
    return str(LOCAL_FILE)


@pytest.fixture
def dataset_and_raw(monkeypatch):
    monkeypatch.setattr(Ortega2020FnirsGrip, "data_path", _local_data_path)
    dataset = Ortega2020FnirsGrip()
    raw = dataset.get_data(subjects=[1])[1]["0fnirs"]["0"]
    return dataset, raw


def test_dataset_metadata():
    dataset = Ortega2020FnirsGrip()
    assert dataset.subject_list == SUBJECTS
    assert dataset.event_id == {"left": 1, "right": 2}
    assert dataset.interval == [-2, 25]


def test_dataset_is_registered():
    assert any(
        isinstance(dataset, Ortega2020FnirsGrip)
        for dataset in FNIRSClassification().datasets
    )


def test_wavelengths_are_interleaved_by_channel():
    wl1 = np.array([[1, 2], [3, 4]])
    wl2 = np.array([[5, 6], [7, 8]])
    np.testing.assert_array_equal(
        _interleave_wavelengths(wl1, wl2),
        [[1, 2], [5, 6], [3, 4], [7, 8]],
    )


@pytest.mark.data
def test_raw_properties(dataset_and_raw):
    _, raw = dataset_and_raw
    assert raw.info["sfreq"] == 12.5
    assert len(raw.ch_names) == 49
    assert len(mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude")) == 48
    assert not np.allclose(raw.get_data(picks="fnirs_cw_amplitude"), 0)


@pytest.mark.data
def test_event_counts(dataset_and_raw):
    _, raw = dataset_and_raw
    labels = raw.get_data(picks="Stim").ravel()
    counts = dict(zip(*np.unique(labels[labels > 0].astype(int), return_counts=True)))
    assert counts == {1: 10, 2: 10}


@pytest.mark.parametrize("subject", SUBJECTS)
@pytest.mark.data
def test_all_subjects_load(monkeypatch, subject):
    monkeypatch.setattr(Ortega2020FnirsGrip, "data_path", _local_data_path)
    raw = Ortega2020FnirsGrip().get_data(subjects=[subject])[subject]["0fnirs"]["0"]
    assert isinstance(raw, mne.io.BaseRaw)
    assert len(raw.ch_names) == 49
