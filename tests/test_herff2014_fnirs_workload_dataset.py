"""Behavioral tests for the Herff2014 fNIRS dataset."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.herff2014_fnirs_workload import Herff2014FnirsWorkload
from fnirsatlas.paradigms import FNIRSClassification

SUBJECTS = list(range(1, 11))
CHANNEL_PAIRS = [
    "S1_D1",
    "S2_D1",
    "S1_D2",
    "S2_D2",
    "S3_D3",
    "S4_D3",
    "S3_D4",
    "S4_D4",
]


@pytest.fixture(scope="module")
def raw():
    return Herff2014FnirsWorkload().get_data(subjects=[1])[1]["0fnirs"]["0"]


def test_dataset_metadata():
    dataset = Herff2014FnirsWorkload()
    assert dataset.subject_list == SUBJECTS
    assert dataset.event_id == {"1-back": 1, "2-back": 2, "3-back": 3}
    assert dataset.interval == [-2, 25]


def test_dataset_is_registered():
    assert any(
        isinstance(dataset, Herff2014FnirsWorkload)
        for dataset in FNIRSClassification().datasets
    )


@pytest.mark.data
def test_raw_properties(raw):
    expected_names = [
        f"{pair} {chromophore}"
        for pair in CHANNEL_PAIRS
        for chromophore in ("hbo", "hbr")
    ]
    assert raw.info["sfreq"] == 25
    assert raw.ch_names == expected_names + ["Stim"]
    assert np.abs(raw.get_data(picks="hbo")).max() < 1e-3


@pytest.mark.data
def test_source_detector_distances(raw):
    picks = mne.pick_types(raw.info, fnirs=True)
    distances = mne.preprocessing.nirs.source_detector_distances(raw.info, picks)
    assert np.all((distances > 0.03) & (distances < 0.04))


@pytest.mark.data
def test_event_counts(raw):
    stim = raw.get_data(picks="Stim").ravel()
    labels = stim[stim > 0].astype(int)
    assert dict(zip(*np.unique(labels, return_counts=True))) == {1: 10, 2: 10, 3: 10}


@pytest.mark.parametrize("subject", SUBJECTS)
@pytest.mark.data
def test_all_subjects_load(subject):
    raw = Herff2014FnirsWorkload().get_data(subjects=[subject])[subject]["0fnirs"]["0"]
    assert isinstance(raw, mne.io.BaseRaw)
    assert len(raw.ch_names) == 17
