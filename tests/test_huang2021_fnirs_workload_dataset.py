"""Behavioral tests for the Huang2021 fNIRS dataset."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.huang2021_fnirs_workload import Huang2021FnirsWorkload
from fnirsatlas.paradigms import FNIRSClassification

SUBJECTS = [
    1,
    5,
    7,
    13,
    14,
    15,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    29,
    31,
    32,
    34,
    35,
    36,
    37,
    38,
    40,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    51,
    52,
    54,
    55,
    56,
    57,
    58,
    60,
    61,
    62,
    63,
    64,
    65,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    91,
    92,
    93,
    94,
    95,
    97,
]
SOURCE_COLUMNS = ["AB_I_O", "AB_I_DO", "CD_I_O", "CD_I_DO"]


@pytest.fixture(scope="module")
def raw():
    return Huang2021FnirsWorkload().get_data(subjects=[1])[1]["0fnirs"]["0"]


def test_dataset_metadata():
    dataset = Huang2021FnirsWorkload()
    assert dataset.subject_list == SUBJECTS
    assert dataset.event_id == {
        "0-back": 1,
        "1-back": 2,
        "2-back": 3,
        "3-back": 4,
    }
    assert dataset.interval == [0, 81.6]


def test_dataset_is_registered():
    assert any(
        isinstance(dataset, Huang2021FnirsWorkload)
        for dataset in FNIRSClassification().datasets
    )


@pytest.mark.data
def test_raw_preserves_released_epochs(raw):
    source = np.genfromtxt(
        Huang2021FnirsWorkload().data_path(1),
        delimiter=",",
        names=True,
        encoding=None,
    )
    source = np.vstack([source[column] for column in SOURCE_COLUMNS])
    assert raw.info["sfreq"] == 5.2
    assert raw.ch_names == [
        "S1_D1 hbo",
        "S1_D1 hbr",
        "S2_D2 hbo",
        "S2_D2 hbr",
        "Stim",
    ]
    events = mne.find_events(raw, shortest_event=0, verbose=False)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0,
        tmax=425 / 5.2,
        baseline=None,
        picks="fnirs",
        preload=True,
        verbose=False,
    )
    expected = source.reshape(4, 16, 426).transpose(1, 0, 2) * 1e-6
    np.testing.assert_allclose(epochs.get_data(copy=False), expected)


@pytest.mark.data
def test_source_detector_distances(raw):
    picks = mne.pick_types(raw.info, fnirs=True)
    distances = mne.preprocessing.nirs.source_detector_distances(raw.info, picks)
    np.testing.assert_allclose(distances, 0.03, atol=0.001)


@pytest.mark.data
def test_paradigm_extracts_all_epochs():
    dataset = Huang2021FnirsWorkload()
    data, labels, _ = FNIRSClassification().get_data(dataset, subjects=[1])
    assert data.shape[0] == 16
    assert dict(zip(*np.unique(labels, return_counts=True))) == {
        "0-back": 4,
        "1-back": 4,
        "2-back": 4,
        "3-back": 4,
    }


@pytest.mark.parametrize("subject", SUBJECTS)
@pytest.mark.data
def test_all_subjects_load(subject):
    raw = Huang2021FnirsWorkload().get_data(subjects=[subject])[subject]["0fnirs"]["0"]
    assert isinstance(raw, mne.io.BaseRaw)
    assert len(mne.find_events(raw, shortest_event=0, verbose=False)) == 16
