"""Tests for the Shin2018bFnirsMA dataset."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.shin2018b_fnirs_ma import Shin2018bFnirsMA

pytestmark = pytest.mark.data

SUBJECTS = list(range(1, 13))
CHANNEL_PAIRS = [
    "S1_D1",
    "S2_D1",
    "S2_D2",
    "S3_D1",
    "S3_D2",
    "S3_D3",
    "S4_D2",
    "S4_D3",
    "S5_D3",
]


@pytest.fixture(scope="module")
def raw():
    return Shin2018bFnirsMA().get_data(subjects=[1])[1]["0fnirs"]["0"]


def test_raw_properties(raw):
    assert raw.info["sfreq"] == 10
    assert raw.ch_names == (
        [f"{pair} hbo" for pair in CHANNEL_PAIRS]
        + [f"{pair} hbr" for pair in CHANNEL_PAIRS]
        + ["Stim"]
    )
    assert not np.allclose(raw.get_data(picks="hbo"), 0)
    assert np.abs(raw.get_data(picks="hbo")).max() < 1e-5


def test_source_detector_distances(raw):
    picks = mne.pick_types(raw.info, fnirs=True)
    distances = mne.preprocessing.nirs.source_detector_distances(raw.info, picks)
    assert np.all((distances > 0.02) & (distances < 0.04))


def test_event_counts(raw):
    stim = raw.get_data(picks="Stim").ravel()
    labels = stim[stim > 0].astype(int)
    assert dict(zip(*np.unique(labels, return_counts=True))) == {1: 30, 2: 30}


@pytest.mark.parametrize("subject", SUBJECTS)
def test_all_subjects_load(subject):
    raw = Shin2018bFnirsMA().get_data(subjects=[subject])[subject]["0fnirs"]["0"]
    assert isinstance(raw, mne.io.BaseRaw)
    assert len(raw.ch_names) == 19
