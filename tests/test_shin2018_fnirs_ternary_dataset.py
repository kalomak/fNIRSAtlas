"""Behavioral tests for the Shin2018FnirsTernary dataset."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.shin2018_fnirs_ternary import Shin2018FnirsTernary
from fnirsatlas.datasets.utils import concentration_length_to_molar

SUBJECTS = list(range(1, 19))


@pytest.fixture
def dataset_and_raw():
    dataset = Shin2018FnirsTernary()
    raw = dataset.get_data(subjects=[1])[1]["0fnirs"]["0"]
    return dataset, raw


@pytest.mark.data
def test_raw_properties(dataset_and_raw):
    _, raw = dataset_and_raw
    assert raw.info["sfreq"] == pytest.approx(13.333333333333334)
    assert len(raw.ch_names) == 33
    assert len(mne.pick_types(raw.info, fnirs="hbo")) == 16
    assert len(mne.pick_types(raw.info, fnirs="hbr")) == 16
    assert not np.allclose(raw.get_data(picks="hbo"), 0)
    assert np.abs(raw.get_data(picks="hbo")).max() < 1e-3


def test_concentration_length_conversion():
    assert concentration_length_to_molar(18.0, 3.0) == pytest.approx(1e-3)


def test_dataset_metadata():
    dataset = Shin2018FnirsTernary()
    assert dataset.subject_list == SUBJECTS
    assert dataset.event_id == {
        "mental_arithmetic": 1,
        "motor_imagery": 2,
        "baseline": 3,
    }


@pytest.mark.data
def test_event_counts(dataset_and_raw):
    _, raw = dataset_and_raw
    labels = raw.get_data(picks="Stim").ravel()
    counts = dict(zip(*np.unique(labels[labels > 0].astype(int), return_counts=True)))
    assert counts == {1: 30, 2: 30, 3: 30}


@pytest.mark.data
def test_subject_7_missing_pair_is_interpolated():
    raw = Shin2018FnirsTernary().get_data(subjects=[7])[7]["0fnirs"]["0"]
    assert "S6_D5 hbo" in raw.ch_names
    assert "S6_D5 hbr" in raw.ch_names
    assert not np.allclose(raw.get_data(picks=["S6_D5 hbo", "S6_D5 hbr"]), 0)


@pytest.mark.parametrize("subject", SUBJECTS)
@pytest.mark.data
def test_all_subjects_load(subject):
    raw = Shin2018FnirsTernary().get_data(subjects=[subject])[subject]["0fnirs"]["0"]
    assert isinstance(raw, mne.io.BaseRaw)
    assert len(raw.ch_names) == 33
