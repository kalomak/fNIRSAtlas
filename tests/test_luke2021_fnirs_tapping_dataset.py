"""Tests for Luke2021FnirsTapping dataset."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.luke2021_fnirs_tapping import (
    EVENTS,
    INTERVAL,
    SFREQ,
    SUBJECTS,
    Luke2021FnirsTapping,
)

N_FNIRS_CHANNELS = 56
N_CHANNELS = N_FNIRS_CHANNELS + 1


@pytest.fixture
def dataset():
    return Luke2021FnirsTapping()


@pytest.fixture
def data(dataset):
    return dataset.get_data(subjects=[1])


class TestDatasetInit:
    def test_subject_list(self, dataset):
        assert dataset.subject_list == SUBJECTS

    def test_paradigm(self, dataset):
        assert dataset.paradigm == "fnirsMotorExecution"

    def test_events(self, dataset):
        assert dataset.event_id == EVENTS

    def test_interval(self, dataset):
        assert dataset.interval == INTERVAL

    def test_code(self, dataset):
        assert dataset.code == "Luke2021FnirsTapping"


@pytest.mark.data
class TestDataStructure:
    def test_returns_dict(self, data):
        assert isinstance(data, dict)

    def test_subject_keys(self, data):
        assert 1 in data

    def test_session_keys(self, data):
        assert "0" in data[1]

    def test_run_key(self, data):
        assert "0" in data[1]["0"]

    def test_raw_type(self, data):
        assert isinstance(data[1]["0"]["0"], mne.io.BaseRaw)


@pytest.mark.data
class TestRawProperties:
    @pytest.fixture
    def raw(self, data):
        return data[1]["0"]["0"]

    def test_sfreq(self, raw):
        assert raw.info["sfreq"] == SFREQ

    def test_n_channels(self, raw):
        assert len(raw.ch_names) == N_CHANNELS

    def test_cw_amplitude_channels(self, raw):
        assert (
            len(mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude"))
            == N_FNIRS_CHANNELS
        )

    def test_stim_channel(self, raw):
        stim = mne.pick_types(raw.info, stim=True)
        assert len(stim) == 1
        assert raw.ch_names[stim[0]] == "Stim"

    def test_data_not_all_zeros(self, raw):
        data_arr = raw.get_data(picks=raw.ch_names[:-1])
        assert not np.allclose(data_arr, 0)


@pytest.mark.data
class TestAnnotations:
    @pytest.fixture
    def stim_data(self, data):
        return data[1]["0"]["0"].get_data(picks="Stim").ravel()

    def test_stim_has_events(self, stim_data):
        assert np.any(stim_data > 0)

    def test_event_labels(self, stim_data):
        unique = set(stim_data[stim_data > 0].astype(int))
        assert unique == set(EVENTS.values())

    def test_event_counts(self, stim_data):
        labels = stim_data[stim_data > 0].astype(int)
        counts = dict(zip(*np.unique(labels, return_counts=True)))
        assert counts == {1: 30, 2: 30, 3: 30}


@pytest.mark.data
class TestAllSubjects:
    @pytest.mark.parametrize("subject", SUBJECTS)
    def test_subject_loads(self, subject):
        data = Luke2021FnirsTapping().get_data(subjects=[subject])
        assert subject in data
        assert "0" in data[subject]
        assert len(data[subject]["0"]["0"].ch_names) == N_CHANNELS
