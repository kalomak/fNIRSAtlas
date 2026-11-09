"""Tests for Bak2019FnirsTapping dataset."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.bak2019_fnirs_tapping import (
    CH_NAMES_ORDERED,
    N_CHANNELS,
    SFREQ,
    Bak2019FnirsTapping,
)

N_TOTAL_CHANNELS = N_CHANNELS * 2 + 1  # HbO + HbR + stim


@pytest.fixture(scope="module")
def dataset():
    return Bak2019FnirsTapping()


@pytest.fixture(scope="module")
def data(dataset):
    return dataset.get_data(subjects=[1])


class TestDatasetInit:
    def test_subject_list(self, dataset):
        assert dataset.subject_list == list(range(1, 31))

    def test_paradigm(self, dataset):
        assert dataset.paradigm == "fnirsMotorExecution"

    def test_events(self, dataset):
        assert dataset.event_id == {
            "Tapping/Right": 1,
            "Tapping/Left": 2,
            "Tapping/Foot": 3,
        }

    def test_interval(self, dataset):
        assert dataset.interval == [-1.98, 15]

    def test_code(self, dataset):
        assert dataset.code == "Bak2019FnirsTapping"

    def test_subject_selection(self):
        ds = Bak2019FnirsTapping(subjects=[1, 3])
        assert ds.subject_list == [1, 3]


@pytest.mark.data
class TestDataStructure:
    def test_returns_dict(self, data):
        assert isinstance(data, dict)

    def test_subject_keys(self, data):
        assert 1 in data

    def test_session_count(self, data):
        assert len(data[1]) == 1

    def test_session_key(self, data):
        assert "0fnirs" in data[1]

    def test_run_key(self, data):
        assert "0" in data[1]["0fnirs"]

    def test_raw_type(self, data):
        raw = data[1]["0fnirs"]["0"]
        assert isinstance(raw, mne.io.RawArray)


@pytest.mark.data
class TestRawProperties:
    @pytest.fixture(scope="class")
    @classmethod
    def raw(cls, data):
        return data[1]["0fnirs"]["0"]

    def test_sfreq(self, raw):
        assert raw.info["sfreq"] == pytest.approx(SFREQ)

    def test_n_channels(self, raw):
        assert len(raw.ch_names) == N_TOTAL_CHANNELS

    def test_hbo_channels(self, raw):
        hbo = mne.pick_types(raw.info, fnirs="hbo")
        assert len(hbo) == N_CHANNELS

    def test_hbr_channels(self, raw):
        hbr = mne.pick_types(raw.info, fnirs="hbr")
        assert len(hbr) == N_CHANNELS

    def test_stim_channel(self, raw):
        stim = mne.pick_types(raw.info, stim=True)
        assert len(stim) == 1
        assert raw.ch_names[stim[0]] == "Stim"

    def test_channel_name_format(self, raw):
        for name in CH_NAMES_ORDERED:
            assert f"{name} hbo" in raw.ch_names
            assert f"{name} hbr" in raw.ch_names

    def test_data_not_all_zeros(self, raw):
        data_arr = raw.get_data()
        assert not np.allclose(data_arr[: N_CHANNELS * 2], 0)


@pytest.mark.data
class TestAnnotations:
    @pytest.fixture(scope="class")
    @classmethod
    def stim_data(cls, data):
        raw = data[1]["0fnirs"]["0"]
        return raw.get_data(picks="Stim").flatten()

    def test_stim_has_events(self, stim_data):
        assert np.any(stim_data > 0)

    def test_event_labels(self, stim_data):
        unique = set(stim_data[stim_data > 0].astype(int))
        assert unique == {1, 2, 3}

    def test_events_per_class(self, stim_data):
        labels = stim_data[stim_data > 0].astype(int)
        for code in (1, 2, 3):
            assert np.sum(labels == code) == 25


@pytest.mark.data
class TestAllSubjects:
    @pytest.mark.parametrize("subject", range(1, 31))
    def test_subject_loads(self, subject):
        ds = Bak2019FnirsTapping()
        data = ds.get_data(subjects=[subject])
        assert subject in data
        assert len(data[subject]) == 1

    @pytest.mark.parametrize("subject", range(1, 31))
    def test_channel_count(self, subject):
        ds = Bak2019FnirsTapping()
        data = ds.get_data(subjects=[subject])
        raw = data[subject]["0fnirs"]["0"]
        assert len(raw.ch_names) == N_TOTAL_CHANNELS
