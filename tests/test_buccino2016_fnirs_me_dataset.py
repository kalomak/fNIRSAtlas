"""Tests for Buccino2016FnirsME dataset."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.buccino2016_fnirs_me import (
    CH_NAMES_ORDERED,
    N_CHANNELS,
    ORIG2MNE,
    SFREQ,
    Buccino2016FnirsME,
)

N_TOTAL_CHANNELS = N_CHANNELS * 2 + 1  # two wavelengths + stim


@pytest.fixture(scope="module")
def dataset():
    return Buccino2016FnirsME()


@pytest.fixture(scope="module")
def data(dataset):
    return dataset.get_data(subjects=[1])


class TestDatasetInit:
    def test_subject_list(self, dataset):
        assert dataset.subject_list == list(range(1, 16))

    def test_paradigm(self, dataset):
        assert dataset.paradigm == "fnirsMotorExecution"

    def test_events(self, dataset):
        assert dataset.event_id == {
            "right_arm": 3,
            "left_arm": 4,
            "right_hand": 7,
            "left_hand": 8,
        }

    def test_interval(self, dataset):
        assert dataset.interval == [-2, 11.8]

    def test_code(self, dataset):
        assert dataset.code == "Buccino2016FnirsME"

    def test_subject_selection(self):
        ds = Buccino2016FnirsME(subjects=[1, 3])
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

    def test_cw_amplitude_channels(self, raw):
        fnirs = mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude")
        assert len(fnirs) == N_CHANNELS * 2

    def test_stim_channel(self, raw):
        stim = mne.pick_types(raw.info, stim=True)
        assert len(stim) == 1
        assert raw.ch_names[stim[0]] == "Stim"

    def test_channel_name_format(self, raw):
        for orig_name in CH_NAMES_ORDERED:
            pair = ORIG2MNE[orig_name]
            assert f"{pair} 760" in raw.ch_names
            assert f"{pair} 850" in raw.ch_names

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
        assert unique == {3, 4, 7, 8}

    def test_events_per_class(self, stim_data):
        labels = stim_data[stim_data > 0].astype(int)
        for code in (3, 4, 7, 8):
            assert np.sum(labels == code) == 25


@pytest.mark.data
class TestParadigmTmax:
    def test_none_vs_explicit_same_shape(self, dataset):
        from fnirsatlas import FNIRSClassification

        interval = dataset.interval
        explicit_tmax = interval[1]
        X_none, _, _ = FNIRSClassification(tmax=None).get_data(dataset, subjects=[1])
        X_exp, _, _ = FNIRSClassification(tmax=explicit_tmax).get_data(
            dataset, subjects=[1]
        )
        assert X_none.shape == X_exp.shape


@pytest.mark.data
class TestAllSubjects:
    @pytest.mark.parametrize("subject", range(1, 16))
    def test_subject_loads(self, subject):
        ds = Buccino2016FnirsME()
        data = ds.get_data(subjects=[subject])
        assert subject in data
        assert len(data[subject]) == 1

    @pytest.mark.parametrize("subject", range(1, 16))
    def test_channel_count(self, subject):
        ds = Buccino2016FnirsME()
        data = ds.get_data(subjects=[subject])
        raw = data[subject]["0fnirs"]["0"]
        assert len(raw.ch_names) == N_TOTAL_CHANNELS
