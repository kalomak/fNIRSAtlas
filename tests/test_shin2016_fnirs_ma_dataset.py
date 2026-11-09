"""Tests for Shin2016FnirsMA dataset."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.shin2016_fnirs import (
    INTERVAL,
    MA_EVENTS,
    MA_SESSION_INDICES,
    N_CHANNEL_PAIRS,
    SFREQ,
    Shin2016FnirsMA,
)

N_FNIRS = N_CHANNEL_PAIRS * 2
N_CHANNELS = N_FNIRS + 1
SESSION_KEYS = [f"{idx}fnirs" for idx in range(1, len(MA_SESSION_INDICES) + 1)]


@pytest.fixture(scope="module")
def dataset():
    return Shin2016FnirsMA(accept=True)


@pytest.fixture(scope="module")
def data(dataset):
    return dataset.get_data(subjects=[1])


class TestDatasetInit:
    def test_subject_list(self, dataset):
        assert dataset.subject_list == list(range(1, 30))

    def test_paradigm(self, dataset):
        assert dataset.paradigm == "fnirsMentalArithmetic"

    def test_events(self, dataset):
        assert dataset.event_id == MA_EVENTS

    def test_interval(self, dataset):
        assert dataset.interval == INTERVAL

    def test_code(self, dataset):
        assert dataset.code == "Shin2016FnirsMA"

    def test_subject_selection(self):
        ds = Shin2016FnirsMA(accept=True, subjects=[1, 3])
        assert ds.subject_list == [1, 3]

    def test_accept_required_for_download(self, tmp_path):
        with pytest.raises(AttributeError, match="accept licence term"):
            Shin2016FnirsMA().data_path(1, path=tmp_path)


@pytest.mark.data
class TestDataStructure:
    def test_returns_dict(self, data):
        assert isinstance(data, dict)

    def test_subject_keys(self, data):
        assert 1 in data

    def test_session_count(self, data):
        assert len(data[1]) == len(SESSION_KEYS)

    def test_session_keys(self, data):
        assert list(data[1]) == SESSION_KEYS

    def test_run_key(self, data):
        first_session = next(iter(data[1].values()))
        assert list(first_session) == ["0"]

    def test_raw_type(self, data):
        raw = data[1][SESSION_KEYS[0]]["0"]
        assert isinstance(raw, mne.io.RawArray)


@pytest.mark.data
class TestRawProperties:
    @pytest.fixture(scope="class")
    @classmethod
    def raw(cls, data):
        return data[1][SESSION_KEYS[0]]["0"]

    def test_sfreq(self, raw):
        assert raw.info["sfreq"] == SFREQ

    def test_n_channels(self, raw):
        assert len(raw.ch_names) == N_CHANNELS

    def test_fnirs_channels(self, raw):
        assert len(mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude")) == N_FNIRS

    def test_stim_channel(self, raw):
        stim = mne.pick_types(raw.info, stim=True)
        assert len(stim) == 1
        assert raw.ch_names[stim[0]] == "Stim"

    def test_channel_pairs(self, raw):
        pair_names = [name.rsplit(" ", 1)[0] for name in raw.ch_names if name != "Stim"]
        counts = {pair: pair_names.count(pair) for pair in set(pair_names)}
        assert len(counts) == N_CHANNEL_PAIRS
        assert set(counts.values()) == {2}

    def test_channel_suffixes(self, raw):
        suffixes = {name.rsplit(" ", 1)[1] for name in raw.ch_names if name != "Stim"}
        assert suffixes == {"760", "850"}

    def test_data_not_all_zeros(self, raw):
        assert not np.allclose(raw.get_data()[:N_FNIRS], 0)


@pytest.mark.data
class TestAnnotations:
    @pytest.fixture(scope="class")
    @classmethod
    def stim_data(cls, data):
        raw = data[1][SESSION_KEYS[0]]["0"]
        return raw.get_data(picks="Stim").flatten()

    def test_stim_has_events(self, stim_data):
        assert np.any(stim_data > 0)

    def test_event_labels(self, stim_data):
        labels = set(stim_data[stim_data > 0].astype(int))
        assert labels == set(MA_EVENTS.values())

    def test_balanced_events(self, stim_data):
        labels = stim_data[stim_data > 0].astype(int)
        counts = np.unique(labels, return_counts=True)[1]
        assert np.array_equal(counts, [10, 10])


@pytest.mark.data
class TestAllSubjects:
    @pytest.mark.parametrize("subject", range(1, 30))
    def test_subject_loads(self, subject):
        data = Shin2016FnirsMA(accept=True).get_data(subjects=[subject])
        assert subject in data
        assert list(data[subject]) == SESSION_KEYS
        raw = data[subject][SESSION_KEYS[0]]["0"]
        assert len(raw.ch_names) == N_CHANNELS
