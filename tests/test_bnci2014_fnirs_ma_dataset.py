"""Tests for Bnci2014FnirsMA dataset."""

import mne
import numpy as np
import pytest
from mne.preprocessing.nirs import source_detector_distances

from fnirsatlas.datasets.bnci2014_fnirs_ma import SESSIONS, SFREQ, Bnci2014FnirsMA

N_HBO = 52
N_HBR = 52
N_CHANNELS = N_HBO + N_HBR + 1  # +1 for stim


@pytest.fixture(scope="module")
def dataset():
    return Bnci2014FnirsMA()


@pytest.fixture(scope="module")
def data(dataset):
    return dataset.get_data(subjects=[1])


class TestDatasetInit:
    def test_subject_list(self, dataset):
        assert dataset.subject_list == list(range(1, 9))

    def test_paradigm(self, dataset):
        assert dataset.paradigm == "fnirsMentalArithmetic"

    def test_events(self, dataset):
        assert dataset.event_id == {"mental_arithmetic": 1, "baseline": 2}

    def test_interval(self, dataset):
        assert dataset.interval == [-2, 15]

    def test_code(self, dataset):
        assert dataset.code == "Bnci2014FnirsMA"

    def test_subject_selection(self):
        ds = Bnci2014FnirsMA(subjects=[1, 3])
        assert ds.subject_list == [1, 3]


@pytest.mark.data
class TestDataStructure:
    def test_returns_dict(self, data):
        assert isinstance(data, dict)

    def test_subject_keys(self, data):
        assert 1 in data

    def test_session_count(self, data):
        expected = len(SESSIONS[1])
        assert len(data[1]) == expected

    def test_session_keys(self, data):
        for sess_num in SESSIONS[1]:
            assert f"{sess_num}fnirs" in data[1]

    def test_run_key(self, data):
        first_session = next(iter(data[1].values()))
        assert "0" in first_session

    def test_raw_type(self, data):
        raw = data[1][f"{SESSIONS[1][0]}fnirs"]["0"]
        assert isinstance(raw, mne.io.RawArray)


@pytest.mark.data
class TestRawProperties:
    @pytest.fixture(scope="class")
    @classmethod
    def raw(cls, data):
        return data[1][f"{SESSIONS[1][0]}fnirs"]["0"]

    def test_sfreq(self, raw):
        assert raw.info["sfreq"] == SFREQ

    def test_n_channels(self, raw):
        assert len(raw.ch_names) == N_CHANNELS

    def test_hbo_channels(self, raw):
        hbo = mne.pick_types(raw.info, fnirs="hbo")
        assert len(hbo) == N_HBO

    def test_hbr_channels(self, raw):
        hbr = mne.pick_types(raw.info, fnirs="hbr")
        assert len(hbr) == N_HBR

    def test_stim_channel(self, raw):
        stim = mne.pick_types(raw.info, stim=True)
        assert len(stim) == 1
        assert raw.ch_names[stim[0]] == "Stim"

    def test_channels_follow_figure_numbering(self, raw):
        # Data column k is figure channel k. Channel 48 lies at Fp1.
        for column, name in ((1, "S1_D1"), (11, "S1_D6"), (48, "S15_D14")):
            assert raw.ch_names[column - 1] == f"{name} hbo"
            assert raw.ch_names[N_HBO + column - 1] == f"{name} hbr"

    def test_source_detector_distance(self, raw):
        distances = source_detector_distances(raw.copy().pick("hbo").info)
        assert np.allclose(distances, 0.03, atol=0.002)

    def test_data_not_all_zeros(self, raw):
        data_arr = raw.get_data()
        assert not np.allclose(data_arr[:104], 0)


@pytest.mark.data
class TestAnnotations:
    @pytest.fixture(scope="class")
    @classmethod
    def stim_data(cls, data):
        raw = data[1][f"{SESSIONS[1][0]}fnirs"]["0"]
        return raw.get_data(picks="Stim").flatten()

    def test_stim_has_events(self, stim_data):
        assert np.any(stim_data > 0)

    def test_event_labels(self, stim_data):
        unique = set(stim_data[stim_data > 0].astype(int))
        assert unique == {1, 2}

    def test_balanced_events(self, stim_data):
        labels = stim_data[stim_data > 0].astype(int)
        n_ma = np.sum(labels == 1)
        n_bl = np.sum(labels == 2)
        assert n_ma == n_bl


@pytest.mark.data
class TestAllSubjects:
    @pytest.mark.parametrize("subject", range(1, 9))
    def test_subject_loads(self, subject):
        ds = Bnci2014FnirsMA()
        data = ds.get_data(subjects=[subject])
        assert subject in data
        assert len(data[subject]) == len(SESSIONS[subject])

    @pytest.mark.parametrize("subject", range(1, 9))
    def test_session_count_matches(self, subject):
        ds = Bnci2014FnirsMA()
        data = ds.get_data(subjects=[subject])
        for sess_num in SESSIONS[subject]:
            raw = data[subject][f"{sess_num}fnirs"]["0"]
            assert len(raw.ch_names) == N_CHANNELS
