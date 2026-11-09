"""Tests for Nemo2024FnirsPerception dataset."""

from pathlib import Path

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.nemo2024_fnirs import (
    EMPE_EVENTS,
    SUBJECTS,
    Nemo2024FnirsPerception,
)

SFREQ = 50.0
N_FNIRS = 48
N_CHANNELS = N_FNIRS + 1
LOCAL_BIDS_ROOT = Path.home() / "mne_data" / "MNE-nemo2024fnirs-data" / "nemo-bids"


def _local_download_subject(
    self, subject, path=None, force_update=False, update_path=None, verbose=None
):
    return str(LOCAL_BIDS_ROOT)


@pytest.fixture
def dataset(monkeypatch):
    monkeypatch.setattr(
        Nemo2024FnirsPerception, "_download_subject", _local_download_subject
    )
    return Nemo2024FnirsPerception()


@pytest.fixture
def data(dataset):
    return dataset.get_data(subjects=[101])


class TestDatasetInit:
    def test_subject_list(self, dataset):
        assert dataset.subject_list == SUBJECTS

    def test_paradigm(self, dataset):
        assert dataset.paradigm == "fnirsEmotion"

    def test_events(self, dataset):
        assert dataset.event_id == EMPE_EVENTS

    def test_interval(self, dataset):
        assert dataset.interval == [-2, 15]

    def test_code(self, dataset):
        assert dataset.code == "Nemo2024FnirsPerception"

    def test_subject_selection(self, monkeypatch):
        monkeypatch.setattr(
            Nemo2024FnirsPerception, "_download_subject", _local_download_subject
        )
        ds = Nemo2024FnirsPerception(subjects=[101, 105])
        assert ds.subject_list == [101, 105]


@pytest.mark.data
class TestDataStructure:
    def test_returns_dict(self, data):
        assert isinstance(data, dict)

    def test_subject_keys(self, data):
        assert 101 in data

    def test_session_count(self, data):
        assert len(data[101]) == 1

    def test_session_keys(self, data):
        assert "0" in data[101]

    def test_run_key(self, data):
        assert "0" in data[101]["0"]

    def test_raw_type(self, data):
        raw = data[101]["0"]["0"]
        assert isinstance(raw, mne.io.BaseRaw)


@pytest.mark.data
class TestRawProperties:
    @pytest.fixture
    def raw(self, data):
        return data[101]["0"]["0"]

    def test_sfreq(self, raw):
        assert raw.info["sfreq"] == pytest.approx(SFREQ)

    def test_n_channels(self, raw):
        assert len(raw.ch_names) == N_CHANNELS

    def test_fnirs_channels(self, raw):
        assert len(mne.pick_types(raw.info, fnirs="fnirs_od")) == N_FNIRS

    def test_stim_channel(self, raw):
        stim = mne.pick_types(raw.info, stim=True)
        assert len(stim) == 1
        assert raw.ch_names[stim[0]] == "Stim"

    def test_data_not_all_zeros(self, raw):
        data_arr = raw.get_data()
        assert not np.allclose(data_arr[:N_FNIRS], 0)


@pytest.mark.data
class TestAnnotations:
    @pytest.fixture
    def stim_data(self, data):
        raw = data[101]["0"]["0"]
        return raw.get_data(picks="Stim").flatten()

    def test_stim_has_events(self, stim_data):
        assert np.any(stim_data > 0)

    def test_event_labels(self, stim_data):
        unique = set(stim_data[stim_data > 0].astype(int))
        assert unique == set(EMPE_EVENTS.values())

    def test_event_counts(self, stim_data):
        labels = stim_data[stim_data > 0].astype(int)
        counts = np.unique(labels, return_counts=True)[1]
        assert np.all(counts == 10)


@pytest.mark.data
class TestAllSubjects:
    @pytest.mark.parametrize("subject", SUBJECTS)
    def test_subject_loads(self, monkeypatch, subject):
        monkeypatch.setattr(
            Nemo2024FnirsPerception, "_download_subject", _local_download_subject
        )
        data = Nemo2024FnirsPerception().get_data(subjects=[subject])
        assert subject in data
        assert "0" in data[subject]
        raw = data[subject]["0"]["0"]
        assert len(raw.ch_names) == N_CHANNELS
