"""Tests for the Luke2021FnirsSpeech dataset."""

from pathlib import Path

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.luke2021_fnirs_speech import Luke2021FnirsSpeech

pytestmark = pytest.mark.data

SUBJECTS = list(range(1, 18))
EVENTS = {"Control": 1, "Noise": 2, "Speech": 3}
SFREQ = 5.208333333333333
LOCAL_BIDS_ROOT = (
    Path.home() / "mne_data" / "fnirsatlas" / "datasets" / "Luke2024a" / "data"
)


def _local_download_subject(
    self, subject, path=None, force_update=False, update_path=None, verbose=None
):
    del self, subject, path, force_update, update_path, verbose
    return str(LOCAL_BIDS_ROOT)


@pytest.fixture(scope="module")
def data():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            Luke2021FnirsSpeech, "_download_subject", _local_download_subject
        )
        return Luke2021FnirsSpeech().get_data(subjects=[1])


class TestRawProperties:
    @pytest.fixture(scope="class")
    @classmethod
    def raw(cls, data):
        return data[1]["01"]["0"]

    def test_sfreq(self, raw):
        assert raw.info["sfreq"] == pytest.approx(SFREQ)

    def test_cw_amplitude_channels(self, raw):
        fnirs = mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude")
        assert len(fnirs) == 66

    def test_stim_channel(self, raw):
        stim = mne.pick_types(raw.info, stim=True)
        assert len(stim) == 1
        assert raw.ch_names[stim[0]] == "Stim"

    def test_channel_name_format(self, raw):
        names_760 = [
            raw.ch_names[pick]
            for pick in mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude")
            if raw.ch_names[pick].endswith(" 760")
        ]
        assert len(names_760) == 33
        for name in names_760:
            assert f"{name[:-4]} 850" in raw.ch_names

    def test_data_not_all_zeros(self, raw):
        fnirs = mne.pick_types(raw.info, fnirs=True)
        assert not np.allclose(raw.get_data(picks=fnirs), 0)


class TestAnnotations:
    @pytest.fixture(scope="class")
    @classmethod
    def stim_data(cls, data):
        return data[1]["01"]["0"].get_data(picks="Stim").ravel()

    def test_stim_has_events(self, stim_data):
        assert np.any(stim_data > 0)

    def test_event_labels(self, stim_data):
        labels = stim_data[stim_data > 0].astype(int)
        assert set(labels) == set(EVENTS.values())

    def test_event_counts(self, stim_data):
        labels = stim_data[stim_data > 0].astype(int)
        assert dict(zip(*np.unique(labels, return_counts=True))) == {
            1: 20,
            2: 20,
            3: 20,
        }


class TestAllSubjects:
    @pytest.mark.parametrize("subject", SUBJECTS)
    def test_subject_loads(self, monkeypatch, subject):
        monkeypatch.setattr(
            Luke2021FnirsSpeech, "_download_subject", _local_download_subject
        )
        data = Luke2021FnirsSpeech().get_data(subjects=[subject])
        assert subject in data
        assert "01" in data[subject]
        raw = data[subject]["01"]["0"]
        assert isinstance(raw, mne.io.BaseRaw)
        assert len(raw.ch_names) == 67
