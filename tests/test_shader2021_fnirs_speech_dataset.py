"""Tests for the Shader2021FnirsSpeech dataset."""

from pathlib import Path

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.shader2021_fnirs_speech import Shader2021FnirsSpeech

pytestmark = pytest.mark.data

SUBJECTS = list(range(1, 9))
EVENTS = {"Audio": 1, "Control": 2, "Video": 3}
N_FNIRS = 104
N_CHANNELS = N_FNIRS + 1
SFREQ = 3.90625
LOCAL_BIDS_ROOT = (
    Path.home()
    / "mne_data/fnirsatlas/datasets/Shader2021"
    / "2021-fNIRS-Audio-visual-speech-Broad-vs-restricted-regions"
)


def _local_download_subject(
    self, subject, path=None, force_update=False, update_path=None, verbose=None
):
    del self, subject, path, force_update, update_path, verbose
    return str(LOCAL_BIDS_ROOT)


@pytest.fixture(scope="module")
def raw():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            Shader2021FnirsSpeech, "_download_subject", _local_download_subject
        )
        return Shader2021FnirsSpeech().get_data(subjects=[1])[1]["01"]["0"]


def test_raw_properties(raw):
    assert raw.info["sfreq"] == pytest.approx(SFREQ)
    assert len(raw.ch_names) == N_CHANNELS
    assert len(mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude")) == N_FNIRS
    assert len(mne.pick_types(raw.info, stim=True)) == 1
    assert not np.allclose(raw.get_data(picks="fnirs_cw_amplitude"), 0)


@pytest.fixture(scope="module")
def stim_data(raw):
    return raw.get_data(picks=mne.pick_types(raw.info, stim=True)).ravel()


def test_event_counts(stim_data):
    labels = stim_data[stim_data > 0].astype(int)
    assert dict(zip(*np.unique(labels, return_counts=True))) == {
        1: 18,
        2: 10,
        3: 18,
    }


@pytest.mark.parametrize("subject", SUBJECTS)
def test_all_subjects_load(monkeypatch, subject):
    monkeypatch.setattr(
        Shader2021FnirsSpeech, "_download_subject", _local_download_subject
    )
    raw = Shader2021FnirsSpeech().get_data(subjects=[subject])[subject]["01"]["0"]
    assert isinstance(raw, mne.io.BaseRaw)
    assert len(raw.ch_names) == N_CHANNELS
