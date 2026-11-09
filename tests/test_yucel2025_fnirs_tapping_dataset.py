"""Tests for the Yucel2025FnirsTapping dataset."""

from pathlib import Path

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.yucel2025_fnirs_tapping import Yucel2025FnirsTapping

SUBJECTS = list(range(1, 11))
EVENTS = {"left2s": 1, "right2s": 2, "left3s": 3, "right3s": 4}
RUNS = {
    "0left2s": "left2s",
    "1right2s": "right2s",
    "2left3s": "left3s",
    "3right3s": "right3s",
}
SFREQ = 8.928571428571429
LOCAL_BIDS_ROOT = (
    Path.home() / "mne_data" / "fnirsatlas" / "datasets" / "Luke2024b" / "data"
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
            Yucel2025FnirsTapping, "_download_subject", _local_download_subject
        )
        return Yucel2025FnirsTapping().get_data(subjects=[1])


def test_init_properties():
    dataset = Yucel2025FnirsTapping()
    assert dataset.subject_list == SUBJECTS
    assert dataset.event_id == EVENTS
    assert dataset.interval == [-2, 9.9]
    assert dataset.paradigm == "fnirsMotorExecution"
    assert dataset.code == "Yucel2025FnirsTapping"


@pytest.mark.data
class TestRawProperties:
    @pytest.fixture(scope="class")
    @classmethod
    def raw(cls, data):
        return data[1]["0"]["0left2s"]

    def test_sfreq(self, raw):
        assert raw.info["sfreq"] == pytest.approx(SFREQ)

    def test_fnirs_channels(self, raw):
        assert len(mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude")) == 136

    def test_stim_channel(self, raw):
        stim = mne.pick_types(raw.info, stim=True)
        assert len(stim) == 1
        assert raw.ch_names[stim[0]] == "Stim"

    def test_data_not_all_zeros(self, raw):
        fnirs = mne.pick_types(raw.info, fnirs=True)
        assert not np.allclose(raw.get_data(picks=fnirs), 0)


@pytest.mark.data
def test_reconstructed_events(data):
    runs = data[1]["0"]
    expected_counts = {"left2s": 25, "right2s": 30, "left3s": 25, "right3s": 30}
    for run, raw in runs.items():
        condition = RUNS[run]
        stim = raw.get_data(picks="Stim").ravel()
        labels = stim[stim > 0].astype(int)
        assert set(labels) == {EVENTS[condition]}
        assert len(labels) == expected_counts[condition]


@pytest.mark.data
class TestAllSubjects:
    @pytest.mark.parametrize("subject", SUBJECTS)
    def test_subject_loads(self, monkeypatch, subject):
        monkeypatch.setattr(
            Yucel2025FnirsTapping, "_download_subject", _local_download_subject
        )
        data = Yucel2025FnirsTapping().get_data(subjects=[subject])
        assert set(data[subject]["0"]) == set(RUNS)
        assert all(
            isinstance(raw, mne.io.BaseRaw) for raw in data[subject]["0"].values()
        )
