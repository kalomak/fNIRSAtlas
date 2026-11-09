"""Tests for the Carlton2026FnirsSqueezing dataset."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.carlton2026_fnirs_squeezing import Carlton2026FnirsSqueezing

SUBJECTS = [
    568,
    577,
    580,
    581,
    583,
    586,
    587,
    588,
    592,
    613,
    618,
    619,
    621,
    633,
    638,
    639,
    640,
]
EVENTS = {"left": 1, "right": 2}
SFREQ = 4.324324324324325
N_FNIRS = 2440
EVENT_COUNT_EXCEPTIONS = {(580, "01"): 13, (638, "01"): 9}


def test_dataset_metadata():
    dataset = Carlton2026FnirsSqueezing()
    assert dataset.subject_list == SUBJECTS
    assert dataset.event_id == EVENTS
    assert dataset.interval == [-2, 15]
    assert dataset.paradigm == "fnirsMotorExecution"
    assert dataset.doi == "10.1117/1.NPh.13.2.025001"


@pytest.fixture(scope="module")
def real_runs():
    return Carlton2026FnirsSqueezing().get_data(subjects=[568])[568]["0"]


@pytest.mark.data
@pytest.mark.manual
class TestRealSubjectData:
    def test_structure(self, real_runs):
        assert set(real_runs) == {"01", "02", "03"}
        assert all(isinstance(raw, mne.io.BaseRaw) for raw in real_runs.values())

    def test_recording_properties(self, real_runs):
        raw = real_runs["01"]
        assert raw.info["sfreq"] == pytest.approx(SFREQ)
        assert len(mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude")) == N_FNIRS
        stim = mne.pick_types(raw.info, stim=True)
        assert len(stim) == 1
        assert raw.ch_names[stim[0]] == "Stim"

    def test_wavelength_pairs(self, real_runs):
        names = real_runs["01"].copy().pick("fnirs_cw_amplitude").ch_names
        assert sum(name.endswith(" 760") for name in names) == 1220
        assert sum(name.endswith(" 850") for name in names) == 1220
        assert {name[:-4] for name in names if name.endswith(" 760")} == {
            name[:-4] for name in names if name.endswith(" 850")
        }

    @pytest.mark.parametrize("run", ["01", "02", "03"])
    def test_signal_is_finite_and_nonzero(self, real_runs, run):
        data = real_runs[run].get_data(picks="fnirs_cw_amplitude")
        assert np.isfinite(data).all()
        assert not np.allclose(data, 0)

    @pytest.mark.parametrize("run", ["01", "02", "03"])
    def test_events(self, real_runs, run):
        raw = real_runs[run]
        stim = raw.get_data(picks="Stim").ravel()
        labels = stim[stim > 0].astype(int)

        assert isinstance(raw, mne.io.BaseRaw)
        assert set(raw.annotations.description) == {"left", "right"}
        assert len(raw.annotations) == 14
        assert set(labels) == {1, 2}
        assert len(labels) == 14
        assert dict(zip(*np.unique(labels, return_counts=True))) == {1: 7, 2: 7}


@pytest.mark.data
@pytest.mark.manual
class TestAllSubjects:
    @pytest.mark.parametrize("subject", SUBJECTS)
    def test_subject_loads(self, subject):
        runs = Carlton2026FnirsSqueezing().get_data(subjects=[subject])[subject]["0"]
        assert set(runs) == {"01", "02", "03"}
        assert all(isinstance(raw, mne.io.BaseRaw) for raw in runs.values())
        assert all(len(raw.ch_names) == N_FNIRS + 1 for raw in runs.values())
        for run, raw in runs.items():
            data = raw.get_data(picks="fnirs_cw_amplitude")
            assert raw.info["sfreq"] == pytest.approx(SFREQ)
            assert np.isfinite(data).all()
            assert np.any(data != 0, axis=1).all()
            assert set(raw.annotations.description) == {"left", "right"}
            assert len(raw.annotations) == EVENT_COUNT_EXCEPTIONS.get(
                (subject, run), 14
            )
