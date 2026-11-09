"""Behavioral tests for the Guevara2024 fNIRS dataset."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.guevara2024_fnirs_parkinson import Guevara2024FnirsParkinson
from fnirsatlas.paradigms import FNIRSClassification

SUBJECTS = list(range(1, 21))


@pytest.fixture(scope="module")
def raw():
    return Guevara2024FnirsParkinson().get_data(subjects=[1])[1]["0fnirs"]["0"]


def test_dataset_metadata():
    dataset = Guevara2024FnirsParkinson()
    assert dataset.subject_list == SUBJECTS
    assert dataset.event_id == {"Tapping/Left": 1, "Tapping/Right": 2}
    assert dataset.interval == [-2, 15]


def test_dataset_is_registered():
    assert any(
        isinstance(dataset, Guevara2024FnirsParkinson)
        for dataset in FNIRSClassification().datasets
    )


@pytest.mark.data
def test_raw_properties(raw):
    assert raw.info["sfreq"] == 50
    assert len(mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude")) == 44
    assert raw.ch_names[:4] == [
        "S1_D1 757",
        "S1_D1 843",
        "S3_D1 757",
        "S3_D1 843",
    ]
    assert raw.ch_names[-1] == "Stim"


@pytest.mark.data
def test_no_samples_at_intensity_floor(raw):
    # Brite recordings start with samples at 2**-16, which are cropped.
    assert np.all(raw.get_data(picks="fnirs") > 2**-16)


@pytest.mark.data
def test_source_detector_distances(raw):
    picks = mne.pick_types(raw.info, fnirs=True)
    distances = mne.preprocessing.nirs.source_detector_distances(raw.info, picks)
    assert np.count_nonzero(distances < 0.02) == 4
    assert np.all((distances > 0.014) & (distances < 0.034))


@pytest.mark.data
def test_events_are_canonical_and_deduplicated(raw):
    assert set(raw.annotations.description) == {"Tapping/Left", "Tapping/Right"}
    assert len(raw.annotations) == 19
    stim = raw.get_data(picks="Stim").ravel()
    assert dict(zip(*np.unique(stim[stim > 0], return_counts=True))) == {
        1.0: 10,
        2.0: 9,
    }


@pytest.mark.data
def test_control18_sampling_rate_correction():
    raw = Guevara2024FnirsParkinson().get_data(subjects=[18])[18]["0fnirs"]["0"]
    assert raw.info["sfreq"] == 12.5
    # Times are relative to the first kept sample, 4 samples (0.32 s) in.
    first_event = np.flatnonzero(raw.get_data(picks="Stim"))[0]
    assert first_event / raw.info["sfreq"] == pytest.approx(26.64)
    assert raw.times[-1] == pytest.approx(657.52, abs=1e-5)


@pytest.mark.parametrize("subject", SUBJECTS)
@pytest.mark.data
def test_all_subjects_load(subject):
    raw = Guevara2024FnirsParkinson().get_data(subjects=[subject])[subject]["0fnirs"][
        "0"
    ]
    assert isinstance(raw, mne.io.BaseRaw)
    assert len(raw.ch_names) == 45
