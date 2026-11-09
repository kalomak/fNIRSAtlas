"""Behavioral tests for the Shin2018 fNIRS datasets."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.shin2018_fnirs import (
    Shin2018FnirsNBack,
    Shin2018FnirsWG,
)

SUBJECTS = list(range(1, 27))
N_PAIRS = 36
# Event counts per subject over all three runs.
EVENT_COUNTS = {
    Shin2018FnirsWG: {1: 30, 2: 30},
    Shin2018FnirsNBack: {1: 9, 2: 9, 3: 9},
}


@pytest.fixture(params=[Shin2018FnirsWG, Shin2018FnirsNBack])
def dataset_and_runs(request):
    dataset = request.param()
    return dataset, dataset.get_data(subjects=[1])[1]["0fnirs"]


@pytest.mark.data
def test_raw_properties(dataset_and_runs):
    _, runs = dataset_and_runs
    assert list(runs) == ["0", "1", "2"]
    for raw in runs.values():
        assert raw.info["sfreq"] == pytest.approx(10.416667)
        assert len(raw.ch_names) == 2 * N_PAIRS + 1
        assert len(mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude")) == 2 * N_PAIRS
        distances = mne.preprocessing.nirs.source_detector_distances(raw.info)
        assert np.all((distances > 0.02) & (distances < 0.035))


@pytest.mark.data
def test_haemoglobin_scale(dataset_and_runs):
    _, runs = dataset_and_runs
    od = mne.preprocessing.nirs.optical_density(runs["0"])
    hbo = mne.preprocessing.nirs.beer_lambert_law(od).get_data(picks="hbo")
    # Subject 1 HbO std is about 0.6 uM. The vendor-converted MATLAB release,
    # which overstates concentrations 4.6-fold, gives 2.5 uM.
    assert 1e-7 < np.median(np.std(hbo, axis=1)) < 1.5e-6


def test_dataset_metadata():
    wg = Shin2018FnirsWG()
    nback = Shin2018FnirsNBack()
    assert wg.subject_list == nback.subject_list == SUBJECTS
    assert wg.event_id == {"word_generation": 1, "baseline": 2}
    assert nback.event_id == {"0-back": 1, "2-back": 2, "3-back": 3}
    assert wg.interval == [-2, 12]
    assert nback.interval == [-2, 42]


@pytest.mark.data
def test_event_counts(dataset_and_runs):
    dataset, runs = dataset_and_runs
    labels = np.concatenate(
        [raw.get_data(picks="Stim").ravel() for raw in runs.values()]
    )
    labels = labels[labels > 0].astype(int)
    counts = dict(zip(*np.unique(labels, return_counts=True)))
    assert counts == EVENT_COUNTS[type(dataset)]


@pytest.mark.parametrize("dataset_class", [Shin2018FnirsWG, Shin2018FnirsNBack])
@pytest.mark.parametrize("subject", SUBJECTS)
@pytest.mark.data
def test_all_subjects_load(dataset_class, subject):
    runs = dataset_class().get_data(subjects=[subject])[subject]["0fnirs"]
    assert len(runs) == 3
    assert all(len(raw.ch_names) == 2 * N_PAIRS + 1 for raw in runs.values())
