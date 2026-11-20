"""Tests for the Liu2020 fNIRS datasets."""

import shutil
from collections import Counter

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.liu2020_fnirs import (
    SFREQ,
    Liu2020FnirsControl,
    Liu2020FnirsStroke,
)

# Extracting the Liu2020 download needs the external 7z tool.
NEEDS_7Z = pytest.mark.xfail(
    shutil.which("7z") is None and shutil.which("7zz") is None,
    reason="7z is not installed",
    raises=RuntimeError,
)


@pytest.mark.parametrize(
    ("dataset", "subjects", "events", "code"),
    [
        (
            Liu2020FnirsControl(),
            list(range(1, 10)),
            {"left hand": 1, "right hand": 2},
            "Liu2020FnirsControl",
        ),
        (
            Liu2020FnirsStroke(),
            list(range(1, 11)),
            {"affected hand": 1, "unaffected hand": 2},
            "Liu2020FnirsStroke",
        ),
    ],
)
def test_init(dataset, subjects, events, code):
    assert dataset.subject_list == subjects
    assert dataset.event_id == events
    assert dataset.interval == [-2, 14.99]
    assert dataset.paradigm == "fnirsMotorExecution"
    assert dataset.code == code


@pytest.mark.data
@NEEDS_7Z
@pytest.mark.parametrize("dataset_class", [Liu2020FnirsControl, Liu2020FnirsStroke])
def test_data(dataset_class):
    raw = dataset_class().get_data(subjects=[1])[1]["0fnirs"]["0"]
    assert isinstance(raw, mne.io.BaseRaw)
    assert raw.info["sfreq"] == pytest.approx(SFREQ)
    assert len(mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude")) == 408
    assert len(mne.pick_types(raw.info, stim=True)) == 1
    assert not np.allclose(raw.get_data(picks="fnirs_cw_amplitude"), 0)
    assert np.isfinite(raw.get_data()).all()
    assert (raw.get_data(picks="fnirs_cw_amplitude") < 0).any()


@pytest.mark.data
@NEEDS_7Z
@pytest.mark.parametrize(
    ("dataset_class", "subject"),
    [(Liu2020FnirsControl, subject) for subject in range(1, 10)]
    + [(Liu2020FnirsStroke, subject) for subject in range(1, 11)],
)
def test_all_subjects_load(dataset_class, subject):
    dataset = dataset_class()
    data = dataset.get_data(subjects=[subject])
    runs = data[subject]["0fnirs"]
    assert set(runs) == {"0", "1"}
    event_codes = np.concatenate(
        [mne.find_events(raw, verbose=False)[:, 2] for raw in runs.values()]
    )
    event_names = {code: name for name, code in dataset.event_id.items()}
    event_counts = Counter(event_names[code] for code in event_codes)
    assert set(event_counts) == set(dataset.event_id)
    assert set(event_counts.values()).issubset({5, 6})
