"""Tests for the processed-data cutoff heuristic."""

import mne
import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from fnirsatlas.datasets import (
    Bak2019FnirsTapping,
    Bnci2014FnirsMA,
    Luke2021FnirsTapping,
)
from scripts.check_processed_data import guess_filter


def _raw(data, sfreq):
    return mne.io.RawArray(data[None], mne.create_info(["S1_D1 hbo"], sfreq, "hbo"))


def test_guess_filter_detects_clear_cutoffs():
    sfreq = 10
    rng = np.random.default_rng(0)
    data = rng.standard_normal(sfreq * 600)
    bandpass = butter(4, [0.01, 0.1], btype="bandpass", fs=sfreq, output="sos")
    filtered = sosfiltfilt(bandpass, data)

    low, high = guess_filter(_raw(filtered, sfreq))

    assert 0.005 < low < 0.03
    assert 0.04 < high < 0.15


def test_guess_filter_rejects_unfiltered_data():
    sfreq = 10
    data = np.random.default_rng(0).standard_normal(sfreq * 600)

    assert guess_filter(_raw(data, sfreq)) is None


def test_guess_filter_requires_both_cutoffs():
    sfreq = 10
    rng = np.random.default_rng(0)
    lowpass = butter(4, 1, btype="lowpass", fs=sfreq, output="sos")
    low_passed = sosfiltfilt(lowpass, rng.standard_normal(sfreq * 600))

    assert guess_filter(_raw(low_passed, sfreq)) is None


@pytest.mark.manual
@pytest.mark.data
def test_filter_guess_accuracy_on_datasets():
    """Report relative cutoff errors on filtered real recordings."""
    dataset_classes = (
        Bak2019FnirsTapping,
        Bnci2014FnirsMA,
        Luke2021FnirsTapping,
    )
    relative_errors = []
    misses = []
    for dataset_class in dataset_classes:
        dataset = dataset_class()
        subject = dataset.subject_list[0]
        sessions = dataset.get_data(subjects=[subject])[subject]
        raw = next(iter(next(iter(sessions.values())).values()))
        picks = mne.pick_types(raw.info, fnirs=True)
        for expected in ((0.01, 0.1), (0.01, 0.2), (0.02, 0.1), (0.02, 0.2)):
            filtered = raw.copy()
            sos = butter(
                4, expected, btype="bandpass", fs=raw.info["sfreq"], output="sos"
            )
            filtered._data[picks] = sosfiltfilt(sos, raw._data[picks], axis=-1)
            guessed = guess_filter(filtered)
            if guessed is None:
                misses.append((dataset.code, expected))
                print(dataset.code, expected, "MISSED")
                continue
            errors = np.abs(np.asarray(guessed) - expected) / expected
            relative_errors.append(errors)
            print(dataset.code, expected, guessed, f"errors={errors * 100}")
    relative_errors = np.asarray(relative_errors)
    print("median relative errors (%)", np.median(relative_errors, axis=0) * 100)
    print("maximum relative errors (%)", relative_errors.max(axis=0) * 100)
    assert not misses
    assert np.all(relative_errors < 0.3)
