"""Tests for the Klinkhammer2026FnirsME dataset."""

import mne
import numpy as np
import pytest
from mne.preprocessing.nirs import source_detector_distances

from fnirsatlas.datasets.klinkhammer2026_fnirs_me import Klinkhammer2026FnirsME

pytestmark = pytest.mark.data

MISSING = {10, 11, 47, 54, 62, 73, 79, 89, 104}
SUBJECTS = [subject for subject in range(1, 110) if subject not in MISSING]
SFREQ = 1 / 0.098304


@pytest.fixture(scope="module")
def data():
    return Klinkhammer2026FnirsME().get_data(subjects=[1])


def test_data_structure(data):
    assert list(data) == [1]
    assert list(data[1]) == ["0"]
    assert sorted(data[1]["0"]) == ["1", "2"]


class TestRawProperties:
    @pytest.fixture(scope="class")
    @classmethod
    def raw(cls, data):
        return data[1]["0"]["1"]

    def test_sfreq(self, raw):
        assert raw.info["sfreq"] == pytest.approx(SFREQ)

    def test_channels(self, raw):
        types = raw.get_channel_types()
        assert types.count("fnirs_cw_amplitude") == 26
        assert types.count("stim") == 1
        assert raw.ch_names[:4] == ["S1_D1 760", "S1_D1 850", "S1_D3 760", "S1_D3 850"]

        distances = source_detector_distances(raw.info, picks="fnirs")
        assert np.sum(distances < 0.015) == 2
        assert np.all(distances[distances >= 0.015] > 0.03)

    def test_data_is_positive_intensity(self, raw):
        assert np.all(raw.get_data(picks="fnirs") > 0)


def test_events(data):
    for raw in data[1]["0"].values():
        stim = raw.get_data(picks="Stim").ravel()
        labels = stim[stim > 0].astype(int)
        # Blocks alternate, starting with hand clenching.
        assert labels.tolist() == [1, 2] * 5


@pytest.mark.parametrize("subject", SUBJECTS)
def test_all_subjects_load(subject):
    runs = Klinkhammer2026FnirsME().get_data(subjects=[subject])[subject]["0"]
    assert sorted(runs) == ["1", "2"]
    for raw in runs.values():
        assert isinstance(raw, mne.io.BaseRaw)
        assert len(raw.ch_names) == 27
