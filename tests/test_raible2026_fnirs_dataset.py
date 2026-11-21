"""Tests for the Raible2026 MULPA fNIRS datasets."""

from collections import Counter

import mne
import numpy as np
import pytest
from mne.preprocessing.nirs import source_detector_distances

from fnirsatlas.datasets.raible2026_fnirs import (
    Raible2026FnirsEmotion,
    Raible2026FnirsME,
    Raible2026FnirsMI,
    Raible2026FnirsMIFQ,
    Raible2026FnirsMusic,
    Raible2026FnirsRest,
    Raible2026FnirsThreat,
)

pytestmark = pytest.mark.data

SUBJECTS = list(range(1, 58))
# Subjects with a recording of the task.
TASK_SUBJECTS = {
    Raible2026FnirsEmotion: [s for s in SUBJECTS if s not in {49, 50, 51, 54, 55}],
    Raible2026FnirsThreat: [s for s in SUBJECTS if s not in {24, 57}],
    Raible2026FnirsMusic: [s for s in SUBJECTS if s != 57],
}
SFREQ = 1 / 0.07936
N_PAIRS = 134
N_SHORT_PAIRS = 32
# Recordings per dataset after excluding stubs and missing tasks.
RECORDINGS = {
    Raible2026FnirsME: 57,
    Raible2026FnirsMI: 57,
    Raible2026FnirsMIFQ: 57,
    Raible2026FnirsEmotion: 52,
    Raible2026FnirsThreat: 55,
    Raible2026FnirsMusic: 110,
    Raible2026FnirsRest: 57,
}
# Event totals across all subjects. Subject 56's motor-imagery recording
# ends during its fifth block. Subject 57's emotion recording is partial.
EVENT_COUNTS = {
    Raible2026FnirsME: {"finger_tapping": 57 * 9, "baseline": 57 * 10},
    Raible2026FnirsMI: {"motor_imagery": 56 * 9 + 4, "baseline": 56 * 10 + 4},
    Raible2026FnirsMIFQ: {"motor_imagery_speed_change": 57 * 9, "baseline": 57 * 10},
    Raible2026FnirsEmotion: {
        "neutral_happy": 308,
        "neutral_sad": 308,
        "neutral_neutral": 307,
    },
    Raible2026FnirsThreat: {"threatening": 275, "non_threatening": 275, "control": 275},
    Raible2026FnirsMusic: {
        "positive_valence_high_arousal": 106,
        "positive_valence_low_arousal": 106,
        "negative_valence_high_arousal": 107,
        "negative_valence_low_arousal": 108,
    },
}


def _raws(data):
    return [
        raw
        for sessions in data.values()
        for runs in sessions.values()
        for raw in runs.values()
    ]


@pytest.fixture(scope="module", params=list(RECORDINGS), ids=lambda c: c.__name__)
def dataset_data(request):
    dataset = request.param()
    return dataset, dataset.get_data()


@pytest.fixture(scope="module")
def raw():
    return Raible2026FnirsME().get_data(subjects=[1])[1]["0"]["0"]


class TestRawProperties:
    def test_structure(self, raw):
        assert isinstance(raw, mne.io.BaseRaw)

    def test_sfreq(self, raw):
        assert raw.info["sfreq"] == pytest.approx(SFREQ)

    def test_cw_amplitude_channels(self, raw):
        names = [
            raw.ch_names[i]
            for i in mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude")
        ]
        assert len(names) == 2 * N_PAIRS
        for name in names:
            if name.endswith(" 760"):
                assert f"{name[:-4]} 850" in raw.ch_names

    def test_short_channels(self, raw):
        distances = source_detector_distances(raw.info, picks="fnirs")
        assert np.sum(distances < 0.015) == 2 * N_SHORT_PAIRS
        assert np.all(distances < 0.06)

    def test_fiducials_in_meters(self, raw):
        nasion = raw.copy().pick("fnirs").get_montage().get_positions()["nasion"]
        assert np.allclose(nasion, [0.0004, 0.0859, -0.0476])

    def test_stim_channel(self, raw):
        stim = mne.pick_types(raw.info, stim=True)
        assert [raw.ch_names[i] for i in stim] == ["Stim"]

    def test_signal_is_positive_intensity(self, raw):
        assert np.all(raw.get_data(picks="fnirs") > 0)


class TestAllSubjects:
    def test_recordings(self, dataset_data):
        dataset, data = dataset_data
        assert set(data) == set(TASK_SUBJECTS.get(type(dataset), SUBJECTS))
        assert len(_raws(data)) == RECORDINGS[type(dataset)]

    def test_event_counts(self, dataset_data):
        dataset, data = dataset_data
        if dataset.interval is None:
            pytest.skip("Recording-only dataset")
        codes = {code: name for name, code in dataset.event_id.items()}
        counts = Counter()
        for raw in _raws(data):
            stim = raw.get_data(picks="Stim").ravel()
            counts.update(codes[int(code)] for code in stim[stim > 0])
        assert counts == EVENT_COUNTS[type(dataset)]
