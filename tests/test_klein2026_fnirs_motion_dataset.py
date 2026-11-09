"""Tests for the Klein2026FnirsMotion dataset."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.klein2026_fnirs_motion import Klein2026FnirsMotion
from fnirsatlas.datasets.preprocessing import mark_bad_segments_by_flat

pytestmark = pytest.mark.data

SUBJECTS = list(range(1, 27))
RUNS = ["0hands", "1headmovement", "2eyes", "3stimulation", "4silence"]
AUDITORY = ["auditory_50db", "auditory_70db", "auditory_90db"]
SFREQ = 10.0


@pytest.fixture(scope="module")
def data():
    return Klein2026FnirsMotion().get_data(subjects=[1])


def test_init_properties():
    dataset = Klein2026FnirsMotion()
    assert dataset.subject_list == SUBJECTS
    assert dataset.code == "Klein2026FnirsMotion"
    assert dataset.paradigm == "fnirsMotionArtifact"
    assert dataset.interval is None
    assert set(dataset.event_id) == {
        *AUDITORY,
        "hand_movement",
        "head_movement",
        "eye_movement",
    }


def test_structure(data):
    assert list(data) == [1]
    assert list(data[1]) == ["0"]
    assert list(data[1]["0"]) == RUNS


class TestRawProperties:
    @pytest.fixture(scope="class")
    @classmethod
    def raw(cls, data):
        return data[1]["0"]["4silence"]

    def test_sfreq(self, raw):
        assert raw.info["sfreq"] == SFREQ

    def test_duration_close_to_91_s(self, raw):
        assert raw.times[-1] == pytest.approx(91, abs=2)

    def test_cw_amplitude_channels(self, raw):
        fnirs = mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude", exclude=[])
        assert len(fnirs) == 16

    def test_constant_pairs_are_flat(self, raw):
        out = mark_bad_segments_by_flat(raw.copy(), threshold=1e-6)
        flat = {
            name.rsplit(" ", 1)[0]
            for names in out.annotations.ch_names
            for name in names
        }
        assert flat == {f"S{s}_D1" for s in range(1, 5)}

    def test_channel_names(self, raw):
        pairs = [f"S{s}_D1" for s in range(1, 5)] + [f"S{s}_D2" for s in range(5, 9)]
        expected = [f"{pair} {wl}" for pair in pairs for wl in (760, 850)]
        assert raw.ch_names == [*expected, "Stim_auditory", "Stim_movement"]


class TestAnnotations:
    def test_silence_has_no_events(self, data):
        raw = data[1]["0"]["4silence"]
        assert len(raw.annotations) == 0
        assert not np.any(raw.get_data(picks="stim"))

    def test_stimulation_has_auditory_pulses(self, data):
        raw = data[1]["0"]["3stimulation"]
        assert list(raw.annotations.description) == AUDITORY
        np.testing.assert_allclose(raw.annotations.duration, 2.0)

    @pytest.mark.parametrize(
        ("run", "label", "source_onset"),
        [
            ("0hands", "hand_movement", 9.0),
            ("1headmovement", "head_movement", 6.0),
            ("2eyes", "eye_movement", 2.0),
        ],
    )
    def test_movement_run_events(self, data, run, label, source_onset):
        raw = data[1]["0"][run]
        descriptions = list(raw.annotations.description)
        assert sorted(descriptions) == sorted([*AUDITORY, label])
        # Onsets are shifted by the dropped startup samples, so the gap to the
        # first auditory pulse (source time 30 s) is preserved.
        onsets = dict(zip(descriptions, raw.annotations.onset, strict=True))
        assert onsets["auditory_50db"] - onsets[label] == pytest.approx(
            30.0 - source_onset
        )

    def test_stim_matches_annotations(self, data):
        raw = data[1]["0"]["0hands"]
        events = mne.find_events(raw, verbose=False)
        np.testing.assert_allclose(
            np.sort(events[:, 0]) / SFREQ,
            np.sort(raw.annotations.onset),
            atol=1 / SFREQ,
        )

    def test_coinciding_onsets_keep_both_events(self):
        # Participant 3 pressed the hand-movement key at the 30 s pulse onset.
        raw = Klein2026FnirsMotion().get_data(subjects=[3])[3]["0"]["0hands"]
        auditory = mne.find_events(raw, "Stim_auditory", verbose=False)
        movement = mne.find_events(raw, "Stim_movement", verbose=False)
        assert auditory[0, 0] == movement[0, 0]
        assert auditory[:, 2].tolist() == [1, 2, 3]
        assert movement[:, 2].tolist() == [4]


class TestAllSubjects:
    @pytest.mark.parametrize("subject", SUBJECTS)
    def test_subject_loads(self, subject):
        data = Klein2026FnirsMotion().get_data(subjects=[subject])
        runs = data[subject]["0"]
        assert list(runs) == RUNS
        for raw in runs.values():
            assert isinstance(raw, mne.io.BaseRaw)
            assert len(raw.ch_names) == 18
