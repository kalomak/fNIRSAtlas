"""Tests for the Bonnal2025FnirsME dataset."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.bonnal2025_fnirs_me import Bonnal2025FnirsME
from fnirsatlas.paradigms import FNIRSClassification

SUBJECTS = list(range(1, 50))
EVENTS = {
    "Distal/Left": 1,
    "Distal/Right": 2,
    "Manipulation/Left": 3,
    "Manipulation/Right": 4,
}
SFREQ = 50.0
N_PAIRS = 24
# Most subjects have 8 blocks per class. Subject 12 (source D_16) has 6.
BLOCKS_PER_CLASS = 8


def test_dataset_metadata():
    dataset = Bonnal2025FnirsME()
    assert dataset.subject_list == SUBJECTS
    assert dataset.event_id == EVENTS
    assert dataset.interval == [-2, 25]
    assert dataset.paradigm == "fnirsMotorExecution"
    assert dataset.code == "Bonnal2025FnirsME"
    assert dataset.doi == "10.1016/j.neuroimage.2025.121578"


@pytest.fixture(scope="module")
def data():
    return Bonnal2025FnirsME().get_data(subjects=[12])


def _label_counts(raw):
    stim = raw.get_data(picks="Stim").ravel()
    return dict(zip(*np.unique(stim[stim > 0].astype(int), return_counts=True)))


@pytest.mark.data
class TestRealSubjectData:
    @pytest.fixture(scope="class")
    @classmethod
    def raw(cls, data):
        return data[12]["0fnirs"]["0"]

    def test_structure(self, data):
        assert list(data) == [12]
        assert list(data[12]) == ["0fnirs"]
        assert list(data[12]["0fnirs"]) == ["0"]

    def test_recording_properties(self, raw):
        assert raw.info["sfreq"] == SFREQ
        picks = mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude")
        assert len(picks) == 2 * N_PAIRS
        # Paper channels 1 and 2 are S1-D1 and S2-D1 in the right patch.
        assert raw.ch_names[:4] == [
            "S1_D1 757",
            "S1_D1 842",
            "S2_D1 757",
            "S2_D1 842",
        ]
        assert raw.ch_names[-1] == "Stim"
        assert np.all(raw.get_data(picks=picks) > 0)

    def test_no_samples_at_intensity_floor(self, raw):
        # Brite recordings start with samples at 2**-16, which are cropped.
        assert np.all(raw.get_data(picks="fnirs") > 2**-16)

    def test_montage_has_30mm_channels(self, raw):
        for ch in raw.copy().pick("fnirs_cw_amplitude").info["chs"]:
            source, detector = ch["loc"][3:6], ch["loc"][6:9]
            assert np.linalg.norm(source - detector) == pytest.approx(0.03, abs=5e-3)

    def test_patches_are_on_opposite_hemispheres(self, raw):
        # Sources 1-5 are the right patch, 6-10 the left (RAS head frame).
        positions = {
            ch["ch_name"].split("_")[0]: ch["loc"][3]
            for ch in raw.copy().pick("fnirs_cw_amplitude").info["chs"]
        }
        assert all(positions[f"S{i}"] > 0 for i in range(1, 6))
        assert all(positions[f"S{i}"] < 0 for i in range(6, 11))

    def test_events(self, raw):
        assert _label_counts(raw) == dict.fromkeys(EVENTS.values(), 6)

    def test_all_epochs_fit(self):
        # Keep bad channels as they are, so only the epoch timing is tested.
        paradigm = FNIRSClassification(
            events=list(EVENTS),
            preprocessing={"max_bad_fraction": None, "interpolate_bads": None},
        )
        _, labels, _ = paradigm.get_data(Bonnal2025FnirsME(), subjects=[30])
        assert dict(zip(*np.unique(labels, return_counts=True))) == dict.fromkeys(
            EVENTS, BLOCKS_PER_CLASS
        )


@pytest.mark.data
@pytest.mark.manual
class TestAllSubjects:
    @pytest.mark.parametrize("subject", SUBJECTS)
    def test_subject_loads(self, subject):
        raw = Bonnal2025FnirsME().get_data(subjects=[subject])[subject]["0fnirs"]["0"]
        assert raw.info["sfreq"] == SFREQ
        assert len(raw.ch_names) == 2 * N_PAIRS + 1
        assert np.isfinite(raw.get_data()).all()
        counts = _label_counts(raw)
        assert list(counts) == list(EVENTS.values())
        # 24-32 blocks per recording, 6-9 per class.
        assert all(6 <= count <= 9 for count in counts.values())
