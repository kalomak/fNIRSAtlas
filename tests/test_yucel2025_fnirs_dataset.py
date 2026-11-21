"""Tests for the Yucel2025FnirsSqueezing and Yucel2025FnirsRest datasets."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.yucel2025_fnirs import (
    Yucel2025FnirsRest,
    Yucel2025FnirsSqueezing,
)

SUBJECTS = [225, 229, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 252, 253, 259, 266, 269, 270, 274, 276, 277, 286, 287, 291, 293, 304, 326, 327, 332, 338, 341, 343, 349, 350, 356, 357, 362, 368, 369, 370, 371, 377, 380, 381, 386, 390, 391, 392, 394, 396, 397, 400, 401, 402, 403, 404, 405, 406, 407, 411, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 437, 438, 439, 440, 441, 442, 443, 449, 450, 451, 452, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 472, 473, 474, 475]  # fmt: skip
EVENT_COUNT_EXCEPTIONS = {253: 29, 270: 29, 274: 29}
# LH recordings that stop before the last RH events.
BOTH_EVENT_COUNT_EXCEPTIONS = {
    249: 29,
    252: 29,
    270: 28,
    274: 28,
    291: 29,
    370: 15,
    394: 28,
}
DEVICES = [("RH",), ("LH",), ("LH", "RH")]


def test_device_selection_sets_code_and_recordings():
    left = Yucel2025FnirsSqueezing(devices=("LH",))
    both = Yucel2025FnirsSqueezing(devices=("RH", "LH"))

    assert left.code == "Yucel2025FnirsSqueezing-lh"
    assert both.code == "Yucel2025FnirsSqueezing-lh-rh"
    assert "sub-225/nirs/sub-225_task-009BallLH_run-1_nirs.snirf" in (
        left._openneuro_file_paths(225)
    )
    assert not any("BallRH" in path for path in left._openneuro_file_paths(225))
    assert {"009BallLH", "010BallRH"} <= {
        path.split("task-")[1].split("_")[0]
        for path in both._openneuro_file_paths(225)
        if "task-" in path
    }


@pytest.fixture(scope="module")
def real_run():
    return Yucel2025FnirsSqueezing().get_data(subjects=[225])[225]["0"]["1"]


@pytest.mark.data
@pytest.mark.manual
class TestRealSubjectData:
    def test_structure(self, real_run):
        assert isinstance(real_run, mne.io.BaseRaw)

    def test_recording_properties(self, real_run):
        assert real_run.info["sfreq"] == pytest.approx(10.172526041666666)
        assert real_run.get_channel_types().count("fnirs_cw_amplitude") == 52
        assert real_run.ch_names[-1] == "Stim"

    def test_events(self, real_run):
        labels = real_run.get_data(picks="Stim").ravel()
        labels = labels[labels > 0].astype(int)

        assert set(real_run.annotations.description) == {"left", "right"}
        assert len(real_run.annotations) == 30
        assert dict(zip(*np.unique(labels, return_counts=True))) == {1: 15, 2: 15}


@pytest.mark.data
@pytest.mark.manual
@pytest.mark.parametrize("subject", SUBJECTS)
def test_all_subjects_load_both_devices(subject):
    raw = Yucel2025FnirsSqueezing(devices=("LH", "RH")).get_data(subjects=[subject])[
        subject
    ]["0"]["1"]
    distances = mne.preprocessing.nirs.source_detector_distances(
        raw.info, picks="fnirs"
    )
    assert len(raw.ch_names) == 105
    assert (distances < 0.015).sum() == 32
    assert len(raw.annotations) == BOTH_EVENT_COUNT_EXCEPTIONS.get(
        subject, EVENT_COUNT_EXCEPTIONS.get(subject, 30)
    )


@pytest.mark.data
@pytest.mark.manual
@pytest.mark.parametrize("devices", DEVICES)
@pytest.mark.parametrize("subject", SUBJECTS)
def test_all_subjects_load_rest(subject, devices):
    runs = Yucel2025FnirsRest(devices=devices).get_data(subjects=[subject])[subject][
        "0"
    ]
    # The sub-236 run 2 recordings have no events, so the devices cannot be aligned.
    skipped = subject == 236 and len(devices) == 2
    assert set(runs) == ({"1"} if skipped else {"1", "2"})
    for raw in runs.values():
        assert len(raw.ch_names) == 52 * len(devices) + 1
        assert set(raw.annotations.description) <= {"rest"}
