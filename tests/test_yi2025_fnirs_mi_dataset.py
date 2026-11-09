"""Tests for the Yi2025FnirsMI dataset."""

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.yi2025_fnirs_mi import EVENTS, Yi2025FnirsMI


def test_dataset_metadata():
    dataset = Yi2025FnirsMI()
    assert dataset.subject_list == list(range(1, 19))
    assert dataset.event_id == EVENTS
    assert dataset.interval == [-2, 14.848]
    assert dataset.paradigm == "fnirsMotorImagery"
    assert dataset.code == "Yi2025FnirsMI"
    assert dataset.doi == "10.1038/s41597-025-05286-0"


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        (1, ["block1-4", "block5", "block6", "block7", "block8"]),
        (2, [f"block{i}" for i in range(1, 9)]),
        (
            10,
            [
                "block1",
                "block2",
                "block4",
                "block5",
                "block6",
                "block7",
                "block8",
            ],
        ),
    ],
)
def test_source_runs(subject, expected):
    assert Yi2025FnirsMI._source_runs(subject) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [("block1", "1block1"), ("block1-4", "1block1to4")],
)
def test_run_key(source, expected):
    assert Yi2025FnirsMI._run_key(source) == expected


@pytest.mark.parametrize(
    ("subject", "run", "removed"),
    [(1, "block1-4", 40), (5, "block6", 1), (2, "block1", 0)],
)
def test_annotation_repairs(subject, run, removed):
    raw = mne.io.RawArray([[0.0] * 100], mne.create_info(["x"], 1, "misc"))
    raw.set_annotations(mne.Annotations(np.arange(50), 1, ["1.0"] * 50))
    Yi2025FnirsMI._repair_annotations(raw, subject, run)
    assert len(raw.annotations) == 50 - removed


@pytest.fixture(scope="module")
def subject_one_data():
    return Yi2025FnirsMI().get_data(subjects=[1])


@pytest.mark.data
@pytest.mark.manual
def test_real_data_structure_and_properties(subject_one_data):
    runs = subject_one_data[1]["0fnirs"]
    assert list(runs) == [
        "1block1to4",
        "5block5",
        "6block6",
        "7block7",
        "8block8",
    ]
    raw = runs["5block5"]
    assert isinstance(raw, mne.io.BaseRaw)
    assert raw.info["sfreq"] == pytest.approx(7.8125)
    assert len(mne.pick_types(raw.info, fnirs="fnirs_cw_amplitude")) == 48
    assert len(mne.pick_types(raw.info, stim=True)) == 1
    assert len(raw.info["dig"]) == 27
    labels = raw.get_data(picks="Stim").ravel()
    assert dict(
        zip(*np.unique(labels[labels > 0].astype(int), return_counts=True))
    ) == {code: 5 for code in EVENTS.values()}


@pytest.mark.data
@pytest.mark.manual
@pytest.mark.parametrize("subject", range(1, 19))
def test_all_subjects_load(subject):
    runs = Yi2025FnirsMI().get_data(subjects=[subject])[subject]["0fnirs"]
    assert len(runs) == len(Yi2025FnirsMI._source_runs(subject))
    assert all(isinstance(raw, mne.io.BaseRaw) for raw in runs.values())
