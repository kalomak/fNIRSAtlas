"""Tests for the Cockx2023FnirsME dataset."""

import pandas as pd
import pytest

from fnirsatlas.datasets.cockx2023_fnirs_me import (
    Cockx2023FnirsME,
    reconstruct_trials,
)


def _trial(number, onset, included, sequence, reason=None):
    reason = "n/a" if included else reason or "erroneous trial"
    rows = [
        [onset - 20, number, "stimulus", None, included, reason],
        [onset, number, "stimulus", None, included, reason],
    ]
    rows.extend(
        [onset + i, number, "finger press", x, included, reason]
        for i, x in enumerate(sequence)
    )
    return rows


def test_reconstruct_trials_uses_task_onset_and_fuzzy_sequence():
    rows = _trial(1, 20, True, "43414124321")
    rows += _trial(2, 40, True, "434")
    rows += _trial(3, 60, False, "434141243212")
    events = pd.DataFrame(
        rows,
        columns=[
            "onset",
            "trial_number",
            "event_type",
            "correct_movement",
            "included_trial",
            "reason_for_exclusion",
        ],
    )
    trials = reconstruct_trials(events, "fingerauto")
    assert trials == [(20.0, "Finger")]


def test_dataset_exclusion_options():
    dataset = Cockx2023FnirsME()
    assert len(dataset.subject_list) == 24
    assert dataset._bids_subject(2) == "10"
    assert Cockx2023FnirsME(apply_subject_exclusions=True).subject_list == [
        subject for subject in range(1, 25) if subject != 2
    ]


@pytest.fixture(scope="module")
def subject_data():
    return Cockx2023FnirsME().get_data(subjects=[1])[1]["0fnirs"]


@pytest.mark.data
def test_raw_properties(subject_data):
    raw = subject_data["0fingerauto"]
    assert raw.info["sfreq"] == 50
    assert raw.get_channel_types().count("fnirs_od") == 72
    assert raw.get_channel_types().count("stim") == 1


@pytest.mark.data
@pytest.mark.manual
@pytest.mark.parametrize("subject", range(1, 25))
def test_all_subjects_load(subject):
    runs = Cockx2023FnirsME().get_data(subjects=[subject])[subject]["0fnirs"]
    assert len(runs) == 8
