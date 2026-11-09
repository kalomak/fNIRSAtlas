"""Tests for the event interval analysis script."""

from argparse import Namespace

import mne
import pandas as pd

import scripts.check_event_intervals as script
from scripts.check_event_intervals import (
    collect_intervals,
    count_excluded_trials,
    interval_distribution,
    summarize,
)


class _Dataset:
    subject_list = [1]
    event_id = {"Finger": 1, "Foot": 2}

    def get_data(self, subjects):
        raw = mne.io.RawArray([[0.0] * 200], mne.create_info(["x"], 1, "misc"))
        raw.set_annotations(
            mne.Annotations([0, 10, 30], [1, 1, 1], ["Finger", "Rest", "Foot"])
        )
        return {subjects[0]: {"0": {"0": raw}}}


def test_collect_intervals_uses_selected_consecutive_events():
    summary = summarize(collect_intervals(_Dataset()))

    assert summary.loc["Finger>Foot", "minimum"] == 30


def test_interval_distribution_reports_actual_shorter_intervals():
    intervals = pd.DataFrame({"interval": [10, 10, 20, 30]})

    distribution = interval_distribution(intervals, quantiles=[0, 0.25, 0.5])

    assert distribution.loc["onset interval (s)"].tolist() == [10, 10, 15]
    assert distribution.loc["overlapping intervals"].tolist() == [0, 0, 2]
    assert distribution.loc["overlapping intervals (%)"].tolist() == [0, 0, 50]
    assert distribution.columns.tolist() == ["0%", "25%", "50%"]
    assert distribution.columns.name == "Excluded %"


def test_count_excluded_trials_for_interval():
    intervals = pd.DataFrame({"interval": [10, 10, 20, 30]})

    assert count_excluded_trials(intervals, [-2, 9]) == 2
