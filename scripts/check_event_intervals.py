"""Summarize intervals between consecutive events in a MOABB dataset."""

import argparse
from itertools import pairwise

import numpy as np
import pandas as pd

from fnirsatlas import datasets

QUANTILES = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def collect_intervals(dataset):
    """Return consecutive event intervals for all dataset recordings."""
    rows = []
    valid_events = set(dataset.event_id)
    for subject in dataset.subject_list:
        for session, runs in dataset.get_data(subjects=[subject])[subject].items():
            for run, raw in runs.items():
                annotations = raw.annotations
                selected = [
                    (onset, str(label))
                    for onset, label in zip(
                        annotations.onset, annotations.description, strict=True
                    )
                    if label in valid_events
                ]
                for previous, current in pairwise(selected):
                    rows.append(
                        {
                            "subject": subject,
                            "session": session,
                            "run": run,
                            "transition": f"{previous[1]}>{current[1]}",
                            "interval": current[0] - previous[0],
                        }
                    )
    return pd.DataFrame(rows)


def summarize(intervals):
    """Return interval statistics grouped by event transition."""
    return intervals.groupby("transition")["interval"].agg(
        n="size", minimum="min", median="median", maximum="max"
    )


def interval_distribution(intervals, quantiles=QUANTILES):
    """Return interval quantiles and overlap counts for their epoch lengths."""
    values = intervals["interval"].to_numpy()
    durations = np.quantile(values, quantiles)
    shorter = np.array([(values < duration).sum() for duration in durations])
    percentiles = np.asarray(quantiles) * 100
    columns = [f"{percentile:g}%" for percentile in percentiles]
    distribution = pd.DataFrame(
        [durations, shorter, shorter / len(values) * 100],
        index=[
            "onset interval (s)",
            "overlapping intervals",
            "overlapping intervals (%)",
        ],
        columns=columns,
    )
    distribution.columns.name = "Excluded %"
    return distribution


def count_excluded_trials(intervals, interval):
    """Count trials whose epoch overlaps the next selected event onset."""
    epoch_length = interval[1] - interval[0]
    if epoch_length <= 0:
        raise ValueError("Interval end must be greater than its start")
    return int((intervals["interval"] < epoch_length).sum())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", help="Exported dataset class, e.g. Bak2019FnirsTapping"
    )
    parser.add_argument(
        "--interval",
        nargs=2,
        type=float,
        metavar=("TMIN", "TMAX"),
        help="Count trials overlapping the next onset for this epoch interval",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    intervals = collect_intervals(datasets.get_dataset(args.dataset))
    if intervals.empty:
        raise ValueError(f"No consecutive selected events found for {args.dataset}")
    if args.interval:
        excluded = count_excluded_trials(intervals, args.interval)
        print(f"Trials excluded by {args.interval}: {excluded}")
        return
    distribution = interval_distribution(intervals)
    formatters = {
        column: lambda value: (
            f"{value:.3f}"
            if isinstance(value, float) and not value.is_integer()
            else f"{value:.0f}"
        )
        for column in distribution.columns
    }
    print(distribution.to_string(formatters=formatters))
    print(f"\nIntervals: {len(intervals)}")


if __name__ == "__main__":
    main()
