"""
Checks the interval between epoch onsets for a given dataset.

Useful for deciding epoch length without creating overlap. Also shows which transitions (previous stimulus class -> current stimulus class) have the smallest intervals.
"""

import logging
from collections import defaultdict
import numpy as np
import pandas as pd
from fnirsatlas.epoching import check_epoch_overlap, create_epochs_from_raw
import argparse
from fnirsatlas.load_parser import load_parser
from fnirsatlas.logging import setup_logging, set_log_level

setup_logging(level="ERROR")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="luke2024b")
parser.add_argument("-p", "--plot", action="store_true")
args, _ = parser.parse_known_args()
dataset_name = args.dataset

parser = load_parser(dataset_name)
subject_ids = parser.get_subject_ids()
min_interval = 10000
n_smallest = 5

intervals = []
cond2label = {}
for subj in subject_ids:
    # skips preprocessing
    raw = parser.convert_to_raw(subj)
    epochs = create_epochs_from_raw(raw=raw, dataset_name=parser.dataset_id, tmax=1)
    cond2label.update(epochs.event_id)
    for i in range(1, len(epochs.events)):
        prev, curr = epochs.events[i - 1], epochs.events[i]
        prev_onset_sec = prev[0] / raw.info["sfreq"]
        curr_onset_sec = curr[0] / raw.info["sfreq"]
        intervals.append([prev_onset_sec, curr_onset_sec, prev[2], curr[2], subj])

df = pd.DataFrame(
    intervals, columns=["onset1", "onset2", "class1", "class2", "subject"]
)
df["interval"] = df["onset2"] - df["onset1"]
label2cond = {v: k for k, v in cond2label.items()}
df["transition"] = df["class1"].map(label2cond) + ">" + df["class2"].map(label2cond)

# Group by transition
summary = []
for transition, group in df.groupby("transition"):
    intervals = group["interval"].values
    med = np.median(intervals)
    # Get the indices of the n_smallest intervals
    smallest_idx = np.argsort(intervals)[:n_smallest]
    smallest_intervals = intervals[smallest_idx]
    smallest_subjects = group.iloc[smallest_idx]["subject"].values
    smallest = list(zip(np.round(smallest_intervals, 2), smallest_subjects))
    summary.append(
        {
            "transition": transition,
            "n": len(intervals),
            "median": med,
            "smallest": smallest,
        }
    )

# Print smallest five intervals with subjects for each transition
print(f"\nSmallest {n_smallest} intervals per transition:")
for row in summary:
    print(f"\nTransition: {row['transition']}")
    for interval, subject in row["smallest"]:
        print(f"  Interval: {interval:6.2f}  Subject: {subject}")

# Print summary
print("\nSummary:")
print("-" * 70)
print(f"{'Transition':<40} | {'N':>4} | {'Median':>8} | {'Min':>8}")
print("-" * 70)
for row in summary:
    print(
        f"{row['transition']:<40} | {row['n']:4d} | {row['median']:8.2f} | {row['smallest'][0][0]:8.2f}"
    )
print("-" * 70)
smallest_overall_ix = np.argmin(df["interval"])
smallest_overall_transition = df.iloc[smallest_overall_ix]["transition"]
print(
    f"{smallest_overall_transition + ' (min)':<40} | {len(df):4d} | {np.median(df['interval']):8.2f} | {np.min(df['interval']):8.2f}"
)

if args.plot:
    import matplotlib.pyplot as plt
    plt.hist(df["interval"], bins=100, range=(0, 50))
    plt.xlabel("Interval (s)")
    plt.ylabel("Count")
    ticks = np.arange(0, 50, 1)
    plt.xticks(ticks, [str(t) if t % 5 == 0 else '' for t in ticks])
    plt.xlim(0, 50)
    plt.grid(True, alpha=0.3)
    plt.title(f"Interval Distribution for {dataset_name}")
    plt.show()
