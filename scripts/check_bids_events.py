"""Check SNIRF annotations against BIDS events.tsv sidecars."""

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd


def mismatch(raw, events):
    """Return why SNIRF annotations and BIDS events differ, or None."""
    annotations = raw.annotations
    if len(annotations) != len(events):
        return f"count: SNIRF={len(annotations)}, BIDS={len(events)}"
    if not np.allclose(annotations.onset, events["onset"]):
        return "onset"
    if not np.allclose(annotations.duration, events["duration"]):
        return "duration"
    if annotations.description.tolist() != events["trial_type"].astype(str).tolist():
        return "trial_type"
    return None


def check_root(root):
    """Print comparison results and return the number of mismatches."""
    snirf_paths = sorted(root.rglob("*_nirs.snirf"))
    if not snirf_paths:
        raise ValueError(f"No SNIRF files found under {root}")
    failures = 0
    for snirf_path in snirf_paths:
        events_path = snirf_path.with_name(
            snirf_path.name.replace("_nirs.snirf", "_events.tsv")
        )
        if not events_path.is_file():
            print(f"MISMATCH {snirf_path}: missing {events_path.name}")
            failures += 1
            continue
        raw = mne.io.read_raw_snirf(snirf_path, preload=False, verbose=False)
        reason = mismatch(raw, pd.read_csv(events_path, sep="\t"))
        print(
            f"{'MATCH' if reason is None else 'MISMATCH'} {snirf_path}: {reason or 'ok'}"
        )
        failures += reason is not None
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bids_root", type=Path)
    args = parser.parse_args()
    raise SystemExit(bool(check_root(args.bids_root)))


if __name__ == "__main__":
    main()
