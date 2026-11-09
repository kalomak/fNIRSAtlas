"""Heuristically check datasets for clear band-pass cutoffs.

Example: uv run python scripts/check_processed_data.py Bnci2014FnirsMA
"""

import argparse

import mne
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, welch

from fnirsatlas import datasets


def guess_filter(raw):
    """Return guessed low/high cutoffs, or None when both are not clear."""
    picks = mne.pick_types(raw.info, fnirs=True, exclude="bads")
    if not len(picks):
        raise ValueError("No fNIRS channels found")
    data = raw.get_data(picks=picks, reject_by_annotation="omit")
    frequencies, psd = welch(data, fs=raw.info["sfreq"], nperseg=data.shape[1], axis=-1)
    spectrum = np.median(psd, axis=0)[1:]
    frequencies = frequencies[1:]
    grid = np.geomspace(frequencies[0], frequencies[-1], min(512, len(frequencies)))
    log_psd = np.interp(
        grid, frequencies, 10 * np.log10(np.maximum(spectrum, np.finfo(float).tiny))
    )
    log_frequency = np.log10(grid)
    slope = np.gradient(gaussian_filter1d(log_psd, 12), log_frequency)
    curvature = gaussian_filter1d(np.gradient(slope, log_frequency), 6)
    knees, _ = find_peaks(-curvature, distance=40)
    knees = knees[(knees >= 15) & (knees < len(grid) - 15)]
    low = knees[(curvature[knees] < -50) & (slope[knees - 15] > 20)]
    if not len(low):
        return None
    low = low[0]
    high = knees[(knees > low) & (curvature[knees] < -300) & (slope[knees + 15] < -60)]
    if not len(high):
        return None
    high = high[0]
    if grid[high] / grid[low] > 50:
        return None
    return grid[low], grid[high]


def check_dataset(dataset):
    """Yield cutoff results for every recording in a dataset."""
    for subject in dataset.subject_list:
        subject_data = dataset.get_data(subjects=[subject])[subject]
        for session, runs in subject_data.items():
            for run, raw in runs.items():
                yield subject, session, run, guess_filter(raw)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Exported dataset class name")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = datasets.get_dataset(args.dataset)
    for subject, session, run, guessed_filter in check_dataset(dataset):
        recording = f"subject={subject} session={session} run={run}"
        if guessed_filter is None:
            print(f"RAW {recording}")
        else:
            low, high = guessed_filter
            print(f"PROCESSED {recording} filter_guess={low:.4g}-{high:.4g} Hz")


if __name__ == "__main__":
    main()
