"""Check that a dataset's haemoglobin amplitude is physiologically plausible.

Loads recordings as the dataset returns them, converts intensity or optical
density to HbO/HbR with MNE (ppf 6), and reports the HbO standard deviation in
the slow (0.01-0.1 Hz), Mayer (0.1-0.2 Hz) and cardiac (0.8-1.6 Hz) bands, in
uM. The median over channels is taken per recording.

Example: uv run python scripts/check_hb_scale.py Bak2019FnirsTapping --subjects 3
"""

import argparse

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from fnirsatlas import datasets
from fnirsatlas.datasets.preprocessing import (
    convert_to_hemoglobin,
    convert_to_optical_density,
)

BANDS = {"slow": (0.01, 0.1), "mayer": (0.1, 0.2), "cardiac": (0.8, 1.6)}
# Slow-band HbO std (uM) of datasets with verified units.
COMMON_SLOW_UM = (0.2, 0.5)
# Slow-band HbO std (uM) outside this range indicates a unit error.
PLAUSIBLE_SLOW_UM = (0.05, 1.0)


def to_haemoglobin(raw):
    """HbO/HbR recording from intensity, optical density or haemoglobin."""
    raw = raw.copy().load_data()
    if "fnirs_cw_amplitude" in raw.get_channel_types():
        raw = convert_to_optical_density(raw)
    return convert_to_hemoglobin(raw)


def band_std_um(raw):
    """Median over finite HbO channels of the per-band standard deviation."""
    data = raw.get_data(picks="hbo")
    data = data[np.isfinite(data).all(axis=1)]
    if len(data) == 0:
        raise ValueError("No finite HbO channels")
    data = data - data.mean(axis=1, keepdims=True)
    nyquist = raw.info["sfreq"] / 2
    result = {}
    for band, (low, high) in BANDS.items():
        # Bands above the recording's Nyquist frequency cannot be measured.
        if high >= nyquist:
            result[band] = np.nan
            continue
        sos = butter(4, [low, high], "bandpass", fs=raw.info["sfreq"], output="sos")
        filtered = sosfiltfilt(sos, data, axis=1)
        result[band] = float(np.median(np.std(filtered, axis=1)) * 1e6)
    return result


def collect(dataset, subjects):
    """One row of band amplitudes per recording."""
    rows = []
    for subject in subjects:
        for session, runs in dataset.get_data(subjects=[subject])[subject].items():
            for run, raw in runs.items():
                bands = band_std_um(to_haemoglobin(raw))
                rows.append(
                    {"subject": subject, "session": session, "run": run, **bands}
                )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", help="Exported dataset class, e.g. Bak2019FnirsTapping"
    )
    parser.add_argument(
        "--subjects", type=int, default=3, help="Number of first subjects to load"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mne.set_log_level("INFO")
    dataset = datasets.get_dataset(args.dataset)
    table = collect(dataset, dataset.subject_list[: args.subjects])
    print(table.to_string(index=False, float_format="{:.3f}".format))
    slow = table["slow"].median()
    print(
        f"\nSlow-band median {slow:.3f} uM. Common {COMMON_SLOW_UM} uM, "
        f"plausible {PLAUSIBLE_SLOW_UM} uM"
    )


if __name__ == "__main__":
    main()
