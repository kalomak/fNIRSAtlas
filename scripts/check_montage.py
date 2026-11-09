"""Check that channel positions are plausible and match the data.

Geometry: channel midpoints are measured from the origin of the head frame,
which lies between the ears. Positions in another frame fall outside
``PLAUSIBLE_ORIGIN_DISTANCE_M``. The median source-detector distance is printed for
comparison with the paper.

Pairing: neighbouring channels correlate more than distant ones. Converts each
recording to HbO with MNE (ppf 6), band-passes it to the
haemodynamic band and regresses out the mean over channels, which removes the
head-wide systemic signal. Channel pairs are then correlated with Spearman
correlation in 30 s windows, median over windows, which limits the effect of
motion bursts and outlier channels.

Neighbours have channel midpoints at most ``NEIGHBOUR_M`` apart, all other
pairs are non-neighbours. AUC is the probability that a neighbour pair
correlates more than a non-neighbour pair. Channel positions that match the
data give an AUC well above 0.5, and wrong positions about 0.5.

Example: uv run python scripts/check_montage.py Shin2018FnirsWG
"""

import argparse

import mne
import numpy as np
import pandas as pd
from check_hb_scale import to_haemoglobin
from scipy.signal import butter, sosfiltfilt
from scipy.stats import mannwhitneyu, rankdata

from fnirsatlas import datasets

BAND_HZ = (0.01, 0.1)
WINDOW_SECONDS = 30.0
# Adjacent channels of a 30 mm grid have midpoints about 21-30 mm apart.
NEIGHBOUR_M = 0.035
# Median AUC of datasets with verified positions.
COMMON_AUC = (0.7, 0.85)
# Median AUC outside this range indicates wrong positions. Shuffled positions
# give 0.45-0.55.
PLAUSIBLE_AUC = (0.6, 1.0)
# Channel midpoint distance from the head origin. Datasets with verified
# positions span 0.076-0.156 m.
PLAUSIBLE_ORIGIN_DISTANCE_M = (0.07, 0.16)


def windowed_spearman(data, sfreq):
    """Median over windows of the Spearman correlation matrix of ``data`` rows."""
    size = int(WINDOW_SECONDS * sfreq)
    if data.shape[1] < size:
        raise ValueError(f"Recording shorter than one {WINDOW_SECONDS} s window")
    matrices = []
    for start in range(0, data.shape[1] - size + 1, size):
        ranks = rankdata(data[:, start : start + size], axis=1)
        matrices.append(np.corrcoef(ranks))
    return np.median(matrices, axis=0)


def local_hbo(raw):
    """Band-passed HbO without the channel mean, and channel midpoints."""
    hb = to_haemoglobin(raw)
    picks = mne.pick_types(hb.info, fnirs="hbo", exclude="bads")
    data = hb.get_data(picks=picks)
    finite = np.isfinite(data).all(axis=1)
    picks, data = picks[finite], data[finite]
    sos = butter(4, BAND_HZ, "bandpass", fs=hb.info["sfreq"], output="sos")
    data = sosfiltfilt(sos, data, axis=1)
    # Regress each channel on the channel mean to remove the global signal.
    mean = data.mean(axis=0)
    data = data - np.outer(data @ mean / (mean @ mean), mean)
    centers = np.array([hb.info["chs"][pick]["loc"][:3] for pick in picks])
    return data, centers, hb.info["sfreq"]


def neighbour_auc(raw):
    """Neighbour vs non-neighbour correlation AUC of one recording."""
    data, centers, sfreq = local_hbo(raw)
    correlation = windowed_spearman(data, sfreq)
    distance = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    upper = np.triu(np.ones_like(distance, dtype=bool), k=1)
    is_near = distance <= NEIGHBOUR_M
    near = correlation[upper & is_near]
    far = correlation[upper & ~is_near]
    if len(near) == 0 or len(far) == 0:
        raise ValueError("No neighbour or no non-neighbour channel pairs")
    auc = mannwhitneyu(near, far).statistic / (len(near) * len(far))
    return {
        "n_near": len(near),
        "n_far": len(far),
        "r_near": np.median(near),
        "r_far": np.median(far),
        "auc": auc,
    }


def geometry(raw):
    """Channel distance from the head origin and median source-detector distance."""
    picks = mne.pick_types(raw.info, fnirs=True, exclude=[])
    loc = np.array([raw.info["chs"][pick]["loc"][:9] for pick in picks])
    origin_distance = np.linalg.norm(loc[:, :3], axis=1)
    return {
        "origin_dist_min": origin_distance.min(),
        "origin_dist_max": origin_distance.max(),
        "sd_median": np.median(np.linalg.norm(loc[:, 3:6] - loc[:, 6:9], axis=1)),
    }


def verdict(name, low, high, plausible):
    """One summary line stating whether ``low`` to ``high`` is within ``plausible``."""
    inside = plausible[0] <= low and high <= plausible[1]
    status = "ok" if inside else "OUTSIDE"
    span = f"{low:.3f}" if low == high else f"{low:.3f} to {high:.3f}"
    return f"{name} {span}, plausible {plausible}: {status}"


def collect(dataset, subjects):
    """One row of geometry and neighbour statistics per recording."""
    rows = []
    for subject in subjects:
        for session, runs in dataset.get_data(subjects=[subject])[subject].items():
            for run, raw in runs.items():
                stats = geometry(raw) | neighbour_auc(raw)
                rows.append(
                    {"subject": subject, "session": session, "run": run, **stats}
                )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", help="Exported dataset class, e.g. Bak2019FnirsTapping"
    )
    parser.add_argument(
        "--subjects", type=int, help="Number of first subjects to load. Default: all"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mne.set_log_level("INFO")
    dataset = datasets.get_dataset(args.dataset)
    table = collect(dataset, dataset.subject_list[: args.subjects])
    print(table.to_string(index=False, float_format="{:.3f}".format))
    auc = table["auc"].median()
    print(
        "\n"
        + verdict(
            "Distance from head origin (m)",
            table["origin_dist_min"].min(),
            table["origin_dist_max"].max(),
            PLAUSIBLE_ORIGIN_DISTANCE_M,
        )
    )
    print(verdict("Median AUC", auc, auc, PLAUSIBLE_AUC) + f". Common {COMMON_AUC}")
    print(f"Median source-detector distance (m) {table['sd_median'].median():.3f}")


if __name__ == "__main__":
    main()
