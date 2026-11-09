"""Print a compact quality table, one row per dataset, using default paradigm thresholds.

Per-channel metrics are cached in ``<results>/quality/<code>.csv.gz``, one row
per channel and run, with the run ``duration`` in seconds. Subjects missing from the cache are computed and appended.
``--force`` recomputes the requested subjects.

Bad data comes from the paradigm's raw pipeline. The SCI and PSP checks run
only on optical density, so for HbO/HbR recordings they run here on optical
density reconstructed with the wavelengths in ``loc[9]``. Their columns are
then informational and not part of ``good``.

Columns: data types, montage, short/all channels ("-" when the dataset
defines no short-channel threshold), median windowed SNR and the percent of
long-channel samples the SNR check marks bad, the percents the invalid-value
and flat checks mark bad, median windowed SCI and PSP of long channels with their bad
percentages, the percent of good long-channel samples, and the number of runs
whose loaded fNIRS data equal an earlier run's (pairs are logged). Channel
counts are per run. Percentages are means over channels and runs.

Example: uv run python scripts/check_dataset_quality.py Luke2021FnirsTapping --subjects 2
"""

import argparse
import hashlib
import logging
import warnings

import mne
import numpy as np
import pandas as pd
from mne.io.constants import FIFF
from mne.preprocessing.nirs._beer_lambert_law import _load_absorption

from fnirsatlas import datasets
from fnirsatlas.benchmark import results_dir
from fnirsatlas.datasets.preprocessing import (
    _cardiac_windows,
    _peak_spectral_power,
    _scalp_coupling_index,
    bad_data_mask,
    mark_bad_segments_by_psp,
    mark_bad_segments_by_sci,
    windowed_snr,
)
from fnirsatlas.paradigms import FNIRSClassification

logger = logging.getLogger(__name__)

RUN_KEYS = ["subject", "session", "run"]
# Column name and minimum width of the printed table.
COLUMNS = {
    "dataset": 22,
    "type": 18,
    "montage": 7,
    "short": 9,
    "snr": 6,
    "snr_bad": 7,
    "invalid_bad": 7,
    "flat_bad": 7,
    "sci": 5,
    "sci_bad": 7,
    "psp": 5,
    "psp_bad": 7,
    "good": 5,
    "dup": 3,
}
# Cache columns added since the first cached format.
CACHE_COLUMNS = {
    "snr_bad",
    "invalid_bad",
    "flat_bad",
    "window_snr",
    "sci",
    "psp",
    "run_hash",
    "duration",
}


def hb_to_od(raw):
    """Optical density at the two wavelengths stored in the HbO/HbR ``loc[9]``.

    Distance and partial pathlength only scale each wavelength, so they are
    omitted. SCI is invariant to that scale.
    """
    raw = raw.copy().pick(["hbo", "hbr"])
    pairs = [name[: -len(" hbo")] for name in raw.ch_names if name.endswith(" hbo")]
    raw.reorder_channels(
        [f"{pair} {kind}" for pair in pairs for kind in ("hbo", "hbr")]
    )
    locs = np.array([ch["loc"][9] for ch in raw.info["chs"]]).reshape(-1, 2)
    wavelengths = locs[0]
    if not np.isfinite(locs).all() or (locs != wavelengths).any():
        raise ValueError(f"Missing or mixed wavelengths in loc[9]: {np.unique(locs)}")
    coef = _load_absorption(wavelengths)  # (wavelength, chromophore)
    hb = raw.get_data().reshape(len(pairs), 2, -1)
    od = np.einsum("wc,pct->pwt", coef, hb).reshape(2 * len(pairs), -1)
    raw.rename_channels(
        {
            name: f"{name.rsplit(' ', 1)[0]} {wavelengths[index % 2]:.0f}"
            for index, name in enumerate(raw.ch_names)
        }
    )
    with raw.info._unlock():
        for ch in raw.info["chs"]:
            ch.update(kind=FIFF.FIFFV_FNIRS_CH, coil_type=FIFF.FIFFV_COIL_FNIRS_OD)
    return mne.io.RawArray(od, raw.info, verbose=False)


def _group_fractions(raw, descriptions=None):
    """Mean bad fraction of the channels of each source-detector group."""
    picks = mne.pick_types(raw.info, meg=False, fnirs=True, exclude=[])
    groups = [raw.ch_names[pick].rsplit(" ", 1)[0] for pick in picks]
    fractions = bad_data_mask(raw, descriptions).mean(axis=1)
    return pd.Series(fractions).groupby(groups).mean().to_dict()


def _optical_density(raw):
    """Optical density of a CW, OD or HbO/HbR recording, or None for other data."""
    types = set(raw.get_channel_types())
    if "fnirs_cw_amplitude" in types:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Negative intensities.*")
            return mne.preprocessing.nirs.optical_density(raw, verbose=False)
    if "fnirs_od" in types:
        return raw
    if {"hbo", "hbr"} <= types:
        return hb_to_od(raw)
    return None


def _coupling_scores(od, window):
    """Median windowed SCI and PSP keyed by source-detector group."""
    windows, names, _ = _cardiac_windows(od, window)
    scores = {
        "sci": _scalp_coupling_index(windows),
        "psp": _peak_spectral_power(windows, od.info["sfreq"]),
    }
    groups = [group[0].rsplit(" ", 1)[0] for group in names]
    with warnings.catch_warnings():
        # Recordings shorter than one window have no scores.
        warnings.filterwarnings("ignore", message="All-NaN slice|Mean of empty")
        return {
            name: dict(zip(groups, np.nanmedian(values, axis=1), strict=True))
            for name, values in scores.items()
        }


def _check_fractions(raw, out, od, paradigm):
    """Bad fraction per check and good fraction, each keyed by source-detector group.

    HbO/HbR recordings get their SCI and PSP checks on ``od``.
    """
    types = set(raw.get_channel_types())
    cw = "fnirs_cw_amplitude" in types
    fractions = {
        "good": {k: 1 - v for k, v in _group_fractions(out).items()},
        "invalid_bad": _group_fractions(out, ["BAD_invalid"]),
        "snr_bad": _group_fractions(out, ["BAD_snr"]) if cw else {},
        "flat_bad": _group_fractions(out, ["BAD_flat"]) if cw else {},
    }
    coupling = od
    if types & {"fnirs_cw_amplitude", "fnirs_od"}:
        coupling = out
    elif od is not None:
        options = paradigm.preprocessing
        window = options["quality_window"]
        mark_bad_segments_by_sci(od, threshold=options["sci_threshold"], window=window)
        mark_bad_segments_by_psp(od, threshold=options["psp_threshold"], window=window)
    for check in ("sci", "psp"):
        fractions[f"{check}_bad"] = (
            {} if coupling is None else _group_fractions(coupling, [f"BAD_{check}"])
        )
    return fractions


def run_rows(raw, pipeline, paradigm):
    """Per-channel metrics of one recording.

    ``pipeline`` is the paradigm's raw pipeline. Fractions are NaN for
    channels the pipeline removes or checks that do not apply.
    """
    picks = mne.pick_types(raw.info, meg=False, fnirs=True, exclude=[])
    distances = mne.preprocessing.nirs.source_detector_distances(raw.info, picks=picks)
    types = raw.get_channel_types(picks=picks)
    data = raw.get_data(picks=picks)
    run_hash = hashlib.sha1(data.tobytes()).hexdigest()[:12]
    window = paradigm.preprocessing["quality_window"]
    with warnings.catch_warnings():
        # Recordings shorter than one window have no SNR.
        warnings.filterwarnings("ignore", message="All-NaN slice")
        snr = np.nanmedian(windowed_snr(raw, picks, window)[0], axis=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Negative intensities.*")
        out = pipeline.transform(raw.copy())
    od = _optical_density(raw)
    fractions = _check_fractions(raw, out, od, paradigm)
    if od is not None:
        fractions |= _coupling_scores(od, window)
    rows = []
    for index, pick in enumerate(picks):
        source_detector = raw.ch_names[pick].rsplit(" ", 1)[0]
        cw = types[index] == "fnirs_cw_amplitude"
        rows.append(
            {
                "channel": raw.ch_names[pick],
                "channel_type": types[index],
                "source_detector": source_detector,
                "sd_distance": distances[index],
                "window_snr": snr[index] if cw else np.nan,
                "run_hash": run_hash,
                "duration": raw.n_times / raw.info["sfreq"],
                **{
                    name: values.get(source_detector, np.nan)
                    for name, values in fractions.items()
                },
            }
        )
    return rows


def _compute_subjects(dataset, subjects):
    """Per-channel rows of every run of the given subjects."""
    paradigm = FNIRSClassification()
    paradigm.prepare_process(dataset)
    pipeline = paradigm._get_raw_pipelines()[0]
    rows = []
    for subject in subjects:
        for session, runs in dataset.get_data(subjects=[subject])[subject].items():
            for run, raw in runs.items():
                ident = {"subject": subject, "session": session, "run": run}
                rows += [{**ident, **row} for row in run_rows(raw, pipeline, paradigm)]
    return pd.DataFrame(rows)


def channel_table(dataset, max_subjects=0, force=False):
    """Cached per-channel metrics of the requested subjects."""
    path = results_dir() / "quality" / f"{dataset.code}.csv.gz"
    subjects = dataset.subject_list[: max_subjects or None]
    cached = pd.DataFrame(columns=["subject"])
    if path.is_file():
        cached = pd.read_csv(path, dtype={"session": str, "run": str})
        if not CACHE_COLUMNS <= set(cached.columns):
            logger.warning("Recomputing %s: cache has old columns", path)
            cached = pd.DataFrame(columns=["subject"])
    if force:
        cached = cached[~cached["subject"].isin(subjects)]
    missing = [subject for subject in subjects if subject not in set(cached["subject"])]
    if missing:
        computed = _compute_subjects(dataset, missing)
        cached = pd.concat([df for df in (cached, computed) if len(df)])
        path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(path, index=False)
    return cached[cached["subject"].isin(subjects)]


def _median(values):
    """Median to two decimals, or "-" without data."""
    if values.isna().all():
        return "-"
    return f"{np.nanmedian(values):.2f}"


def _percent(values):
    """Mean of fractions in percent, or "-" without data."""
    if values.isna().all():
        return "-"
    return f"{100 * np.nanmean(values):.0f}%"


def _per_run(counts):
    """Distinct per-run values, e.g. "40" or "36,40" when runs differ."""
    return ",".join(str(value) for value in sorted(set(counts)))


def _short_channels(table, short_threshold):
    """Mask of short channels. Without distances every channel counts as long."""
    if short_threshold is None or not np.isfinite(table["sd_distance"]).any():
        return pd.Series(False, index=table.index)
    return table["sd_distance"] < short_threshold


def duplicate_runs(table):
    """Runs whose ``run_hash`` equals an earlier run's, keyed to that earlier run.

    Keys and values are ``"subject/session/run"``.
    """
    runs = table.drop_duplicates(RUN_KEYS)
    names = runs[RUN_KEYS].astype(str).agg("/".join, axis=1)
    first = names.groupby(runs["run_hash"]).transform("first")
    is_copy = runs["run_hash"].duplicated()
    return dict(zip(names[is_copy], first[is_copy], strict=True))


def summarize(code, table, short_threshold):
    """Column values summarising the per-channel metrics of a dataset."""
    by_run = table.groupby(RUN_KEYS)
    has_montage = by_run["sd_distance"].apply(lambda d: (d > 0).any()).all()
    is_short = _short_channels(table, short_threshold)
    short = "-"
    if has_montage and short_threshold is not None:
        n_short = is_short.groupby([table[key] for key in RUN_KEYS]).sum()
        short = f"{_per_run(n_short)}/{_per_run(by_run.size())}"
    long = table[~is_short]
    duplicates = duplicate_runs(table)
    if duplicates:
        pairs = "\n".join(f"  {copy} = {first}" for copy, first in duplicates.items())
        logger.warning("%s duplicate runs:\n%s", code, pairs)
    return {
        "dataset": code,
        "type": ",".join(sorted(set(table["channel_type"]))),
        "montage": "yes" if has_montage else "no",
        "short": short,
        "snr": _median(table["window_snr"]),
        "snr_bad": _percent(long["snr_bad"]),
        "invalid_bad": _percent(long["invalid_bad"]),
        "flat_bad": _percent(long["flat_bad"]),
        "sci": _median(long["sci"]),
        "sci_bad": _percent(long["sci_bad"]),
        "psp": _median(long["psp"]),
        "psp_bad": _percent(long["psp_bad"]),
        "good": _percent(long["good"]),
        "dup": str(len(duplicates)),
    }


def format_row(values):
    """Align values, in ``COLUMNS`` order, to the column widths."""
    return " ".join(
        f"{value:<{max(width, len(name))}}"
        for value, (name, width) in zip(values, COLUMNS.items(), strict=True)
    ).rstrip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", help="Exported dataset class names")
    parser.add_argument(
        "--subjects", type=int, default=0, help="Subjects per dataset, 0 for all"
    )
    parser.add_argument("--force", action="store_true", help="Recompute cached rows")
    args = parser.parse_args()
    mne.set_log_level("INFO")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    print(format_row(COLUMNS))
    for code in args.datasets:
        dataset = datasets.get_dataset(code)
        table = channel_table(dataset, args.subjects, args.force)
        options = FNIRSClassification().options(dataset)
        short_threshold = options["short_channel_threshold"]
        row = summarize(dataset.code, table, short_threshold)
        print(format_row(row.values()), flush=True)


if __name__ == "__main__":
    main()
