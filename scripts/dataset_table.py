"""Print a table of datasets from their METADATA, one row per dataset.

Recording hours are read from the quality cache that
``scripts/check_dataset_quality.py`` writes, so run it for the datasets first.
Epochs are the METADATA trials per subject, summed over subjects. Epoch length is the
dataset interval, empty for datasets without one.

Example: uv run python scripts/dataset_table.py Bak2019FnirsTapping Bnci2014FnirsMA
"""

import argparse
from pathlib import Path

import pandas as pd

from fnirsatlas import datasets
from fnirsatlas.benchmark import results_dir

RUN_KEYS = ["subject", "session", "run"]


def n_pairs(acquisition):
    """Number of source-detector pairs, from channels per pair."""
    types = acquisition.channel_types
    if "hbo" in types:
        return types["hbo"]
    return acquisition.n_channels // len(acquisition.wavelengths_nm)


def citation(metadata):
    """Paper DOI, or the ``paper`` external link without one."""
    return metadata.documentation.doi or metadata.external_links["paper"]


def epoch_length(dataset):
    """Length in seconds of the dataset interval, or NA without one."""
    if dataset.interval is None:
        return pd.NA
    start, stop = dataset.interval
    return round(stop - start, 2)


def n_epochs(metadata):
    """Total trials from ``n_trials``, an int for every subject or a dict per subject."""
    n_trials = metadata.data_structure.n_trials
    if isinstance(n_trials, dict):
        return sum(n_trials.values())
    return n_trials * metadata.participants.n_subjects


def describe(dataset):
    """Table columns that come from METADATA and the dataset class."""
    metadata = dataset.metadata
    acquisition = metadata.acquisition
    return {
        "dataset": dataset.code,
        "task": metadata.experiment.task_type,
        "subjects": metadata.participants.n_subjects,
        "pairs": n_pairs(acquisition),
        "short": acquisition.n_short_channels,
        "sfreq": round(acquisition.sampling_rate, 2),
        "type": ",".join(acquisition.channel_types),
        "device": acquisition.hardware,
        "license": metadata.documentation.license,
        "doi": citation(metadata),
        "epoch_s": epoch_length(dataset),
        "epochs": n_epochs(metadata),
        "classes": metadata.experiment.n_classes,
    }


def recording_hours(dataset):
    """Total recording hours over all runs in the quality cache."""
    path = results_dir() / "quality" / f"{dataset.code}.csv.gz"
    table = pd.read_csv(path, usecols=[*RUN_KEYS, "duration"])
    missing = set(dataset.subject_list) - set(table["subject"])
    if missing:
        raise ValueError(f"{path} lacks subjects {sorted(missing)}")
    return table.drop_duplicates(RUN_KEYS)["duration"].sum() / 3600


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets", nargs="*", help="Exported dataset class names. Default: all"
    )
    parser.add_argument("--output", type=Path, help="TSV path. Default: print")
    return parser.parse_args(argv)


def main(argv=None):
    """Print the dataset table."""
    args = parse_args(argv)
    rows = []
    for name in args.datasets or datasets.list_datasets():
        dataset = datasets.get_dataset(name)
        rows.append(describe(dataset) | {"hours": round(recording_hours(dataset), 1)})
    table = pd.DataFrame(rows)
    if args.output:
        table.to_csv(args.output, sep="\t", index=False)
        return
    with pd.option_context("display.max_colwidth", 40, "display.width", None):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
