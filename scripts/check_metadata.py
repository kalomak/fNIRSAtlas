"""Check a dataset's METADATA.

Checks that the required fields are set, runs MOABB's metadata validator,
compares events and paradigm with the dataset, and compares sampling rate,
fNIRS channel count, wavelengths (cw and od data only) and short-separation
pair count with the first run of the first subject. Prints each problem.

Example: uv run --extra data python scripts/check_metadata.py Buccino2016FnirsME
"""

import argparse

import mne
import numpy as np
from mne.preprocessing.nirs import source_detector_distances
from moabb.datasets.metadata import validate_metadata_against_dataset

from fnirsatlas import datasets
from fnirsatlas.paradigms.fnirs import FNIRSClassification

# METADATA fields that every dataset must set, as "<section>.<field>".
REQUIRED_FIELDS = [
    "acquisition.hardware",
    "acquisition.other_modalities",
    "documentation.data_url",
    "documentation.license",
]


def mismatch(name, metadata_value, actual_value):
    """Mismatch message, or None if the values agree."""
    if metadata_value == actual_value:
        return None
    return f"{name}: METADATA {metadata_value!r}, actual {actual_value!r}"


def check_required(metadata):
    """Messages for required fields that are not set."""
    messages = []
    for path in REQUIRED_FIELDS:
        section, name = path.split(".")
        if getattr(getattr(metadata, section), name) is None:
            messages.append(f"{path}: required but not set")
    return messages


def check_dataset(dataset):
    """Mismatches between METADATA and the dataset class."""
    experiment = dataset.metadata.experiment
    messages = validate_metadata_against_dataset(dataset, dataset.metadata)
    return messages + [
        mismatch("events", experiment.events, dataset.event_id),
        mismatch("class_labels", experiment.class_labels, list(dataset.event_id)),
        mismatch("n_classes", experiment.n_classes, len(dataset.event_id)),
        mismatch("paradigm", experiment.paradigm, dataset.paradigm),
    ]


def first_raw(dataset):
    """First run of the first subject's first session."""
    subject = dataset.subject_list[0]
    sessions = dataset.get_data([subject])[subject]
    return next(iter(next(iter(sessions.values())).values()))


def n_short_pairs(dataset, raw):
    """Number of source-detector pairs closer than the short-channel threshold."""
    picks = mne.pick_types(raw.info, fnirs=True)
    distances = source_detector_distances(raw.info, picks=picks)
    # Channel names are "<source>_<detector> <wavelength or chromophore>".
    pairs = [raw.ch_names[i].split()[0] for i in picks]
    pair_distances = np.array(list(dict(zip(pairs, distances)).values()))
    threshold = FNIRSClassification().options(dataset)["short_channel_threshold"]
    return int((pair_distances < threshold).sum())


def check_recording(dataset, raw):
    """Mismatches between METADATA acquisition and a recording."""
    acquisition = dataset.metadata.acquisition
    picks = mne.pick_types(raw.info, fnirs=True)
    messages = [
        mismatch(
            "sampling_rate",
            round(acquisition.sampling_rate, 6),
            round(raw.info["sfreq"], 6),
        ),
        mismatch("n_channels", acquisition.n_channels, len(picks)),
        mismatch(
            "n_short_channels",
            acquisition.n_short_channels,
            n_short_pairs(dataset, raw),
        ),
    ]
    # Hemoglobin channels carry placeholder wavelengths, so only cw and od
    # channels are compared.
    optical = mne.pick_types(raw.info, fnirs=["fnirs_cw_amplitude", "fnirs_od"])
    if len(optical):
        wavelengths = sorted({float(raw.info["chs"][i]["loc"][9]) for i in optical})
        messages.append(
            mismatch("wavelengths_nm", acquisition.wavelengths_nm, wavelengths)
        )
    return messages


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", help="Exported dataset class, e.g. Buccino2016FnirsME"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mne.set_log_level("INFO")
    dataset = datasets.get_dataset(args.dataset)
    if dataset.metadata is None:
        raise ValueError(f"{args.dataset} has no METADATA")
    messages = (
        check_required(dataset.metadata)
        + check_dataset(dataset)
        + check_recording(dataset, first_raw(dataset))
    )
    messages = [message for message in messages if message is not None]
    for message in messages:
        print(message)
    print(f"{args.dataset}: {len(messages)} problems")


if __name__ == "__main__":
    main()
