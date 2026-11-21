"""Klinkhammer2026 hand-clenching and finger-tapping fNIRS dataset (Zenodo)."""

import shutil
import tempfile
from pathlib import Path

import h5py
import mne
from moabb.datasets import download as dl
from moabb.datasets.metadata.schema import (
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)

from .bids_base import BaseBIDSFNIRSDataset, get_bids_cache_root
from .metadata import FnirsAcquisitionMetadata
from .utils import extract_archive, paired_fnirs_channel_order

# No paper yet: the data descriptor is under review, so this is the data DOI.
DOI = "10.5281/zenodo.22911602"
URL = (
    "https://zenodo.org/api/records/22911602/files/"
    "bids_fnirs_sensorimotor_dataset.zip/content"
)
CODE = "Klinkhammer2026FnirsME"
PARADIGM = "fnirsMotorExecution"
TASK = "motor"
ARCHIVE_ROOT = "bids_fnirs_sensorimotor_dataset"
# Participant IDs run from 1 to 109. Nine IDs are absent from the release.
MISSING_SUBJECTS = {10, 11, 47, 54, 62, 73, 79, 89, 104}
SUBJECTS = [s for s in range(1, 110) if s not in MISSING_SUBJECTS]
EVENTS = {"hand_clenching": 1, "finger_tapping": 2}
# SNIRF time step. The BIDS sidecar rounds it to 10.1725 Hz.
SFREQ = 1 / 0.098304
RUNS = ["1", "2"]
# Each run has 5 alternating blocks per class.
TRIALS_PER_CLASS_AND_RUN = 5
# 10 s task blocks are followed by 20 s rest plus up to 2 s jitter. Onsets are
# at least 26.935 s apart.
INTERVAL = [-2, 24.9]
# The raw SNIRF files store intensity but label it with the SNIRF dataType for
# processed data and the non-standard dataTypeLabel "RAW", which MNE rejects.
# They are read as continuous-wave amplitude instead.
SNIRF_PROCESSED = 99999
SNIRF_CW_AMPLITUDE = 1


class Klinkhammer2026FnirsME(BaseBIDSFNIRSDataset):
    """Klinkhammer 2026 hand-clenching and finger-tapping fNIRS dataset.

    Right-hand rhythmic fist clenching versus sequential thumb-to-finger
    tapping over the left sensorimotor cortex, in alternating 10 s blocks that
    always start with hand clenching [1]_.

    Notes
    -----
    Release v1.0.0 holds a standard-density 5x5 optode subset of the recorded
    8x8 montage. The remaining channels are to be released later.

    References
    ----------
    .. [1] Klinkhammer, S., Näher, T., Raible, S., Lührs, M., Klein, F., and
           Sorger, B. (2026). A BIDS-Formatted fNIRS Dataset of Hand Clenching
           and Finger Tapping. Zenodo. https://doi.org/10.5281/zenodo.22911602
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"fnirs_cw_amplitude": 26},
            hardware="NIRx NIRSport2",
            cap_manufacturer="EasyCap",
            # Recorded in Maastricht, Netherlands.
            line_freq=50.0,
            montage="10-5",
            wavelengths_nm=[760.0, 850.0],
            n_sources=5,
            n_detectors=5,
            n_short_channels=1,
            # optodes.tsv template positions in MNI152NLin2009aSym, identical
            # for all subjects.
            sensor_positions="3d",
            other_modalities=["accelerometer", "gyroscope"],
        ),
        # From participants.tsv.
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            gender={"female": 78, "male": 22},
            handedness={"right": 91, "left": 9},
            age_mean=23.4,
            age_std=6.6,
            age_min=19.0,
            age_max=59.0,
        ),
        experiment=ExperimentMetadata(
            paradigm=PARADIGM,
            task_type="right-hand clenching and finger tapping",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            trial_duration=10.0,
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=DOI,
            investigators=[
                "Simona Klinkhammer",
                "Tim Näher",
                "Sophie Raible",
                "Michael Lührs",
                "Franziska Klein",
                "Bettina Sorger",
            ],
            institution="Maastricht University",
            country="NL",
            repository="Zenodo",
            data_url=f"https://doi.org/{DOI}",
            license="CC BY 4.0",
            publication_year=2026,
        ),
        data_structure=DataStructureMetadata(
            n_trials=len(RUNS) * len(EVENTS) * TRIALS_PER_CLASS_AND_RUN,
            n_trials_per_class={
                label: len(RUNS) * TRIALS_PER_CLASS_AND_RUN for label in EVENTS
            },
            trials_context="per subject",
        ),
        runs_per_session=len(RUNS),
        file_format="BIDS SNIRF",
        external_links={
            "code": "https://github.com/s-klinkhammer/OpenFNIRS-sensorimotor",
        },
    )

    _bids_task = TASK
    _bids_subject_format = "{:03d}"
    _annotation_rename_map = {"1": "hand_clenching", "2": "finger_tapping"}  # noqa: RUF012

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=EVENTS,
            code=CODE,
            interval=INTERVAL,
            paradigm=PARADIGM,
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        return str(self._download_data(path, force_update, verbose))

    def _load_raw(self, bids_path, subject):
        raw = _read_snirf_as_cw_amplitude(bids_path.fpath)
        # SNIRF lists all 760 nm channels before all 850 nm channels.
        raw.reorder_channels(paired_fnirs_channel_order(raw.ch_names))
        return raw

    @staticmethod
    def _download_data(path, force_update, verbose):
        cache = get_bids_cache_root(CODE, path)
        root = cache / "data"
        marker = root / "participants.tsv"
        if marker.is_file() and not force_update:
            return root
        archive = Path(dl.data_dl(URL, CODE, path, force_update, verbose))
        extracted = cache / "extracted"
        if extracted.is_dir():
            shutil.rmtree(extracted)
        if root.is_dir():
            shutil.rmtree(root)
        extract_archive(archive, extracted)
        (extracted / ARCHIVE_ROOT).rename(root)
        extracted.rmdir()
        return root


def _read_snirf_as_cw_amplitude(path):
    """Read a raw recording as continuous-wave amplitude.

    The source file is not modified. A temporary copy is relabelled and read.
    """
    path = Path(path)
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / path.name
        shutil.copyfile(path, copy)
        with h5py.File(copy, "r+") as source:
            for name, group in source["nirs/data1"].items():
                if not name.startswith("measurementList"):
                    continue
                if group["dataType"][0] != SNIRF_PROCESSED:
                    raise ValueError(f"Unexpected dataType in {path}: {name}")
                group["dataType"][...] = SNIRF_CW_AMPLITUDE
        return mne.io.read_raw_snirf(copy, preload=True, verbose=False)
