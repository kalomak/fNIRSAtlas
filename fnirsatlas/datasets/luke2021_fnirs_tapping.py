"""Luke2021 fNIRS finger tapping dataset in BIDS format."""

import shutil

import mne
import numpy as np
import pandas as pd
from moabb.datasets import download as dl
from moabb.datasets.metadata.schema import (
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)

from .bids_base import (
    BaseBIDSFNIRSDataset,
    build_bids_fnirs_file_paths,
    get_bids_cache_root,
)
from .metadata import FnirsAcquisitionMetadata
from .utils import download_lock, extract_archive

URL = (
    "https://zenodo.org/api/records/6575155/files/"
    "rob-luke/BIDS-NIRS-Tapping-v0.1.0.zip/content"
)
CODE = "Luke2021FnirsTapping"
TASK = "tapping"
SUBJECTS = [1, 2, 3, 4, 5]
EVENTS = {"Control": 1, "Tapping/Left": 2, "Tapping/Right": 3}
SFREQ = 7.8125
# The endpoint is the last complete sample before 15 s. Onsets are at least
# 19.072 s apart.
INTERVAL = [-2, np.floor(15 * SFREQ) / SFREQ]
EXTRACTED_DIR = "rob-luke-BIDS-NIRS-Tapping-e262df8"


class Luke2021FnirsTapping(BaseBIDSFNIRSDataset):
    """Luke2021 fNIRS finger tapping dataset in BIDS format.

    Left-hand tapping, right-hand tapping and control, each cued for 5 s by an
    auditory cue [1]_.

    References
    ----------
    .. [1] Luke, R., & McAlpine, D. (2021). fNIRS Finger Tapping Data in BIDS
           Format. Zenodo. https://doi.org/10.5281/zenodo.6575155
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"fnirs_cw_amplitude": 56},
            # Device lines of the NIRx .hdr files
            hardware="NIRx NIRScout (16x24 and NIRScoutX 64x32)",
            line_freq=50.0,
            montage="custom",
            wavelengths_nm=[760.0, 850.0],
            n_sources=8,
            n_detectors=16,
            n_short_channels=8,
            sensor_positions="3d",
            other_modalities=[],
        ),
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            gender={"female": 3, "male": 2},
            age_min=20.0,
            age_max=53.0,
        ),
        experiment=ExperimentMetadata(
            paradigm="fnirsMotorExecution",
            task_type="finger tapping",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            trial_duration=5.0,
            stimulus_type="auditory cue",
            stimulus_modalities=["auditory"],
            primary_modality="auditory",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.6575155",
            investigators=["Robert Luke", "David McAlpine"],
            repository="Zenodo",
            data_url="https://zenodo.org/records/6575155",
            # Zenodo record license id. The record names no specific license.
            license="other-open",
            publication_year=2021,
        ),
        data_structure=DataStructureMetadata(
            n_trials=90,
            n_trials_per_class={label: 30 for label in EVENTS},
            trials_context="per subject",
        ),
        file_format="BIDS SNIRF",
    )
    _bids_task = TASK
    _bids_subject_format = "{:02d}"

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=EVENTS,
            code=CODE,
            interval=INTERVAL,
            paradigm="fnirsMotorExecution",
            doi="10.5281/zenodo.6575155",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        return str(self._prepare_bids_root(subject, path, force_update, verbose))

    def _load_raw(self, bids_path, subject):
        """Load a recording and replace its annotations from BIDS events."""
        raw = super()._load_raw(bids_path, subject)
        raw.set_annotations(self._load_annotations(bids_path, raw.info["meas_date"]))
        return raw

    def _prepare_bids_root(self, subject, path, force_update, verbose):
        bids_root = get_bids_cache_root(CODE, path) / "data"
        files = build_bids_fnirs_file_paths(
            self._bids_subject(subject), None, [(TASK, None)]
        )
        expected = bids_root / next(
            path for path in files if path.endswith("_nirs.snirf")
        )
        # All subjects share one archive.
        with download_lock(bids_root):
            if expected.is_file() and not force_update:
                return bids_root
            archive_path = dl.data_dl(URL, CODE, path, force_update, verbose)
            if bids_root.is_dir():
                shutil.rmtree(bids_root)
            extract_archive(archive_path, bids_root.parent)
            (bids_root.parent / EXTRACTED_DIR).rename(bids_root)
        return bids_root

    @classmethod
    def _load_annotations(cls, bids_path, meas_date):
        events_tsv = bids_path.copy().update(suffix="events", extension=".tsv").fpath
        metadata = pd.read_csv(events_tsv, sep="\t")
        metadata = metadata[metadata["trial_type"].isin(cls._valid_trial_types())]
        return mne.Annotations(
            onset=metadata["onset"].to_numpy(dtype=float),
            duration=metadata["duration"].to_numpy(dtype=float),
            description=metadata["trial_type"].astype(str).to_numpy(),
            orig_time=meas_date,
        )

    @staticmethod
    def _valid_trial_types():
        return np.array(list(EVENTS))
