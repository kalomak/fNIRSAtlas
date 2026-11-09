"""Yucel2025 FRESH finger tapping fNIRS dataset."""

import shutil
from pathlib import Path

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
from .utils import extract_archive

URL = "https://osf.io/download/juxy8/"
CODE = "Yucel2025FnirsTapping"
DOI = "10.1038/s42003-025-08412-1"
DATA_DOI = "10.17605/OSF.IO/B4WCK"
PARADIGM = "fnirsMotorExecution"
TASK = "FRESHMOTOR"
SUBJECTS = list(range(1, 11))
EVENTS = {"left2s": 1, "right2s": 2, "left3s": 3, "right3s": 4}
RUN_CONDITIONS = {
    "0left2s": "left2s",
    "1right2s": "right2s",
    "2left3s": "left3s",
    "3right3s": "right3s",
}
SFREQ = 8.928571428571429
# Trials per subject in most source recordings. Subjects 3, 5, 6, 7 and 9
# differ by 1 to 4 trials in some runs.
N_TRIALS_PER_CLASS = {"left2s": 25, "right2s": 30, "left3s": 25, "right3s": 30}
# 10 of 1,051 consecutive within-run intervals overlap this 11.9-second epoch.
INTERVAL = [-2, 9.9]


class Yucel2025FnirsTapping(BaseBIDSFNIRSDataset):
    """FRESH motor dataset (Dataset II).

    Left- or right-hand finger tapping for 2 or 3 s, one condition per run,
    with 10 to 20 s rest between trials [1]_.

    Notes
    -----
    The paper reports 30 repetitions of 2 s and 25 of 3 s tapping. The source
    runs have 30 right-hand and 25 left-hand trials, and 1,091 of 1,100
    planned trials.

    References
    ----------
    .. [1] Yücel, M. A., Luke, R., Mesquita, R. C., von Lühmann, A., et al.
           (2025). fNIRS reproducibility varies with data quality, analysis
           pipelines, and researcher experience. Communications Biology, 8,
           1149.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"fnirs_cw_amplitude": 136},
            hardware="NIRx NIRScout",
            # Recorded in Campinas, Brazil.
            line_freq=60.0,
            montage="custom",
            wavelengths_nm=[760.0, 850.0],
            n_sources=14,
            n_detectors=32,
            n_short_channels=4,
            # Per-subject Polhemus Fastrak digitisation.
            sensor_positions="3d",
            other_modalities=[],
        ),
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            handedness={"right": len(SUBJECTS)},
            age_min=20.0,
            age_max=35.0,
        ),
        experiment=ExperimentMetadata(
            paradigm=PARADIGM,
            task_type="finger tapping",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=DOI,
            investigators=[
                "Meryem A. Yücel",
                "Robert Luke",
                "Rickson C. Mesquita",
                "Alexander von Lühmann",
            ],
            institution="University of Campinas",
            country="BR",
            repository="OSF",
            data_url=f"https://doi.org/{DATA_DOI}",
            license="CC BY 4.0",
            publication_year=2025,
        ),
        data_structure=DataStructureMetadata(
            n_trials=sum(N_TRIALS_PER_CLASS.values()),
            n_trials_per_class=N_TRIALS_PER_CLASS,
            trials_context="per subject",
        ),
        runs_per_session=len(RUN_CONDITIONS),
        file_format="BIDS SNIRF",
    )
    _bids_task = TASK
    _bids_subject_format = "{:02d}"
    preprocessing_options = {"short_channel_threshold": 0.02}  # noqa: RUF012

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

    def _get_single_subject_data(self, subject):
        source_sessions = super()._get_single_subject_data(subject)
        runs = {}
        for condition, source_runs in source_sessions.items():
            if set(source_runs) != {"0"}:
                raise ValueError(f"Expected one {condition} recording")
            run = next(
                key for key, value in RUN_CONDITIONS.items() if value == condition
            )
            runs[run] = source_runs["0"]
        return {"0": runs}

    def _load_raw(self, bids_path, subject):
        raw = super()._load_raw(bids_path, subject)
        condition = bids_path.session
        # The BIDS session identifies the condition. Every source annotation says motor.
        raw.annotations.rename({"motor": condition})
        # Source files differ only by floating-point noise in the stored frequency.
        with raw.info._unlock():
            raw.info["sfreq"] = SFREQ
        return raw

    def get_additional_metadata(self, subject, session, run):
        """Return BIDS events with the condition reconstructed from its session."""
        if session != "0" or run not in RUN_CONDITIONS:
            raise ValueError(f"Unknown logical recording {session}/{run}")
        condition = RUN_CONDITIONS[run]
        matches = [
            path for path in self.bids_paths(subject) if path.session == condition
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {run} BIDS path; found {len(matches)}")
        metadata = self._read_bids_events(matches[0])
        metadata["trial_type"] = condition
        metadata["value"] = EVENTS[condition]
        return metadata.assign(subject=subject, session=session, run=run)

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        bids_root = get_bids_cache_root(CODE, path) / "data"
        expected = bids_root / self._recording_path(subject)
        if expected.is_file() and not force_update:
            return str(bids_root)
        archive_path = Path(dl.data_dl(URL, CODE, path, force_update, verbose))
        bids_root.parent.mkdir(parents=True, exist_ok=True)
        if bids_root.is_dir():
            shutil.rmtree(bids_root)
        extracted = bids_root.parent / "FRESH-Motor"
        if extracted.is_dir():
            shutil.rmtree(extracted)
        extract_archive(archive_path, bids_root.parent)
        if extracted.is_dir():
            extracted.rename(bids_root)
        if not expected.is_file():
            raise FileNotFoundError(f"Archive does not contain {expected}")
        return str(bids_root)

    def _recording_path(self, subject):
        bids_subject = self._bids_subject(subject)
        return Path(
            f"sub-{bids_subject}/ses-left2s/nirs/"
            f"sub-{bids_subject}_ses-left2s_task-{TASK}_nirs.snirf"
        )
