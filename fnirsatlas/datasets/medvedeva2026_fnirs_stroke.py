"""Medvedeva2026 multisession post-stroke hand-movement fNIRS dataset."""

import re
import shutil
import warnings
from pathlib import Path

import mne
from moabb.datasets import download as dl
from moabb.datasets.metadata.schema import (
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)

from .base import BaseFNIRSDataset
from .metadata import FnirsAcquisitionMetadata
from .utils import extract_archive, paired_fnirs_channel_order

URL = "https://ndownloader.figshare.com/files/60210935"
CODE = "Medvedeva2026FnirsStroke"
DOI = "10.1038/s41597-026-06803-5"
DATA_DOI = "10.6084/m9.figshare.28904942"
PARADIGM = "fnirsMotorExecution"
SFREQ = 13.56336784362793
N_PAIRS = 70
WAVELENGTHS = [760.0, 850.0]
# Source session numbers 1-6. Subjects have 3-6 sessions.
N_SESSIONS = 6
SUBJECTS = list(range(1, 17))
EVENTS = {"left_hand": 1, "right_hand": 2, "ignore_left": 3, "ignore_right": 4}
# File names encode paretic side, subject, session, target hand and run,
# e.g. "R_02_session2_right1.fif".
FILE_PATTERN = re.compile(r"[LR]_(\d+)_session(\d)_(left|right)(\d)\.fif")
# Source events name the flashing LED. The LED above the run's target hand
# prompts a button press with that hand. The other LED must be ignored.
FLASH_SIDES = {"flash_l": "left", "flash_r": "right"}
# All flashes are at least 11.133 s apart. The paper epochs [-0.1, 11].
INTERVAL = [-2, 9.1]


class Medvedeva2026FnirsStroke(BaseFNIRSDataset):
    """Stimulus-response hand movements by post-stroke patients.

    Patients press a button with the run's target hand when the LED above it
    flashes and ignore flashes of the LED above the other hand. Runs hold 12
    flashes, 6 per LED, and sessions span two weeks of inpatient
    rehabilitation [1]_.

    Notes
    -----
    The authors removed 30 recordings with many artefacts or incorrect
    triggers before release, mainly from the first three patients. Two runs
    hold 11 flashes. One run's last flash lacks data and is dropped at
    epoching. The data is optical density converted from raw intensity by the
    authors. Raw intensity is not released.

    References
    ----------
    .. [1] Medvedeva, A., Syrov, N., Yakovlev, L., Alieva, Y.,
           Berkmush-Antipova, A., Ivanova, G., Shusharina, N., and Kaplan, A.
           (2026). Multisession fNIRS-EEG data of Post-Stroke Motor Recovery.
           Recordings During Intact and Paretic Hand Movements. Scientific
           Data, 13, 448. https://doi.org/10.1038/s41597-026-06803-5
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            # Optical density, converted from intensity by the authors with MNE.
            channel_types={"fnirs_od": len(WAVELENGTHS) * N_PAIRS},
            hardware="NIRx NIRSport 16x16 and NIRSport 8x8",
            # Recorded in Moscow, Russia.
            line_freq=50.0,
            montage="custom",
            wavelengths_nm=WAVELENGTHS,
            n_sources=24,
            # The paper lists 24 detectors. A malfunctioning detector and its
            # mirror were excluded, leaving 22 in the data.
            n_detectors=22,
            n_short_channels=0,
            # Positions in each FIF, identical across subjects.
            sensor_positions="3d",
            # Separate archives, recorded with an NVX-52 amplifier. EEG has 8
            # channels.
            other_modalities=["EEG", "EMG", "ECG"],
        ),
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            gender={"male": 10, "female": 6},
            handedness={"right": 16},
            age_mean=57.2,
            age_std=9.68,
            age_min=42.0,
            age_max=71.0,
            # Paretic side 10 left and 6 right. 13 subacute and 3 chronic, 13
            # ischemic and 3 hemorrhagic. Clinical scores (ARAT, Fugl-Meyer) in
            # "Patients scales.xlsx".
            health_status="patients",
            clinical_population="stroke",
        ),
        experiment=ExperimentMetadata(
            paradigm=PARADIGM,
            task_type="stimulus-response hand movement",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            # Flash duration.
            trial_duration=0.2,
            stimulus_type="LED flash",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=DOI,
            investigators=[
                "Alexandra Medvedeva",
                "Nikolay Syrov",
                "Lev Yakovlev",
                "Yana Alieva",
                "Artemiy Berkmush-Antipova",
                "Galina Ivanova",
                "Natalia Shusharina",
                "Alexander Kaplan",
            ],
            institution="Skolkovo Institute of Science and Technology",
            country="RU",
            repository="Figshare",
            data_url=f"https://doi.org/{DATA_DOI}",
            license="CC BY 4.0",
            publication_year=2026,
        ),
        data_structure=DataStructureMetadata(
            # Median subject. Subjects have 108-432 flashes, 5,638 in total.
            n_trials=396,
            n_trials_per_class={label: 99 for label in EVENTS},
            trials_context="per subject",
        ),
        sessions_per_subject=N_SESSIONS,
        # At least 3 runs per target hand. Released sessions hold 2-8.
        runs_per_session=6,
        file_format="FIF",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=N_SESSIONS,
            events=EVENTS,
            code=CODE,
            interval=INTERVAL,
            paradigm=PARADIGM,
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the directory holding every subject's FIF recordings."""
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        return str(self._download_data(path, force_update, verbose))

    def _get_single_subject_data(self, subject):
        recordings = [self._read_run(source) for source in self._subject_files(subject)]
        # Target hands alternate in a randomized order, so runs are keyed by
        # their recording order within the session, followed by the source
        # target hand and run number, e.g. "3left2".
        recordings.sort(key=lambda recording: recording[2].info["meas_date"])
        sessions = {}
        for session, source_run, raw in recordings:
            key = f"{session}fnirs"
            index = len(sessions.get(key, {}))
            self._add_run(sessions, key, raw, f"{index}{source_run}")
        return sessions

    def _subject_files(self, subject):
        root = Path(self.data_path(subject))
        files = sorted(root.glob(f"[LR]_{subject:02d}_session*.fif"))
        if not files:
            raise FileNotFoundError(f"No Medvedeva2026 files for subject {subject}")
        return files

    def _read_run(self, source):
        """Return the source session, source run name and labelled recording."""
        match = FILE_PATTERN.fullmatch(source.name)
        if match is None:
            raise ValueError(f"Unexpected Medvedeva2026 file name: {source}")
        _, session, target, run = match.groups()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*does not conform to MNE naming conventions.*"
            )
            raw = mne.io.read_raw_fif(source, preload=True, verbose=False)
        # Some channel pairs store 850 nm before 760 nm.
        raw.reorder_channels(paired_fnirs_channel_order(raw.ch_names))
        labels = [
            _label(FLASH_SIDES[flash], target) for flash in raw.annotations.description
        ]
        raw.set_annotations(
            mne.Annotations(
                raw.annotations.onset,
                raw.annotations.duration,
                labels,
                orig_time=raw.annotations.orig_time,
            )
        )
        return session, f"{target}{run}", self._standardize_raw(raw)

    @staticmethod
    def _download_data(path, force_update, verbose):
        data_dir = Path(dl.get_dataset_path(CODE, path))
        root = data_dir / f"MNE-{CODE.lower()}-data" / "FNIRS_FIF"
        marker = root.parent / ".extracted"
        if marker.is_file() and not force_update:
            return root
        archive = Path(dl.data_dl(URL, CODE, path, force_update, verbose))
        marker.unlink(missing_ok=True)
        if root.is_dir():
            shutil.rmtree(root)
        extract_archive(archive, root)
        marker.touch()
        return root


def _label(flash_side, target_side):
    """Name a flash as a press with the target hand or an ignored flash."""
    if flash_side == target_side:
        return f"{flash_side}_hand"
    return f"ignore_{flash_side}"
