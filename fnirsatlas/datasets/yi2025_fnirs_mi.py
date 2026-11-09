"""Yi2025 unilateral upper-limb motor-imagery fNIRS dataset."""

import logging
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
from .utils import extract_archive

logger = logging.getLogger(__name__)

URL = "https://ndownloader.figshare.com/files/42320127"
CODE = "Yi2025FnirsMI"
DOI = "10.1038/s41597-025-05286-0"
SFREQ = 7.8125
SUBJECTS = list(range(1, 19))
EVENTS = {
    "hand_open": 1,
    "wrist_flexion": 2,
    "wrist_abduction": 3,
    "elbow_pronation": 4,
    "elbow_flexion": 5,
    "shoulder_pronation": 6,
    "shoulder_abduction": 7,
    "shoulder_flexion": 8,
}
# The paper evaluates up to 15 s. 14.848 s is 116 samples. Onsets are at
# least 17.92 s apart.
INTERVAL = [-2, 14.848]


class Yi2025FnirsMI(BaseFNIRSDataset):
    """Eight-class unilateral upper-limb motor-imagery recordings.

    Kinesthetic motor imagery of eight right-arm movements of the hand, wrist,
    elbow and shoulder for 4 s, cued by text and video, with simultaneous EEG
    [1]_.

    Notes
    -----
    Subject 10 block 3 is excluded because its wavelength-pair scalp-coupling
    correlations are near -1. The final 40 duplicate annotations of subject 1
    block 1-4 and the first annotation of subject 5 block 6 are removed, leaving
    5,735 of the 5,760 published trials.

    References
    ----------
    .. [1] Yi, W., Chen, J., Wang, D., Hu, X., Xu, M., Li, F., Wu, S., & Qian, J.
           (2025). A multi-modal dataset of electroencephalography and functional
           near-infrared spectroscopy recordings for motor imagery of multi-types
           of joints from unilateral upper limb. Scientific Data, 12, 953.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"fnirs_cw_amplitude": 48},
            # Device line of the NIRx .hdr files.
            hardware="NIRx NIRScout 16x24",
            # Recorded in China. The paper's EEG notch filter is at 50 Hz.
            line_freq=50.0,
            montage="custom",
            wavelengths_nm=[760.0, 850.0],
            n_sources=8,
            n_detectors=8,
            n_short_channels=0,
            # Optode coordinates from each recording's NIRS_probeInfo.mat.
            sensor_positions="3d",
            # The 64-channel EEG is not loaded.
            other_modalities=["EEG"],
        ),
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            gender={"male": 8, "female": 10},
            handedness={"right": 18},
            age_min=22.0,
            age_max=27.0,
        ),
        experiment=ExperimentMetadata(
            paradigm="fnirsMotorImagery",
            task_type="unilateral upper-limb motor imagery",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            trial_duration=4.0,
            stimulus_type="text and video cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=DOI,
            investigators=[
                "Weibo Yi",
                "Jiaming Chen",
                "Dan Wang",
                "Xinkang Hu",
                "Meng Xu",
                "Fangda Li",
                "Shuhan Wu",
                "Jin Qian",
            ],
            institution="Beijing University of Technology",
            country="CN",
            repository="figshare",
            data_url="https://doi.org/10.6084/m9.figshare.24123303.v3",
            license="CC BY 4.0",
            publication_year=2025,
        ),
        data_structure=DataStructureMetadata(
            n_trials=320,
            n_trials_per_class={label: 40 for label in EVENTS},
            trials_context="per subject",
        ),
        file_format="NIRx",
        # Best binary pair, hand open/close versus shoulder abduction/adduction.
        # SVM on fNIRS mean and slope, within-subject 5-fold cross-validation
        # (Table 2).
        performance={"accuracy_percent": 61.87},
        # Eight blocks of 40 trials. Subject 1 stores blocks 1-4 as one run.
        runs_per_session=8,
        external_links={"code": "https://github.com/Bill-Chen-BJ/FJMI_EEG_fNIRS"},
    )

    _annotation_rename_map = {f"{code}.0": name for name, code in EVENTS.items()}  # noqa: RUF012

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=EVENTS,
            code=CODE,
            interval=INTERVAL,
            paradigm="fnirsMotorImagery",
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the native NIRx recording directory for one subject."""
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        root = self._download_data(path, force_update, verbose)
        subject_dir = root / f"subject{subject}" / "fNIRS"
        if not subject_dir.is_dir():
            raise FileNotFoundError(f"Missing Yi2025 data for subject {subject}")
        return subject_dir

    def _get_single_subject_data(self, subject):
        subject_dir = Path(self.data_path(subject))
        runs = {
            self._run_key(run): self._load_run(subject_dir / run, subject, run)
            for run in self._source_runs(subject)
        }
        return {"0fnirs": runs}

    def _load_run(self, path, subject, run):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Extraction of measurement date from NIRX file failed.*",
            )
            warnings.filterwarnings(
                "ignore", message="A single dat file was expected.*"
            )
            raw = mne.io.read_raw_nirx(path, preload=True, verbose=False)
        self._repair_annotations(raw, subject, run)
        return self._standardize_raw(raw)

    @staticmethod
    def _repair_annotations(raw, subject, run):
        """Apply verified recording-specific annotation corrections."""
        if subject == 1 and run == "block1-4":
            raw.annotations.delete(
                range(len(raw.annotations) - 40, len(raw.annotations))
            )
            logger.debug("Removed 40 duplicate Yi2025 subject 1 annotations")
        elif subject == 5 and run == "block6":
            # subject5/fNIRS/notes.txt requires dropping the first of 41 trials.
            raw.annotations.delete(0)
            logger.debug("Removed extra Yi2025 subject 5 block 6 annotation")

    @staticmethod
    def _source_runs(subject):
        # Subject 1 stores its first four blocks in one NIRx recording.
        if subject == 1:
            return ["block1-4", "block5", "block6", "block7", "block8"]
        runs = [f"block{i}" for i in range(1, 9)]
        # Subject 10 block 3 has wavelength-pair SCI values near -1.
        if subject == 10:
            runs.remove("block3")
        return runs

    @staticmethod
    def _run_key(source_run):
        block = source_run.removeprefix("block").replace("-", "to")
        return f"{block.split('to')[0]}block{block}"

    def _download_data(self, path, force_update, verbose):
        data_dir = Path(dl.get_dataset_path(CODE, path))
        base_dir = data_dir / f"MNE-{CODE.lower()}-data"
        root = base_dir / "FineMI"
        marker = base_dir / ".extracted"
        if marker.is_file() and not force_update:
            return root
        archive_path = Path(dl.data_dl(URL, CODE, path, force_update, verbose))
        marker.unlink(missing_ok=True)
        if root.is_dir():
            shutil.rmtree(root)
        extract_archive(archive_path, base_dir)
        self._validate_extraction(root)
        marker.touch()
        return root

    @classmethod
    def _validate_extraction(cls, root):
        missing = [
            root / f"subject{subject}" / "fNIRS" / run / "NIRS.hdr"
            for subject in SUBJECTS
            for run in cls._source_runs(subject)
            if not (root / f"subject{subject}" / "fNIRS" / run / "NIRS.hdr").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Yi2025 archive is missing {missing[0]}")
