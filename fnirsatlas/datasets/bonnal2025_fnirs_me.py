"""Bonnal2025 handedness hand-motor fNIRS dataset (Zenodo)."""

import shutil
from pathlib import Path

import mne
import numpy as np
from moabb.datasets import download as dl
from moabb.datasets.metadata.schema import (
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)
from scipy.io import loadmat

from .base import BaseFNIRSDataset
from .metadata import FnirsAcquisitionMetadata
from .preprocessing import _crop_leading_clipped_samples
from .utils import extract_archive, homer_nirs_intensity

# Channels 1-12 (S1-S5, D1-D4) cover the right and 13-24 (S6-S10, D5-D8) the
# left premotor and sensorimotor cortex.
MONTAGE_PATH = Path(__file__).parent / "montages" / "bonnal2025_montage_3d.json"
URL = "https://zenodo.org/api/records/17308191/files/{}/content"
CODE = "Bonnal2025FnirsME"
# Source file IDs per archive. Right-handers are "D_XX" in RH_nirs.zip and
# left-handers "G_XX" in LH_nirs.zip. The spreadsheet's LH_29 has no
# recording in the release. Subject 12 is D_16 and subject 49 is G_28.
GROUPS = {
    "RH_nirs.zip": (
        "D",
        [*range(1, 5), 6, 7, 8, 10, 11, 12, 13, *range(16, 24), *range(25, 31)],
    ),
    "LH_nirs.zip": (
        "G",
        [1, 2, 3, 5, 6, 7, *range(9, 14), *range(15, 24), *range(25, 29)],
    ),
}
# Subjects 1-25 are right-handed, 26-49 left-handed.
SOURCE_FILES = [
    Path(archive).stem + f"/{prefix}_{number:02d}.nirs"
    for archive, (prefix, numbers) in GROUPS.items()
    for number in numbers
]
SUBJECTS = list(range(1, len(SOURCE_FILES) + 1))
EVENTS = {
    "Distal/Left": 1,
    "Distal/Right": 2,
    "Manipulation/Left": 3,
    "Manipulation/Right": 4,
}
# French hand initials mark the distal (simple) task, English ones the
# manipulation (complex) task. Matched by recomputing the released per-subject
# laterality indices from the files' Homer averages. "W" marks each block end.
CONDITION_EVENTS = {
    "G": "Distal/Left",
    "D": "Distal/Right",
    "L": "Manipulation/Left",
    "R": "Manipulation/Right",
}
SFREQ = 50.0
# Brite recordings start with samples at 2**-16, the lowest exported intensity.
BRITE_FLOOR = 2**-16
TRIAL_DURATION = 20
BLOCKS_PER_CLASS = 8
DOI = "10.1016/j.neuroimage.2025.121578"
# The 20 s block plus the paper's 5 s post-offset analysis window. Onsets are
# at least 33.6 s apart in the source data.
INTERVAL = [-2, 25]


class Bonnal2025FnirsME(BaseFNIRSDataset):
    """Bonnal2025 distal and manipulation hand tasks in right- and left-handers.

    Hand opening/closing at 0.5 Hz paced by a metronome (distal) and moving
    sticks on a nine-hole peg board (manipulation), each with the left or right
    hand in 20 s blocks with 20-30 s rest, in randomized condition order,
    after a 120 s baseline [1]_.

    Notes
    -----
    Subjects 1-25 are right-handed and 26-49 left-handed. The authors excluded 9
    recruits for poor signal quality before release. The release lacks one of
    the 25 analysed left-handers (the spreadsheet's ``LH_29``). Most subjects
    have 8 blocks per class. Subject 12 has 6 and subject 49 7 per class.
    Subjects 3 and 26 are unbalanced in the source markers, with 7/7/9/9 and
    8/9/8/7 trials for ``Distal/Left``, ``Distal/Right``,
    ``Manipulation/Left`` and ``Manipulation/Right``. Intensity is stored in
    steps of 1/65536 (16-bit).

    References
    ----------
    .. [1] Bonnal, J., Pila, O., Papin, C., Lebkowski, L., Sarrazin, M.,
           Bravo, R., & Prieur, F. (2025). Handedness and task demands modulate
           motor cortex lateralization: A cross-sectional fNIRS study.
           NeuroImage, 323, 121578.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            # Time vector t. The paper states 10 Hz.
            sampling_rate=SFREQ,
            channel_types={"fnirs_cw_amplitude": 48},
            hardware="Artinis Brite 24",
            # Recorded in Orléans, France.
            line_freq=50.0,
            montage="custom",
            # SD.Lambda. The paper states 760 and 850 nm.
            wavelengths_nm=[757.0, 842.0],
            n_sources=10,
            n_detectors=8,
            n_short_channels=0,
            # montage-toolkit positions, constructed from Fig. 1C of the paper
            # (channels on the MNI brain) and the flat 3 cm grid in
            # SD.SrcPos/SD.DetPos. The digitized positions were not released.
            sensor_positions="3d",
            # 13 aux columns without labels in the source files.
            other_modalities=["unlabelled aux"],
        ),
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            # data_lateralinirs.xlsx without LH_29, which has no recording.
            gender={"male": 22, "female": 27},
            handedness={"right": 25, "left": 24},
            age_mean=22.6,
            age_std=3.9,
            age_min=19.0,
            age_max=35.0,
        ),
        experiment=ExperimentMetadata(
            paradigm="fnirsMotorExecution",
            task_type="hand movement",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            trial_duration=float(TRIAL_DURATION),
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=DOI,
            investigators=[
                "Julien Bonnal",
                "Ophélie Pila",
                "Coraline Papin",
                "Lucie Lebkowski",
                "Mathilde Sarrazin",
                "Raquel Bravo",
                "Fabrice Prieur",
            ],
            institution="Université d'Orléans",
            country="FR",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.17308191",
            license="CC BY 4.0",
            publication_year=2025,
        ),
        data_structure=DataStructureMetadata(
            n_trials=len(EVENTS) * BLOCKS_PER_CLASS,
            n_trials_per_class={label: BLOCKS_PER_CLASS for label in EVENTS},
            trials_context="per subject",
        ),
        file_format="Homer .nirs",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=EVENTS,
            code=CODE,
            interval=INTERVAL,
            paradigm="fnirsMotorExecution",
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the subject's Homer ``.nirs`` recording path."""
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        root = self._download_data(path, force_update, verbose)
        return root / SOURCE_FILES[subject - 1]

    def _get_single_subject_data(self, subject):
        source = loadmat(
            self.data_path(subject), squeeze_me=True, struct_as_record=False
        )
        raw = self._read_raw(source)
        raw.set_annotations(self._task_annotations(source))
        raw = _crop_leading_clipped_samples(raw, floor=BRITE_FLOOR)
        return self._single_run_session(self._standardize_raw(raw))

    def _read_raw(self, source):
        data, names = homer_nirs_intensity(source)
        times = np.asarray(source["t"], dtype=float).ravel()
        if times[0] != 0 or not np.allclose(np.diff(times), 1 / SFREQ):
            raise ValueError("Bonnal2025 t is not a 50 Hz grid starting at 0 s")
        # d is raw Brite intensity. The device-reported wavelengths (757 and
        # 842 nm) are kept in the channel names.
        raw = self._make_raw(data, names, ["fnirs_cw_amplitude"] * len(names), SFREQ)
        # set_montage needs the wavelengths that metadata population adds.
        raw = self._populate_fnirs_metadata(raw)
        raw.set_montage(self._load_montage_from_json(MONTAGE_PATH)[0])
        return raw

    @staticmethod
    def _task_annotations(source):
        # s is a (samples, conditions) onset matrix with one column per
        # CondNames entry. Onsets are sample times from t.
        stim = np.asarray(source["s"])
        times = np.asarray(source["t"], dtype=float).ravel()
        onsets, labels = [], []
        for column, condition in enumerate(np.atleast_1d(source["CondNames"])):
            if condition == "W":
                continue
            samples = np.flatnonzero(stim[:, column])
            onsets.extend(times[samples])
            labels.extend([CONDITION_EVENTS[condition]] * len(samples))
        return mne.Annotations(onsets, TRIAL_DURATION, labels)

    def _download_data(self, path, force_update, verbose):
        extract_dir = (
            Path(dl.get_dataset_path(CODE, path)) / f"MNE-{CODE.lower()}-data" / "data"
        )
        marker = extract_dir / ".extracted"
        if marker.is_file() and not force_update:
            return extract_dir
        if extract_dir.is_dir():
            shutil.rmtree(extract_dir)
        for archive_name in GROUPS:
            archive_path = Path(
                dl.data_dl(
                    URL.format(archive_name),
                    CODE,
                    path,
                    force_update,
                    verbose,
                    fname=archive_name,
                )
            )
            target = extract_dir / Path(archive_name).stem
            target.mkdir(parents=True)
            extract_archive(archive_path, target)
        missing = [name for name in SOURCE_FILES if not (extract_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Bonnal2025 archive is missing {missing[0]}")
        marker.touch()
        return extract_dir
