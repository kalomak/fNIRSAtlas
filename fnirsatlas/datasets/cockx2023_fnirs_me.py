"""Cockx2023 automatic and non-automatic finger/foot movement dataset."""

import shutil
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

from .bids_base import BaseBIDSFNIRSDataset, get_bids_cache_root
from .metadata import FnirsAcquisitionMetadata
from .utils import extract_archive, override_raw_sfreq

URL = "https://data.ru.nl/api/collectionfiles/di/id/di_231884707/files/download"
CODE = "Cockx2023FnirsME"
BIDS_SUBJECTS = [
    "06",
    "10",
    "19",
    "21",
    "27",
    "31",
    "33",
    "34",
    "38",
    "43",
    "44",
    "52",
    "53",
    "63",
    "67",
    "69",
    "74",
    "75",
    "77",
    "84",
    "87",
    "88",
    "90",
    "95",
]
SUBJECTS = list(range(1, len(BIDS_SUBJECTS) + 1))
AUTHOR_EXCLUDED_SUBJECTS = [BIDS_SUBJECTS.index("10") + 1]
EVENTS = {"Finger": 1, "Foot": 2}
TASKS = (
    "fingerauto",
    "fingerautodual",
    "fingernonauto",
    "fingernonautodual",
    "footauto",
    "footautodual",
    "footnonauto",
    "footnonautodual",
)
SEQUENCES = ("212321324241", "434141243212")
# Each task run has 11 trials.
TRIALS_PER_RUN = 11
# Valid trial onsets within a run are at least 35.608 seconds apart.
INTERVAL = [-2, 15]
SFREQ = 50.0


def levenshtein_distance(left, right):
    """Return the edit distance between two movement sequences."""
    previous = list(range(len(right) + 1))
    for i, left_value in enumerate(left):
        current = [i + 1]
        for j, right_value in enumerate(right):
            current.append(
                min(
                    previous[j + 1] + 1,
                    current[j] + 1,
                    previous[j] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def reconstruct_trials(events, task, exclude_practice_trials=False):
    """Convert Cockx movement-level event rows into classification trials.

    Rows are grouped by ``trial_number``. The second stimulus row is the movement
    onset, the first is the preceding baseline. Recorded correct movements are
    matched against either expected 12-movement sequence using Levenshtein
    distance. Unreconstructable trials are discarded. The recording task
    determines whether the resulting onset is labeled Finger or Foot.

    Returns
    -------
    list of tuple
        ``(onset_seconds, class_name)`` for each valid trial.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown Cockx2023 task: {task}")
    trials = []
    label = "Finger" if task.startswith("finger") else "Foot"
    for trial_number, group in events.groupby("trial_number", sort=False):
        reasons = set(group["reason_for_exclusion"].dropna())
        is_practice = reasons == {"test trial"}
        if not group["included_trial"].astype(bool).all() and (
            not is_practice or exclude_practice_trials
        ):
            continue
        movements = group["correct_movement"].dropna().astype(int).astype(str)
        sequence = "".join(movements.iloc[:12])
        distance = min(
            levenshtein_distance(sequence, expected) for expected in SEQUENCES
        )
        if distance > len(sequence) / 2:
            continue
        stimuli = group[group["event_type"] == "stimulus"]
        if len(stimuli) < 2:
            raise ValueError(f"Missing task onset in trial {trial_number}")
        trials.append((float(stimuli.iloc[1]["onset"]), label))
    return trials


class Cockx2023FnirsME(BaseBIDSFNIRSDataset):
    """Cockx2023 automatic and non-automatic finger and foot movement dataset.

    Right-hand finger tapping or right-foot stepping of a learned 12-movement
    sequence, cued by a red fixation cross, under automatic and non-automatic
    sequence and single- and dual-task conditions [1]_.

    Parameters
    ----------
    apply_subject_exclusions : bool
        Apply the authors' participant-quality exclusions. Default is False.
    exclude_practice_trials : bool
        Exclude trials called test trials in the source data. Default is False.

    Notes
    -----
    The authors excluded ``sub-10`` for channel quality and the first trial of
    each run, labelled ``test trial``. Both are included by default. Trials
    marked erroneous and trials whose sequence cannot be reconstructed are
    excluded.

    ``sub-69`` stores the same source and detector position for the short pair
    ``S21_D9``. Its zero distance marks it as short, so short-channel removal
    also removes this pair.

    References
    ----------
    .. [1] Cockx, H., Oostenveld, R., Tabor, M., Savenco, E., van Setten, A.,
           Cameron, I., & van Wezel, R. (2023). fNIRS is sensitive to leg
           activity in the primary motor cortex after systemic artifact
           correction. NeuroImage, 269, 119880.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"fnirs_od": 72},
            # Two devices, one using half of its optodes.
            hardware="Artinis Brite24",
            # Recorded in Nijmegen, NL. The BIDS sidecars give no
            # PowerLineFrequency.
            line_freq=50.0,
            montage="custom",
            wavelengths_nm=[760.0, 850.0],
            n_sources=24,
            n_detectors=12,
            n_short_channels=12,
            # Per-subject positions from 3D head scans, stored in the SNIRF files.
            sensor_positions="3d",
            # Video and Xsens IMU recordings are not part of the data record.
            other_modalities=[],
        ),
        # From participants.tsv. The paper reports the 23 included subjects.
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            gender={"female": 14, "male": 10},
            # All Dutch Handedness Questionnaire scores are positive.
            handedness={"right": 24},
            age_mean=24.0,
            age_std=2.5,
            age_min=20.0,
            age_max=29.0,
        ),
        experiment=ExperimentMetadata(
            paradigm="fnirsMotorExecution",
            task_type="sequential finger tapping and foot stepping",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            # One self-paced 12-movement sequence at about 1.5 Hz.
            trial_duration=8.0,
            stimulus_type="fixation cross",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.neuroimage.2023.119880",
            investigators=[
                "Helena Cockx",
                "Robert Oostenveld",
                "Merel Tabor",
                "Ecaterina Savenco",
                "Arne van Setten",
                "Ian Cameron",
                "Richard van Wezel",
            ],
            institution="Radboud University",
            country="NL",
            repository="Radboud Data Repository",
            data_url="https://doi.org/10.34973/vesb-mh30",
            # Repository record and LICENSE.txt. dataset_description.json
            # states ODC-BY.
            license="CC BY-SA 4.0",
            publication_year=2023,
        ),
        # Design values. Erroneous trials are dropped, leaving 76-88 per subject.
        data_structure=DataStructureMetadata(
            n_trials=len(TASKS) * TRIALS_PER_RUN,
            n_trials_per_class={
                label: len(TASKS) * TRIALS_PER_RUN // len(EVENTS) for label in EVENTS
            },
            trials_context="per subject",
        ),
        runs_per_session=len(TASKS),
        file_format="BIDS SNIRF",
    )

    preprocessing_options = {"short_channel_threshold": 0.02}  # noqa: RUF012

    def __init__(
        self,
        subjects=None,
        sessions=None,
        *,
        apply_subject_exclusions=False,
        exclude_practice_trials=False,
    ):
        available_subjects = SUBJECTS
        if apply_subject_exclusions:
            available_subjects = [
                subject
                for subject in SUBJECTS
                if subject not in AUTHOR_EXCLUDED_SUBJECTS
            ]
        self.exclude_practice_trials = exclude_practice_trials
        super().__init__(
            subjects=available_subjects,
            sessions_per_subject=1,
            events=EVENTS,
            code=CODE,
            interval=INTERVAL,
            paradigm="fnirsMotorExecution",
            doi="10.1016/j.neuroimage.2023.119880",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def _bids_subject(self, subject):
        return BIDS_SUBJECTS[subject - 1]

    def _get_single_subject_data(self, subject):
        runs = {}
        for bids_path in self.bids_paths(subject):
            raw = self._load_raw(bids_path, subject)
            run = f"{TASKS.index(bids_path.task)}{bids_path.task}"
            runs[run] = self._standardize_raw(raw)
        return {"0fnirs": runs}

    def _load_raw(self, bids_path, subject):
        del subject
        raw = mne.io.read_raw_snirf(bids_path.fpath, preload=True, verbose=False)
        # SNIRF-derived rates differ from 50 Hz only by floating-point noise.
        raw = override_raw_sfreq(raw, SFREQ)
        trials = reconstruct_trials(
            self._read_bids_events(bids_path),
            bids_path.task,
            exclude_practice_trials=self.exclude_practice_trials,
        )
        raw.set_annotations(
            mne.Annotations(
                onset=[onset for onset, _ in trials],
                duration=[0] * len(trials),
                description=[label for _, label in trials],
            )
        )
        return raw

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        del update_path
        root = get_bids_cache_root(CODE, path)
        expected = root / "bids_raw" / f"sub-{self._bids_subject(subject)}"
        marker = root / ".extracted"
        if marker.is_file() and expected.is_dir() and not force_update:
            return str(root / "bids_raw")
        archive_path = Path(dl.data_dl(URL, CODE, path, force_update, verbose))
        archive_copy = root.parent / f".{CODE}.tar.gz"
        if force_update and root.is_dir():
            shutil.copy2(archive_path, archive_copy)
            shutil.rmtree(root)
            archive_path = archive_copy
        root.mkdir(parents=True, exist_ok=True)
        try:
            extract_archive(archive_path, root)
        finally:
            if archive_path == archive_copy:
                archive_copy.unlink(missing_ok=True)
        if not expected.is_dir():
            raise FileNotFoundError(f"Archive does not contain {expected}")
        marker.touch()
        return str(root / "bids_raw")
