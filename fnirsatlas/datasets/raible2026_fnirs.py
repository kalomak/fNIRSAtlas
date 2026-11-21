"""Raible2026 MULtiple PAradigms (MULPA) fNIRS datasets (Zenodo)."""

import shutil
from dataclasses import replace
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
from .utils import download_lock, extract_archive, merge_repeated_annotations

URL = "https://zenodo.org/api/records/23163739/files/MULPA_v1.1.zip/content"
# All MULPA datasets share one archive.
DOWNLOAD_CODE = "Raible2026Fnirs"
DOI = "10.64898/2026.06.06.728412"
DATA_DOI = "10.5281/zenodo.21033498"
ARCHIVE_ROOT = "MULPA_v1.1"
SUBJECTS = list(range(1, 58))
# Subjects without a loaded recording of the task: missing from the source or
# fully in EXCLUDED.
EMOTION_MISSING = frozenset({49, 50, 51, 54, 55})
VISUAL_MISSING = frozenset({24, 57})
MUSIC_MISSING = frozenset({57})
# SNIRF time step of 0.07936 s. The paper reports 12.6 Hz.
SFREQ = 1 / 0.07936
WAVELENGTHS_NM = [760.0, 850.0]
# 102 long and 32 short source-detector pairs.
N_PAIRS = 134
N_SHORT_PAIRS = 32
ME_PARADIGM = "fnirsMotorExecution"
MI_PARADIGM = "fnirsMotorImagery"
EMOTION_PARADIGM = "fnirsEmotion"
VISUAL_PARADIGM = "fnirsVisual"
REST_PARADIGM = "fnirsRestingState"
ME_EVENTS = {"finger_tapping": 1, "baseline": 2}
MI_EVENTS = {"motor_imagery": 1, "baseline": 2}
MIFQ_EVENTS = {"motor_imagery_speed_change": 1, "baseline": 2}
EMOTION_EVENTS = {"neutral_happy": 1, "neutral_sad": 2, "neutral_neutral": 3}
VISUAL_EVENTS = {"threatening": 1, "non_threatening": 2, "control": 3}
MUSIC_EVENTS = {
    "positive_valence_high_arousal": 1,
    "positive_valence_low_arousal": 2,
    "negative_valence_high_arousal": 3,
    "negative_valence_low_arousal": 4,
}
REST_EVENTS = {"start": 1, "end": 2}
# Source annotation to event name. Start and End mark the processing window of
# each recording.
EVENT_NAMES = {
    "Start": "start",
    "End": "end",
    "Baseline": "baseline",
    "MotorAction": "finger_tapping",
    "MentalImagery": "motor_imagery",
    "MentalImageryFreqChange": "motor_imagery_speed_change",
    "Neutral-Happy": "neutral_happy",
    "Neutral-Sad": "neutral_sad",
    "Neutral-Neutral": "neutral_neutral",
    "Threatening": "threatening",
    "Non-Threatening": "non_threatening",
    "Control": "control",
    "PosValence_HighArousal": "positive_valence_high_arousal",
    "PosValence_LowArousal": "positive_valence_low_arousal",
    "NegValence_HighArousal": "negative_valence_high_arousal",
    "NegValence_LowArousal": "negative_valence_low_arousal",
}
# Recordings as (subject, BIDS file stem) that are not loaded.
EXCLUDED = {
    # Stubs under 4 s that hold only Start and End markers.
    (51, "sub-51_task-emotion"),
    (54, "sub-54_task-emotion"),
    (52, "sub-52_task-music_run-1"),
    # Holds one trial and ends 17 s after its onset, before its epoch ends.
    (55, "sub-55_task-emotion"),
    # Faces are shown about 2 s instead of the 12-13 s morph, so trials start
    # 16.6-19.0 s apart and overlap the next trial's response.
    (50, "sub-50_task-emotion"),
}
# Motor blocks and their baselines start at least 15.951 s apart.
MOTOR_INTERVAL = [-2, 13.9]
# Emotion trials start at least 27.141 s apart.
EMOTION_INTERVAL = [-2, 25.1]
# Visual blocks start at least 22.459 s apart.
VISUAL_INTERVAL = [-2, 20.4]
# Music blocks start at least 38.966 s apart.
MUSIC_INTERVAL = [-2, 36.9]
# Stimuli within a visual or music block start at most 8.1 s apart. Blocks of
# one class start at least 16.4 s apart.
MAX_STIMULUS_INTERVAL = 10
ACQUISITION = FnirsAcquisitionMetadata(
    sampling_rate=SFREQ,
    channel_types={"fnirs_cw_amplitude": len(WAVELENGTHS_NM) * N_PAIRS},
    # Two cascaded NIRSport2 systems.
    hardware="NIRx NIRSport2",
    software="Aurora 2023.9.3",
    # Recorded in Maastricht, the Netherlands.
    line_freq=50.0,
    montage="custom",
    wavelengths_nm=WAVELENGTHS_NM,
    n_sources=32,
    # 28 long-separation and 32 short-separation (about 8 mm) detectors.
    n_detectors=60,
    n_short_channels=N_SHORT_PAIRS,
    # Template positions in MNI152NLin2009aSym from the SNIRF files, identical
    # for all subjects.
    sensor_positions="3d",
    # SNIRF aux channels. The paper records EMG at the right hand, the SNIRF
    # labels its channels ExGa1-4.
    other_modalities=[
        "PPG",
        "heart rate",
        "SpO2",
        "respiration",
        "GSR",
        "ECG",
        "EMG",
        "temperature",
        "accelerometer",
        "gyroscope",
    ],
)
# From participants.tsv, whose age summary matches the paper. Sex, gender and
# handedness are n/a for 5 subjects.
PARTICIPANTS = ParticipantMetadata(
    n_subjects=len(SUBJECTS),
    gender={"female": 34, "male": 18},
    handedness={"right": 44, "left": 7, "ambidextrous": 1},
    age_mean=26.93,
    age_std=8.67,
    age_min=19.0,
    age_max=79.0,
)
DOCUMENTATION = DocumentationMetadata(
    doi=DOI,
    # dataset_description.json.
    investigators=[
        "Sophie Raible",
        "João Pereira",
        "Foivos Kostogiannis",
        "Bruno Direito",
        "Teresa Sousa",
        "Manu da Cunha Seiffert",
        "Rik Lavicka",
        "Vendija Skeltona",
        "Daniëlle Evenblij",
        "Assunta Ciarlo",
        "Armin Heinecke",
        "Jacqueline Gädtke",
        "Zeus Tipado",
        "David M. A. Mehler",
        "Simon H. Kohl",
        "Miguel Castelo-Branco",
        "Rainer Goebel",
        "Michael Lührs",
        "Bettina Sorger",
    ],
    institution="Maastricht University",
    country="NL",
    repository="Zenodo",
    data_url=f"https://doi.org/{DATA_DOI}",
    license="CC BY-NC 4.0",
    publication_year=2026,
)


def _metadata(
    paradigm: str,
    events: dict[str, int],
    n_trials_per_class: dict[str, int],
    task_type: str,
    trial_duration: float,
    stimulus_type: str,
    stimulus_modality: str,
    runs_per_session: int = 1,
    missing: frozenset[int] = frozenset(),
) -> DatasetMetadata:
    """METADATA of one MULPA task, with ``n_trials_per_class`` per subject.

    ``missing`` subjects have no recording of the task. The other participant
    counts cover all subjects.
    """
    return DatasetMetadata(
        acquisition=ACQUISITION,
        participants=replace(PARTICIPANTS, n_subjects=len(SUBJECTS) - len(missing)),
        experiment=ExperimentMetadata(
            paradigm=paradigm,
            task_type=task_type,
            events=events,
            n_classes=len(events),
            class_labels=list(events),
            trial_duration=trial_duration,
            stimulus_type=stimulus_type,
            stimulus_modalities=[stimulus_modality],
            primary_modality=stimulus_modality,
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DOCUMENTATION,
        data_structure=DataStructureMetadata(
            n_trials=sum(n_trials_per_class.values()),
            n_trials_per_class=n_trials_per_class,
            trials_context="per subject",
        ),
        runs_per_session=runs_per_session,
        file_format="BIDS SNIRF",
    )


def _motor_metadata(
    paradigm: str, events: dict[str, int], task_type: str
) -> DatasetMetadata:
    """METADATA of a motor task: 9 blocks of 16 s and 10 baselines."""
    task, baseline = events
    return _metadata(
        paradigm,
        events,
        # The run starts with a 20 s baseline.
        {task: 9, baseline: 10},
        task_type=task_type,
        trial_duration=16.0,
        stimulus_type="hand icon",
        stimulus_modality="visual",
    )


class BaseRaible2026Fnirs(BaseBIDSFNIRSDataset):
    """MULPA recordings of one task, read from the shared BIDS archive.

    SNIRF stimulus names are renamed by ``EVENT_NAMES``. They match the BIDS
    ``events.tsv`` sidecars, except that sidecars can list events after the
    end of a truncated recording.
    """

    _bids_subject_format = "{:02d}"
    _bids_task = None  # Set by subclasses.
    _missing = frozenset()  # Subjects without a recording of the task.
    _annotation_rename_map = EVENT_NAMES

    def __init__(self, subjects, sessions, **kwargs):
        super().__init__(
            subjects=[subject for subject in SUBJECTS if subject not in self._missing],
            sessions_per_subject=1,
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
            **kwargs,
        )

    def _get_single_subject_data(self, subject):
        # Only the music task has two runs.
        runs = {
            bids_path.run or "0": self._standardize_raw(
                self._load_raw(bids_path, subject)
            )
            for bids_path in self.bids_paths(subject)
            if (subject, bids_path.basename.removesuffix("_nirs.snirf")) not in EXCLUDED
        }
        return {"0": runs} if runs else {}

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        base_dir = get_bids_cache_root(DOWNLOAD_CODE, path)
        bids_root = base_dir / ARCHIVE_ROOT
        # Inside the archive root so another archive version is extracted anew.
        marker = bids_root / ".extracted"
        with download_lock(base_dir):
            if marker.is_file() and not force_update:
                return str(bids_root)
            archive_path = Path(
                dl.data_dl(URL, DOWNLOAD_CODE, path, force_update, verbose)
            )
            marker.unlink(missing_ok=True)
            if bids_root.is_dir():
                shutil.rmtree(bids_root)
            extract_archive(archive_path, base_dir)
            if not (bids_root / "participants.tsv").is_file():
                raise FileNotFoundError(f"MULPA archive lacks {bids_root}")
            marker.touch()
        return str(bids_root)


class Raible2026FnirsME(BaseRaible2026Fnirs):
    """Raible2026 MULPA finger tapping fNIRS dataset.

    Bimanual finger tapping versus rest (MULPA ``motoraction``). Nine 16 s
    tapping blocks alternate with 15-18 s fixation-cross baselines [1]_.

    References
    ----------
    .. [1] Raible, S., Pereira, J., et al. (2026). All signals considered: Data
           quality partially explains inter-individual task differences in a
           large, open fNIRS dataset. bioRxiv.
    """

    METADATA = _motor_metadata(ME_PARADIGM, ME_EVENTS, "bimanual finger tapping")
    _bids_task = "motoraction"

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects,
            sessions,
            events=ME_EVENTS,
            code="Raible2026FnirsME",
            interval=MOTOR_INTERVAL,
            paradigm=ME_PARADIGM,
        )


class Raible2026FnirsMI(BaseRaible2026Fnirs):
    """Raible2026 MULPA motor imagery fNIRS dataset.

    Bimanual motor imagery versus rest (MULPA ``motorimagery``). Nine 16 s
    imagery blocks alternate with 15-18 s fixation-cross baselines [1]_.

    Notes
    -----
    Subject 56's recording ends during its fifth block.

    References
    ----------
    .. [1] Raible, S., Pereira, J., et al. (2026). All signals considered: Data
           quality partially explains inter-individual task differences in a
           large, open fNIRS dataset. bioRxiv.
    """

    METADATA = _motor_metadata(MI_PARADIGM, MI_EVENTS, "bimanual motor imagery")
    _bids_task = "motorimagery"

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects,
            sessions,
            events=MI_EVENTS,
            code="Raible2026FnirsMI",
            interval=MOTOR_INTERVAL,
            paradigm=MI_PARADIGM,
        )


class Raible2026FnirsMIFQ(BaseRaible2026Fnirs):
    """Raible2026 MULPA motor imagery with speed change fNIRS dataset.

    Bimanual motor imagery with a speed change versus rest (MULPA
    ``motorimageryfreqchange``). Imagined movement speeds up until an auditory
    cue 8 s into each 16 s block, then slows down. Nine blocks alternate with
    15-18 s fixation-cross baselines [1]_.

    References
    ----------
    .. [1] Raible, S., Pereira, J., et al. (2026). All signals considered: Data
           quality partially explains inter-individual task differences in a
           large, open fNIRS dataset. bioRxiv.
    """

    METADATA = _motor_metadata(
        MI_PARADIGM, MIFQ_EVENTS, "bimanual motor imagery with speed change"
    )
    _bids_task = "motorimageryfreqchange"

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects,
            sessions,
            events=MIFQ_EVENTS,
            code="Raible2026FnirsMIFQ",
            interval=MOTOR_INTERVAL,
            paradigm=MI_PARADIGM,
        )


class Raible2026FnirsEmotion(BaseRaible2026Fnirs):
    """Raible2026 MULPA emotion recognition fNIRS dataset.

    Faces morphing from neutral to happy, sad or neutral (MULPA ``emotion``).
    Six 12-13 s trials per class, separated by 15-17 s baselines [1]_.

    Notes
    -----
    Not loaded: emotion recordings of subjects 50, 51, 54 and 55 (see
    ``EXCLUDED``). Subject 49 has no emotion recording. Subject 57's recording
    is partial.

    References
    ----------
    .. [1] Raible, S., Pereira, J., et al. (2026). All signals considered: Data
           quality partially explains inter-individual task differences in a
           large, open fNIRS dataset. bioRxiv.
    """

    METADATA = _metadata(
        EMOTION_PARADIGM,
        EMOTION_EVENTS,
        {label: 6 for label in EMOTION_EVENTS},
        task_type="emotion recognition",
        # Paper value. The events last 12.2-12.8 s.
        trial_duration=12.0,
        stimulus_type="morphing facial expressions",
        stimulus_modality="visual",
        missing=EMOTION_MISSING,
    )
    _bids_task = "emotion"
    _missing = EMOTION_MISSING

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects,
            sessions,
            events=EMOTION_EVENTS,
            code="Raible2026FnirsEmotion",
            interval=EMOTION_INTERVAL,
            paradigm=EMOTION_PARADIGM,
        )


class BaseBlockRaible2026Fnirs(BaseRaible2026Fnirs):
    """MULPA task whose source marks every stimulus of a block.

    Consecutive stimuli of one class form a block, and one event marks each
    block.
    """

    def _load_raw(self, bids_path, subject):
        raw = super()._load_raw(bids_path, subject)
        return raw.set_annotations(
            merge_repeated_annotations(raw.annotations, MAX_STIMULUS_INTERVAL)
        )


class Raible2026FnirsThreat(BaseBlockRaible2026Fnirs):
    """Raible2026 MULPA visual threat fNIRS dataset.

    Threatening or non-threatening animal faces, or grass and flower control
    images (MULPA ``visual``). Five 7.5 s blocks per class, each of five 1.5 s
    images, separated by 15-17 s baselines. One event marks each block [1]_.

    Notes
    -----
    Subjects 24 and 57 have no visual recording.

    References
    ----------
    .. [1] Raible, S., Pereira, J., et al. (2026). All signals considered: Data
           quality partially explains inter-individual task differences in a
           large, open fNIRS dataset. bioRxiv.
    """

    METADATA = _metadata(
        VISUAL_PARADIGM,
        VISUAL_EVENTS,
        {label: 5 for label in VISUAL_EVENTS},
        task_type="threat perception",
        trial_duration=7.5,
        stimulus_type="animal face and scene images",
        stimulus_modality="visual",
        missing=VISUAL_MISSING,
    )
    _bids_task = "visual"
    _missing = VISUAL_MISSING

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects,
            sessions,
            events=VISUAL_EVENTS,
            code="Raible2026FnirsThreat",
            interval=VISUAL_INTERVAL,
            paradigm=VISUAL_PARADIGM,
        )


class Raible2026FnirsMusic(BaseBlockRaible2026Fnirs):
    """Raible2026 MULPA music listening fNIRS dataset.

    Music listening in four valence-arousal classes (MULPA ``music``), eyes
    closed. Each of two runs holds one block per class of three 7 s excerpts,
    8 s apart, separated by 15-17 s baselines. One event marks each block, so
    a subject has at most two trials per class [1]_.

    Notes
    -----
    Too few trials per class for within-session evaluation. Not loaded: run 1
    of subject 52 (see ``EXCLUDED``). Missing from the source: run 1
    of subject 57 and run 2 of subjects 23 and 57.

    References
    ----------
    .. [1] Raible, S., Pereira, J., et al. (2026). All signals considered: Data
           quality partially explains inter-individual task differences in a
           large, open fNIRS dataset. bioRxiv.
    """

    METADATA = _metadata(
        EMOTION_PARADIGM,
        MUSIC_EVENTS,
        {label: 2 for label in MUSIC_EVENTS},
        task_type="music listening",
        # Three 7 s excerpts with 1 s gaps.
        trial_duration=23.0,
        stimulus_type="music excerpts",
        stimulus_modality="auditory",
        runs_per_session=2,
        missing=MUSIC_MISSING,
    )
    _bids_task = "music"
    _missing = MUSIC_MISSING

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects,
            sessions,
            events=MUSIC_EVENTS,
            code="Raible2026FnirsMusic",
            interval=MUSIC_INTERVAL,
            paradigm=EMOTION_PARADIGM,
        )


class Raible2026FnirsRest(BaseRaible2026Fnirs):
    """Raible2026 MULPA resting-state fNIRS dataset.

    Five-minute fixation-cross resting state (MULPA ``restingstate``) [1]_.
    Start and End annotations mark the resting period.

    Notes
    -----
    Recording-only: one resting period per subject, no classes.

    References
    ----------
    .. [1] Raible, S., Pereira, J., et al. (2026). All signals considered: Data
           quality partially explains inter-individual task differences in a
           large, open fNIRS dataset. bioRxiv.
    """

    METADATA = _metadata(
        REST_PARADIGM,
        REST_EVENTS,
        {label: 1 for label in REST_EVENTS},
        task_type="eyes-open resting state",
        trial_duration=300.0,
        stimulus_type="fixation cross",
        stimulus_modality="visual",
    )
    _bids_task = "restingstate"

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects,
            sessions,
            events=REST_EVENTS,
            code="Raible2026FnirsRest",
            interval=None,
            paradigm=REST_PARADIGM,
        )
