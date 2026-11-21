"""Shadaydeh2025 fNIRS finger and foot motor-imitation hyperscanning dataset."""

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
from .utils import extract_archive

URL = "https://osf.io/download/wzxng/"
CODE = "Shadaydeh2025FnirsHyperscanning"
TASK = "MotorImitation"
BIDS_SUBJECTS = [
    f"{dyad:02d}{partner}"
    for dyad in range(1, 11)
    for partner in "ab"
    if (dyad, partner) != (7, "b")
]
SUBJECTS = list(range(1, len(BIDS_SUBJECTS) + 1))
EVENTS = {"Finger": 1, "Foot": 2}
# Onsets are at least 79.921 s apart.
INTERVAL = [-2, 30]
# Five videos per class in each of the two roles.
TRIALS_PER_CLASS = 10
SFREQ = 10.172526
PARADIGM = "fnirsMotorExecution"
DOI = "10.1111/ejn.70252"


class Shadaydeh2025FnirsHyperscanning(BaseBIDSFNIRSDataset):
    """Shadaydeh2025 finger and foot motor-imitation hyperscanning dataset.

    Dyads seated face to face. The Model copies a 20 s video of right-hand
    sequential finger tapping or right-foot tapping at about 1.5 Hz, the
    Imitator copies the Model. Each video follows a 60 s fixation-cross rest.
    Partners switch roles halfway. Optodes cover the left and right primary
    motor (M1) and premotor cortex (PMC) [1]_.

    Notes
    -----
    Model and Imitator roles and actor identity are pooled into the two classes.
    The source dataset holds 10 of the 11 recorded dyads, the authors excluded
    one dyad for data quality. Participant ``07b`` is absent.

    References
    ----------
    .. [1] Shadaydeh, M., Noering, V., Franz, M., Chand, T., Croy, I., &
           Denzler, J. (2025). Directionality of Interpersonal Neural Influence
           in Functional Near-Infrared Spectroscopy Hyperscanning: Feasibility
           of Information-Theoretic Causality Analysis in Motor Tasks. European
           Journal of Neuroscience, 62(6), e70252.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"fnirs_cw_amplitude": 56},
            hardware="NIRx NIRSport2",
            # PowerLineFrequency of the BIDS sidecars.
            line_freq=50.0,
            montage="custom",
            wavelengths_nm=[760.0, 850.0],
            n_sources=8,
            # Includes the 8 short-separation detectors.
            n_detectors=15,
            n_short_channels=8,
            # SNIRF probe positions.
            sensor_positions="3d",
            other_modalities=[],
        ),
        # Paper values for all 22 recruited participants (11 dyads). The release
        # has 19 and its participants.tsv is all n/a, so their counts are unknown.
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            gender={"female": 21, "male": 1},
            age_mean=23.15,
            age_std=2.58,
        ),
        experiment=ExperimentMetadata(
            paradigm=PARADIGM,
            task_type="finger and foot tapping imitation",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            trial_duration=20.0,
            # The Model watches the video, the Imitator watches the Model.
            stimulus_type="video",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=DOI,
            investigators=[
                "Maha Shadaydeh",
                "Vanessa Noering",
                "Marcel Franz",
                "Tara Chand",
                "Ilona Croy",
                "Joachim Denzler",
            ],
            institution="Friedrich Schiller University Jena",
            country="DE",
            repository="OSF",
            data_url="https://doi.org/10.17605/OSF.IO/N4JQC",
            license="CC BY 4.0",
            publication_year=2025,
        ),
        data_structure=DataStructureMetadata(
            n_trials=len(EVENTS) * TRIALS_PER_CLASS,
            n_trials_per_class={label: TRIALS_PER_CLASS for label in EVENTS},
            trials_context="per subject",
        ),
        file_format="BIDS SNIRF",
    )

    _bids_task = TASK
    # Source codes 4/5 are finger/foot of actor A, 6/7 of actor B. Codes 1
    # (task block start), 8/9 (rest), and 11 (baseline, added 20 s after rest)
    # are not used.
    _annotation_rename_map = {  # noqa: RUF012
        "4": "Finger",
        "5": "Foot",
        "6": "Finger",
        "7": "Foot",
    }

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

    def _bids_subject(self, subject):
        """Convert a MOABB subject ID to the source BIDS participant ID."""
        return BIDS_SUBJECTS[subject - 1]

    # Partner "b" SNIRF files omit annotations; their verified BIDS sidecars
    # contain the recording-specific events.
    def _load_raw(self, bids_path, subject):
        """Load a recording, restoring annotations missing from partner files."""
        raw = super()._load_raw(bids_path, subject)
        if len(raw.annotations):
            return raw
        events = self._read_bids_events(bids_path)
        raw.set_annotations(
            mne.Annotations(
                events["onset"],
                events["duration"],
                events["trial_type"],
                orig_time=raw.info["meas_date"],
            )
        )
        return raw

    # MOABB's BIDS implementation cannot match recordings without session and
    # run entities, which this dataset omits.
    # NTH: potential abstraction
    def get_additional_metadata(self, subject, session, run):
        """Return selected BIDS events for one recording."""
        matches = [
            path
            for path in self.bids_paths(subject)
            if (path.session or "0") == session and (path.run or "0") == run
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one BIDS path; found {len(matches)}")
        metadata = self._read_bids_events(matches[0])
        metadata["trial_type"] = metadata["trial_type"].replace(
            self._annotation_rename_map
        )
        metadata = metadata[metadata["trial_type"].isin(self.event_id)].copy()
        metadata["value"] = metadata["trial_type"].map(self.event_id)
        return metadata.assign(subject=subject, session=session, run=run)

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        bids_root = get_bids_cache_root(CODE, path) / "BIDS"
        expected = bids_root / self._recording_path(subject)
        if expected.is_file() and not force_update:
            return str(bids_root)
        archive_path = Path(dl.data_dl(URL, CODE, path, force_update, verbose))
        if force_update and bids_root.is_dir():
            shutil.rmtree(bids_root)
        bids_root.parent.mkdir(parents=True, exist_ok=True)
        extract_archive(archive_path, bids_root.parent)
        if not expected.is_file():
            raise FileNotFoundError(f"Archive does not contain {expected}")
        return str(bids_root)

    def _recording_path(self, subject):
        bids_subject = self._bids_subject(subject)
        return Path(
            f"sub-{bids_subject}/nirs/sub-{bids_subject}_task-{TASK}_nirs.snirf"
        )
