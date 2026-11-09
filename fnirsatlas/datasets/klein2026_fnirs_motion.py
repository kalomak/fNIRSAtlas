"""Klein2026 paired artifact-benchmarking fNIRS dataset (Figshare)."""

import json
from pathlib import Path

import h5py
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

from .base import BaseFNIRSDataset
from .metadata import FnirsAcquisitionMetadata
from .utils import extract_archive

URL = "https://ndownloader.figshare.com/files/67790217"
CODE = "Klein2026FnirsMotion"
# No paper DOI exists. The archive README names a Data Descriptor without one.
DOI = "10.6084/m9.figshare.31771234"
PARADIGM = "fnirsMotionArtifact"
ARCHIVE_ROOT = "paired_fnirs_scientific_data_figshare"
# The fNIRS data of every run of participants 27-30 is identical to a run of
# participants 1-21, e.g. silence_27 to silence_1 and hands_27 to hands_3.
SUBJECTS = list(range(1, 27))
# Recording order within the single session of each participant.
CONDITIONS = ["hands", "headmovement", "eyes", "stimulation", "silence"]
RUNS = {condition: f"{index}{condition}" for index, condition in enumerate(CONDITIONS)}
MOVEMENT_EVENTS = {
    "hands": "hand_movement",
    "headmovement": "head_movement",
    "eyes": "eye_movement",
}
AUDITORY_CONDITIONS = {"stimulation", *MOVEMENT_EVENTS}
AUDITORY_EVENTS = ["auditory_50db", "auditory_70db", "auditory_90db"]
EVENTS = {
    "auditory_50db": 1,
    "auditory_70db": 2,
    "auditory_90db": 3,
    "hand_movement": 4,
    "head_movement": 5,
    "eye_movement": 6,
}
# Movement onsets can coincide with auditory pulses, so each stream gets
# its own stim channel.
STIM_CHANNELS = {
    "Stim_auditory": AUDITORY_EVENTS,
    "Stim_movement": list(MOVEMENT_EVENTS.values()),
}
# Out is exported on a 500 Hz Simulink grid. The device samples at a
# nominal 10 Hz.
EXPORT_SFREQ = 500.0
SFREQ = 10.0
# Source time by which the startup plateau of device output has ended in
# every record. The plateau ends by 1.6 s.
SETTLE_TIME = 2.0
# Runs with a movement onset at source time 1 s. Their plateau ends by 0.66 s.
EARLY_SETTLE_TIME = {("eyes", 9): 0.8, ("headmovement", 23): 0.8}
N_PAIRS = 8
WAVELENGTHS = (760, 850)
N_SOURCES = 8
N_DETECTORS = 2
# Pairs T1-R1 ... T4-R1 (right) and T5-R2 ... T8-R2 (left).
PAIRS = [
    (transmitter, 1 if transmitter <= 4 else 2)
    for transmitter in range(1, N_SOURCES + 1)
]


class Klein2026FnirsMotion(BaseFNIRSDataset):
    """Klein2026 paired auditory and instructed-movement dataset.

    Participants stayed still through five 92 s runs in one session: a single
    self-timed hand, head or eye movement during auditory stimulation,
    auditory stimulation only, and silence. Auditory runs have 2 s pulses at
    50, 70 and 90 dB SPL at 30, 45 and 60 s. The operator marked each
    movement onset by key press in whole seconds [1]_.

    Notes
    -----
    Participants 27-30 duplicate runs of earlier participants and are
    excluded. The startup plateau of device output, which ends by 1.6 s, is
    dropped from each run. Runs ``eyes_9`` and ``headmovement_23`` are cut at
    0.8 s to keep their movement onsets at 1 s. Pairs ``S1_D1`` to ``S4_D1``
    are constant in all but 12 runs, ``S5_D2`` in 2 runs and ``S8_D2`` in 9
    runs. These are marked bad by the flat check.
    Recording-only: movement onsets often coincide with auditory pulses,
    auditory onsets are the published schedule rather than a trigger trace,
    and movement duration was not recorded.

    References
    ----------
    .. [1] Klein, L., Mimra, T., Augustynek, M., and Knocikova, J. A. (2026).
           A paired fNIRS dataset for artifact-aware benchmarking with induced
           motion and auditory stimulation. figshare.
           https://doi.org/10.6084/m9.figshare.31771234
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"fnirs_cw_amplitude": len(WAVELENGTHS) * N_PAIRS},
            # Not named in the data record. Exported through Simulink.
            hardware="unknown",
            # Recorded in Ostrava, Czech Republic.
            line_freq=50.0,
            montage="custom",
            wavelengths_nm=[float(w) for w in WAVELENGTHS],
            n_sources=N_SOURCES,
            n_detectors=N_DETECTORS,
            n_short_channels=0,
            # Only the hemisphere of each pair is known. All pairs are 30 mm.
            sensor_positions="none",
            # 16 EEG outputs without fitted electrodes or gel, unusable, and
            # one auxiliary output of unknown algorithm and unit.
            other_modalities=["EEG", "unlabelled aux"],
        ),
        participants=ParticipantMetadata(n_subjects=len(SUBJECTS)),
        experiment=ExperimentMetadata(
            paradigm=PARADIGM,
            task_type="instructed movement during auditory stimulation",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            # Auditory pulse duration in protocol_triggers.tsv. Movement
            # duration was not recorded.
            trial_duration=2.0,
            stimulus_type="auditory pulses",
            stimulus_modalities=["auditory"],
            primary_modality="auditory",
            # Auditory pulses follow a fixed schedule. Movements are self-timed.
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=DOI,
            investigators=[
                "Lukáš Klein",
                "Tomas Mimra",
                "Martin Augustynek",
                "Juliana A. Knocikova",
            ],
            # ORCID records of the first two authors.
            institution="VSB - Technical University of Ostrava",
            country="CZ",
            repository="figshare",
            data_url=f"https://doi.org/{DOI}",
            license="CC BY 4.0",
            publication_year=2026,
        ),
        data_structure=DataStructureMetadata(
            n_trials=len(AUDITORY_CONDITIONS) * len(AUDITORY_EVENTS)
            + len(MOVEMENT_EVENTS),
            n_trials_per_class={
                **{label: len(AUDITORY_CONDITIONS) for label in AUDITORY_EVENTS},
                **{label: 1 for label in MOVEMENT_EVENTS.values()},
            },
            trials_context="per subject",
        ),
        runs_per_session=len(CONDITIONS),
        # MATLAB v7.3 (HDF5) files holding the Simulink Out matrix.
        file_format="MAT",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=EVENTS,
            code=CODE,
            interval=None,
            paradigm=PARADIGM,
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        root = self._download_data(path, force_update, verbose)
        return [
            str(root / "raw" / f"{condition}_{subject}.mat") for condition in CONDITIONS
        ]

    def _get_single_subject_data(self, subject):
        root = self._download_data(None, False, None)
        movement_onsets = json.loads(
            (root / "derivatives" / "artifact_onsets.json").read_text()
        )
        pulses = pd.read_csv(root / "derivatives" / "protocol_triggers.tsv", sep="\t")
        sessions = {}
        for condition, source in zip(CONDITIONS, self.data_path(subject), strict=True):
            settle = EARLY_SETTLE_TIME.get((condition, subject), SETTLE_TIME)
            raw, start = self._read_record(Path(source), settle)
            onsets, durations, labels = _run_events(
                condition, subject, pulses, movement_onsets
            )
            # Source onsets refer to Out(:,1). The loaded run starts later,
            # after the startup plateau.
            raw.set_annotations(
                mne.Annotations(np.asarray(onsets) - start, durations, labels)
            )
            self._add_stim_channels(raw)
            self._add_run(sessions, "0", self._standardize_raw(raw), RUNS[condition])
        return sessions

    def _read_record(self, source, settle_time):
        # MATLAB v7.3 files are HDF5. h5py returns Out transposed, so rows
        # are samples: time, 16 fNIRS voltages in mV, then EEG and auxiliary
        # outputs.
        with h5py.File(source, "r") as handle:
            values = handle["Out"][()]
        if values.shape[1] != 35:
            raise ValueError(f"Unexpected Klein2026 Out shape {values.shape}: {source}")
        time = values[:, 0]
        signal = values[:, 1 : 1 + 2 * N_PAIRS]
        first = np.searchsorted(time, settle_time)
        # Columns alternate 760 and 850 nm for each pair.
        names = [
            f"S{source_id}_D{detector} {wavelength}"
            for source_id, detector in PAIRS
            for wavelength in WAVELENGTHS
        ]
        raw = self._make_raw(
            signal[first:].T, names, ["fnirs_cw_amplitude"] * len(names), EXPORT_SFREQ
        )
        raw.resample(SFREQ, method="polyphase", verbose=False)
        # No optode coordinates are released. All pairs are 30 mm.
        return self._populate_fnirs_metadata(raw), time[first]

    def _add_stim_channels(self, raw):
        samples = raw.time_as_index(raw.annotations.onset, use_rounding=True)
        descriptions = raw.annotations.description
        for name, labels in STIM_CHANNELS.items():
            keep = np.isin(descriptions, labels)
            stim = self._make_stim_raw(
                raw.n_times,
                SFREQ,
                samples[keep],
                [EVENTS[label] for label in descriptions[keep]],
                name,
            )
            raw.add_channels([stim], force_update_info=True)

    @staticmethod
    def _download_data(path, force_update, verbose):
        base_dir = Path(dl.get_dataset_path(CODE, path)) / f"MNE-{CODE.lower()}-data"
        root = base_dir / ARCHIVE_ROOT
        if (root / "times.txt").is_file() and not force_update:
            return root
        archive_path = Path(dl.data_dl(URL, CODE, path, force_update, verbose))
        extract_archive(archive_path, base_dir)
        return root


def _run_events(condition, subject, pulses, movement_onsets):
    """Return source onsets, durations and labels for one run."""
    onsets, durations, labels = [], [], []
    # Every sound-present run follows the same schedule of three 2 s pulses.
    if condition in AUDITORY_CONDITIONS:
        onsets += pulses["onset_s"].tolist()
        durations += pulses["duration_s"].tolist()
        labels += [
            f"auditory_{level:.0f}db" for level in pulses["nominal_level_db_spl"]
        ]
    # One operator key press per instructed run, in whole seconds.
    if condition in MOVEMENT_EVENTS:
        onsets.append(movement_onsets[condition][subject - 1])
        durations.append(0.0)
        labels.append(MOVEMENT_EVENTS[condition])
    return onsets, durations, labels
