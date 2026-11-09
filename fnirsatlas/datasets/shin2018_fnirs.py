"""Shin et al. 2018 word-generation and n-back fNIRS datasets."""

import re
from pathlib import Path
from typing import ClassVar

import numpy as np
from moabb.datasets import download as dl
from moabb.datasets.metadata.schema import (
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)

from fnirsatlas.datasets.base import BaseFNIRSDataset
from fnirsatlas.datasets.metadata import FnirsAcquisitionMetadata
from fnirsatlas.datasets.utils import download_lock, extract_archive

# NIRx NIRScout wavelengths, also stated in the source files.
WAVELENGTHS = (760, 850)
URL = "https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/NIRS/VP{subject:03d}.zip"
DOWNLOAD_CODE = "Shin2018Fnirs"
SUBJECTS = list(range(1, 27))
MONTAGE_PATH = Path(__file__).parent / "montages" / "shin2018_montage_3d.json"
RUNS_PER_EXPERIMENT = 3
# SamplingRate of the NIRx .hdr files.
SFREQ = 10.416667
N_PAIRS = 36
DOI = "10.1038/sdata.2018.3"
WG_EVENTS = {"word_generation": 1, "baseline": 2}
NBACK_EVENTS = {"0-back": 1, "2-back": 2, "3-back": 3}
# Onsets are at least 28.224 s apart.
WG_INTERVAL = [-2, 12]
# Onsets are at least 67.776 s apart.
NBACK_INTERVAL = [-2, 42]
ACQUISITION = FnirsAcquisitionMetadata(
    sampling_rate=SFREQ,
    channel_types={"fnirs_cw_amplitude": 2 * N_PAIRS},
    # Device field of the NIRx .hdr files.
    hardware="NIRx NIRScout 16x24",
    # Recorded at TU Berlin, DE.
    line_freq=50.0,
    montage="custom",
    wavelengths_nm=[float(wavelength) for wavelength in WAVELENGTHS],
    n_sources=16,
    n_detectors=16,
    n_short_channels=0,
    # montage-toolkit positions: channel centers at the 10-05 positions of the
    # paper's channel labels, optodes inferred at 30 mm separation and numbered as
    # in the NIRx .hdr S-D-Mask.
    sensor_positions="3d",
    # EEG and EOG are in separate archives.
    other_modalities=["EEG", "EOG"],
)
PARTICIPANTS = ParticipantMetadata(
    n_subjects=len(SUBJECTS),
    gender={"male": 9, "female": 17},
    handedness={"right": 26},
    age_mean=26.1,
    age_std=3.5,
    age_min=17.0,
    age_max=33.0,
)
DOCUMENTATION = DocumentationMetadata(
    doi=DOI,
    investigators=[
        "Jaeyoung Shin",
        "Alexander von Lühmann",
        "Do-Won Kim",
        "Jan Mehnert",
        "Han-Jeong Hwang",
        "Klaus-Robert Müller",
    ],
    institution="Technische Universität Berlin",
    country="DE",
    repository="DepositOnce",
    data_url="https://doi.org/10.14279/depositonce-5830.2",
    license="CC BY 4.0",
    publication_year=2018,
)
EXTERNAL_LINKS = {
    "website": "http://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/",
    "code": "https://github.com/JaeyoungShin/simultaneous_EEG-NIRS",
}


class BaseShin2018Fnirs(BaseFNIRSDataset):
    """Shared loader for the Shin 2018 raw NIRx intensity recordings.

    Each subject archive holds nine session folders, three per experiment.
    Every folder of the selected experiment becomes one run.
    """

    _source_event_map: ClassVar[dict[int, int]]

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        subject_dir = self._subject_dir(path, subject)
        # NBack and WG share each subject archive.
        with download_lock(subject_dir):
            if subject_dir.is_dir() and not force_update:
                return str(subject_dir)
            url = URL.format(subject=subject)
            archive_path = Path(
                dl.data_dl(url, DOWNLOAD_CODE, path, force_update, verbose)
            )
            subject_dir.mkdir(parents=True, exist_ok=True)
            extract_archive(archive_path, subject_dir)
        return str(subject_dir)

    def _get_single_subject_data(self, subject):
        session_dirs = self._experiment_dirs(Path(self.data_path(subject)))
        sessions = {}
        for run, session_dir in enumerate(session_dirs):
            self._add_run(sessions, "0fnirs", self._read_session(session_dir), str(run))
        return sessions

    def _experiment_dirs(self, subject_dir):
        """Session folders whose triggers contain every code of the experiment."""
        codes = set(self._source_event_map)
        dirs = [
            session_dir
            for session_dir in sorted(subject_dir.iterdir())
            if session_dir.is_dir()
            and codes <= set(self._read_events(session_dir)[1].tolist())
        ]
        if len(dirs) != RUNS_PER_EXPERIMENT:
            raise ValueError(f"Expected 3 sessions in {subject_dir}, found {dirs}")
        return dirs

    def _read_session(self, session_dir):
        header = self._read_header(session_dir)
        pairs, wavelengths = header["pairs"], header["wavelengths"]
        montage, channel_map = self._load_montage_from_json(MONTAGE_PATH)
        if [f"S{s}_D{d}" for s, d in pairs] != list(channel_map.values()):
            raise ValueError(f"Shin2018 S-D-Mask differs from montage: {session_dir}")
        columns = [(source - 1) * 16 + detector - 1 for source, detector in pairs]
        intensity = [
            np.loadtxt(self._session_file(session_dir, f".wl{index}"))[:, columns].T
            for index in (1, 2)
        ]
        ch_names = [
            f"S{source}_D{detector} {wavelength}"
            for source, detector in pairs
            for wavelength in wavelengths
        ]
        # Interleave the two wavelengths of each pair.
        data = np.stack(intensity, axis=1).reshape(len(ch_names), -1)
        samples, codes = self._read_events(session_dir)
        keep = np.isin(codes, list(self._source_event_map))
        labels = [self._source_event_map[code] for code in codes[keep]]
        stim = self._make_stim_raw(
            data.shape[1], header["sfreq"], samples[keep], labels
        )
        raw = self._make_raw(
            np.vstack([data, stim.get_data()]),
            [*ch_names, "Stim"],
            ["fnirs_cw_amplitude"] * len(ch_names) + ["stim"],
            header["sfreq"],
        )
        # The montage needs the wavelengths set by the metadata step.
        raw = self._populate_fnirs_metadata(raw)
        raw.set_montage(montage)
        return self._standardize_raw(raw)

    @classmethod
    def _read_header(cls, session_dir):
        text = cls._session_file(session_dir, ".hdr").read_text()
        sfreq = float(_header_field(text, r"^SamplingRate=([^\n]+)$"))
        wavelengths = [
            int(value)
            for value in _header_field(text, r'^Wavelengths="([^"]+)"').split()
        ]
        mask_text = _header_field(text, r'^S-D-Mask="#\n(.*?)#"')
        mask = np.array(mask_text.split(), dtype=int).reshape(16, 16)
        pairs = [(int(s) + 1, int(d) + 1) for s, d in zip(*np.nonzero(mask))]
        if wavelengths != list(WAVELENGTHS):
            raise ValueError(f"Unexpected Shin2018 wavelengths in {session_dir}")
        return {"sfreq": sfreq, "wavelengths": wavelengths, "pairs": pairs}

    @classmethod
    def _read_events(cls, session_dir):
        """Trigger samples and codes. Each row is a sample and its trigger bits."""
        events = np.loadtxt(cls._session_file(session_dir, ".evt"), dtype=int, ndmin=2)
        codes = events[:, 1:] @ (1 << np.arange(events.shape[1] - 1))
        return events[:, 0], codes

    @staticmethod
    def _session_file(session_dir, suffix):
        return session_dir / f"NIRS-{session_dir.name}{suffix}"

    @staticmethod
    def _subject_dir(path, subject):
        return (
            Path(dl.get_dataset_path(DOWNLOAD_CODE, path))
            / f"MNE-{DOWNLOAD_CODE.lower()}-data"
            / f"VP{subject:03d}"
        )


class Shin2018FnirsWG(BaseShin2018Fnirs):
    """Shin 2018 word-generation versus baseline fNIRS dataset.

    After a 2 s cue, either generating words starting with the shown letter or
    gazing at a fixation cross for 10 s [1]_.

    Notes
    -----
    The discrimination/selection response experiment is not loaded.

    References
    ----------
    .. [1] Shin, J., von Lühmann, A., Kim, D.-W., Mehnert, J., Hwang, H.-J., &
           Müller, K.-R. (2018). Simultaneous acquisition of EEG and NIRS during
           cognitive tasks for an open access dataset. Scientific Data, 5, 180003.
    """

    METADATA = DatasetMetadata(
        acquisition=ACQUISITION,
        participants=PARTICIPANTS,
        experiment=ExperimentMetadata(
            paradigm="fnirsWordGeneration",
            task_type="word generation",
            events=WG_EVENTS,
            n_classes=len(WG_EVENTS),
            class_labels=list(WG_EVENTS),
            trial_duration=10.0,
            stimulus_type="letter or fixation cross cue",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DOCUMENTATION,
        data_structure=DataStructureMetadata(
            n_trials=60,
            n_trials_per_class={label: 30 for label in WG_EVENTS},
            trials_context="per subject",
        ),
        runs_per_session=RUNS_PER_EXPERIMENT,
        file_format="NIRx",
        external_links=EXTERNAL_LINKS,
        # Peak of within-subject HbO+HbR sLDA over 5 s sliding windows, 10x5-fold
        # cross-validation.
        performance={"accuracy_percent": 74.3},
    )
    _source_event_map: ClassVar = {1: 1, 2: 2}

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=WG_EVENTS,
            code="Shin2018FnirsWG",
            interval=WG_INTERVAL,
            paradigm="fnirsWordGeneration",
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )


class Shin2018FnirsNBack(BaseShin2018Fnirs):
    """Shin 2018 0-back, 2-back, and 3-back fNIRS dataset.

    After a 2 s instruction, a 40 s series of digits shown every 2 s, answering
    whether each matches the digit n positions back [1]_.

    Notes
    -----
    The discrimination/selection response experiment is not loaded.

    References
    ----------
    .. [1] Shin, J., von Lühmann, A., Kim, D.-W., Mehnert, J., Hwang, H.-J., &
           Müller, K.-R. (2018). Simultaneous acquisition of EEG and NIRS during
           cognitive tasks for an open access dataset. Scientific Data, 5, 180003.
    """

    METADATA = DatasetMetadata(
        acquisition=ACQUISITION,
        participants=PARTICIPANTS,
        experiment=ExperimentMetadata(
            paradigm="fnirsNBack",
            task_type="n-back working memory",
            events=NBACK_EVENTS,
            n_classes=len(NBACK_EVENTS),
            class_labels=list(NBACK_EVENTS),
            trial_duration=40.0,
            stimulus_type="digits",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DOCUMENTATION,
        data_structure=DataStructureMetadata(
            n_trials=27,
            n_trials_per_class={label: 9 for label in NBACK_EVENTS},
            trials_context="per subject",
        ),
        runs_per_session=RUNS_PER_EXPERIMENT,
        file_format="NIRx",
        external_links=EXTERNAL_LINKS,
    )
    _source_event_map: ClassVar = {7: 1, 8: 2, 9: 3}

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=NBACK_EVENTS,
            code="Shin2018FnirsNBack",
            interval=NBACK_INTERVAL,
            paradigm="fnirsNBack",
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )


def _header_field(text, pattern):
    """First group of a multi-line NIRx header ``pattern``."""
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError(f"Missing NIRx header field {pattern!r}")
    return match.group(1)
