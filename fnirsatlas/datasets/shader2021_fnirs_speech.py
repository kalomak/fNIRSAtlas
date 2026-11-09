"""Shader2021 auditory and visual speech fNIRS dataset."""

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

URL = "https://osf.io/download/xwerv/"
CODE = "Shader2021FnirsSpeech"
DOI = "10.1016/j.heares.2021.108256"
TASK = "AudioVisualBroadVsRestricted"
SUBJECTS = list(range(1, 9))
SFREQ = 3.90625
EVENTS = {"Audio": 1, "Control": 2, "Video": 3}
# Selected events are at least 26.88 seconds apart in the source data.
INTERVAL = [-2, 19.968]
EXTRACTED_DIR = "2021-fNIRS-Audio-visual-speech-Broad-vs-restricted-regions"


class Shader2021FnirsSpeech(BaseBIDSFNIRSDataset):
    """Shader2021 auditory-only and visual-only speech fNIRS dataset.

    Segments of a connected children's story, presented either auditory-only
    or visual-only (lipreading video), and silent control trials [1]_.

    Notes
    -----
    Source ``15.0`` (``Misc``) markers are excluded.

    References
    ----------
    .. [1] Shader, M. J., Luke, R., Gouailhardou, N., & McKay, C. M. (2021).
           The use of broad vs restricted regions of interest in functional
           near-infrared spectroscopy for measuring cortical activation to
           auditory-only and visual-only speech. Hearing Research, 406, 108256.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"fnirs_cw_amplitude": 104},
            hardware="NIRx NIRScout",
            # BIDS PowerLineFrequency.
            line_freq=50.0,
            montage="custom",
            wavelengths_nm=[760.0, 850.0],
            n_sources=16,
            # 15 long-separation and 8 short-separation detectors, BIDS optodes.tsv.
            n_detectors=23,
            n_short_channels=8,
            # BIDS optodes.tsv. Not digitised, placed by the 10-20 system.
            sensor_positions="3d",
            other_modalities=[],
        ),
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            # participants.tsv. The paper does not report sex.
            gender={"male": 4, "female": 4},
            age_mean=29.8,
            age_std=3.1,
            age_min=24.0,
            age_max=34.0,
        ),
        experiment=ExperimentMetadata(
            paradigm="fnirsAudiovisual",
            task_type="speech perception",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            # Mean speech segment duration. Control trials are 10 s of silence.
            trial_duration=12.5,
            stimulus_type="connected speech",
            stimulus_modalities=["auditory", "visual"],
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=DOI,
            investigators=[
                "Maureen J. Shader",
                "Robert Luke",
                "Nathalie Gouailhardou",
                "Colette M. McKay",
            ],
            institution="Bionics Institute",
            country="AU",
            repository="OSF",
            data_url="https://doi.org/10.17605/OSF.IO/U23RB",
            license="CC BY 4.0",
            publication_year=2021,
        ),
        data_structure=DataStructureMetadata(
            n_trials=46,
            n_trials_per_class={"Audio": 18, "Control": 10, "Video": 18},
            trials_context="per subject",
        ),
        file_format="BIDS SNIRF",
    )

    _bids_task = TASK
    _bids_session = "01"
    _bids_subject_format = "{:02d}"
    _annotation_rename_map = {  # noqa: RUF012
        "1.0": "Audio",
        "2.0": "Video",
        "3.0": "Control",
    }

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=EVENTS,
            code=CODE,
            interval=INTERVAL,
            paradigm="fnirsAudiovisual",
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        bids_root = get_bids_cache_root(CODE, path) / EXTRACTED_DIR
        expected = bids_root / self._recording_path(subject)
        if expected.is_file() and not force_update:
            return str(bids_root)
        archive_path = Path(dl.data_dl(URL, CODE, path, force_update, verbose))
        if bids_root.is_dir():
            shutil.rmtree(bids_root)
        bids_root.parent.mkdir(parents=True, exist_ok=True)
        extract_archive(archive_path, bids_root.parent)
        if not expected.is_file():
            raise FileNotFoundError(f"Archive does not contain {expected}")
        return str(bids_root)

    def _recording_path(self, subject):
        bids_subject = self._bids_subject(subject)
        return Path(
            f"sub-{bids_subject}/ses-01/nirs/"
            f"sub-{bids_subject}_ses-01_task-{TASK}_nirs.snirf"
        )
