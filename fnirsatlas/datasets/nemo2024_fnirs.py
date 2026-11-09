"""NEMO fNIRS emotion recognition datasets."""

from pathlib import Path

from moabb.datasets import download as dl
from moabb.datasets.metadata.schema import (
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)

from .bids_base import BaseBIDSFNIRSDataset
from .metadata import FnirsAcquisitionMetadata
from .utils import download_lock, extract_archive

MONTAGE_PATH = Path(__file__).parent / "montages" / "nemo_montage_3d.json"
URL = "https://osf.io/download/pfst8/"
DOWNLOAD_CODE = "Nemo2024Fnirs"
SUBJECTS = [
    101,
    105,
    107,
    108,
    109,
    112,
    113,
    119,
    120,
    121,
    123,
    124,
    125,
    126,
    127,
    129,
    130,
    131,
    133,
    134,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
]
EMPE_EVENTS = {"HANV": 1, "HAPV": 2, "LANV": 3, "LAPV": 4}
AFIM_EVENTS = {
    "afim_HANV": 6,
    "afim_HAPV": 8,
    "afim_LANV": 5,
    "afim_LAPV": 7,
}
# Onsets are at least 18.66 s apart.
EMPE_INTERVAL = [-2, 15]
# Onsets are at least 34.78 s apart.
AFIM_INTERVAL = [-2, 30]
DOI = "10.1109/TAFFC.2023.3315971"
ACQUISITION = FnirsAcquisitionMetadata(
    sampling_rate=50.0,
    channel_types={"fnirs_od": 48},
    hardware="Artinis Brite 24",
    # PowerLineFrequency of the BIDS sidecars.
    line_freq=50.0,
    montage="custom",
    wavelengths_nm=[756.0, 853.0],
    n_sources=10,
    n_detectors=8,
    n_short_channels=0,
    # montage-toolkit positions, constructed from the paper's probe image and the
    # 2D layout in the SNIRF files.
    sensor_positions="3d",
    other_modalities=[],
)
PARTICIPANTS = ParticipantMetadata(
    n_subjects=len(SUBJECTS),
    gender={"male": 18, "female": 12, "non-binary": 1},
    handedness={"right": 27, "left": 4},
    age_mean=31.7,
    age_std=7.5,
    age_min=21.0,
    age_max=52.0,
)
DOCUMENTATION = DocumentationMetadata(
    doi=DOI,
    investigators=["Michiel Spapé", "Kalle Mäkelä", "Tuukka Ruotsalo"],
    institution="University of Helsinki",
    country="FI",
    repository="OSF",
    data_url="https://osf.io/pd9rv",
    # License of the BIDS dataset_description.json. The OSF record lists "Other".
    license="CC BY-NC",
    publication_year=2024,
)
EXTERNAL_LINKS = {"zenodo": "https://doi.org/10.5281/zenodo.8333978"}


class BaseNemo2024Fnirs(BaseBIDSFNIRSDataset):
    """Base class for NEMO fNIRS datasets sharing the same BIDS archive."""

    _bids_task = None  # Set by subclasses: "empe" or "afim"

    def _load_raw(self, bids_path, subject):
        """Load one recording and replace its 2D layout with the 3D montage."""
        raw = super()._load_raw(bids_path, subject)
        raw.set_montage(self._load_montage_from_json(MONTAGE_PATH)[0])
        return raw

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        data_dir = Path(dl.get_dataset_path(DOWNLOAD_CODE, path))
        base_dir = data_dir / f"MNE-{DOWNLOAD_CODE.lower()}-data"
        bids_root = base_dir / "nemo-bids"
        bids_sub_str = self._bids_subject_str(subject)
        expected = (
            bids_root
            / f"{bids_sub_str}/nirs/{bids_sub_str}_task-{self._bids_task}_nirs.snirf"
        )
        # Imagery and Perception share one archive.
        with download_lock(base_dir):
            if expected.is_file() and not force_update:
                return str(bids_root)
            archive_path = Path(
                dl.data_dl(URL, DOWNLOAD_CODE, path, force_update, verbose)
            )
            base_dir.mkdir(parents=True, exist_ok=True)
            extract_archive(archive_path, base_dir)
        return str(bids_root)


class Nemo2024FnirsPerception(BaseNemo2024Fnirs):
    """NEMO fNIRS emotional perception dataset (EMPE task).

    Viewing IAPS images from four arousal/valence quadrants for 14 s each, after
    a 4 s fixation baseline [1]_.

    Notes
    -----
    The last two trials of subject 126 overlap and the recording ends
    abruptly, so they are excluded.

    References
    ----------
    .. [1] Spapé, M., Mäkelä, K., & Ruotsalo, T. (2024). NEMO: A Database for
           Emotion Analysis Using Functional Near-Infrared Spectroscopy. IEEE
           Transactions on Affective Computing, 15(3), 1166-1177.
    """

    METADATA = DatasetMetadata(
        acquisition=ACQUISITION,
        participants=PARTICIPANTS,
        experiment=ExperimentMetadata(
            paradigm="fnirsEmotion",
            task_type="emotional perception",
            events=EMPE_EVENTS,
            n_classes=len(EMPE_EVENTS),
            class_labels=list(EMPE_EVENTS),
            trial_duration=14.0,
            stimulus_type="IAPS images",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DOCUMENTATION,
        data_structure=DataStructureMetadata(
            n_trials=40,
            n_trials_per_class={label: 10 for label in EMPE_EVENTS},
            trials_context="per subject",
        ),
        file_format="BIDS SNIRF",
        external_links=EXTERNAL_LINKS,
    )
    _bids_task = "empe"

    def _load_raw(self, bids_path, subject):
        raw = super()._load_raw(bids_path, subject)
        if subject == 126:
            # Subject 126 has two overlapping epochs and an abrupt recording end.
            # We assume these epochs are faulty and remove them.
            raw.annotations.delete(
                range(len(raw.annotations) - 2, len(raw.annotations))
            )
        return raw

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=EMPE_EVENTS,
            code="Nemo2024FnirsPerception",
            interval=EMPE_INTERVAL,
            paradigm="fnirsEmotion",
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )


class Nemo2024FnirsImagery(BaseNemo2024Fnirs):
    """NEMO fNIRS affective imagery dataset (AFIM task).

    Imagining a scenario from one of four arousal/valence quadrants for 12 s,
    after reading its description and a 4 s countdown [1]_.

    References
    ----------
    .. [1] Spapé, M., Mäkelä, K., & Ruotsalo, T. (2024). NEMO: A Database for
           Emotion Analysis Using Functional Near-Infrared Spectroscopy. IEEE
           Transactions on Affective Computing, 15(3), 1166-1177.
    """

    METADATA = DatasetMetadata(
        acquisition=ACQUISITION,
        participants=PARTICIPANTS,
        experiment=ExperimentMetadata(
            paradigm="fnirsEmotion",
            task_type="affective imagery",
            events=AFIM_EVENTS,
            n_classes=len(AFIM_EVENTS),
            class_labels=list(AFIM_EVENTS),
            trial_duration=12.0,
            stimulus_type="textual scenario description",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DOCUMENTATION,
        data_structure=DataStructureMetadata(
            n_trials=24,
            n_trials_per_class={label: 6 for label in AFIM_EVENTS},
            trials_context="per subject",
        ),
        file_format="BIDS SNIRF",
        external_links=EXTERNAL_LINKS,
    )
    _bids_task = "afim"

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=AFIM_EVENTS,
            code="Nemo2024FnirsImagery",
            interval=AFIM_INTERVAL,
            paradigm="fnirsEmotion",
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )
