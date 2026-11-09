"""Herff et al. 2014 n-back workload fNIRS dataset."""

from pathlib import Path

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

from fnirsatlas.datasets.base import BaseFNIRSDataset
from fnirsatlas.datasets.metadata import FnirsAcquisitionMetadata

# Artinis Oxymon Mark III wavelengths, also stated in the paper.
WAVELENGTHS = (765, 856)
CODE = "Herff2014FnirsWorkload"
SUBJECTS = list(range(1, 11))
EVENTS = {"1-back": 1, "2-back": 2, "3-back": 3}
# Onsets are at least 64.24 s apart. The 2 s before onset fall in the 5 s
# instruction period, not rest. 25 s is the paper's best three-class window.
INTERVAL = [-2, 25]
SFREQ = 25.0
DATA_URL = "https://www.csl.uni-bremen.de/CorpusData/download.php?crps=fNIRS"
CHANNEL_PAIRS = [
    "S1_D1",
    "S2_D1",
    "S1_D2",
    "S2_D2",
    "S3_D3",
    "S4_D3",
    "S3_D4",
    "S4_D4",
]
MONTAGE_PATH = Path(__file__).parent / "montages" / "herff2014_montage_3d.json"


class Herff2014FnirsWorkload(BaseFNIRSDataset):
    """Herff et al. 2014 prefrontal n-back workload fNIRS dataset.

    Ten trials each of 1-, 2- and 3-back letter tasks, each 44 s after a 5 s
    instruction and followed by 15 s of rest [1]_.

    Notes
    -----
    The 10 s relax trials in the source data are not loaded as a class.
    The source files require registration at the distributor.

    References
    ----------
    .. [1] Herff, C., Heger, D., Fortmann, O., Hennrich, J., Putze, F., &
           Schultz, T. (2014). Mental workload during n-back task - quantified
           in the prefrontal cortex using fNIRS. Frontiers in Human
           Neuroscience, 7, 935.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"hbo": len(CHANNEL_PAIRS), "hbr": len(CHANNEL_PAIRS)},
            hardware="Artinis Oxymon Mark III",
            # Recorded at the Karlsruhe Institute of Technology, DE.
            line_freq=50.0,
            montage="custom",
            wavelengths_nm=[float(w) for w in WAVELENGTHS],
            n_sources=4,
            n_detectors=4,
            n_short_channels=0,
            # montage-toolkit positions, constructed from Fig. 1 of the paper and
            # the Fp1/Fp2 reference positions.
            sensor_positions="3d",
            other_modalities=[],
        ),
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            gender={"male": 6, "female": 4},
            handedness={"right": 8, "left": 2},
            age_mean=22.0,
        ),
        experiment=ExperimentMetadata(
            paradigm="fnirsNBack",
            task_type="n-back working memory",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            trial_duration=44.0,
            stimulus_type="letter sequence",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnhum.2013.00935",
            investigators=[
                "Christian Herff",
                "Dominic Heger",
                "Ole Fortmann",
                "Johannes Hennrich",
                "Felix Putze",
                "Tanja Schultz",
            ],
            institution="Karlsruhe Institute of Technology",
            country="DE",
            repository="Cognitive Systems Lab, University of Bremen",
            data_url=DATA_URL,
            # Neither the download page nor the archive readme states a license.
            license="unknown",
            publication_year=2014,
        ),
        data_structure=DataStructureMetadata(
            n_trials=30,
            n_trials_per_class={label: 10 for label in EVENTS},
            trials_context="per subject",
        ),
        file_format="MATLAB",
        # Within-subject three-class LDA on HbO/HbR slopes of 25 s windows, 10-fold
        # cross-validation.
        performance={"accuracy_percent": 50.3},
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=EVENTS,
            code=CODE,
            interval=INTERVAL,
            paradigm="fnirsNBack",
            doi="10.3389/fnhum.2013.00935",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        del force_update, update_path, verbose
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        data_file = self._data_file(path, subject)
        if data_file.is_file():
            return str(data_file)
        raise FileNotFoundError(
            f"Herff2014 requires registration at {DATA_URL}. "
            f"Place subject{subject}.mat at {data_file}"
        )

    def _get_single_subject_data(self, subject):
        source = loadmat(
            self.data_path(subject), squeeze_me=True, struct_as_record=False
        )
        return self._single_run_session(
            self._convert_subject(source["data"], source["label"])
        )

    def _convert_subject(self, data, labels):
        ch_names = [
            f"{pair} {chromophore}"
            for pair in CHANNEL_PAIRS
            for chromophore in ("hbo", "hbr")
        ]
        # Source rows are unfiltered HbO/HbR concentration changes in micromolar.
        raw = self._make_raw(
            data * 1e-6,
            ch_names,
            ["hbo", "hbr"] * len(CHANNEL_PAIRS),
            SFREQ,
            montage=self._load_montage_from_json(MONTAGE_PATH)[0],
        )
        labels = np.asarray(labels).ravel()
        onsets = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
        onsets = onsets[np.isin(labels[onsets], list(EVENTS.values()))]
        stim = self._make_stim_raw(raw.n_times, SFREQ, onsets, labels[onsets])
        raw.add_channels([stim])
        raw = self._populate_fnirs_metadata(raw, default_wavelengths=WAVELENGTHS)
        return self._standardize_raw(raw)

    @staticmethod
    def _data_file(path, subject):
        return (
            Path(dl.get_dataset_path(CODE, path))
            / f"MNE-{CODE.lower()}-data"
            / f"subject{subject}.mat"
        )
