import warnings
import numpy as np
import mne
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import logging

from tqdm import tqdm
from fnirsatlas.utils import get_tqdm_env_kwargs, recursive_update
from fnirsatlas.epoching import create_epochs_from_raw
from fnirsatlas.montage import load_montage_from_file, has_saved_montage
from fnirsatlas.preprocessing import process_raw
from fnirsatlas.config import get_dataset_path, get_classification_tasks

logger = logging.getLogger(__name__)


class BaseFNIRSParser(ABC):
    """Abstract base class for fNIRS dataset parsers."""

    dataset_id: str = None  # Should be set in each subclass

    def __init__(self, dataset_path=None, target_sfreq: float = 5.0):
        self.dataset_path = dataset_path
        self.target_sfreq = target_sfreq  # Default downsampling to 5Hz
        if self.dataset_path is None:
            self.dataset_path = get_dataset_path(self.dataset_id)
        self.cond2label = None
        self.montage = None
        if has_saved_montage(self.dataset_id):
            self.montage = load_montage_from_file(self.dataset_id)
        self.default_wls = [760, 850]

    @abstractmethod
    def load_data(self, subject_id: str, session: Any = None) -> Dict:
        pass

    @abstractmethod
    def get_subject_ids(self) -> List[str]:
        """Get list of available subject IDs."""
        pass

    @property
    def dataset_name(self):
        dataset_name = self.dataset_id
        if hasattr(self, "experiment"):
            dataset_name += f"__{self.experiment}"
        return dataset_name

    def extract_channels(
        self, data: Dict, session: Any = None
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        pass

    def extract_events(
        self, data: Dict, session: Any = None
    ) -> Tuple[np.ndarray, Dict]:
        pass

    def get_montage(self) -> mne.channels.DigMontage:
        """Get the montage for the dataset."""
        self.montage = load_montage_from_file(self.dataset_id)
        if self.montage is not None:
            logger.debug(f"Montage loaded for dataset '{self.dataset_id}'")
        else:
            logger.debug(f"No montage file available for dataset '{self.dataset_id}'")
        return self.montage

    def get_sampling_frequency(self, data: Dict) -> float:
        """Extract sampling frequency from data."""
        return data["sfreq"]

    def get_sessions(self, subject_id: str) -> list[str]:
        """Get available sessions for a subject."""
        return [0]

    def convert_to_raw(self, subject_id: str) -> mne.io.RawArray:
        """Return the raw object."""
        sessions = self.get_sessions(subject_id)
        raw_list = []
        for session in sessions:
            data = self.load_data(subject_id, session=session)
            raw_list.append(self._raw_from_session(data))
        raw = self._concatenate_raws(raw_list)
        raw.info["subject_info"] = {"his_id": str(subject_id)}
        return raw
        
    def _concatenate_raws(self, raw_list):
        """concatenate a list of raw objects. Also combines metadata."""
        temp = {}
        for raw in raw_list:
            temp = recursive_update(temp, raw.info.get("temp", {}))
        raw = mne.concatenate_raws(raw_list, verbose=False)
        raw.info["temp"] = temp
        return raw

    def set_montage(self, raw):
        if self.montage is not None:
            raw.set_montage(self.montage)
        return raw

    def set_annotations(self, raw, events, event_desc):
        if events.size > 0:
            annot_from_events = mne.annotations_from_events(
                events=events,
                event_desc=event_desc,
                sfreq=raw.info["sfreq"],
                verbose=False,
            )
            raw.set_annotations(annot_from_events)
            # store event id to preserve event labels
            event_id = {str(v): int(k) for k, v in event_desc.items()}
            raw.info["temp"] = {"event_id": event_id}
        if "Stim" in raw.ch_names:
            raw.drop_channels(["Stim"])
        return raw

    def _raw_from_session(self, data: Dict) -> mne.io.RawArray:
        ch_data, ch_names, ch_types = self.extract_channels(data)
        events, event_desc = self.extract_events(data)
        sfreq = self.get_sampling_frequency(data)
        trig = np.zeros((1, ch_data.shape[1]))
        if events.size > 0:
            event_times = events[:, 0]
            event_ids = events[:, 2]
            trig[0, event_times] = event_ids
        raw_data = np.vstack([ch_data, trig])
        ch_names.append("Stim")  # TODO can we remove Stim?
        ch_types.append("stim")
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        raw = mne.io.RawArray(data=raw_data, info=info, verbose=False)
        raw = self.set_channel_wavelengths(raw)
        raw = self.set_montage(raw)
        raw = self.set_annotations(raw, events, event_desc)
        return raw

    def _create_epochs(self, raw):
        return create_epochs_from_raw(
            raw=raw,
            dataset_name=self.dataset_name,
        )

    def _process_raw(self, raw):
        return process_raw(raw, dataset_name=self.dataset_name)

    def get_epochs(
        self,
        subject_ids=None,
        downsample: bool = True,
        disable_tqdm: bool = False,
    ) -> dict[str, mne.Epochs]:
        """
        Loads all subjects and creates epochs for each subject using the standard interface.
        Returns a subject->epochs dict.
        """
        if subject_ids is None:
            subject_ids = self.get_subject_ids()

        ed = {}
        cond2label = {}
        for subject_id in tqdm(
            subject_ids,
            disable=disable_tqdm,
            desc=f"{'Processing':11} {self.dataset_name[:12]:<12}",
            **get_tqdm_env_kwargs(),
        ):
            sessions = self.get_sessions(subject_id)
            all_epochs = []
            for session in sessions:
                logger.info(f"Processing subject {subject_id}, session {session}...")
                data = self.load_data(subject_id, session=session)
                raw = self._raw_from_session(data)
                raw = self._process_raw(raw)
                if downsample:
                    raw = self.downsample(raw)
                epochs = self._create_epochs(raw)
                cond2label.update(epochs.event_id)
                all_epochs.append(epochs)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message=".*Concatenation of Annotations within Epochs is not supported yet.*",
                )
                ed[subject_id] = mne.concatenate_epochs(all_epochs, verbose=False)
        logger.debug(f"Changed cond2label to {cond2label}")
        self.cond2label = cond2label  # MNE remaps events
        return ed

    def downsample(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Downsample raw data to target sampling frequency."""
        return raw.copy().resample(self.target_sfreq, npad="auto")

    def set_channel_wavelengths(self, raw):
        """Set wavelength information for fNIRS channels in raw.info['chs']."""
        for ch in raw.info["chs"]:
            if (
                ch["ch_name"] != "Stim"
                and ch["kind"] == mne.io.constants.FIFF.FIFFV_FNIRS_CH
                and np.isnan(ch["loc"][9])
            ):
                if "hbo" in ch["ch_name"].lower():
                    wavelength = self.default_wls[0]
                elif "hbr" in ch["ch_name"].lower():
                    wavelength = self.default_wls[1]
                else:
                    wavelength = float(ch["ch_name"].split()[-1])
                ch["loc"][9] = wavelength
                logger.debug(f"Set wavelength {wavelength} for {ch['ch_name']}")
        return raw

    def get_classification_tasks(self):
        """
        Return a list of available classification tasks for this dataset, loaded from classification_tasks.yaml via config utility.
        """
        return get_classification_tasks(
            self.dataset_name,
            cond2label=getattr(self, "cond2label", {}),
        )
