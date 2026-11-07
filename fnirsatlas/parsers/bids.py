"""
BIDS dataset parser for fNIRS data.

Hierarchy:
Subject → [Session] → DataType → Task → [Run]
"""

import warnings
import re
import pandas as pd
from pathlib import Path
import logging
import mne
import mne.io
from mne_bids import BIDSPath, read_raw_bids, find_matching_paths, get_entities_from_fname
from typing import List, Dict, Tuple, Any
import numpy as np
from .base import BaseFNIRSParser
from fnirsatlas.channels import (
    reorder_chs,
    validate_chs,
    chs_to_mne_format,
    to_mne_ch_order,
    set_channel_types_and_coils,
)

logger = logging.getLogger(__name__)

class BIDSParser(BaseFNIRSParser):
    """Generic BIDS parser for fNIRS datasets."""

    dataset_id = "bids"

    def __init__(self, dataset_path: Path | None = None):
        super().__init__(dataset_path)
        self.root_path = self.dataset_path

    def _find_matching_paths(self, **kwargs) -> List[BIDSPath]:
        if "subject" in kwargs:
            kwargs["subjects"] = kwargs.pop("subject")
        elif "session" in kwargs:
            kwargs["sessions"] = kwargs.pop("session")
        elif "task" in kwargs:
            kwargs["tasks"] = kwargs.pop("task")
        elif "run" in kwargs:
            kwargs["runs"] = kwargs.pop("run")
        if "subjects" in kwargs and "sub-" in kwargs["subjects"]:
            kwargs["subjects"] = self._to_bids_sub_id(kwargs["subjects"])
        return find_matching_paths(root=self.root_path, datatypes="nirs", **kwargs)

    def _to_bids_sub_id(self, sub_id: str) -> str:
        return sub_id.replace("sub-", "")

    def _from_bids_sub_id(self, sub_id: str) -> str:
        return f"sub-{sub_id}"

    def _get_bids_sub_ids(self) -> List[str]:
        return list(sorted(set([path.subject for path in self._find_matching_paths()])))

    def get_subject_ids(self) -> List[str]:
        """Get list of available subject IDs from BIDS structure (full BIDS name, e.g., 'sub-01')."""
        sub_ids = self._get_bids_sub_ids()
        return [self._from_bids_sub_id(sub_id) for sub_id in sub_ids]

    def get_sessions(self, subject: str) -> list[str]:
        """We treat (ses, task, run) tuples, i.e. a single .snirf file, as separate sessions."""
        bids_sub_id = self._to_bids_sub_id(subject)
        path_filters = {}
        if hasattr(self, "experiment") and self.experiment is not None:
            path_filters["tasks"] = self.experiment
        bids_paths = self._find_matching_paths(subjects=bids_sub_id, extensions=".snirf", **path_filters)
        sessions = set([path.basename for path in bids_paths])
        if None in sessions:
            sessions.remove(None)
        if len(sessions) == 0:
            return [""]
        return list(sorted(sessions))

    def get_experiments(self, subject: str, session: str = None) -> List[str]:
        """Get available experiments for a subject and session (if any)."""
        bids_sub_id = self._to_bids_sub_id(subject)
        bids_paths = self._find_matching_paths(subjects=bids_sub_id, sessions=session)
        experiments = set([path.task for path in bids_paths])
        if None in experiments:
            experiments.remove(None)
        return list(sorted(experiments))

    def get_runs(self, subject: str, session: str, experiment: str) -> List[str]:
        bids_sub_id = self._to_bids_sub_id(subject)
        bids_paths = self._find_matching_paths(subjects=bids_sub_id, sessions=session, tasks=experiment)
        runs = set([path.run for path in bids_paths])
        if None in runs:
            runs.remove(None)
        return list(sorted(runs))

    def load_data(
        self, subject_id: str, session: str = None, experiment: str = None
    ) -> Dict:
        run = None
        if session.endswith("_nirs.snirf"):
            entities = get_entities_from_fname(session)
            session = entities.get("session", None)
            experiment = entities.get("task", None)
            run = entities.get("run", None)
        if experiment is None:
            experiments = self.get_experiments(subject_id, session)
            if len(experiments) != 1:
                logger.warning(f"{len(experiments)} experiments found for subject {subject_id} and session {session}: {experiments}. Using the first one.")
            experiment = experiments[0]
        return {
            "subject_id": subject_id,
            "session": session,
            "experiment": experiment,
            "run": run,
        }

    def read_channels_tsv(self, data: Dict) -> pd.DataFrame:
        subject_id_bids = data["subject_id"]
        session = data.get("session", None)
        experiment = data["experiment"]
        run = data.get("run", None)
        path = self._find_matching_paths(subjects=subject_id_bids, sessions=session, tasks=experiment, runs=run, suffixes="channels", extensions=".tsv")
        if len(path) != 1:
            raise ValueError(f"Found {len(path)} channels files for session {data}: {path}")
        return pd.read_csv(path[0].fpath, sep="\t")

    def get_ch2type(self, channels_df: pd.DataFrame) -> dict:
        ch_type_map = {}
        for _, row in channels_df.iterrows():
            ch_name = row["name"]
            ch_type = row["type"].lower()
            match ch_type:
                case "nirscwopticaldensity":
                    ch_type_map[ch_name] = "fnirs_od"
                case "nirscwamplitude":
                    ch_type_map[ch_name] = "fnirs_cw_amplitude"
                case "nirshbo":
                    ch_type_map[ch_name] = "hbo"
                case "nirshbr":
                    ch_type_map[ch_name] = "hbr"
                case "eeg":
                    ch_type_map[ch_name] = "eeg"
                case "stim":
                    ch_type_map[ch_name] = "stim"
                case _:
                    ch_type_map[ch_name] = "misc"
        return ch_type_map

    def read_and_map_channels(self, data: Dict, raw: mne.io.Raw) -> mne.io.Raw:
        """Set channel types according to BIDS channels.tsv file.
        Also renames channels to MNE format if necessary.
        """
        channels_df = self.read_channels_tsv(data)
        ch_type_map = self.get_ch2type(channels_df)
        raw = set_channel_types_and_coils(raw, ch_type_map)
        raw = chs_to_mne_format(raw)
        raw = BIDSParser._maybe_override_bids_chs(raw)
        return raw

    def apply_metadata(self, metadata: pd.DataFrame, data: dict = None) -> pd.DataFrame:
        """Hook method for applying a transformation to the metadata after loading."""
        return metadata

    def get_events_from_tsv(
        self,
        data: Dict,
    ) -> pd.DataFrame:
        subject_id = data["subject_id"]
        session = data["session"]
        experiment = data["experiment"]
        run = data.get("run", None)
        events_path = self._find_matching_paths(subjects=subject_id, sessions=session, tasks=experiment, runs=run, suffixes="events", extensions=".tsv")
        if len(events_path) != 1:
            raise ValueError(f"Found {len(events_path)} events files for session {data}: {events_path}")
        metadata = pd.read_csv(events_path[0].fpath, sep="\t")
        return metadata

    def get_cond2label_from_metadata(self, metadata: pd.DataFrame) -> dict:
        if "trial_type" not in metadata.columns or "value" not in metadata.columns:
            raise ValueError(
                "Failed to create cond2label mapping, missing `trial_type` or `value` columns in `events.tsv`. You can specify `self.cond2label` for a custom mapping."
            )
        md = metadata[["trial_type", "value"]]
        md = md.dropna()
        md = md.drop_duplicates()
        md["trial_type"] = md["trial_type"].astype(str)
        md["value"] = md["value"].astype(int)
        return dict(zip(md["trial_type"], md["value"]))

    def extract_events(self, data: dict) -> tuple:
        subject_id = data["subject_id"]
        metadata = self.get_events_from_tsv(data)
        na_ix = metadata["onset"].isna()
        if na_ix.any():
            metadata = metadata[~na_ix]
            logger.info(f"Removed {na_ix.sum()} rows with missing onset times.")
        metadata = self.apply_metadata(metadata, data)
        if self.cond2label is None:
            self.cond2label = {}
        events_in_md = metadata["trial_type"].unique()
        events_missing = set(events_in_md) - set(self.cond2label.keys())
        if events_missing:
            self.cond2label.update(self.get_cond2label_from_metadata(metadata))
        sfreq = self.get_sampling_frequency(data)
        events = []
        event_desc = {}
        if "trial_type" in metadata.columns:
            metadata["trial_type"] = metadata["trial_type"].fillna("")
        for i, (idx, row) in enumerate(metadata.iterrows()):
            onset = row["onset"]
            event_type = row["trial_type"]
            event_type_str = str(event_type).strip()
            event_id = self.cond2label[event_type_str]
            sample_ix = int(onset * sfreq)
            events.append([sample_ix, 0, event_id])
            event_desc[event_id] = str(event_type_str)
        return np.array(events), event_desc

    def get_sampling_frequency(self, data: Dict) -> float:
        return data["sfreq"]

    @staticmethod
    def _overwrite_bids_with_snirf(
        bids_raw: mne.io.Raw, snirf_raw: mne.io.Raw
    ) -> mne.io.Raw:
        """
        Replaces BIDS channels with SNIRF channels while preserving BIDS metadata and events.
        
        Takes channel info and signal data from SNIRF, but keeps metadata and events from BIDS.
        """
        bids_raw.load_data(verbose=False)
        snirf_raw.load_data(verbose=False)
        if bids_raw._data.shape != snirf_raw._data.shape:
            raise ValueError(
                f"Data shapes must match: BIDS {bids_raw._data.shape} != SNIRF {snirf_raw._data.shape}"
            )
        bids_raw._data = snirf_raw._data.copy()
        with bids_raw.info._unlock():
            bids_raw.info["chs"] = snirf_raw.info["chs"].copy()
            bids_raw.info["ch_names"] = snirf_raw.info["ch_names"].copy()
            bids_raw.info["nchan"] = snirf_raw.info["nchan"]
        return bids_raw

    @staticmethod
    def _maybe_override_bids_chs(raw: mne.io.Raw) -> mne.io.Raw:
        """
        BIDS channels are generally unreliable since mne_bids.read:_handle_channels_reading:L720 simply renames channels based on the BIDS tsv file, often resulting in mismatched channel names and locations.

        Currently falls back to .snirf channels if the BIDS channels are not valid. Snirf channels are more reliable since they are programmatically constructed from sources and detectors in the .snirf file, not a human-edited tsv file.

        Could add ch correction method from `fnirsatlas.channels.try_to_correct_ch_order` but it's not tested and not needed currently.
        """
        
        if validate_chs(raw.info["chs"]):
            logger.debug("BIDS channels are valid.")
            return raw  # no need to correct

        mne_ch_order = to_mne_ch_order(raw.info["chs"])
        if validate_chs(reorder_chs(raw.info["chs"], mne_ch_order)):
            logger.debug("BIDS channels are valid after reordering.")
            raw.reorder_channels(mne_ch_order)
            return raw

        raw_snirf = mne.io.read_raw_snirf(raw.filenames[0], verbose=False)
        snirf_mne_ch_order = to_mne_ch_order(raw_snirf.info["chs"])
        raw_snirf.reorder_channels(snirf_mne_ch_order)
        if validate_chs(reorder_chs(raw_snirf.info["chs"], snirf_mne_ch_order)):
            logger.debug(".snirf channels are valid. Using .snirf channels.")
            raw = BIDSParser._overwrite_bids_with_snirf(raw, raw_snirf)
            return raw
        else:
            logger.debug(".snirf channels are not valid.")

        logger.warning(
            "Could not fix BIDS channels. Fix either the BIDS channels.tsv or the .snirf file. Falling back to .snirf channels."
        )
        raw = BIDSParser._overwrite_bids_with_snirf(raw, raw_snirf)
        return raw

    def _raw_from_session(self, data: dict):
        subject_id = data["subject_id"]
        session = data["session"]
        experiment = data["experiment"]
        run = data.get("run", None)
        bidspath = BIDSPath(
            subject=subject_id.replace("sub-", ""),
            session=session if session else None,
            task=experiment,
            root=self.root_path,
            datatype="nirs",
            run=run,
        )
        with mne.utils.use_log_level("ERROR"):
            # We handle channel mismatch manually later
            raw = read_raw_bids(bidspath, on_ch_mismatch="rename").load_data()

        data["sfreq"] = raw.info["sfreq"]
        # Extract events from BIDS file
        events, event_desc = self.extract_events(data)
        raw = self.set_annotations(raw, events, event_desc)
        raw = self.read_and_map_channels(data, raw)
        return raw
