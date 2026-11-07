"""
Yi2025 dataset parser for NIRx format data.

subject 10, block 3 has SCI values (correlations between chs in same location but different wl in 0.7-1.5 Hz range) very close to -1. Could one of the data types be flipped?

Trial exclusions follow:
https://github.com/Bill-Chen-BJ/FJMI_EEG_fNIRS/blob/master/fNIRS_decoding.py
"""

import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import mne
from .base import BaseFNIRSParser
import re
import warnings

logger = logging.getLogger(__name__)


class Yi2025Parser(BaseFNIRSParser):
    dataset_id = "yi2025"
    """Parser for Yi2025 FineMI fNIRS dataset."""

    def __init__(self, dataset_path: Path = None):
        super().__init__(dataset_path)

        self.cond2label = {
            "hand_open": 1,  # hand open/close
            "wrist_flexion": 2,  # wrist flexion/extension
            "wrist_abduction": 3,  # wrist abduction/adduction
            "elbow_pronation": 4,  # elbow pronation/supination
            "elbow_flexion": 5,  # elbow flexion/extension
            "shoulder_pronation": 6,  # shoulder pronation/supination
            "shoulder_abduction": 7,  # shoulder abduction/adduction
            "shoulder_flexion": 8,  # shoulder flexion/extension
        }

    def get_sessions(self, subject_id: str) -> list[str]:
        sessions = [f"block{i}" for i in range(1, 9)]
        if subject_id == "1":
            sessions = ["block1-4"] + sessions[4:]
        # Remove subject 10 block 3 due to all SCI values being close to -1
        if subject_id == "10":
            sessions = sessions[:2] + sessions[3:]
        return sessions

    def load_data(self, subject_id: str, session: Any = None) -> Dict:
        """Load data for a specific subject using exact original logic."""
        data_dir = self.dataset_path / f"subject{subject_id}" / "fNIRS" / session
        return {
            "subject_id": subject_id,
            "session": session,
            "data_dir": data_dir,
        }

    def _raw_from_session(self, data: Dict) -> mne.io.RawArray:
        """Load raw object from session."""
        data_dir = data["data_dir"]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Extraction of measurement date from NIRX file failed.*",
            )
            warnings.filterwarnings(
                "ignore",
                message="A single dat file was expected in the specified path, but got 0.*",
            )
            raw = mne.io.read_raw_nirx(str(data_dir), preload=True, verbose=False)

        # Eliminations from the original implementation
        subject_id = data["subject_id"]
        session = data["session"]
        if subject_id == "1" and session == "block1-4":
            idx_to_remove = np.arange(-40, 0)
            raw.annotations.delete(idx_to_remove)
            raw.crop_by_annotations()
            logger.info(f"Deleted last 40 annotations from subject 1 block1-4")

        elif subject_id == "5" and session == "block6":
            if len(raw.annotations) > 0:
                raw.annotations.delete(0)
                raw.crop_by_annotations()
                logger.info(f"Deleted first annotation from subject 5 block6")

        # rename annotations
        rename_map = {f"{v}.0": k for k, v in self.cond2label.items()}
        raw.annotations.rename(rename_map)

        return raw

    def get_sampling_frequency(self, data: Dict) -> float:
        return 7.8125

    def get_subject_ids(self) -> List[str]:
        """Get list of available subject IDs."""
        subject_ids = []

        if not self.dataset_path.exists():
            return subject_ids

        for item in self.dataset_path.iterdir():
            if item.is_dir() and item.name.startswith("subject"):
                # Extract subject number
                subject_num = item.name.replace("subject", "")
                if subject_num.isdigit():
                    subject_ids.append(subject_num)

        return sorted(subject_ids, key=int)
