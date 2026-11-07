"""
Cockx2023 dataset parser for BIDS format data.
This dataset has special channel mapping requirements due to naming errors in the original data.
All conditions (fingerauto, fingerautodual, etc.) are treated as part of one unified dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Any
from .bids import BIDSParser

logger = logging.getLogger(__name__)


def levenshtein_dist(s1, s2):
    """Calculate Levenshtein distance between two sequences."""
    if type(s1) != str:
        s1 = "".join([str(i) for i in s1])
    if len(s1) < len(s2):
        return levenshtein_dist(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def find_closest_sequence(sequence, sequences):
    """Find the closest sequence from a list of sequences."""
    distances = [levenshtein_dist(sequence, s) for s in sequences]
    min_ix = np.argmin(distances)
    min_dist = distances[min_ix]
    if min_dist > len(sequence) / 2:
        raise ValueError(f"No sequence found for {sequence} (min distance: {min_dist})")
    return sequences[min_ix]


def get_tapping_sequences(md, stimulus_ix, field):
    """Extract tapping sequences from metadata."""
    sequences = []
    for i, ix in enumerate(stimulus_ix):
        if ix:
            sequence = "".join(
                md[field].iloc[i + 1 :].dropna()[:12].astype(int).astype(str)
            )
            sequences.append(sequence)
    return sequences


class Cockx2023Parser(BIDSParser):
    """Parser for Cockx2023 fNIRS dataset (BIDS format with special channel mapping)."""

    dataset_id = "cockx2023"
    EXPERIMENTS = [
        "fingerauto",
        "fingerautodual",
        "fingernonauto",
        "fingernonautodual",
        "footauto",
        "footautodual",
        "footnonauto",
        "footnonautodual",
    ]

    def __init__(self, dataset_path: Path = None, experiment = None):
        super().__init__(dataset_path)
        self.cond2label = {
            "fingerauto_a": 1,
            "fingerauto_b": 2,
            "fingerautodual_a": 3,
            "fingerautodual_b": 4,
            "fingernonauto_a": 5,
            "fingernonauto_b": 6,
            "fingernonautodual_a": 7,
            "fingernonautodual_b": 8,
            "footauto_a": 9,
            "footauto_b": 10,
            "footautodual_a": 11,
            "footautodual_b": 12,
            "footnonauto_a": 13,
            "footnonauto_b": 14,
            "footnonautodual_a": 15,
            "footnonautodual_b": 16,
        }

    def get_sessions(self, subject_id: str) -> list:
        return self.get_experiments(subject_id)

    def get_experiments(self, subject_id: str, session: Any = None) -> list:
        # Find all experiments for a subject by scanning the nirs directory
        subject_dir = self.root_path / subject_id / "nirs"
        if not subject_dir.exists():
            return []
        experiments = set()
        for file in subject_dir.glob("*.snirf"):
            parts = file.stem.split("_")
            for part in parts:
                if part.startswith("task-"):
                    experiment = part.split("-")[1]
                    if f"task-{experiment}_" in file.stem:
                        experiments.add(experiment)
        return sorted(list(experiments))

    def get_events_from_tsv(self, data: dict) -> pd.DataFrame:
        subject_id = data["subject_id"]
        experiment = data["experiment"]
        tapping_sequences = {
            "a": "212321324241",
            "b": "434141243212",
        }
        events_path = (
            self.root_path
            / subject_id
            / "nirs"
            / f"{subject_id}_task-{experiment}_events.tsv"
        )
        if not events_path.exists():
            logger.error(f"Events file not found: {events_path}")
            return np.array([]), {}
        md = pd.read_csv(events_path, sep="\t")
        stm_offset = 1
        stm_ix = pd.concat(
            [pd.Series([False] * stm_offset), (md["trial_number"].diff() != 0)]
        ).reset_index(drop=True)[: len(md)]
        md_clean = md[stm_ix]
        md_clean = md_clean.dropna(subset=["onset"])
        possible_sequences = list(tapping_sequences.values())
        seq2label = {v: k for k, v in tapping_sequences.items()}
        extracted_sequences = get_tapping_sequences(md, stm_ix, "correct_movement")
        trial_types = []
        remove_ix = []
        for i, (idx, row) in enumerate(md_clean.iterrows()):
            extracted_seq = extracted_sequences[i]
            try:
                seq_label = find_closest_sequence(extracted_seq, possible_sequences)
            except ValueError:
                logger.debug(
                    f"Could not recognize sequence {extracted_seq} in {subject_id} {experiment}. Skipping..."
                )
                remove_ix.append(idx)
                continue
            trial_type = f"{experiment}_{seq2label[seq_label]}"
            trial_types.append(trial_type)
        md_clean = md_clean.drop(remove_ix)
        md_clean["trial_type"] = trial_types
        return md_clean

    def apply_metadata(self, metadata: pd.DataFrame, data: dict = None) -> pd.DataFrame:
        """The authors exclude the first trial of each session as a 'test trial'. Some other trials are excluded as 'erroneous trials'."""
        exlude_reasons = metadata["reason_for_exclusion"].unique()
        # you can set which exclude reasons to exclude
        # exlude_reasons = ["test trial", "erroneous trial"]
        exclusions = metadata[
            ["trial_number", "trial_type", "included_trial", "reason_for_exclusion"]
        ]
        exclusions = exclusions[exclusions["included_trial"] == False]
        exclusions = exclusions[exclusions["reason_for_exclusion"].isin(exlude_reasons)]
        metadata = metadata[~metadata["trial_number"].isin(exclusions["trial_number"])]
        for _, row in exclusions.iterrows():
            logger.debug(
                f"Excluding trial {row['trial_number']} ({row['trial_type']}), reason: {row['reason_for_exclusion']}."
            )
        return metadata

    def load_data(self, subject_id: str, session: Any = None):
        return {"subject_id": subject_id, "session": None, "experiment": session}
