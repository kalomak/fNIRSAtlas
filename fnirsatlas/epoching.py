"""
Epoching utilities for fNIRS data.
"""

from collections import defaultdict
import logging
import pandas as pd
import numpy as np
import mne
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
import warnings
from fnirsatlas.preprocessing import beer_lambert_law
from fnirsatlas.config import get_dataset_config, get_preprocessing_config

logger = logging.getLogger(__name__)


def create_epochs_from_raw(
    raw,
    events=None,
    event_name_mapping=None,
    event_metadata=None,
    tmin=None,
    tmax=None,
    reject_criteria=None,
    verbose=None,
    check_overlaps=None,
    dataset_name=None,
    **kwargs,
):
    """
    Creates preprocessed MNE epoch objects for a subject.

    1. Marks bad epochs based on maximum peak-to-peak signal amplitude (PTP)
    2. Saves `is_bad_epoch` and `bad_epoch_reason` to epochs.metadata.
    3. Removes epochs with missing data.
    4. Optionally checks for overlapping epochs and raises an error if found.

    Parameters
    ----------
    raw : mne.io.Raw
            Processed MNE raw object.
    events : array
            Events array.
    event_name_mapping : dict
            Mapping from event names to event codes.
    event_metadata : pd.DataFrame
            Event metadata.
    tmin : float, optional
            Time before event to include in epoch. If None, will use config value.
    tmax : float, optional
            Time after event to include in epoch. If None, will use config value.
    reject_criteria : dict, optional
            Criteria for rejecting epochs. Keys are channel types and values are the maximum PTP.
            If None, will use config value.
    verbose : bool, optional
            Whether to print progress. If None, will use config value.
    check_overlaps : bool, optional
            Whether to check for overlapping epochs and raise an error if found. If None, will use config value.
    dataset_name : str, optional
            Name of the dataset for dataset-specific config overrides.

    Returns
    -------
    epochs : mne.Epochs
            The epochs object.
    """
    # Load preprocessing config
    preproc_config = get_preprocessing_config()

    # Check if tmin/tmax should be overridden from dataset config
    if dataset_name:
        try:
            config = get_dataset_config(dataset_name)
            if tmin is None and "tmin" in config:
                tmin = config["tmin"]
            if tmax is None and "tmax" in config:
                tmax = config["tmax"]
            if check_overlaps is None and "allow_overlaps" in config:
                check_overlaps = not config["allow_overlaps"]
        except ValueError:
            logger.debug(
                f"Dataset {dataset_name} interval settings not found in config, using defaults"
            )
            pass

    tmin = tmin if tmin is not None else preproc_config.get("tmin", -2)
    tmax = tmax if tmax is not None else preproc_config.get("tmax", 15)
    reject_criteria = (
        reject_criteria
        if reject_criteria is not None
        else preproc_config.get("reject_criteria")
    )
    verbose = verbose if verbose is not None else preproc_config.get("verbose", False)
    check_overlaps = (
        check_overlaps
        if check_overlaps is not None
        else not preproc_config.get("allow_overlaps", False)
    )

    # Robust subject extraction
    subject = None
    if raw.info.get("subject_info") and raw.info["subject_info"] is not None:
        subject = raw.info["subject_info"].get("his_id")
    if not subject:
        subject = "sub-synthetic"
    subject = subject if subject.startswith("sub-") else f"sub-{subject}"

    if events is None or event_name_mapping is None:
        if "temp" in raw.info and "event_id" in raw.info["temp"]:
            event_name_mapping = raw.info["temp"]["event_id"]
        events, event_name_mapping = mne.events_from_annotations(
            raw, event_id=event_name_mapping, verbose=verbose
        )

    # drop duplicates
    _, ix = np.unique(events, axis=0, return_index=True)
    events = events[np.sort(ix)]

    if event_metadata is None:
        event_metadata = pd.DataFrame(events[:, ::2], columns=["onset", "value"])

    event_metadata["subject"] = subject

    event_repeated = kwargs.pop("event_repeated", "merge")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_name_mapping,
        metadata=event_metadata,
        reject=reject_criteria,
        tmin=tmin,
        tmax=tmax,
        verbose=False,
        preload=True,
        event_repeated=event_repeated,
        **kwargs,
    )

    # Check for overlaps if requested
    if check_overlaps:
        overlap_result = check_epoch_overlap(epochs)
        
        # Some datasets use multiple markers for same-event epochs, so we drop them by default.
        # This can be a bit dangerous so we warn the user. Should probably remove the warning 
        # for known datasets.
        drop_indices = overlap_result.get("same_event_overlap_indices", [])
        if drop_indices:
            logger.warning(f"Dropping {len(drop_indices)} overlapping same-event epochs")
            epochs = epochs.drop(drop_indices, verbose=False)
        overlap_result = check_epoch_overlap(epochs)
        if overlap_result["has_overlaps"]:
            raise ValueError(overlap_result["error_msg"])

    return epochs


def check_epoch_overlap(epochs):
    """
    Check if epochs are overlapping by examining event timing and epoch duration.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object to check for overlaps.

    Returns
    -------
    dict
        Dictionary containing overlap information including:
        - has_overlaps: bool indicating if any overlaps were found
        - overlap_pairs: list of tuples with overlapping epoch indices
        - overlap_info: detailed information about each overlap
        - same_event_overlap_indices: sorted list of indices to drop for same-event overlaps
        - error_msg: error message string if overlaps remain, else None
    """
    events = epochs.events
    tmin, tmax = epochs.tmin, epochs.tmax
    duration = tmax - tmin
    sfreq = epochs.info["sfreq"]

    # Get event labels
    code_to_label = {v: k for k, v in getattr(epochs, "event_id", {}).items()}
    event_labels = [code_to_label.get(e[2], f"Unknown_{e[2]}") for e in events]

    overlap_pairs, overlap_info, intervals = [], [], []
    same_event_overlap_indices = []
    for i in range(len(events) - 1):
        onset1, onset2 = events[i, 0] / sfreq, events[i + 1, 0] / sfreq
        intervals.append(onset2 - onset1)
        start1, end1 = onset1 + tmin, onset1 + tmax
        start2 = onset2 + tmin
        if end1 > start2:
            overlap_duration = end1 - start2
            if overlap_duration > 0.001:
                overlap_pairs.append((i, i + 1))
                overlap_info.append(
                    {
                        "epoch_indices": (i, i + 1),
                        "overlap_duration": overlap_duration,
                        "current_epoch": {
                            "onset": onset1,
                            "start": start1,
                            "end": end1,
                            "label": event_labels[i],
                        },
                        "next_epoch": {
                            "onset": onset2,
                            "start": start2,
                            "end": onset2 + tmax,
                            "label": event_labels[i + 1],
                        },
                    }
                )
                # Collect same-event overlap indices (drop the latter one)
                if event_labels[i] == event_labels[i + 1]:
                    same_event_overlap_indices.append(i + 1)

    has_overlaps = bool(overlap_pairs)
    min_interval = min(intervals) if intervals else float("inf")

    # Logging and error message construction
    error_msg = None
    if has_overlaps:
        # Construct error message for raising
        error_msg = f"Overlapping epochs detected!\n"
        error_msg += f"  - Epoch duration: {duration:.2f}s (tmin={tmin:.1f}s, tmax={tmax:.1f}s)\n"
        error_msg += f"  - Found {len(overlap_pairs)} overlapping epoch pairs\n"
        for i, overlap in enumerate(overlap_info[:3]):
            error_msg += f"  - Overlap {i+1}: Epochs {overlap['epoch_indices']} overlap by {overlap['overlap_duration']:.2f}s\n"
        if len(overlap_info) > 3:
            error_msg += f"  - ... and {len(overlap_info) - 3} more overlaps\n"
        error_msg += f"\nConsider reducing tmax or adjusting event timing."

    return {
        "has_overlaps": has_overlaps,
        "overlap_pairs": overlap_pairs,
        "overlap_info": overlap_info,
        "min_interval": min_interval,
        "same_event_overlap_indices": sorted(set(same_event_overlap_indices)),
        "error_msg": error_msg,
    }
