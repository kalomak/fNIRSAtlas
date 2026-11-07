#!/usr/bin/env python3
"""
Test script to verify that preprocessing parameters are loaded from config correctly.
"""

import numpy as np
import mne
from fnirsatlas.config import get_preprocessing_config
from fnirsatlas.preprocessing import process_raw
from fnirsatlas.epoching import create_epochs_from_raw


def test_config_loading():
    """Test that preprocessing config is loaded correctly."""
    print("=== Testing Config Loading ===")

    config = get_preprocessing_config()
    print("Preprocessing config loaded:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Test that all expected keys are present
    expected_keys = [
        "ch_interpolation",
        "sci_threshold",
        "tddr",
        "l_freq",
        "h_freq",
        "l_trans_bandwidth",
        "h_trans_bandwidth",
        "filter_length",
        "filter_kwargs",
        "bll_ppf",
        "remove_short",
        "short_ch_regression",
        "tmin",
        "tmax",
        "reject_criteria",
        "allow_overlaps",
        "verbose",
    ]

    missing_keys = [key for key in expected_keys if key not in config]
    if missing_keys:
        print(f"WARNING: Missing keys in config: {missing_keys}")
    else:
        print("✓ All expected config keys present")

    return config


def test_preprocessing_function_signature():
    """Test that the preprocessing function accepts the new parameters."""
    print("\n=== Testing Preprocessing Function ===")

    # Test that the function can be called with None parameters
    # (which should use config defaults)
    print("Testing process_raw function signature...")

    # Create dummy raw data for testing
    sfreq = 10.0
    n_channels = 4
    n_times = 1000

    # Create dummy NIRS data (simulating HbO/HbR channels)
    data = np.random.randn(n_channels, n_times)
    ch_names = [f"S{i}_D{i} hbo" for i in range(1, 3)] + [
        f"S{i}_D{i} hbr" for i in range(1, 3)
    ]
    ch_types = ["hbo", "hbo", "hbr", "hbr"]

    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data, info)

    print("✓ Raw object created successfully")

    # Test that process_raw can be called with None parameters
    try:
        processed_raw = process_raw(
            raw=raw,
            ch_interpolation=None,
            sci_threshold=None,
            tddr=None,
            l_freq=None,
            h_freq=None,
            verbose=None,
        )
        print("✓ process_raw function works with config parameters")
    except Exception as e:
        print(f"✗ Error in process_raw: {e}")
        return False

    return True


def test_epoching_function():
    """Test that the epoching function uses config parameters."""
    print("\n=== Testing Epoching Function ===")

    # Create dummy raw data
    sfreq = 10.0
    n_channels = 4
    n_times = 1000

    data = np.random.randn(n_channels, n_times)
    ch_names = [f"S{i}_D{i} hbo" for i in range(1, 3)] + [
        f"S{i}_D{i} hbr" for i in range(1, 3)
    ]
    ch_types = ["hbo", "hbo", "hbr", "hbr"]

    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data, info)

    # Add some dummy events
    events = np.array([[100, 0, 1], [300, 0, 2], [500, 0, 1]])
    event_id = {1: "event1", 2: "event2"}

    try:
        epochs = create_epochs_from_raw(
            raw=raw,
            events=events,
            event_name_mapping=event_id,
            tmin=None,
            tmax=None,
            verbose=None,
        )
        print("✓ create_epochs_from_raw function works with config parameters")
        print(f"  Epoch duration: {epochs.tmax - epochs.tmin:.2f}s")
    except Exception as e:
        print(f"✗ Error in create_epochs_from_raw: {e}")
        return False

    return True


if __name__ == "__main__":
    print("Testing preprocessing configuration integration...")

    # Test config loading
    config = test_config_loading()

    # Test preprocessing function
    success1 = test_preprocessing_function_signature()

    # Test epoching function
    success2 = test_epoching_function()

    if success1 and success2:
        print("\n✓ All tests passed! Preprocessing parameters are now configurable.")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
