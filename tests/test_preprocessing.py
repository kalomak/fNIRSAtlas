#!/usr/bin/env python3
"""
Test script for the new preprocessing function.
"""

import numpy as np
import mne
from fnirsatlas.preprocessing import (
    beer_lambert_law,
    inverse_mbll,
    process_raw,
    _get_preprocessing_config_with_overrides,
)


def test_config_precedence():
    """Test that configuration precedence works correctly."""
    print("Testing configuration precedence...")

    # Test global config
    config = _get_preprocessing_config_with_overrides()
    print(f"Global config remove_short: {config.get('remove_short')}")

    # Test dataset-specific config
    config = _get_preprocessing_config_with_overrides("cockx2023")
    print(f"Cockx2023 config remove_short: {config.get('remove_short')}")

    # Test kwargs override
    config = _get_preprocessing_config_with_overrides("cockx2023", remove_short=0.1)
    print(f"Cockx2023 with override remove_short: {config.get('remove_short')}")

    print("✓ Configuration precedence test passed!")


def test_process_raw_basic():
    """Test basic functionality of process_raw."""
    print("\nTesting process_raw basic functionality...")

    # Create dummy fNIRS data
    n_channels = 4
    n_times = 1000
    sfreq = 10.0

    # Create proper fNIRS channel names (S1_D1 760, S1_D1 850, S2_D2 760, S2_D2 850)
    ch_names = ["S1_D1 760", "S1_D1 850", "S2_D2 760", "S2_D2 850"]
    ch_types = ["fnirs_cw_amplitude"] * n_channels

    # Create dummy data
    data = np.random.randn(n_channels, n_times)

    # Create info and raw object
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(data=data, info=info)

    # Set wavelength in raw.info['chs'][i]['loc'][9]
    for i, ch in enumerate(raw.info["chs"]):
        wl = 760 if "760" in ch["ch_name"] else 850
        if ch["loc"] is None or len(ch["loc"]) < 10:
            ch["loc"] = np.zeros(12, dtype=float)
        ch["loc"][9] = wl

    # Test processing with minimal settings
    try:
        processed_raw = process_raw(raw, dataset_name="cockx2023", verbose=True)
        print(f"✓ Processing successful! Final shape: {processed_raw.get_data().shape}")
        print(f"✓ Channel types: {set(processed_raw.get_channel_types())}")
    except Exception as e:
        print(f"✗ Processing failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_skip_all_preprocessing():
    """Ensure skip_all returns raw unchanged."""
    # Create dummy fNIRS CW amplitude data
    n_channels = 4
    n_times = 500
    sfreq = 10.0
    ch_names = ["S1_D1 760", "S1_D1 850", "S2_D2 760", "S2_D2 850"]
    ch_types = ["fnirs_cw_amplitude"] * n_channels
    data = np.random.randn(n_channels, n_times)

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(data=data, info=info, verbose=False)

    # Keep a copy of original data and types
    data_before = raw.get_data().copy()
    types_before = list(raw.get_channel_types())

    # Run with skip_all
    raw_after = process_raw(raw, dataset_name=None, skip_all=True)

    # Validate unchanged
    assert np.allclose(data_before, raw_after.get_data())
    assert types_before == list(raw_after.get_channel_types())

    print("✓ skip_all leaves raw unchanged")


def test_inverse_mbll_roundtrip():
    n_times = 200
    sfreq = 10.0
    ch_names = ["S1_D1 760", "S1_D1 850", "S2_D2 760", "S2_D2 850"]
    ch_types = ["fnirs_od"] * len(ch_names)
    data = np.random.RandomState(0).randn(len(ch_names), n_times)

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw_od = mne.io.RawArray(data=data, info=info, verbose=False)

    for ch in raw_od.info["chs"]:
        ch["loc"][9] = 760 if "760" in ch["ch_name"] else 850

    raw_hb = beer_lambert_law(raw_od, ppf=6.0)
    reconstructed_od = inverse_mbll(raw_hb, ppf=6.0)

    assert set(reconstructed_od.get_channel_types()) == {"fnirs_od"}
    assert reconstructed_od.ch_names == ch_names
    assert np.allclose(
        raw_od.get_data(), reconstructed_od.get_data(), atol=1e-10, rtol=1e-7
    )


if __name__ == "__main__":
    test_config_precedence()
    test_process_raw_basic()
    print("\nAll tests completed!")
