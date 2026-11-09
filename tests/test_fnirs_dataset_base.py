"""Tests for shared fNIRS dataset base helpers."""

import json

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.base import BaseFNIRSDataset


class RecordingOnlyFnirs(BaseFNIRSDataset):
    """Minimal dataset without a fixed-epoch paradigm."""

    def __init__(self):
        super().__init__(
            subjects=[1],
            sessions_per_subject=1,
            events={"block": 1, "event": 2},
            code="RecordingOnlyFnirs",
            interval=None,
            paradigm=None,
            return_all_modalities=True,
        )

    def _get_single_subject_data(self, subject):
        raw = self._make_raw(
            data=np.ones((1, 20)),
            ch_names=["S1_D1 hbo"],
            ch_types=["hbo"],
            sfreq=2.0,
        )
        raw.set_annotations(
            mne.Annotations(
                onset=[0.0, 1.0],
                duration=[5.0, 0.5],
                description=["block", "event"],
            )
        )
        return self._single_run_session(self._standardize_raw(raw))

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return []


def test_recording_only_dataset_preserves_variable_annotations():
    raw = RecordingOnlyFnirs().get_data()[1]["0fnirs"]["0"]

    assert raw.annotations.description.tolist() == ["block", "event"]
    assert raw.annotations.duration.tolist() == [5.0, 0.5]


def test_standardize_annotation_free_recording_adds_empty_stim():
    raw = BaseFNIRSDataset._make_raw(
        data=np.ones((1, 20)),
        ch_names=["S1_D1 hbo"],
        ch_types=["hbo"],
        sfreq=2.0,
    )

    raw = RecordingOnlyFnirs()._standardize_raw(raw)

    assert raw.ch_names[-1] == "Stim"
    assert not raw.get_data(picks="Stim").any()


def test_make_stim_raw_sets_requested_samples():
    raw = BaseFNIRSDataset._make_stim_raw(
        n_times=8,
        sfreq=10.0,
        samples=np.array([1, 4, 6]),
        labels=np.array([2, 1, 2]),
    )

    assert raw.ch_names == ["Stim"]
    assert raw.info["sfreq"] == 10.0
    assert raw.get_data().shape == (1, 8)
    assert np.array_equal(raw.get_data().ravel(), [0, 2, 0, 0, 1, 0, 2, 0])


def test_make_stim_raw_rejects_mismatched_samples_and_labels():
    with pytest.raises(ValueError, match="same length"):
        BaseFNIRSDataset._make_stim_raw(
            n_times=8,
            sfreq=10.0,
            samples=np.array([1, 4]),
            labels=np.array([2]),
        )


def test_load_montage_from_json(tmp_path):
    montage_path = tmp_path / "montage.json"
    montage_dict = {
        "ch_pos": {"S1": [0.0, 0.1, 0.2], "D1": [0.1, 0.2, 0.3]},
        "coord_frame": "head",
        "nasion": [0.0, 0.1, 0.0],
        "lpa": [-0.1, 0.0, 0.0],
        "rpa": [0.1, 0.0, 0.0],
    }
    channel_map = {"ch1": "S1_D1"}
    montage_path.write_text(
        json.dumps({"montage": montage_dict, "channel_map": channel_map})
    )

    montage, loaded_map = BaseFNIRSDataset._load_montage_from_json(montage_path)

    assert loaded_map == channel_map

    positions = montage.get_positions()
    assert set(positions["ch_pos"]) == {"S1", "D1"}
    assert np.allclose(positions["ch_pos"]["S1"], [0.0, 0.1, 0.2])


def test_make_head_montage_uses_supplied_positions_and_fiducials():
    sensor_pos = {"S1": np.array([0.0, 0.1, 0.2]), "D1": np.array([0.1, 0.2, 0.3])}
    fiducials = {
        "nasion": np.array([0.0, 0.1, 0.0]),
        "lpa": np.array([-0.1, 0.0, 0.0]),
        "rpa": np.array([0.1, 0.0, 0.0]),
    }

    montage = BaseFNIRSDataset._make_head_montage(sensor_pos, fiducials)

    positions = montage.get_positions()
    assert set(positions["ch_pos"]) == {"S1", "D1"}
    assert np.allclose(positions["nasion"], fiducials["nasion"])


def test_make_raw_applies_optional_montage():
    montage = mne.channels.make_dig_montage(
        ch_pos={"Cz": [0.0, 0.1, 0.2]},
        coord_frame="head",
        nasion=[0.0, 0.1, 0.0],
        lpa=[-0.1, 0.0, 0.0],
        rpa=[0.1, 0.0, 0.0],
    )

    raw = BaseFNIRSDataset._make_raw(
        data=np.array([[1.0, 2.0, 3.0]]),
        ch_names=["Cz"],
        ch_types=["eeg"],
        sfreq=10.0,
        montage=montage,
    )

    assert raw.info["sfreq"] == 10.0
    assert raw.ch_names == ["Cz"]
    assert np.allclose(raw.get_data(), [[1.0, 2.0, 3.0]])
    assert raw.get_montage() is not None


def test_single_run_session_wraps_raw():
    raw = BaseFNIRSDataset._make_raw(
        data=np.array([[1.0, 2.0]]),
        ch_names=["Stim"],
        ch_types=["stim"],
        sfreq=10.0,
    )

    sessions = BaseFNIRSDataset._single_run_session(
        raw, session_key="2fnirs", run_key="3"
    )

    assert list(sessions) == ["2fnirs"]
    assert list(sessions["2fnirs"]) == ["3"]
    assert sessions["2fnirs"]["3"] is raw


def test_add_run_to_sessions_updates_nested_mapping():
    raw_a = BaseFNIRSDataset._make_raw(
        data=np.array([[1.0, 2.0]]),
        ch_names=["Stim"],
        ch_types=["stim"],
        sfreq=10.0,
    )
    raw_b = BaseFNIRSDataset._make_raw(
        data=np.array([[3.0, 4.0]]),
        ch_names=["Stim"],
        ch_types=["stim"],
        sfreq=10.0,
    )

    sessions = {}
    BaseFNIRSDataset._add_run(sessions, "1fnirs", raw_a)
    BaseFNIRSDataset._add_run(sessions, "1fnirs", raw_b, run_key="1")

    assert list(sessions["1fnirs"]) == ["0", "1"]
    assert sessions["1fnirs"]["0"] is raw_a
    assert sessions["1fnirs"]["1"] is raw_b


def test_populate_fnirs_metadata_sets_cw_wavelength_and_coil_type():
    raw = BaseFNIRSDataset._make_raw(
        data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        ch_names=["S1_D1 760", "S1_D1 850"],
        ch_types=["fnirs_cw_amplitude", "fnirs_cw_amplitude"],
        sfreq=10.0,
    )

    BaseFNIRSDataset._populate_fnirs_metadata(raw)

    assert raw.info["chs"][0]["loc"][9] == 760
    assert raw.info["chs"][1]["loc"][9] == 850
    assert raw.info["chs"][0]["kind"] == mne.io.constants.FIFF.FIFFV_FNIRS_CH
    assert (
        raw.info["chs"][0]["coil_type"]
        == mne.io.constants.FIFF.FIFFV_COIL_FNIRS_CW_AMPLITUDE
    )


def test_populate_fnirs_metadata_can_fill_hb_wavelengths_from_defaults():
    raw = BaseFNIRSDataset._make_raw(
        data=np.array([[1.0, 2.0], [3.0, 4.0]]),
        ch_names=["S1_D1 hbo", "S1_D1 hbr"],
        ch_types=["hbo", "hbr"],
        sfreq=10.0,
    )

    BaseFNIRSDataset._populate_fnirs_metadata(raw, default_wavelengths=(760, 850))

    assert raw.info["chs"][0]["loc"][9] == 760
    assert raw.info["chs"][1]["loc"][9] == 850
