"""Tests for dataset loading utilities."""

import io
import logging
import tarfile
import zipfile

import mne
import numpy as np
import pytest
from filelock import Timeout

from fnirsatlas.datasets.utils import (
    combine_devices,
    download_lock,
    extract_archive,
    merge_repeated_annotations,
    standardize_raw,
    warn_if_optodes_are_planar,
)


def _raw_with_optode_heights(source_z, detector_z):
    raw = mne.io.RawArray(
        np.ones((1, 10)), mne.create_info(["S1_D1 hbo"], 1.0, "hbo"), verbose=False
    )
    loc = raw.info["chs"][0]["loc"]
    loc[3:6] = (0.0, 0.0, source_z)
    loc[6:9] = (0.03, 0.0, detector_z)
    return raw


def test_warn_if_optodes_are_planar_warns_for_flat_positions(caplog):
    with caplog.at_level(logging.WARNING):
        warn_if_optodes_are_planar(_raw_with_optode_heights(0.0, 0.0))

    assert "two-dimensional" in caplog.text


def test_warn_if_optodes_are_planar_accepts_3d_positions(caplog):
    with caplog.at_level(logging.WARNING):
        warn_if_optodes_are_planar(_raw_with_optode_heights(0.08, 0.079))

    assert caplog.text == ""


def test_warn_if_optodes_are_planar_ignores_missing_positions(caplog):
    raw = _raw_with_optode_heights(np.nan, np.nan)
    raw.info["chs"][0]["loc"][3:9] = np.nan
    with caplog.at_level(logging.WARNING):
        warn_if_optodes_are_planar(raw)

    assert caplog.text == ""


def test_standardize_raw_adds_stim_channel_from_annotations():
    raw = mne.io.RawArray(np.zeros((1, 10)), mne.create_info(["hbo"], 1, "hbo"))
    raw.set_annotations(mne.Annotations([2], [0], ["Event"]))

    out = standardize_raw(raw, {"Event": 1})

    assert out.get_channel_types() == ["hbo", "stim"]
    assert out.get_data(picks="stim")[0, 2] == 1


def test_standardize_raw_maps_annotations_before_adding_stim_channel():
    raw = mne.io.RawArray(np.zeros((1, 10)), mne.create_info(["hbo"], 1, "hbo"))
    raw.set_annotations(
        mne.Annotations([2, 4, 6], [0, 0, 1], ["1", "Other", "BAD noisy"])
    )

    out = standardize_raw(raw, {"Event": 1}, annotation_rename_map={"1": "Event"})

    assert out.annotations.description.tolist() == ["Event", "Other", "BAD noisy"]
    assert out.get_data(picks="stim")[0, 2] == 1


def test_standardize_raw_places_events_on_recordings_with_first_samp():
    raw = mne.io.RawArray(
        np.zeros((1, 100)), mne.create_info(["hbo"], 10, "hbo"), first_samp=5
    )
    raw.set_annotations(mne.Annotations([2.0], [1.5], ["Event"]))

    out = standardize_raw(raw, {"Event": 1})

    events = mne.find_events(out, "Stim", verbose=False)
    assert events[:, 0].tolist() == [raw.first_samp + 20]
    np.testing.assert_allclose(out.annotations.onset, raw.first_time + 2.0)
    assert out.annotations.duration.tolist() == [1.5]


def test_standardize_raw_warns_when_events_share_a_sample(caplog):
    raw = mne.io.RawArray(np.zeros((1, 10)), mne.create_info(["hbo"], 1, "hbo"))
    raw.set_annotations(mne.Annotations([2, 2], [0, 0], ["A", "B"]))

    with caplog.at_level(logging.WARNING):
        standardize_raw(raw, {"A": 1, "B": 2})

    assert "share onset samples" in caplog.text
    assert raw.annotations.description.tolist() == ["A", "B"]


def test_standardize_raw_drops_annotations_after_last_sample(caplog):
    raw = mne.io.RawArray(np.zeros((1, 10)), mne.create_info(["hbo"], 10, "hbo"))
    raw.set_annotations(mne.Annotations([0.2, 0.96], [0, 0], ["Event", "Event"]))

    with caplog.at_level(logging.WARNING):
        out = standardize_raw(raw, {"Event": 1})

    assert out.annotations.onset.tolist() == [0.2]
    assert out.get_data(picks="stim")[0].nonzero()[0].tolist() == [2]
    assert "after the last sample" in caplog.text


def test_standardize_raw_keeps_existing_stim_channel():
    raw = mne.io.RawArray(np.zeros((1, 10)), mne.create_info(["Stim"], 1, "stim"))
    out = standardize_raw(raw, {"Event": 1})
    assert out is raw
    assert out.ch_names == ["Stim"]


def _zip_archive(path):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data/recording.txt", "signal")
    return path


def _tar_archive(path, member_name="data/recording.txt"):
    content = b"signal"
    member = tarfile.TarInfo(member_name)
    member.size = len(content)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(member, io.BytesIO(content))
    return path


@pytest.mark.parametrize(
    "make_archive, name",
    # Downloads are often stored without a file extension.
    [(_zip_archive, "download"), (_tar_archive, "download.tar.gz")],
)
def test_extract_archive_extracts_then_deletes_the_archive(
    tmp_path, make_archive, name
):
    archive = make_archive(tmp_path / name)
    destination = tmp_path / "extracted"

    extract_archive(archive, destination)

    assert (destination / "data/recording.txt").read_text() == "signal"
    assert not archive.exists()


def test_extract_archive_keeps_the_archive_when_extraction_fails(tmp_path):
    archive = _tar_archive(tmp_path / "unsafe.tar.gz", "../outside.txt")

    with pytest.raises(ValueError, match="outside"):
        extract_archive(archive, tmp_path / "extracted")

    assert archive.exists()
    assert not (tmp_path / "outside.txt").exists()


def test_download_lock_blocks_a_second_loader(tmp_path):
    with download_lock(tmp_path / "data"), pytest.raises(Timeout):
        download_lock(tmp_path / "data").acquire(timeout=0)


def test_merge_repeated_annotations_merges_close_repeats_only():
    annotations = mne.Annotations(
        onset=[0, 1.5, 3, 10, 30, 31.5],
        duration=[1, 1, 1, 15, 1, 1],
        description=["a", "a", "a", "rest", "a", "a"],
    )

    merged = merge_repeated_annotations(annotations, max_interval=2)

    assert list(merged.description) == ["a", "rest", "a"]
    np.testing.assert_allclose(merged.onset, [0, 10, 30])
    np.testing.assert_allclose(merged.duration, [4, 15, 2.5])


def test_merge_repeated_annotations_splits_distant_repeats():
    annotations = mne.Annotations(
        onset=[0, 1, 20], duration=[1, 1, 1], description=["a", "a", "a"]
    )

    merged = merge_repeated_annotations(annotations, max_interval=2)

    np.testing.assert_allclose(merged.onset, [0, 20])
    np.testing.assert_allclose(merged.duration, [2, 1])


def _device_raw(n_times, onsets, descriptions):
    names = ["S1_D1 760", "S1_D1 850", "S2_D9 760", "S2_D9 850"]
    info = mne.create_info(names, 10.0, "fnirs_cw_amplitude")
    raw = mne.io.RawArray(np.ones((4, n_times)), info, verbose=False)
    raw.set_annotations(mne.Annotations(onsets, 1.0, descriptions))
    return raw


def test_combined_devices_are_aligned_on_first_event():
    reference = _device_raw(400, [10.0, 20.0, 35.0], ["2", "3", "2"])
    other = _device_raw(400, [15.0, 25.0, 25.0], ["2", "3", "3"])

    raw = combine_devices(reference, other)

    assert raw.ch_names == [
        "S1_D1 760",
        "S1_D1 850",
        "S2_D9 760",
        "S2_D9 850",
        "S3_D10 760",
        "S3_D10 850",
        "S4_D18 760",
        "S4_D18 850",
    ]
    assert raw.n_times == 350
    assert raw.annotations.description.tolist() == ["2", "3"]
    np.testing.assert_allclose(raw.annotations.onset - raw.first_time, [10, 20])


def test_combined_devices_with_mismatched_events_raise():
    reference = _device_raw(400, [10.0, 20.0], ["2", "3"])
    other = _device_raw(400, [15.0, 25.0], ["2", "2"])

    with pytest.raises(ValueError, match="events"):
        combine_devices(reference, other)


def test_combined_devices_without_events_raise():
    reference = _device_raw(400, [10.0, 20.0], ["2", "3"])
    other = _device_raw(400, [15.0], ["2"])
    other.set_annotations(None)

    with pytest.raises(ValueError, match="no events"):
        combine_devices(reference, other)
