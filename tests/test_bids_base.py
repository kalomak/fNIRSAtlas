"""Tests for shared BIDS fNIRS utilities."""

from fnirsatlas.datasets.bids_base import (
    BaseBIDSFNIRSDataset,
    build_bids_fnirs_file_paths,
    get_bids_cache_root,
)


class _TestBIDSDataset(BaseBIDSFNIRSDataset):
    def _download_subject(self, subject, path, force_update, update_path, verbose):
        raise NotImplementedError


def test_cache_root_uses_dataset_code(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "fnirsatlas.datasets.bids_base.get_dataset_path", lambda code, path: tmp_path
    )

    root = get_bids_cache_root("Luke2021FnirsTapping")

    assert root == tmp_path / "MNE-luke2021fnirstapping-data"


def test_paths_with_session_and_run():
    files = build_bids_fnirs_file_paths("170", "02", [("BallSqueezing", "01")])

    assert files[0].startswith("sub-170/ses-02/")


def test_paths_without_session_or_run():
    files = build_bids_fnirs_file_paths("01", None, [("tapping", None)])

    assert "sub-01/nirs/sub-01_task-tapping_nirs.snirf" in files
