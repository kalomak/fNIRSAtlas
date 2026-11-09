"""Tests for getting datasets by name."""

import pytest

import fnirsatlas


def test_list_datasets_names_exported_datasets():
    names = fnirsatlas.list_datasets()
    assert "Bnci2014FnirsMA" in names
    assert "BaseFNIRSDataset" not in names


def test_get_dataset_returns_named_dataset():
    assert fnirsatlas.get_dataset("Bnci2014FnirsMA").code == "Bnci2014FnirsMA"


def test_get_dataset_unknown_name_lists_datasets():
    with pytest.raises(ValueError, match="Bnci2014FnirsMA"):
        fnirsatlas.get_dataset("bnci")
