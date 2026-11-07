import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open
import numpy as np

from fnirsatlas.results import Results
from fnirsatlas.utils import normalize_seeds, format_seeds, build_result_path

_real_open = open


@pytest.fixture(autouse=True)
def mock_results_dir():
    with patch("fnirsatlas.config.get_results_cache_dir", return_value=Path("results")):
        yield


def mock_results_data(
    setting="cross", dataset_name="shin2019", task="BL|MA", n_classes=2, **kwargs
):
    metrics = {}
    if setting == "within":
        metrics["within_acc"] = [0.7, 0.8]
        metrics["within_kappa"] = [0.5, 0.6]
    elif setting == "cross":
        metrics["cross_acc"] = [0.3, 0.4]
        metrics["cross_kappa"] = [0.1, 0.2]
    return {
        f"{dataset_name}": {
            f"{task}": {
                "n_classes": n_classes,
                "n_epochs": 1080,
                "mapping": {"Idle State": 0, "Mental Arithmetic": 1},
                **metrics,
            }
        },
        "model_name": "lda",
    }


def open_mock_multi(path, mode=None):
    ps = str(path)
    if "cockx2023" in ps:
        dataset_name = "cockx2023"
        setting = "within"
    elif "bnci" in ps:
        dataset_name = "bnci"
        setting = "cross"
    elif "bak2019" in ps:
        dataset_name = "bak2019"
        setting = "cross"
    else:
        return _real_open(path, mode or "r")
    if setting in ps:
        d = mock_results_data(setting, dataset_name)
    else:
        d = {}
    djson = json.dumps(d)
    return mock_open(read_data=djson)(path, mode)


class TestResults:
    """Test cases for the Results class."""

    def test_get_args_str(self):
        """Test argument string generation."""
        path = Results("bak2019").get_file_path()
        assert path.name == "results.json"
        assert path.parts[-2] == "42"  # seed
        assert path.parts[-3] == "bak2019"  # dataset
        assert path.parts[-4] == "lda"  # model
        assert path.parts[-5] == "none"  # search strategy
        assert path.parts[-6] == "cross"  # setting

    def test_infer_classification_setting(self):
        """Test inference of classification setting from metric."""
        assert Results._infer_classification_settings("cross_acc") == "cross"
        assert Results._infer_classification_settings("within_acc") == "within"
        assert Results._infer_classification_settings("other_metric") is None

    def test_get_saved_results_success(self):
        """Test successful loading of saved results."""
        mock_json = json.dumps(mock_results_data())
        with (
            patch("builtins.open", mock_open(read_data=mock_json)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = Results("shin2019", "cross_acc", "lda", seeds=[42]).load()
            assert "BL|MA" in result.task_dict

    def test_load_method_success(self):
        """Test the load() method successfully loads results and sets attributes."""
        mock_json = json.dumps(mock_results_data())
        with (
            patch("builtins.open", mock_open(read_data=mock_json)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            results = Results("shin2019", "cross_acc", "lda", seeds=[42])
            results.load()
            assert results.file_path is not None
            assert results.task_dict is not None
            assert "BL|MA" in results.task_dict
            assert results.task2metric is not None
            assert "BL|MA" in results.task2metric
            assert "cross_acc" in results.task_dict["BL|MA"]
            assert len(results.task2metric["BL|MA"]) == 2  # [0.5, 0.6] from mock data

    def test_specific_overrides(self):
        """Test that cross results are used when both files available"""
        with (
            patch("builtins.open", side_effect=open_mock_multi),
            patch("pathlib.Path.exists", return_value=True),
        ):
            results = Results("bnci", "cross_acc", "lda", seeds=[42])
            results.load()
            assert "cross_acc" in results.task_dict["BL|MA"]
            assert results.task2metric["BL|MA"][0] == 0.3

    def test_load_single_seed_file(self):
        """Test loading from single seed file."""
        mock_json = json.dumps(
            mock_results_data(setting="cross", dataset_name="shin2019")
        )
        with (
            patch("builtins.open", mock_open(read_data=mock_json)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            results = Results("shin2019", "cross_acc", "lda", seeds=[42])
            results.load()
            assert "cross_acc" in results.task_dict["BL|MA"]
            assert results.task2metric["BL|MA"][0] == 0.3

    def test_missing_file(self):
        """Test error when file not found."""
        with (
            patch("builtins.open", side_effect=open_mock_multi),
            patch("pathlib.Path.exists", return_value=False),
        ):
            results = Results("nonexistent", "cross_acc", "lda", seeds=[42])
            with pytest.raises(FileNotFoundError, match="No split files found"):
                results.load()


class TestSmartLoading:
    """Test cases for smart loading with individual seed files."""

    @staticmethod
    def create_test_results_dir(tmp_path, dataset="test_dataset"):
        """Create a test results directory with sample files."""
        results_dir = tmp_path / "results"
        results_dir.mkdir(exist_ok=True)
        return results_dir

    @staticmethod
    def create_result_file(
        dataset, seeds, setting="cross", model="lda", task="TestTask"
    ):
        """Create a single result file."""
        filepath = build_result_path(setting, model, dataset, seeds)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            dataset: {
                task: {
                    "n_classes": 2,
                    "n_epochs": 100,
                    "mapping": {"Class0": 0, "Class1": 1},
                    "cross_acc": [0.6, 0.65, 0.7],
                    "cross_kappa": [0.2, 0.3, 0.4],
                }
            }
        }
        filepath.write_text(json.dumps(data))
        return filepath

    def test_smart_load_individual_seed_files(self, tmp_path, monkeypatch):
        """Test smart loading from individual seed files."""
        monkeypatch.chdir(tmp_path)
        dataset = "test_dataset"

        self.create_result_file(dataset, [0], setting="cross")
        self.create_result_file(dataset, [1], setting="cross")
        self.create_result_file(dataset, [2], setting="cross")

        results = Results(dataset, "cross_acc", "lda", seeds=[0, 1, 2])
        results.load()

        assert results.task_dict is not None
        assert "TestTask" in results.task_dict
        assert "merged" in results.file_path
        assert len(results.task2metric["TestTask"]) == 9

    def test_smart_load_combines_metrics(self, tmp_path, monkeypatch):
        """Test that metrics are properly concatenated from individual files."""
        monkeypatch.chdir(tmp_path)
        dataset = "test_dataset"

        self.create_result_file(dataset, [0], setting="cross")
        self.create_result_file(dataset, [1], setting="cross")

        results = Results(dataset, "cross_acc", "lda", seeds=[0, 1])
        results.load()

        expected_acc = [0.6, 0.65, 0.7, 0.6, 0.65, 0.7]
        np.testing.assert_array_equal(
            results.task2metric["TestTask"], np.array(expected_acc)
        )

    def test_always_merges_individual_seed_files(self, tmp_path, monkeypatch):
        """Test that individual seed files are always merged, even if combined file exists."""
        monkeypatch.chdir(tmp_path)
        dataset = "test_dataset"

        # Create both combined and individual files
        self.create_result_file(dataset, [0, 1], setting="cross")
        self.create_result_file(dataset, [0], setting="cross")
        self.create_result_file(dataset, [1], setting="cross")

        results = Results(dataset, "cross_acc", "lda", seeds=[0, 1])
        results.load()

        # Should always merge now, not use combined file
        assert "merged" in str(results.file_path)
        assert len(results.task2metric["TestTask"]) == 6  # 3 from each file

    def test_missing_individual_seed_file(self, tmp_path, monkeypatch):
        """Test strict mode errors when an individual seed file is missing."""
        monkeypatch.chdir(tmp_path)
        dataset = "test_dataset"

        self.create_result_file(dataset, [0], setting="cross")

        results = Results(
            dataset,
            "cross_acc",
            "lda",
            seeds=[0, 1, 2],
            accept_partial=False,
        )
        with pytest.raises(FileNotFoundError):
            results.load()

    def test_missing_individual_seed_file_defaults_to_partial(
        self, tmp_path, monkeypatch
    ):
        """Test default behavior loads available seeds when one is missing."""
        monkeypatch.chdir(tmp_path)
        dataset = "test_dataset"

        self.create_result_file(dataset, [0], setting="cross")
        self.create_result_file(dataset, [2], setting="cross")

        results = Results(dataset, "cross_acc", "lda", seeds=[0, 1, 2])
        results.load()

        expected_acc = [0.6, 0.65, 0.7, 0.6, 0.65, 0.7]
        np.testing.assert_array_equal(
            results.task2metric["TestTask"], np.array(expected_acc)
        )

    def test_accept_partial_individual_seed_files(self, tmp_path, monkeypatch):
        """Test loading available seed files when accept_partial is enabled."""
        monkeypatch.chdir(tmp_path)
        dataset = "test_dataset"

        self.create_result_file(dataset, [0], setting="cross")
        self.create_result_file(dataset, [2], setting="cross")

        results = Results(
            dataset,
            "cross_acc",
            "lda",
            seeds=[0, 1, 2],
            accept_partial=True,
        )
        results.load()

        expected_acc = [0.6, 0.65, 0.7, 0.6, 0.65, 0.7]
        np.testing.assert_array_equal(
            results.task2metric["TestTask"], np.array(expected_acc)
        )

    def test_accept_partial_still_fails_when_no_seed_files(self, tmp_path, monkeypatch):
        """Test accept_partial still raises when nothing can be loaded."""
        monkeypatch.chdir(tmp_path)

        results = Results(
            "test_dataset",
            "cross_acc",
            "lda",
            seeds=[0, 1],
            accept_partial=True,
        )
        with pytest.raises(FileNotFoundError, match="No results found"):
            results.load()

    def test_smart_load_preserves_metadata(self, tmp_path, monkeypatch):
        """Test that metadata is preserved from first file during merge."""
        monkeypatch.chdir(tmp_path)
        dataset = "test_dataset"

        self.create_result_file(dataset, [0], setting="cross")
        self.create_result_file(dataset, [1], setting="cross")

        results = Results(dataset, "cross_acc", "lda", seeds=[0, 1])
        results.load()

        assert results.task_dict["TestTask"]["n_classes"] == 2
        assert results.task_dict["TestTask"]["n_epochs"] == 100
        assert results.task_dict["TestTask"]["mapping"] == {"Class0": 0, "Class1": 1}

    def test_smart_load_with_multiple_tasks(self, tmp_path, monkeypatch):
        """Test smart loading when files contain multiple tasks."""
        monkeypatch.chdir(tmp_path)
        dataset = "test_dataset"

        for seeds in [[0], [1]]:
            filepath = build_result_path("cross", "lda", dataset, seeds)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            data = {
                dataset: {
                    "Task1": {
                        "n_classes": 2,
                        "n_epochs": 100,
                        "mapping": {"C0": 0, "C1": 1},
                        "cross_acc": [0.7, 0.75],
                        "cross_kappa": [0.4, 0.5],
                    },
                    "Task2": {
                        "n_classes": 3,
                        "n_epochs": 100,
                        "mapping": {"C0": 0, "C1": 1, "C2": 2},
                        "cross_acc": [0.6, 0.65],
                        "cross_kappa": [0.4, 0.45],
                    },
                }
            }
            filepath.write_text(json.dumps(data))

        results = Results(dataset, "cross_acc", "lda", seeds=[0, 1])
        results.load()

        assert "Task1" in results.task_dict
        assert "Task2" in results.task_dict
        assert len(results.task2metric["Task1"]) == 4
        assert len(results.task2metric["Task2"]) == 4

    @staticmethod
    def create_split_file(
        dataset, seed, split_ix, setting="cross", model="lda", task="TestTask"
    ):
        """Create a single split result file."""
        filepath = build_result_path(setting, model, dataset, [seed], split_ix)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            dataset: {
                task: {
                    "n_classes": 2,
                    "n_epochs": 100,
                    "mapping": {"Class0": 0, "Class1": 1},
                    "cross_acc": [0.5 + split_ix * 0.1],
                    "cross_kappa": [0.1 + split_ix * 0.05],
                }
            }
        }
        filepath.write_text(json.dumps(data))
        return filepath

    def test_load_fallback_to_split_files(self, tmp_path, monkeypatch):
        """Test that Results.load() falls back to split files when main file not found."""
        monkeypatch.chdir(tmp_path)
        dataset = "test_dataset"
        seed = 42

        # Create only split files, no main file
        for split_ix in range(5):
            self.create_split_file(dataset, seed, split_ix)

        results = Results(dataset, "cross_acc", "lda", seeds=[seed])
        results.load()

        assert results.task_dict is not None
        assert "TestTask" in results.task_dict
        assert "merged" in str(results.file_path)
        assert "split files" in str(results.file_path)

        # Verify all splits were combined
        expected_acc = [0.5, 0.6, 0.7, 0.8, 0.9]
        np.testing.assert_array_almost_equal(
            results.task2metric["TestTask"], np.array(expected_acc)
        )

    def test_load_split_files_combines_kappa(self, tmp_path, monkeypatch):
        """Test that split files are properly combined for cross_kappa metric."""
        monkeypatch.chdir(tmp_path)
        dataset = "test_dataset"
        seed = 42

        # Create split files
        for split_ix in range(3):
            self.create_split_file(dataset, seed, split_ix)

        # Load with cross_kappa metric
        results = Results(dataset, "cross_kappa", "lda", seeds=[seed])
        results.load()

        expected_kappa = [0.1, 0.15, 0.2]
        np.testing.assert_array_almost_equal(
            results.task2metric["TestTask"], np.array(expected_kappa)
        )

    def test_load_split_files_preserves_metadata(self, tmp_path, monkeypatch):
        """Test that metadata is preserved when loading from split files."""
        monkeypatch.chdir(tmp_path)
        dataset = "test_dataset"
        seed = 42

        for split_ix in range(2):
            self.create_split_file(dataset, seed, split_ix)

        results = Results(dataset, "cross_acc", "lda", seeds=[seed])
        results.load()

        assert results.task_dict["TestTask"]["n_classes"] == 2
        assert results.task_dict["TestTask"]["n_epochs"] == 100
        assert results.task_dict["TestTask"]["mapping"] == {"Class0": 0, "Class1": 1}
