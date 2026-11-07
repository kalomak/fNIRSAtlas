# Project Structure

This repository is organized around a reproducible pipeline from raw fNIRS data to benchmark results.

## `fnirsatlas/`

Core Python package.

- `parsers/`: Dataset-specific parsers plus shared parser base classes.
- `download.py`: Dataset download helpers used by the scripts.
- `preprocessing.py`: Beer-Lambert conversion, filtering, TDDR, SCI, and downsampling.
- `epoching.py`: Epoch extraction from continuous recordings.
- `feature_extraction.py`: Feature transforms and sklearn-compatible preprocessing helpers.
- `classification.py`: Benchmark execution, CV logic, and baseline model entry points.
- `processed_dataset.py`: Processed dataset loading utilities.
- `datasets/`: Lightweight NumPy and PyTorch-facing dataset loaders.
- `io/`: Lower-level dataset wrappers and adapters such as `.to_torch()` and `.to_mne()`.
- `config.py`, `utils.py`, `logging.py`: Configuration, shared helpers, and logging setup.

## `scripts/`

Command-line entry points and analysis helpers.

- `run_all.py`: Orchestrates download, processing, and classification.
- `download_dataset.py`: Download one dataset.
- `process_dataset.py`: Process one dataset into `.npz` output.
- `classify_dataset.py`: Run benchmark classification for datasets or suites.
- `compare_models.py`, `wilcoxon_test.py`, `friedman_test.py`: Statistical comparison scripts.
- `display_results.py`, `display_search_results.py`: Result inspection utilities.
- `examples/`: Minimal runnable examples for custom sklearn and torch models.
- `checks/`: Validation and debugging scripts for data and results.
- `paper/`: Paper-specific tables, plots, and result-generation helpers.

## `docs/`

Project documentation.

- `configuration.md`: Config system and override behavior.
- `classification.md`: Benchmark API and CV behavior.
- `models.md`: Built-in models and custom model examples.
- `model_comparison.md`: Statistical comparison workflows.
- `processed_dataset.md`: Processed `.npz` format and adapters.
- `preprocessing.md`: Preprocessing pipeline details.
- `epoch_intervals.md`: Dataset-specific epoch timing.
- `parser_info.md`: Parser implementation notes.
- `adding_a_dataset.md`, `adding_a_model.md`: Contributor guides.
- `datasets/`: One Markdown file per dataset.

## `fnirsatlas/configs/`

YAML and metadata files that define datasets, tasks, preprocessing settings, and parameter grids.

## `notebooks/`

Jupyter notebooks that mirror the example workflows with additional explanation.

## `tests/`

Pytest-based unit and integration tests for classification, preprocessing, parsers, results, and model plumbing.
