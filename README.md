<p align="center"><img src="./docs/assets/fnirsatlas-logo.svg" width="160" alt="fNIRSAtlas logo"></p>

# fNIRSAtlas

[![PyPI](https://img.shields.io/pypi/v/fnirsatlas)](https://pypi.org/project/fnirsatlas/)
[![arXiv](https://img.shields.io/badge/arXiv-2611.10082-b31b1b)](https://arxiv.org/abs/2611.10082)

fNIRSAtlas is a Python package for parsing, preprocessing, and classifying fNIRS datasets with a unified, extensible structure. It provides the largest fNIRS benchmark with multiple suites (motor, Mental Arithmetic (MA), Motor Imagery (MI), Emotion, MISC) and fully automatic pipeline from raw data to classification. The processed data can be easily exported with no dependencies.

## Table of Contents

- [Features](#features)
- [Usage](#usage)
  - [Installation](#installation)
  - [Quick start](#quick-start)
  - [Loading datasets programmatically](#loading-datasets-programmatically)
  - [Running your own model](#running-your-own-model)
  - [Running scripts](#running-scripts)
  - [Running the full benchmark](#running-the-full-benchmark)
- [Examples](#examples)
- [Processed data usage](#processed-data-usage)
- [Configuration](#configuration)
- [Logging](#logging)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)

## Features

- Largest benchmark/collection of fNIRS datasets: 20+ datasets, multiple suites (Motor, MA, MI, Emotion, MISC)
- Completely automatic: download, preprocessing, classification
- Transparent and reproducible: all steps from raw data to classification
- Common format for processed data that [can be used with no dependencies](#processed-data-usage)
- Cross- and within-subject classification with nested cross-validation (default: outer CV=seeds x 5-fold, inner CV=4-fold) for robust evaluation and hyperamater search, [details](docs/classification.md)
- Built-in baseline models: LDA and EEGNet for comparison, [details](docs/models.md)
- [Run with your own models](#running-your-own-model) (e.g. PyTorch, scikit-learn)
- [Statistical comparison between models](docs/model_comparison.md)
- [Jupyter notebooks and examples](#examples)
- [Configurable pipeline](#configuration): preprocessing steps, dataset specific settings, cross-validation, etc.

## Usage

### Installation

```bash
# Install everything (recommended for new users)
pip install "fnirsatlas[all]"
```

The package is split into optional dependency groups if you only need a subset:

```bash
# Read pre-processed .npz data only (no MNE, no heavy deps)
pip install fnirsatlas

# + download and process raw datasets (adds mne, mne-bids, mne-nirs, datalad, git-annex)
pip install "fnirsatlas[process]"

# + classify / run benchmarks (adds scikit-learn, scipy, statsmodels, pandas)
pip install "fnirsatlas[classify]"

# + deep learning models (adds torch, einops)
pip install "fnirsatlas[dl]"
```

For GPU-enabled PyTorch, install torch separately following https://pytorch.org/get-started/locally/ before installing `fnirsatlas[dl]`.

### Quick start

```bash
pip install "fnirsatlas[process,classify]"
```

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from fnirsatlas.datasets import load_dataset
from fnirsatlas.classification import run_benchmark
from fnirsatlas.feature_extraction import WindowedFeatureExtractor

# Downloads (44 MB) and processes the dataset on first use
X, y, groups, info = load_dataset("luke2021", download="yes")
print(X.shape, info["cond2label"])


def get_clf(**kwargs):
    return make_pipeline(
        WindowedFeatureExtractor(n_windows=6, features=["mean"]),
        StandardScaler(),
        LogisticRegression(),
    )


# Cross-subject classification, prints a summary table
results = run_benchmark(datasets=["luke2021"], get_clf=get_clf)
```

```text
(450, 40, 76) {'Control': 1, 'Tapping/Left': 2, 'Tapping/Right': 3}

Summary:
dataset          task                           #classes  #epochs     Cross%    Cross κ
---------------------------------------------------------------------------------------
Luke2021         FingerTap[L|R]                        2      300   84.0±9.2  68.0±18.5
---------------------------------------------------------------------------------------
MEAN                                                                84.0±9.2  68.0±18.5
```

### Loading datasets programmatically

You can load specific datasets via `fnirsatlas.datasets`.

- NumPy/scikit-learn interface:

```python
from fnirsatlas.datasets import load_dataset

# download: "ask" (default) prompts if raw data is missing, "yes" downloads, "no" never downloads
X, y, groups, info = load_dataset("bnci", download="yes")
```

- PyTorch-style interface (requires torch):

```python
from fnirsatlas.datasets import Bnci

ds = Bnci()                  # or Bnci('/path/to/*.npz')
X, y, g = ds[0]              # (tensor, tensor, subject-id)
dl = ds.dataloader(batch_size=32)
```

- Class API with adapters:

```python
import numpy as np
from fnirsatlas.io import fNIRSAtlasDataset

ds = fNIRSAtlasDataset("bnci")
X, y, groups, info = ds.as_tuple()
torch_ds = ds.to_torch()   # requires torch
epochs = ds.to_mne()       # requires mne
```

### Running your own model

#### sklearn example
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from fnirsatlas.classification import run_benchmark
from fnirsatlas.feature_extraction import WindowedFeatureExtractor


def get_clf(**kwargs):
    random_state = kwargs.pop("random_state", 42)
    return Pipeline(
        [
            ("extractor", WindowedFeatureExtractor()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=random_state, max_iter=100000)),
        ]
    )


results = run_benchmark(
    get_clf=get_clf,
    classification_setting="cross",
    suite="ma",
    seed=42,
    n_jobs=1,
    param_grid={
        "extractor__n_windows": [1, 6],
        "clf__C": [1e-3, 1, 1e3],
    },
)
```

Full example with explanations: [`custom_sklearn`](notebooks/custom_sklearn.ipynb)

#### pytorch example
```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from fnirsatlas.dl.DLModelBase import DLModelBase
from fnirsatlas.feature_extraction import StandardScalerND, ChannelSplitter
from fnirsatlas.classification import run_benchmark


class SimpleNet(DLModelBase):
    def __init__(self, T, n_chs, num_classes, p_dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        self.T = T
        self.n_chs = n_chs
        self.num_classes = num_classes
        self.p_dropout = p_dropout
        self.layers = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=(1, T)),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Conv2d(4, 8, kernel_size=(n_chs, 1)),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Flatten(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x):
        return self.layers(x)

    def _on_epoch_end(self, optimizer):
        if not self.epoch % 10:
            print(f"Epoch: {self.epoch}, LR: {optimizer.param_groups[0]['lr']}")


def get_clf(**kwargs):
    random_state = kwargs["random_state"]
    num_classes = np.unique(kwargs["y"]).shape[0]
    ch_names = kwargs["info"]["ch_names"]
    T = kwargs["X"].shape[-1]
    n_chs = len(ch_names) // 2
    torch.manual_seed(random_state)
    return Pipeline(
        [
            ("splitter", ChannelSplitter(ch_names)),
            ("scaler", StandardScalerND(dim=(0, 2, 3))),
            ("clf", SimpleNet(T, n_chs, num_classes)),
        ]
    )


res = run_benchmark(
    get_clf=get_clf,
    suite="ma",
    classification_setting="cross",
    seed=42,
    param_grid={
        "clf__p_dropout": [0.1, 0.5, 0.9],
    },
)
```

Full example with explanations: [`custom_torch`](notebooks/custom_torch.ipynb)

### Running scripts

The `scripts/` folder has command-line scripts for running the full benchmark. It is not part of the pip package. To use it, clone the repository and set up the environment:

```bash
git clone https://github.com/kalomak/fNIRSAtlas
cd fNIRSAtlas
# Create .venv with all optional deps and pytest
uv sync --extra all --group dev
```

If you prefer conda, `environment.yml` installs the same extras:

```bash
conda env create -f environment.yml
conda activate fnirsatlas
```

Run all script commands below from the repository root. With conda, drop the `uv run` prefix.

#### Running the full benchmark

`scripts/run_all.py` is the main script that runs:
1. `scripts/download_dataset.py`, which downloads the datasets to `paths.raw_datasets` (default `~/mne_data/fnirsatlas/datasets`).
2. `scripts/process_dataset.py`, which processes the datasets and saves them as `.npz` files to `paths.processed_datasets` (default `~/mne_data/fnirsatlas/processed_datasets`).
3. `scripts/classify_dataset.py`, which classifies the datasets and prints a summary table. Base results are cached under the configured `paths.results_cache` directory as `.json` files.

```bash
# Run by default: download, process, and classify all datasets
uv run python scripts/run_all.py

# Skip classification step (if you only want the processed data)
uv run python scripts/run_all.py --skip-classification

# Run a suite
uv run python scripts/run_all.py --suite ma

# Run with specific datasets
uv run python scripts/run_all.py -d bak2019 luke2021
```

**Note**: If you get warnings for datasets that are missing / require manual download, follow the instructions in the warning to download the data. They are generally very easy to download.

##### CLI options

`run_all.py`
- `-d, --datasets`: Specific datasets to process
- `-s, --suite`: Suite name to process (motor, ma, mi, nback, emotion, misc)
- `-j, --n_jobs`: Number of parallel jobs (default: 8)
- `--seed`: Random seed for classification (default: 42)
- `-m, --model`: Classification model: lda or eegnet (default: lda)
- `--skip-classification`: Skip classification step
- `-c, --classification_setting`: Classification type to use: cross or within (default: cross)
- `--search-strategy`: Hyperparameter search strategy: none, nested, single, or block (default: none)

The `run_all.py` pipeline consists of three steps that can also be run individually:

1. `download_dataset.py` `-d/--dataset`
2. `process_dataset.py` `-d/--dataset`
3. `classify_dataset.py` `-d/--datasets` `-s/--suite` `--seed` `-m/--model`
    - `-j/--n_jobs`: Number of parallel jobs (default: 1)
    - `--seed`: Random seed for reproducibility (default: 42)
    - `--output`: Output file path for saved results
    - `-c, --classification_setting`: Classification type to use: cross or within (default: cross)
    - `--search-strategy`: Hyperparameter search strategy: none, nested, single, or block

#### Automated run with Snakemake

Snakemake downloads, processes, and classifies the datasets:

```bash
# --restart-times 3 retries download jobs that fail randomly
uv run --extra all snakemake --cores all --rerun-incomplete --restart-times 3
```

By default this runs a small subset without hyperparameter search. The datasets, models, seeds, and settings are in [`fnirsatlas/configs/quick_run_snakemake.yaml`](fnirsatlas/configs/quick_run_snakemake.yaml). To run the full benchmark, use the commented-out full run in that file. **The full run is not recommended without a cluster.**

Result tables are saved to `.snakemake/display/{setting}.txt`.

To run the workflow as one Slurm job:

```bash
bash scripts/slurm/snakemake_single_sbatch.job.sh --account <account> --partition <partition>
```
### Examples

Examples are available as both Jupyter notebooks and Python scripts. Notebooks are at `notebooks/` and have some extra explanations. Scripts are at `scripts/examples/`.

- [`custom_sklearn`](notebooks/custom_sklearn.ipynb) - Benchmark a custom sklearn model with RandomForest
- [`custom_torch`](notebooks/custom_torch.ipynb) - Benchmark a custom PyTorch CNN model
- [`custom_torch_with_curve`](notebooks/custom_torch_with_curve.ipynb) - Run a PyTorch model with learning curves and no sklearn pipeline
- [`custom_vs_baseline`](notebooks/custom_vs_baseline.ipynb) - Compare custom model to LDA baseline
- [`visualize_features`](notebooks/visualize_features.ipynb) - Visualize feature extraction
- [`classify`](notebooks/classify.ipynb) - Full custom pipeline for a single dataset

## Processed data usage

After processing ([see how](#usage)), processed data is saved to `paths.processed_datasets` (default `~/mne_data/fnirsatlas/processed_datasets`).
You can load datasets with the IO API:

```python
import numpy as np
from fnirsatlas.io import fNIRSAtlasDataset

ds = fNIRSAtlasDataset("bnci")  # or fNIRSAtlasDataset("/path/to/bnci_processed.npz")
X, y, groups, info = ds.as_tuple()
print(f"Data shape: {X.shape}")
print(f"Class counts: {np.unique(y, return_counts=True)}")
print(f"Class mapping: {info['cond2label']}")

# optional adapters (dynamic imports)
torch_ds = ds.to_torch()  # requires torch
epochs_dict = ds.to_mne()  # requires mne
```

If you are not planning to use the raw data, you can delete it after processing. Raw datasets are very large compared to the processed data (total 60GB vs 800MB). Raw data is downloaded to `paths.raw_datasets`.

More on the processed data format, e.g. how to convert it back to mne epochs, see [`processed_dataset`](docs/processed_dataset.md).

## Configuration

Default settings are in [`fnirsatlas/configs/config.yaml`](fnirsatlas/configs/config.yaml). To change them, write a YAML file containing only the keys to change:

```yaml
datasets:
  herff2014:
    path: "/data/Herff2014"
```

Set `FNIRSATLAS_CONFIG` to the path of this file:

```bash
FNIRSATLAS_CONFIG=/path/to/my_config.yaml python my_script.py
```

More on the configuration, see [Configuration](docs/configuration.md).

## Logging

The package uses centralized logging configuration. Log levels can be controlled via the `FNIRSATLAS_LOG_LEVEL` environment variable:

```bash
# Default WARNING level
uv run python scripts/classify_dataset.py -d bnci

# Debug level for detailed output
FNIRSATLAS_LOG_LEVEL=DEBUG uv run python scripts/classify_dataset.py -d bnci
```

Available levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

## Documentation

- [Configuration](docs/configuration.md)
- [Classification](docs/classification.md)
- [Models](docs/models.md)
- [Model comparison](docs/model_comparison.md)
- [Adding a dataset](docs/adding_a_dataset.md)
- [Existing parser info](docs/parser_info.md)
- [Classification tasks](docs/classification_tasks.md)
- [Processed dataset](docs/processed_dataset.md)
- [Dataset summaries](docs/datasets/)
- [Preprocessing pipeline](docs/preprocessing.md)
- [Epoch intervals](docs/epoch_intervals.md)
- [Project structure](docs/project_structure.md)

## Contributing

See [Contributing](CONTRIBUTING.md). In short:
- Make a bug report or feature request [here](https://github.com/kalomak/fNIRSAtlas/issues)
- [Create a pull request](https://github.com/kalomak/fNIRSAtlas/pulls) for a fix, feature etc.
- [Request a dataset or report a correction](https://github.com/kalomak/fNIRSAtlas/issues/new/choose) to a dataset or task
- [Add a new dataset](docs/adding_a_dataset.md) through a pull request
- Contact the author if you need help with any of the above or have any questions

## Citation

```bibtex
@inproceedings{makela2026fnirsatlas,
  title     = {{fNIRSAtlas}: A Large-Scale Benchmark for Functional Near-Infrared Spectroscopy Classification},
  author    = {M{\"a}kel{\"a}, Kalle and Ruotsalo, Tuukka},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  eprint    = {2611.10082},
  archivePrefix = {arXiv}
}
```
