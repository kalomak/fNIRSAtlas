<p align="center"><img src="./docs/assets/fnirsatlas-logo.svg" width="160" alt="fNIRSAtlas logo"></p>

# fNIRSAtlas

[![PyPI](https://img.shields.io/pypi/v/fnirsatlas)](https://pypi.org/project/fnirsatlas/)
[![arXiv](https://img.shields.io/badge/arXiv-2611.10082-b31b1b)](https://arxiv.org/abs/2611.10082)

fNIRSAtlas is a Python package for parsing, preprocessing, and classifying fNIRS datasets with a unified, extensible structure. It provides the largest fNIRS benchmark, covering motor, mental arithmetic, motor imagery, emotion, and other tasks, and a fully automatic pipeline from raw data to classification.

## Features

- 20+ public fNIRS datasets
- Mental arithmetic, motor, n-back, and emotion classification paradigms
- Within-session, cross-session, and cross-subject evaluation
- YAML and Python pipelines for scikit-learn, PyTorch, and Braindecode
- Dataset caching, result persistence, statistics, and plots
- Export of fNIRS data in a consistent format

## Usage

### Installation

Requires Python 3.11 or later and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
uv sync --extra dl            # native PyTorch models
uv sync --extra braindecode   # Braindecode models
```

### Run a benchmark

```python
from moabb.evaluations import CrossSubjectEvaluation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification, WindowedFeatureExtractor

evaluation = CrossSubjectEvaluation(
    paradigm=FNIRSClassification(), datasets=[Bnci2014FnirsMA()]
)
pipeline = make_pipeline(
    WindowedFeatureExtractor(),
    StandardScaler(),
    LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
)
results = evaluation.process({"LDA": pipeline})
```

Evaluations accept scikit-learn-compatible [pipelines](docs/pipelines.md). Data is downloaded and cached on first use. Raw results are stored in `results/` and summaries in `benchmark/`. Existing runs are resumed unless `overwrite=True` is passed to the evaluation. See [Benchmarking](docs/benchmark.md) for more options.

### Access data

```python
import fnirsatlas

fnirsatlas.list_datasets()  # available dataset names
dataset = fnirsatlas.get_dataset("Bnci2014FnirsMA")
X, labels, metadata = fnirsatlas.FNIRSClassification().get_data(dataset, subjects=[1])
```

See [Data access](docs/data_access.md) for continuous and epoched output options.

## Examples

See the runnable [examples](examples/).

## Documentation

See the [documentation](docs/).

## Contributing

See [Contributing](CONTRIBUTING.md) for the development setup, tests, and checks. In short:

- [Open an issue](https://github.com/kalomak/fNIRSAtlas/issues/new/choose) to report a bug, suggest an improvement, request a feature or dataset, report a dataset correction, or ask a question.
- [Open a pull request](https://github.com/kalomak/fNIRSAtlas/pulls) for a fix or feature.
- [Add a dataset](docs/adding_datasets.md) through a pull request.

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
