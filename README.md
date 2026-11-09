# fnirsatlas

fnirsatlas is the largest fNIRS dataset collection and benchmark.

## Features

- 20+ public fNIRS datasets through the MOABB dataset API
- Mental arithmetic, motor, n-back, and emotion classification paradigms
- Within-session, cross-session, and cross-subject evaluation
- YAML and Python pipelines for scikit-learn, PyTorch, and Braindecode
- Dataset caching, result persistence, statistics, and plots
- Easily export fNIRS data in consistent format

## Usage

### Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). From the repository root:

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

Evaluations accept sklearn-compatible [pipelines](/docs/pipelines.md).
Data is downloaded cached on first use. Raw results are stored in `results/`. Summaries are written to `benchmark/`. Existing runs are resumed unless `overwrite=True` is passed to the evaluation. See [Benchmarking](docs/benchmark.md) for more options.

### Access data

```python
import fnirsatlas

fnirsatlas.list_datasets()  # available dataset names
dataset = fnirsatlas.get_dataset("Bnci2014FnirsMA")
X, labels, metadata = fnirsatlas.FNIRSClassification().get_data(dataset, subjects=[1])
```

See [Data access](docs/data_access.md) for continuous and epoched output options.

## Examples

Runnable examples are in `examples/`.

## Documentation

See the [documentation](docs/index.rst).

## Cite FNIRSAtlas

If you use FNIRSAtlas in your experiments, please cite the software and related publication.

### Software Citation

#### APA Format

```text
Mäkelä, K. (2026). FNIRSAtlas: A Large-Scale Benchmark Suite for Near-Infrared
Spectroscopy Analysis (Version 0.1.0) [Computer software].
```

#### BibTeX Format

```bibtex
@software{Makela_FNIRSAtlas_2026,
  author  = {Mäkelä, Kalle},
  title   = {FNIRSAtlas: A Large-Scale Benchmark Suite for Near-Infrared Spectroscopy Analysis},
  year    = {2026},
  version = {0.1.0},
}
```

### Scientific Publication

TBD.

## Development

See the [development guide](development.md).
