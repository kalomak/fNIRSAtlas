# Getting Started

## Installation

fnirsatlas requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).
From the repository root, install the core package:

```bash
uv sync
```

Install optional dependencies when needed:

```bash
uv sync --extra dl            # Native PyTorch models
uv sync --extra braindecode   # Braindecode models
```

## Run A Benchmark

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

See [Benchmarking](benchmark.md) for evaluation and result-storage options.

## Access Data


```python
from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification

dataset = Bnci2014FnirsMA(subjects=[1])
paradigm = FNIRSClassification()
X, labels, metadata = paradigm.get_data(dataset, subjects=[1])
```

Data is downloaded and cached on first use. See [Data access](data_access.md) for
continuous and epoched output options.
