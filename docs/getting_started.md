# Getting Started

## Installation

fnirsatlas requires Python 3.11 or newer. Install the pre-release from PyPI:

```bash
pip install --pre fnirsatlas
```

Install optional dependencies when needed:

```bash
pip install --pre "fnirsatlas[dl]"            # Native PyTorch models
```

For development, install from source with [uv](https://docs.astral.sh/uv/).

```bash
git clone -b v2 https://github.com/kalomak/fNIRSAtlas.git
cd fNIRSAtlas
uv sync --all-extras
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
