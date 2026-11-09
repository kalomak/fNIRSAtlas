# Pipelines

`fnirsatlas` uses [MOABB's pipeline format](https://moabb.neurotechx.com/docs/auto_examples/tutorials/tutorial_3_benchmarking_multiple_pipelines.html) unchanged. A pipeline definition is a YAML or Python file, named explicitly when running the benchmark:

```bash
uv run python scripts/run_fnirs_benchmark.py -p pipelines/Windowed_ANN.yml pipelines/dl/fnirsnet24.py
```

## YAML definitions

```yaml
name: Windowed + Random Forest
paradigms:
  - FNIRSClassification
pipeline:
  - name: WindowedFeatureExtractor
    from: fnirsatlas.pipelines
    parameters:
      n_windows: 6
  - name: RandomForestClassifier
    from: sklearn.ensemble
    parameters:
      n_estimators: 100
```

## Python definitions

A Python definition must expose a `PIPELINE` mapping:

[WindowedFeatureExtractor](generated/fnirsatlas.pipelines.WindowedFeatureExtractor.rst)
computes channel features over temporal windows for conventional classifiers.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fnirsatlas.pipelines import WindowedFeatureExtractor

PIPELINE = {
    "name": "Windowed + LDA",
    "paradigms": ["FNIRSClassification"],
    "pipeline": make_pipeline(
        WindowedFeatureExtractor(n_windows=6),
        StandardScaler(),
        LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
    ),
}
```

To construct and evaluate a model directly instead of defining a loader file, see [Running a Custom Model](custom_models.md).


## PyTorch models

Install the optional dependency with `uv sync --extra dl`. Pass an
uninstantiated PyTorch module class to
{class}`fnirsatlas.pipelines.dl.FNIRSClassifier`:

```python
from moabb.evaluations import WithinSessionEvaluation
from sklearn.pipeline import make_pipeline

from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification
from fnirsatlas.pipelines import StandardScalerND
from fnirsatlas.pipelines.dl import EEGNet, FNIRSClassifier

classifier = make_pipeline(
    StandardScalerND(dim=(0, 2, 3)),
    FNIRSClassifier(module=EEGNet, module__F1=8, max_epochs=20),
)
evaluation = WithinSessionEvaluation(
    paradigm=FNIRSClassification(),
    datasets=[Bnci2014FnirsMA()],
)
results = evaluation.process({"EEGNet": classifier})
```

## Hyperparameter search

Add `param_grid` beside `pipeline`. YAML example:

```yaml
param_grid:
  windowedfeatureextractor__n_windows:
    - 4
    - 6
    - 8
```

The equivalent Python entry is:

```python
PIPELINE["param_grid"] = {
    "windowedfeatureextractor__n_windows": [4, 6, 8],
}
```

MOABB uses this grid for grid search. Pass `--optuna` to search the same space with Optuna. See [MOABB's hyperparameter-search example](https://moabb.neurotechx.com/docs/auto_examples/how_to_benchmark/plot_benchmark_grid_search.html).

## Braindecode models

Braindecode is available as an independent optional integration:

```bash
uv sync --extra braindecode
```

Its sklearn-compatible estimators can also be passed directly to MOABB:

```python
from braindecode import EEGClassifier
from braindecode.models import EEGNet
from moabb.evaluations import WithinSessionEvaluation

from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification


classifier = EEGClassifier(module=EEGNet, max_epochs=20, train_split=None)
evaluation = WithinSessionEvaluation(
    paradigm=FNIRSClassification(),
    datasets=[Bnci2014FnirsMA()],
)
results = evaluation.process({"Braindecode EEGNet": classifier})
```
