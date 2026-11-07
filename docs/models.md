# Models

## Running your own model

### Sklearn example

The input data is a 3D array of shape (n_samples, n_channels, n_timepoints) and sklearn models expect a 2D array of shape (n_samples, n_features). You can use `WindowedFeatureExtractor`, which extracts windowed statistical features from each channel and returns a flattened array, or alternatively simply flatten the data before passing it to other sklearn components.
After that, you can use any sklearn model.

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

This example is available in `scripts/examples/custom_sklearn.py`.

### Pytorch example

To run a pytorch model on the benchmark suite, you can simply extend [DLModelBase](models.md#dlmodelbase) class. You can use kwargs passed to `get_clf` to derive model parameters.

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

This example is available in `scripts/examples/custom_torch.py`.

## DLModelBase

Is a base class for deep learning models.

It has a sklearn-like interface, with `fit`, `predict`, `predict_proba` methods.
See [pytorch example](models.md#pytorch-example) to see how to extend it.

To update training behavior at the end of each epoch, override `_on_epoch_end()` in
your model subclass. See `scripts/examples/custom_torch.py` for an example.

You can also run it with a more typical pytorch training loop which validates on each epoch end `.fit(X_train, y_train, X_val=X_val, y_val=y_val)`. See `scripts/examples/custom_torch_with_curve.py` for an example.


## Baseline models

These are the implementations of the baseline models that are used in the original paper.

### LDA

```python
Pipeline([
	("extractor", WindowedFeatureExtractor(n_windows=6, features=["mean"])),
	("scaler", StandardScalerND()),
	("clf", LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")),
])
```

1. Extract features. This extracts mean from 6 non-overlapping windows for each channel individually. The returned feature vector is flattened to shape (n_samples, 6 * n_channels).
2. Normalize. Z-normalize the features. StandardScalerND is a wrapper around sklearn's StandardScaler that works with multi-dimensional arrays.
3. Classify. Linear discriminant analysis with shrinkage regularization. Regularization strength is determined by ledoit-wolf lemma.

### Logistic regression

Logistic regression uses the SAGA solver with the same windowed mean features and feature-wise normalization as LDA. The parameter search evaluates 1 and 5 windows, L1 and L2 penalties, and `C` values 0.00001, 0.01, 1, and 100.

### SVM

SVM uses an RBF kernel with `gamma="scale"`, the same windowed mean features, and feature-wise normalization. The parameter search evaluates 1 and 5 windows and `C` from 0.0001 to 1000.

### Random forest

Random forest uses the same windowed mean features and feature-wise normalization. The parameter search evaluates 1 and 5 windows, 10 to 1000 trees, and minimum leaf sizes of 1 and 2. Each forest is seeded and fitted serially to avoid nested parallelism.

### ANN

```python
Pipeline([
	("extractor", WindowedFeatureExtractor(n_windows=1, features=["mean", "slope"])),
	("scaler", StandardScaler()),
	("clf", MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, random_state=seed)),
])
```

1. Extract features. This uses a single window with mean and slope features per channel.
2. Normalize. Standardize the resulting 2D feature matrix before classification.
3. Classify. A small one-hidden-layer MLP with 20 hidden units.

This built-in baseline is a simple approximation inspired by `naseer2015comparison`, "Comparison of Artificial Neural Network and Support Vector Machine Classifications for fNIRS-based BCI". The paper reports the feature type and `20 hidden neurons`, but leaves the full ANN architecture under-specified, so this implementation intentionally stays minimal.

### EEGNet

```python
Pipeline([
	("ch_splitter", ChannelSplitter(ch_names=ch_names)),
	("scaler", StandardScalerND(dim=(0, 2, 3))),
	("clf", EEGNet(n_chs, T, n_ch_types, normalize=False, **kwargs)),
])
```

1. Split channels by type. Just reformats the data from (n_samples, n_total_chs, T) to (n_samples, n_ch_types, n_chs_per_type, T).
2. Normalize. Z-normalize the features. StandardScalerND is a wrapper around sklearn's StandardScaler that works with multi-dimensional arrays.
3. We adapted the EEGNet implementation from huang2021. The difference is that it is more flexible, works with different data shapes, sfreq, etc. This extends [DLModelBase](models.md#dlmodelbase) class.

### Baseline Model Parameter Grids

Enable parameter grid search by selecting a non-default `--search-strategy`:

```bash
uv run python scripts/classify_dataset.py -m lda --suite ma -c cross --seed 42 --search-strategy nested
```

Parameter grids are defined in `fnirsatlas/configs/param_grids.yaml`:

```yaml
lda:
  extractor__n_windows: [1, 3, 5, 15]
  extractor__features: [["mean"], ["slope"], ["std"], ["mean", "slope"]]

logreg:
  extractor__n_windows: [1, 5]
  clf__penalty: ["l1", "l2"]
  clf__C: [0.00001, 0.01, 1, 100]

svm:
  extractor__n_windows: [1, 5]
  clf__C: [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

rf:
  extractor__n_windows: [1, 5]
  clf__n_estimators: [10, 50, 100, 1000]
  clf__min_samples_leaf: [1, 2]

knn:
  extractor__n_windows: [1, 5]
  clf__n_neighbors: [1, 3, 5, 7]
  clf__weights: ["uniform", "distance"]

ann:
  clf__activation: ["relu", "tanh"]
  clf__alpha: [1, 5]
  clf__hidden_layer_sizes: [[20], [256]]
  clf__learning_rate_init: [0.001, 0.01]

eegnet:
  clf__F1: [4, 16]
  clf__kernel_size_1: [3, 10]
  clf__lr: [0.01, 0.001]
  clf__p_dropout: [0.25, 0.5]
```

To add your own baseline model, see [adding_a_model.md](adding_a_model.md).
