# Classification

The classification module provides functionality to run machine learning benchmarks on fNIRS datasets.

## Usage

The main entry point is `run_benchmark()` in `fnirsatlas/classification.py`:

```python
from fnirsatlas.classification import run_benchmark

results = run_benchmark(
    get_clf=get_clf,  # function that returns a classifier pipeline
    suite="motor",  # or specific datasets=["dataset1", "dataset2"]
    classification_setting="cross",  # "cross" or "within"
    seed=42,  # random seed for reproducibility
    n_jobs=1,  # number of parallel jobs
    param_grid={  # Optional hyperparameter search for GridSearchCV
        "extractor__n_windows": [1, 6],
        "clf__C": [1e-3, 1, 1e3],
    }
)
```

See `scripts/examples/custom_sklearn.py` for a full example.

To evaluate multiple random initializations, run the benchmark multiple times with different `seed` values and compare or aggregate the saved results afterward.

## Cross-Validation

- **Cross-subject**: Uses `GroupKFold` to evaluate generalization across subjects.
- **Within-subject**: Uses `StratifiedKFold` for each subject individually.
- **Hyperparameter search**: Nested CV with `GridSearchCV` (inner folds: 4)

### Configuration

Outer and inner cross-validation can be configured in `config.yaml`:

```yaml
cv:
  cross:
    class: sklearn.model_selection.GroupKFold
    kwargs:
      n_splits: 5
      shuffle: true
      random_state: null
    inner:
      class: sklearn.model_selection.GroupKFold
      kwargs:
        n_splits: 4
  within:
    class: sklearn.model_selection.StratifiedKFold
    kwargs:
      n_splits: 5
      shuffle: true
      random_state: null
    inner:
      class: sklearn.model_selection.StratifiedKFold
      kwargs:
        n_splits: 4
```

## Models

See [Models](models.md) for baseline implementations (LDA, EEGNet) and custom model examples. 
