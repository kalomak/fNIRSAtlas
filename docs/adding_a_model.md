# Adding a Baseline Model

How to add a new built-in baseline model (accessible via `get_baseline_model("name")`).
For running your own model without registering it, see [models.md](models.md#running-your-own-model).

## 1. Define the model

Create `fnirsatlas/dl/YourModel.py`.

- Extend `DLModelBase` (handles fit/predict/sklearn compat).
- List all tunable hyperparams as explicit `__init__` args (not hidden in `**kwargs`) so sklearn's `get_params`/`set_params` can see them.
- Add a pipeline factory function that derives data-dependent params (`n_chs`, `T`, `num_classes`) from `**kwargs`.

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.pipeline import Pipeline
from fnirsatlas.dl.DLModelBase import DLModelBase
from fnirsatlas.feature_extraction import StandardScalerND, ChannelSplitter

class YourModel(DLModelBase):
    def __init__(self, n_chs, T, n_ch_types=2, num_classes=2,
                 F1=4, p_dropout=0.5, lr=1e-2, **kwargs):
        super().__init__(lr=lr, **kwargs)
        self.n_chs = n_chs
        self.T = T
        self.n_ch_types = n_ch_types
        self.num_classes = num_classes
        self.F1 = F1
        self.p_dropout = p_dropout
        self.lr = lr
        # ... define layers ...

    def forward(self, x):
        return self.layers(x)

def create_your_model_pipeline(**kwargs):
    torch.manual_seed(kwargs.get("random_state", 42))
    num_classes = np.unique(kwargs["y"]).shape[0]
    ch_names = kwargs["info"]["ch_names"]
    n_ch_types = len(np.unique([c.split()[1] for c in ch_names]))
    n_chs = len(ch_names) // n_ch_types
    T = kwargs["X"].shape[-1]
    batch_size = 8 if kwargs.get("classification_setting") == "within" else 32
    return Pipeline([
        ("ch_splitter", ChannelSplitter(ch_names)),
        ("scaler", StandardScalerND()),
        ("clf", YourModel(
            n_chs, T, n_ch_types,
            num_classes=num_classes, batch_size=batch_size,
        )),
    ])
```

## 2. Register in classification.py

Add an `elif` branch with a lazy import in `get_baseline_model()`:

```python
# fnirsatlas/classification.py, in get_baseline_model():
elif model_name.lower() == "yourmodel":
    try:
        from fnirsatlas.dl.YourModel import create_your_model_pipeline
        default_params = get_model_params("yourmodel")
        default_params.update(kwargs)
        return create_your_model_pipeline(**default_params)
    except ImportError:
        raise ImportError("Install pytorch: pip install torch")
```

## 3. Add param grid

Append to `fnirsatlas/configs/param_grids.yaml`. Prefix params with the pipeline step name (`clf__`):

```yaml
yourmodel:
  clf__F1: [4, 8]
  clf__lr: [0.01, 0.001]
  clf__p_dropout: [0.25, 0.5]
```

## 4. Add tests

Add to `tests/test_dl_pipeline.py`:

1. **Sklearn compatibility test** -- pipeline fit + predict on dummy data.
2. **Param grid test** -- iterate all grid combos, check forward pass shapes and `set_params` works.

See `test_tscnn_sklearn_compatibility` and `test_tscnn_stcnn_param_grid` for reference.

## 5. Run

```bash
# no grid search
uv run python scripts/classify_dataset.py -m yourmodel -d bnci -c cross
# with grid search
uv run python scripts/classify_dataset.py -m yourmodel --suite ma -c cross --seed 42 --search-strategy nested
```
