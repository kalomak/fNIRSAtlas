"""
Datasets module for fnirsatlas.

Provides familiar interface for loading datasets:

```python
from fnirsatlas.datasets import load_dataset
X, y, groups, info = load_dataset("nemo")
print(X.shape, y.shape, groups.shape)
```

Or pytorch-style datasets:

```python
from fnirsatlas.datasets import Nemo
ds = Nemo()
X, y, groups = ds[0]
ds.dataloader()
```
"""

from .load import load_dataset  # noqa: F401

__all__ = ["load_dataset"]


def __getattr__(name: str):
    """Lazy expose dataset classes via fnirsatlas.datasets."""
    lower = name.lower()
    try:
        from .torch import make_single_dataset_class

        return make_single_dataset_class(lower, class_name=name)
    except ModuleNotFoundError as e:
        raise ImportError(
            "PyTorch not available. Use fnirsatlas.datasets.load_dataset for NumPy arrays, "
            "or pip install torch to use Dataset classes."
        ) from e
