# Processed Dataset Format

Processed datasets are saved as `.npz` files in `paths.processed_datasets` (default `~/mne_data/fnirsatlas/processed_datasets`). Each file contains:

- `X`: Data array (n_epochs, n_channels, n_timepoints)
- `y`: Class labels (n_epochs,)
- `groups`: Subject IDs (n_epochs,)
- `info`: Metadata dictionary

## Simple Loading Example

```python
import numpy as np
from fnirsatlas.utils import get_processed_data_path

data = np.load(get_processed_data_path("bnci"), allow_pickle=True)
X, y, groups, info = data["X"], data["y"], data["groups"], data["info"].item()
```

## Class API with adapters

```python
from fnirsatlas.io import fNIRSAtlasDataset

ds = fNIRSAtlasDataset("nemo__empe")
X, y, groups, info = ds.as_tuple()

torch_ds = ds.to_torch()  # dynamic torch import
epochs_dict = ds.to_mne()  # dynamic mne import
```

## Converting processed data to MNE Epochs

```python
import mne
from fnirsatlas.processed_dataset import from_dataset, load_processed_dataset
from fnirsatlas.utils import get_processed_data_path

dataset_id = "nemo__empe"
X, y, groups, info = load_processed_dataset(get_processed_data_path(dataset_id))
ed = from_dataset(X, y, groups, info)  # Dict[subject_id, mne.Epochs]
epochs = mne.concatenate_epochs(list(ed.values()))
```
