# Montages

Custom 3D montages. A parser sets the montage of its dataset if a file exists. Files are named `<dataset_id>_montage_3d.json`.

## Usage

Load a montage as `mne.channels.DigMontage`:

```python
from fnirsatlas.montage import load_montage_from_file

montage = load_montage_from_file("herff2014")
```

Save an MNE montage to JSON:

```python
import json
from fnirsatlas.montage import montage_to_dict

with open("fnirsatlas/montages/<dataset_id>_montage_3d.json", "w") as f:
    json.dump(montage_to_dict(montage), f, indent=2)
```
