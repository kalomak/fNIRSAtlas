# Adding a dataset

This tutorial adds a dataset `smith2025` with left and right finger tapping. A parser converts the source files into an MNE `Raw` object with events. Preprocessing, epoching, and classification then work without dataset-specific code.

Files touched:

| File | Purpose |
|---|---|
| `fnirsatlas/configs/config.yaml` | Data location and dataset settings |
| `fnirsatlas/configs/downloads.yaml` | Download instructions |
| `fnirsatlas/parsers/smith2025.py` | Parser |
| `fnirsatlas/parsers/__init__.py` | Parser registration |
| `fnirsatlas/dataset_catalog.py` | Suite membership |
| `fnirsatlas/configs/classification_tasks.yaml` | Classification tasks |
| `tests/test_smith2025_parser.py` | Tests for dataset-specific logic |
| `docs/datasets/smith2025.md` | Dataset summary |

Set up the environment as in [Running scripts](../README.md#running-scripts) and run all commands from the repository root.

## 1. Configure location and download

Name the dataset after the first author and year of the paper that publishes it.

Add the dataset to `fnirsatlas/configs/config.yaml`:

```yaml
datasets:
  smith2025:
    subpath: "Smith2025/data"   # relative to paths.raw_datasets
    description: "Smith2025 finger tapping"
    original_data_type: "fnirs_cw_amplitude"   # data type in the source files
```

Add the download to `fnirsatlas/configs/downloads.yaml`:

```yaml
smith2025:
  type: automatic
  url: https://zenodo.org/records/<id>/files/data.zip
  destination: datasets/Smith2025/
  filename: Smith2025.zip
  extractions:
    - archive: Smith2025.zip
      destination: data
```

Use `type: manual` or `type: request` if the data cannot be downloaded by script. See [Downloads](downloads.md) for the other fields, e.g. `parts` and `renames`.

Download and check that the files land in `paths.raw_datasets` + `subpath`:

```bash
uv run python scripts/download_dataset.py -d smith2025
```

Use the rawest data available: intensity > optical density > haemoglobin concentration.

## 2. Write the parser

### BIDS datasets

Subclass `BIDSParser`. It reads SNIRF files and events from the BIDS structure, so usually only the label mapping and event filtering are dataset-specific. See `fnirsatlas/parsers/luke2021.py`:

```python
from pathlib import Path
import pandas as pd
from .bids import BIDSParser


class Smith2025Parser(BIDSParser):
    """Smith2025 finger tapping parser."""

    dataset_id = "smith2025"

    def __init__(self, dataset_path: Path = None):
        super().__init__(dataset_path)
        self.cond2label = {"Tapping/Left": 1, "Tapping/Right": 2}

    def apply_metadata(self, metadata: pd.DataFrame, data: dict = None) -> pd.DataFrame:
        """Drop events that are not trials."""
        return metadata[metadata["trial_type"] != "Start"]
```

### Other formats

Subclass `BaseFNIRSParser` and implement four methods. See `fnirsatlas/parsers/bnci.py`.

```python
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .base import BaseFNIRSParser

# Source-detector pairs in channel order, from the paper's montage figure.
PAIRS = [(1, 1), (1, 2), (2, 1), (2, 2)]


class Smith2025Parser(BaseFNIRSParser):
    """Smith2025 finger tapping parser."""

    dataset_id = "smith2025"

    def __init__(self, dataset_path: Path = None):
        super().__init__(dataset_path)
        self.cond2label = {"Tapping/Left": 1, "Tapping/Right": 2}

    def get_subject_ids(self) -> list[str]:
        return [f"S{i:02d}" for i in range(1, 21)]

    def load_data(self, subject_id: str, session=None) -> dict:
        mat = loadmat(self.dataset_path / f"{subject_id}.mat", squeeze_me=True)
        return {"intensity": mat["d"], "onsets": mat["onsets"], "codes": mat["codes"], "sfreq": float(mat["fs"])}

    def extract_channels(self, data: dict) -> tuple:
        # Source files store (n_times, n_channels). MNE expects (n_channels, n_times).
        ch_data = data["intensity"].T
        ch_names = [f"S{s}_D{d} {wl}" for s, d in PAIRS for wl in (760, 850)]
        ch_types = ["fnirs_cw_amplitude"] * len(ch_names)
        return ch_data, ch_names, ch_types

    def extract_events(self, data: dict) -> tuple:
        # MNE events: (onset sample, 0, event code)
        onsets = np.round(data["onsets"] * data["sfreq"]).astype(int)
        events = np.column_stack([onsets, np.zeros_like(onsets), data["codes"]])
        event_desc = {code: name for name, code in self.cond2label.items()}
        return events, event_desc
```

- Channel names follow `S<source>_D<detector> <wavelength>` for intensity and optical density, and `S<source>_D<detector> hbo|hbr` for haemoglobin.
- Channel types are MNE fNIRS types: `fnirs_cw_amplitude`, `fnirs_od`, `hbo`, `hbr`.
- Override `get_sessions(subject_id)` if a subject has several recordings. Sessions are concatenated per subject.
- If the dataset has several experiments, set `EXPERIMENTS = ["MA", "MI"]` and take an `experiment` argument. The dataset name is then `<dataset>__<experiment>`, e.g. `shin2016__ma`. See `fnirsatlas/parsers/shin2016.py`.
- Comment non-obvious parsing decisions, e.g. event shifts or excluded subjects, with their source.

### Montage

If the source files have no sensor positions, add `fnirsatlas/montages/smith2025_montage_3d.json`. `BaseFNIRSParser` subclasses set it automatically. Parsers that build the raw object themselves, e.g. BIDS parsers, call `self.set_montage(raw)` (see `fnirsatlas/parsers/nemo.py`). See the [montages README](../fnirsatlas/montages/README.md).

## 3. Register the parser

In `fnirsatlas/parsers/__init__.py`, import the parser and add it to `PARSER_MAP` and `__all__`:

```python
from .smith2025 import Smith2025Parser

PARSER_MAP = {
    ...
    "smith2025": Smith2025Parser,
}
```

Add the dataset to a suite in `SUITE_MAP` in `fnirsatlas/dataset_catalog.py`, e.g. `"motor"`.

## 4. Define classification tasks

Add the tasks to `fnirsatlas/configs/classification_tasks.yaml`. Keys are condition names from `cond2label`:

```yaml
smith2025:
  classification_tasks:
    - description: "FingerTap[L|R]"
      mapping:
        Tapping/Left: 0
        Tapping/Right: 1
```

See [Classification tasks](classification_tasks.md) for multi-condition mappings and per-experiment tasks.

## 5. Set the epoch interval

Set expected interval, defaults to `tmin=-2`, `tmax=15` seconds. Check whether consecutive epochs overlap:

```bash
uv run python scripts/checks/check_overlap.py -d smith2025 -p
```

If they do, set a shorter `tmax` in `config.yaml` and comment the reason:

```yaml
  smith2025:
    ...
    tmax: 12
```

See [Epoch intervals](epoch_intervals.md).

## 6. Process and classify

```bash
uv run python scripts/process_dataset.py -d smith2025
uv run python scripts/dataset_overview.py -d smith2025
uv run python scripts/classify_dataset.py -d smith2025
```

Check the overview against the paper: subjects, epochs per class, channels. If the paper reports classification accuracy, compare. Large differences usually come from event timing or labels.

## 7. Test

Add `tests/test_smith2025_parser.py` for dataset-specific logic, e.g. event filtering, label mapping, or subject selection. Tests should not require the raw data. See `tests/test_gao2024_parser.py`:

```python
import pandas as pd
from fnirsatlas.parsers.smith2025 import Smith2025Parser


def test_start_events_are_dropped(tmp_path):
    parser = Smith2025Parser(tmp_path)
    metadata = pd.DataFrame({"trial_type": ["Start", "Tapping/Left"]})

    assert parser.apply_metadata(metadata)["trial_type"].tolist() == ["Tapping/Left"]
```

```bash
uv run pytest
```

## 8. Document

- Add `docs/datasets/smith2025.md` following `docs/datasets/luke2021.md`: subjects, classes, device, data link, license, citation.
- Add the dataset to [Downloads](downloads.md) and [Parser info](parser_info.md).

## 9. Open a pull request

See [Contributing](../CONTRIBUTING.md). Checklist:

- [ ] Download works from a clean `paths.raw_datasets`
- [ ] Parser registered in `PARSER_MAP` and `SUITE_MAP`
- [ ] Classification tasks defined
- [ ] No overlapping epochs, or the overlap is documented
- [ ] Overview matches the paper
- [ ] Tests pass
- [ ] Dataset documented
