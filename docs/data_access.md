# Data access

## By name

```python
import fnirsatlas

fnirsatlas.list_datasets()  # available dataset names
dataset = fnirsatlas.get_dataset("Bnci2014FnirsMA")
```

## Paradigm data

This is the intended path for labeled and epoched data.
Uses a paradigms preprocessing and segmentation settings.

```python
from fnirsatlas import FNIRSClassification

paradigm = FNIRSClassification()
X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

epochs, labels, metadata = paradigm.get_data(dataset, subjects=[1], return_epochs=True)

raws, labels, metadata = paradigm.get_data(dataset, subjects=[1], return_raws=True)
```

```{note}
Bad channels are interpolated in Epochs but not in raws.
```

## Standardized continuous data

```python
from fnirsatlas import Bnci2014FnirsMA

dataset = Bnci2014FnirsMA(subjects=[1])
recordings = dataset.get_data(subjects=[1])
raw = recordings[1]["0fnirs"]["0"]  # subject/session/run
```

## Processed continuous data

```python
from moabb.datasets.bids_interface import StepType
from moabb.datasets.preprocessing import FixedPipeline

from fnirsatlas.datasets.preprocessing import interpolate_bads_step, preprocess_fnirs

pipeline = FixedPipeline(
    [
        (
            StepType.RAW,
            preprocess_fnirs(
                sci_threshold=0.5,
                tddr=True,
                ppf=6.0,
                l_freq=0.01,
                h_freq=0.1,
            ),
        ),
        # Optional. Replaces bad channels instead of only marking them.
        (StepType.RAW, interpolate_bads_step()),
    ]
)

recordings = dataset.get_data(subjects=[1], process_pipeline=pipeline)
raw = recordings[1]["0fnirs"]["0"]  # subject/session/run
```

The output has the same subject/session/run structure and contains processed
continuous `Raw` objects with their annotations attached.

