# What's new in v2

fNIRSAtlas v2 is a major rewrite that builds on [MOABB](https://moabb.neurotechx.com/docs/) wherever possible. Goals:

- Avoid duplicating work already done in MOABB, e.g. dataset download, epoching, evaluation loops and results storage.
- Use MOABB features:
    - [evaluation splits](https://moabb.neurotechx.com/docs/api.html#evaluations), including [transfer learning splits](https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectSplitter.html)
    - [statistical comparison of pipelines](https://moabb.neurotechx.com/docs/generated/moabb.analysis.meta_analysis.compute_dataset_statistics.html)
    - [dataset and result caching](https://moabb.neurotechx.com/docs/generated/moabb.datasets.base.CacheConfig.html)
- Share one structure across neuroimaging modalities for easier skill transfer and future integration.

## Changes

- Within-subject evaluation is replaced by within-session and cross-session evaluation.
- The classification task concept is removed. Most datasets classify all classes against each other by default. Subsets are selected with the paradigm `events`, e.g. `Bak2019FnirsTapping` (left, right and foot tapping): `MotorExecution()` classifies all three, `MotorExecution(events=["Tapping/Left", "Tapping/Right"])` left vs right.
- Preprocessing and hyperparameter search are part of the pipeline config.
- Processed `.npz` files, `fNIRSAtlasDataset` and `load_processed_dataset` are replaced by MOABB data access. See [Data access](data_access.md).

    ```python
    import fnirsatlas

    dataset = fnirsatlas.get_dataset("Bnci2014FnirsMA")
    paradigm = fnirsatlas.FNIRSClassification()
    # v1 groups: metadata.subject
    X, labels, metadata = paradigm.get_data(dataset, subjects=[1])
    epochs, labels, metadata = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
    raws = dataset.get_data(subjects=[1])  # continuous, subject/session/run
    ```

- Parsers are replaced by datasets (MOABB `BaseDataset` subclasses):

    | v1 parser | v2 dataset |
    |---|---|
    | `bak2019` | `Bak2019FnirsTapping` |
    | `bnci` | `Bnci2014FnirsMA` |
    | `buccino2016` | `Buccino2016FnirsME` |
    | `cockx2023` | `Cockx2023FnirsME` |
    | `gao2024` | `Gao2023FnirsSqueezing` |
    | `guevara2024` | `Guevara2024FnirsParkinson` |
    | `herff2014` | `Herff2014FnirsWorkload` |
    | `huang2021` | `Huang2021FnirsWorkload` |
    | `luke2021` | `Luke2021FnirsTapping` |
    | `luke2024a` | `Luke2021FnirsSpeech` |
    | `luke2024b` | `Yucel2025FnirsTapping` |
    | `nemo__afim` | `Nemo2024FnirsImagery` |
    | `nemo__empe` | `Nemo2024FnirsPerception` |
    | `ortega2020` | `Ortega2020FnirsGrip` |
    | `shader2021` | `Shader2021FnirsSpeech` |
    | `shin2016__ma` | `Shin2016FnirsMA` |
    | `shin2016__mi` | `Shin2016FnirsMI` |
    | `shin2018__nback` | `Shin2018FnirsNBack` |
    | `shin2018__wg` | `Shin2018FnirsWG` |
    | `shin2018b` | `Shin2018bFnirsMA` |
    | `shin2019` | `Shin2018FnirsTernary` |
    | `yi2025` | `Yi2025FnirsMI` |

## New features

- Everything MOABB offers, e.g. [evaluations](https://moabb.neurotechx.com/docs/api.html#evaluations), [statistics](https://moabb.neurotechx.com/docs/api.html#statistics), [Braindecode](https://braindecode.org) pipelines and structured dataset metadata.
- Easier dataset addition: more detailed instructions ([Adding datasets](adding_datasets.md), `.agents/skills/adding-datasets`) with automated data validation scripts (`scripts/check_event_intervals.py`, `scripts/check_hb_scale.py`).
