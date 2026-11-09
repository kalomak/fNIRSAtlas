# Adding Datasets

Follow [MOABB's dataset tutorial](https://moabb.neurotechx.com/docs/auto_examples/tutorials/tutorial_4_adding_a_dataset.html) for the general dataset interface. This page covers the fNIRS-specific requirements.

Use `BaseBIDSFNIRSDataset` for BIDS/SNIRF datasets and `BaseFNIRSDataset` for native or tabular formats. Existing implementations in `fnirsatlas/datasets/` provide format-specific examples.

A MOABB-evaluable dataset must:

- declare events, subjects, an epoch interval, and an alphanumeric paradigm prefixed with `fnirs`;
- return MNE `Raw` objects as `{session: {run: raw}}` for each subject;
- expose fNIRS channels using MNE channel types such as `hbo`, `hbr`, or `fnirs_cw_amplitude`;
- preserve stimulus labels and timing;
- be exported from `fnirsatlas.datasets` and `fnirsatlas`.

Add real-data tests for metadata, returned structure, channels, and events. Check the epoch interval with:

```bash
uv run python scripts/check_event_intervals.py <DatasetClass>
```

Finally, run a smoke benchmark:

```bash
uv run python scripts/run_fnirs_benchmark.py \
  --datasets <dataset.code> --evaluations WithinSession
```

## Recording-only datasets

Use the same `BaseFNIRSDataset` and `dataset.get_data()` API when continuous
recordings cannot be represented as meaningful fixed-size epochs. Set
`paradigm=None` and `interval=None`. These datasets remain available directly
but are not in `BENCHMARK_DATASETS`.
