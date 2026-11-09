# Adding Datasets

```{tip}
Using a coding agent? See the [adding-datasets skill](https://github.com/kalomak/fNIRSAtlas/blob/main/.agents/skills/adding-datasets/SKILL.md).
```
[MOABB's dataset tutorial](https://moabb.neurotechx.com/docs/auto_examples/tutorials/tutorial_4_adding_a_dataset.html) explains the general dataset interface.

Each dataset is one module in [`fnirsatlas/datasets/`](https://github.com/kalomak/fNIRSAtlas/tree/main/fnirsatlas/datasets). We recommend using the existing module closest to your source format as a reference:

| Source format | Base class | Example |
|---|---|---|
| BIDS with SNIRF | `BaseBIDSFNIRSDataset` | [`Gao2023FnirsSqueezing`](../fnirsatlas/datasets/gao2023_fnirs_squeezing.py), [`Luke2021FnirsTapping`](../fnirsatlas/datasets/luke2021_fnirs_tapping.py) |
| Native fNIRS format (`.snirf`, NIRx) | `BaseFNIRSDataset` | [`Guevara2024FnirsParkinson`](../fnirsatlas/datasets/guevara2024_fnirs_parkinson.py), [`Yi2025FnirsMI`](../fnirsatlas/datasets/yi2025_fnirs_mi.py) |
| Homer `.nirs` | `BaseFNIRSDataset` | [`Bonnal2025FnirsME`](../fnirsatlas/datasets/bonnal2025_fnirs_me.py) |
| `.mat`, `.csv`, or another raw format | `BaseFNIRSDataset` | [`Bak2019FnirsTapping`](../fnirsatlas/datasets/bak2019_fnirs_tapping.py), [`Bnci2014FnirsMA`](../fnirsatlas/datasets/bnci2014_fnirs_ma.py) |
| Pre-epoched arrays | `BaseFNIRSDataset` | [`Huang2021FnirsWorkload`](../fnirsatlas/datasets/huang2021_fnirs_workload.py) |
| Continuous recordings, no fixed-size epochs (`interval=None`) | `BaseFNIRSDataset` | [`Klein2026FnirsMotion`](../fnirsatlas/datasets/klein2026_fnirs_motion.py) |

## Tests

Copy [`tests/test_luke2021_fnirs_speech_dataset.py`](../tests/test_luke2021_fnirs_speech_dataset.py) to `tests/test_<dataset>.py` and replace its expected values with those of your dataset. Run it with the real data, including the `data` and `manual` tests that are skipped by default:

```bash
uv run pytest tests/test_<dataset>.py -m ""
```

## Checks

Run the check scripts in [`scripts/`](https://github.com/kalomak/fNIRSAtlas/tree/main/scripts) and fix or document what they flag:

| Command | Flags |
|---|---|
| `uv run python scripts/check_event_intervals.py <DatasetClass>` | Event spacing per epoch length. Choose the epoch interval from it |
| `uv run python scripts/check_bids_events.py <bids_root>` | BIDS only: `.snirf` events that differ from the BIDS events |
| `uv run python scripts/check_hb_scale.py <DatasetClass>` | Hemoglobin values outside the plausible range: unit error or spikes |
| `uv run python scripts/check_montage.py <DatasetClass>` | Wrong channel positions |
| `uv run python scripts/check_processed_data.py <DatasetClass>` | Data that is already filtered |
| `uv run python scripts/check_dataset_quality.py <DatasetClass>` | Flat channels and duplicate runs |
| `uv run --extra data python scripts/check_metadata.py <DatasetClass>` | `METADATA` that does not match the data |

## Benchmark

Check that the dataset runs end to end and scores as expected:

```bash
uv run python scripts/run_fnirs_benchmark.py \
  --datasets <dataset.code> --evaluations CrossSubject
```
