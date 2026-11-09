---
name: adding-datasets
description: Add and validate fNIRS datasets in fnirsatlas.
---

# Adding datasets

## 1. Draft METADATA from the paper

- Obtain the paper describing the dataset. If not easily obtainable, ask the user.
- Create a new file in `fnirsatlas/datasets/`. Naming convention: first author and year of the paper that publishes the dataset, `fnirs` and a one-word descriptive suffix, e.g. `buccino2016_fnirs_me.py` / `Buccino2016FnirsME`. Prefer established shorthands (`ME`, `MI`, `MA`, `WG`) over spelled-out words.
- Write a class skeleton with only the docstring and `METADATA = DatasetMetadata(...)`, with `FnirsAcquisitionMetadata` as `acquisition`, from the paper, before loading any data. See `Buccino2016FnirsME` and `Luke2021FnirsTapping`.
    - Fill values only from sources: paper, data files, repository record. Comment the source of non-obvious values.
    - `hardware`: specific model. If the paper does not name it, check metadata, raw file headers, etc.
    - `documentation.license`: from the repository record.
    - `documentation.doi`: paper DOI. Data DOI only if there is no paper.
    - Docstring: title line, short task description, `Notes` with exclusions, `References`.
- Paper's classification accuracy, if any, in `performance`, for step 6.

## 2. Inspect the data

Open a few source files and their metadata. Correct and supplement `METADATA` where the data differs from the paper, and note the difference for the report.
- Data type (cw/od/hb) and raw unit, for loading in step 3.
    - Use the rawest format available: intensity > optical density > hemoglobin. Check all download options.
- Whether the events support fixed-size classification epochs. If not, explain the limitations and confirm Recording-only status with the user. Document the reasons in `Notes`.

### BIDS

Check `.snirf` BIDS events match: `uv run python scripts/check_bids_events.py <bids_root>`.

## 3. Implement the dataset module

### Reference implementations

See most suitable reference implementation(s).

| Source format | Base class | What to implement | Loading pattern | Example |
|---|---|---|---|---|
| BIDS with SNIRF | `BaseBIDSFNIRSDataset` | BIDS search attributes and download logic; override `_get_single_subject_data()` only for dataset-specific handling | The base class discovers files with `bids_paths()`, reads SNIRF, and preprocesses each run | `Gao2023FnirsSqueezing`, `Luke2021FnirsTapping` |
| Native fNIRS format (`.snirf`, NIRx) | `BaseFNIRSDataset` | `data_path()` and `_get_single_subject_data()` | Read with the corresponding MNE reader, call `_standardize_raw()`, then return the session/run mapping | `Guevara2024FnirsParkinson`, `Yi2025FnirsMI` |
| Homer `.nirs` | `BaseFNIRSDataset` | `data_path()` and `_get_single_subject_data()` | `loadmat()`, read intensity with `homer_nirs_intensity()`, events from `s`/`CondNames`, then call `_standardize_raw()` | `Bonnal2025FnirsME` |
| `.mat`, `.csv`, or another raw format | `BaseFNIRSDataset` | `data_path()`, format-specific parsing, and `_get_single_subject_data()` | Parse arrays and events, create data and stim `RawArray` objects with `_make_raw()` and `_make_stim_raw()`, then call `_standardize_raw()` | `Bak2019FnirsTapping`, `Bnci2014FnirsMA`, `Shin2016Fnirs*` |
| Pre-epoched arrays | `BaseFNIRSDataset` | `data_path()`, epoch validation, and `_get_single_subject_data()` | Concatenate equal-length epochs into a `RawArray`, preserve boundaries with a stim event per epoch, and skip loader preprocessing | `Huang2021FnirsWorkload`; MOABB's `Hinss2021` |
| Recording-only | Choose suitable | download logic, and source annotation loading | Preserve continuous recordings and annotations without creating epochs | `Klein2026FnirsMotion` |

### Units

MNE expects `hbo`/`hbr` in M. Convert from the documented unit with
`concentration_length_to_molar()` , with a comment.

| Device / software | Unit | Example |
|---|---|---|
| Hitachi ETG-4000 | mM mm | `Bnci2014FnirsMA` |
| Shimadzu | mM cm | `Bak2019FnirsTapping` |

### Constructor

Call `super().__init__()` with dataset metadata. Key parameters:
- **`subjects`**: full list of subject IDs
- **`events`**: dict mapping event names to integer codes
- **`interval`**: `[tmin, tmax]` relative to stimulus onset; initially use `[-2, 15]`, then adjust it as described below. Recording-only datasets: `None`.
- **`paradigm`**: a BIDS-safe task family prefixed with `"fnirs"`, such as
  `"fnirsMotorExecution"`, `"fnirsMotorImagery"`, or `"fnirsNBack"`.

### Register the dataset

Add the dataset in these places:
- `fnirsatlas/datasets/__init__.py`
- `fnirsatlas/__init__.py`
- `croissant.json`: one `cr:FileObject` per download URL

Classification datasets also go in `BENCHMARK_DATASETS`, which paradigms use by default.

### `data_path()`

Returns the path to the data file for a given subject.

- Use MOABB's `data_dl()` to download and cache from a public URL.
    - Only use this path to download data. Do not download data manually.
    - BIDS/OpenNeuro: override `_download_subject()` with `download_openneuro_dataset_files()` instead. See `Carlton2026FnirsSqueezing`.
- The `sign` parameter determines the cache directory under `~/mne_data/`. Use self.code for this if possible.
- Use a marker file (not directory existence) to detect prior extraction. See `Bak2019FnirsTapping` or `BaseNemo2024Fnirs` for the pattern.
- For datasets distributed as archives, implement a `_download_data()` helper that downloads and extracts.

Verify this implementation before moving on by using it to download the full dataset.

```bash
uv run python -c "from fnirsatlas import DatasetClass; DatasetClass().get_data()"
```

- Make a copy of the raw zip before so you can test the final implementation without re-downloading the data.
- If the source data is more than 20GB, only use a partition for testing.

### `_get_single_subject_data()`

Loads all sessions for one subject. Must return:
```
{session_key: {run_key: mne.io.Raw}}
```

**For `BaseFNIRSDataset`**: Build `RawArray` objects using the base class helpers:
- `_make_raw(data, ch_names, ch_types, sfreq, montage)` — creates `RawArray`
- `_make_stim_raw(n_times, sfreq, samples, labels)` — creates stim channel
- `_populate_fnirs_metadata(raw, default_wavelengths=None)` — populates required fNIRS channel metadata
- `_standardize_raw(raw)` — standardizes event annotations and ensures a stim channel exists
- `_single_run_session(raw)` / `_add_run(sessions, key, raw)` — wraps in session/run dict
- Channel types: `"hbo"`, `"hbr"`, `"stim"` (or `"fnirs_cw_amplitude"` for raw intensity)
- Channel names: `"S{x}_D{y} hbo"` / `"S{x}_D{y} hbr"` convention

**For `BaseBIDSFNIRSDataset`**: Override `_get_single_subject_data()` only if you need custom annotation/stim handling. The base class reads SNIRF files automatically.

This part is where most hard parsing decisions are made, therefore use verbose
comment style, e.g.:

```python
# ...
# Each CSV starts with device metadata. The numeric table begins after
# the row whose first column is "Time".
skip_header = self._header_line(source) + 1
values = np.genfromtxt(source, delimiter=",", skip_header=skip_header)
if values.ndim != 2 or values.shape[1] < 2 + 2 * N_PAIRS:
    raise ValueError(f"Unexpected Liu2020 CSV shape {values.shape}: {source}")

# Start from first non-zero row.
values = self._drop_startup_samples(values)

# Skip Time/Marker, select the 204 pairs at two wavelengths, and
# transpose CSV (samples, channels) into MNE (channels, samples).
data = values[:, 2 : 2 + 2 * N_PAIRS].T

# The CSV only identifies CH1...CH204. MNE requires each wavelength
# channel to use "S#_D# wavelength" syntax, so assign one synthetic
# source-detector pair per exported channel. Fallback distances are populated below.
names = [
    f"S{pair}_D{pair} {wavelength}"
    for pair in range(1, N_PAIRS + 1)
    for wavelength in WAVELENGTHS
]
# ...
```

### Montage

If the dataset does not have sensor locations, ask the user to create a montage based on the paper description of it.

Multiple montages:
- Across subjects or runs: one dataset class per montage. MOABB stacks all subjects into one array. See `Liu2020Fnirs*`.
- Within the same run: check whether merging is possible.

## 4. Test

- Use real data for testing.
- Do not skip any for the dataset you are adding, run even manual and data requiring tests.

Create `tests/test_<dataset>.py` covering:
1. Init properties (subject list, paradigm, events, interval, code)
2. Data structure (`get_data()` returns `{subject: {session: {run: Raw}}}`)
3. Raw properties (sampling frequency, channel counts/types/names)
4. Stim channel (events present, correct labels, expected counts)
5. All subjects load (parametrize across subjects)

See `./../../tests/test_luke2021_fnirs_speech_dataset.py` for a complete example.

If adding the dataset exposes a general issue in the package, make a test and fix for it with mock data and commit the fix before the dataset.

## 5. Dataset validation

###  Determine the epoch interval

For classification datasets, once the dataset loads with the temporary
`[-2, 15]` interval, run:

```bash
uv run python scripts/check_event_intervals.py <DatasetClass>
```

The script reports the interval distribution, corresponding `tmax` values, and how many consecutive intervals would overlap each epoch length.

Alternatively, pass `--interval TMIN TMAX` to report only the trials excluded by a candidate interval. Choose the tradeoff between epoch length and overlap, rounding `tmax` down when needed for safety. Document the choice as a comment.

```bash
uv run python scripts/check_event_intervals.py <DatasetClass> --interval -2 9.9
```

Mark intervals at the top of the file with a comment:
```python
# Selected events are at least 20 seconds apart in the source data.
INTERVAL = [-2, 15]
```

### Automated checks

1. Slow-band median outside the plausible range: unit error or spikes (`Bonnal2025FnirsME`).
`uv run python scripts/check_hb_scale.py <DatasetClass>`

2. `OUTSIDE` distance from head origin, or AUC near 0.5: wrong channel positions.
`uv run python scripts/check_montage.py <DatasetClass>`

3. Band-pass cutoffs: data already filtered. Document it.
`uv run python scripts/check_processed_data.py <DatasetClass>`

4. Bad data: compare with the paper.
`uv run python scripts/check_dataset_quality.py <DatasetClass>`
`flat_bad > 0`: constant channels, e.g. disconnected optodes. Check that the loader does not create them.
`dup > 0`: runs with identical data. Exclude the duplicate runs in the loader and document the pairs (e.g. `Klein2026FnirsMotion`).

5. `METADATA` mismatches with the dataset and its first recording.
`uv run --extra data python scripts/check_metadata.py <DatasetClass>`

Report these to the user at the end.

## 6. Smoke test

Skip this benchmark for recording-only datasets.

```bash
uv run python scripts/run_fnirs_benchmark.py \
  --datasets <dataset.code> --evaluations CrossSubject
```

If accuracy is far below the paper, check in order:
1. Stim samples vs source events.
2. Source events vs raw-file triggers.
3. Label source.
4. Step 5 checks.
5. Paper details again.

## 7. Double check

- Double check that the dataset's tests match the example test file approximately.

## 8. Report

- Dataset basics. What was the objective of the study? How are we using it?
    - So e.g. the objective was to test BCI with clinical vs healthy population, we only use healthy. Or the study was aimed at studying dyads but we only use the recordings to study finger tapping.
- If something differs from expectations.
- Exclusion decisions should be documented as comments, see `../../fnirsatlas/datasets/klein2026_fnirs_motion.py` for an example.
- Smoke check results (vs expectations).
- Interval selection logic.
- Automated checks outputs.

## 9. Feedback

- If this document could be improved in some way, suggest.
    - Obvious fixes, typos, etc.
    - More instructions needed.
    - Contradictions.
    - Do remember that this document should be as compact as possible.
