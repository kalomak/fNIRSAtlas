# Preprocessing

## Steps

### Dataset standardization

- Standardizes annotations.
- Selects known events.
- Creates a stimulus channel when absent.
- Applies dataset-specific format corrections.

### Signal preprocessing

`preprocess_fnirs` constructs these ordered steps. A step that does not apply
to the channel type returns the data unchanged, and a `None` threshold
disables a check.

1. Invalid value check: Mark non-finite samples, and CW amplitude samples at or
   below zero, as bad per channel.
2. Flat channel check: Mark CW channel windows of `quality_window=10.0` s as
   bad when the amplitude standard deviation divided by its absolute mean is
below `flat_threshold=1e-6`, e.g. a disconnected optode.
3. SNR check: Mark CW channel windows of `quality_window` as bad when the
   amplitude mean divided by its standard deviation is at most
`snr_threshold=10.0`.
4. Convert continuous-wave amplitude to optical density when needed.
5. Remove dataset-configured short channels.
6. SCI check: Mark optical-density windows of `quality_window` with scalp
   coupling index below `sci_threshold=0.5` as bad. SCI and PSP use the
   0.5-2.5 Hz cardiac band (30-150 bpm) of Pollonini et al. 2016. Below
   5.6 Hz sampling, the upper edge is lowered to Nyquist minus 0.3 Hz. Below
   3.6 Hz sampling, both checks are skipped with a warning.
7. PSP check: Mark optical-density windows of `quality_window` with cardiac
   peak spectral power below `psp_threshold=0.02` as bad. PSP follows
   Pollonini et al. 2016 (PHOEBE), with the peak searched within the cardiac
   band. The threshold is chosen empirically and is lenient compared with
   PHOEBE's 0.1. On the benchmark datasets, 0.1 also rejects windows with a
   clear pulse, while 0.02 rejects mostly noise and motion that the other
   checks miss. TDDR corrects moderate motion spikes later anyway.
8. Apply TDDR.
9. Convert optical density to HbO/HbR using PPF 6. Missing source-detector
   distances produce a warning and use a 3 cm fallback.
10. Apply the paradigm's order-3 Butterworth IIR filter. Exactly one band is
    supported. The default is 0.01-0.1 Hz.
11. Resample to the paradigm's target rate.
12. Mark windows of `quality_window` with standard deviation above `max_std`
    as bad. Disabled by default.

### Epoching

1. Creates epochs.
2. Drops epochs where more than `max_bad_fraction=0.5` of the fNIRS channels
   are bad and fewer than `min_good_pairs=8` source-detector pairs have no bad
   channel. The defaults are very lenient to keep as much data as possible.
   A channel is bad in an epoch when it is in `info["bads"]` or a
   channel-specific `BAD_` annotation of its source-detector group overlaps
   the epoch.
3. Interpolates bad channels from the good channels of the same wavelength or
   chromophore. The default `interpolate_bads=interpolate_dist` averages them
   with weights `softmax(-d / temperature)` over the distances `d` between
   channel midpoints, with `temperature=0.005` m.
   `interpolate_bads=interpolate_nearest` copies the closest one.
   `interpolate_bads=interpolate_zero` sets bad channels to zero.
   `interpolate_bads=None` skips interpolation and keeps the channels bad.
4. Applies baseline correction.
5. Crops epochs.

### Array conversion

Without `return_epochs`:

1. Converts epochs to an array of shape
   `(n_epochs, n_chromophores, n_source_detector_groups, n_times)`, HbO first.
2. Multiplies the array by the dataset's `unit_factor`, by default `1e6`.

## Configuration

Preprocessing options and their defaults are in
`fnirsatlas.paradigms.fnirs.PREPROCESSING`. Pass the ones to change:

```python
from fnirsatlas.datasets.preprocessing import interpolate_nearest
from fnirsatlas.paradigms import FNIRSClassification

paradigm = FNIRSClassification(
    preprocessing={"sci_threshold": 0.3, "interpolate_bads": interpolate_nearest},
)
```

Override `_get_raw_pipelines` in a custom paradigm when a different signal
preprocessing pipeline is required.

### Dataset specific configuration

A dataset can override preprocessing options for its own data:

```python
class Luke2021FnirsSpeech(BaseFNIRSDataset):
    preprocessing_options = {"short_channel_threshold": 0.02}
```

## Data quality table

```bash
uv run python scripts/check_dataset_quality.py Luke2021FnirsTapping Bak2019FnirsTapping
```

Prints one row per dataset with the percent of data each check above marks bad, using the paradigm's default thresholds. Per-channel metrics are cached in `<results>/quality/`, and `--force` recomputes them.
