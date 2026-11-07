## Configuration

Default settings are in [`fnirsatlas/configs/config.yaml`](../fnirsatlas/configs/config.yaml). To change them, write a YAML file containing only the keys to change:

```yaml
paths:
  raw_datasets: "/data/fnirs/raw"
  processed_datasets: "/data/fnirs/processed"
  results_cache: "/data/fnirs/results"
datasets:
  herff2014:
    path: "/data/Herff2014"
preprocessing:
  l_freq: 0.01
```

Set `FNIRSATLAS_CONFIG` to the path of this file:

```bash
FNIRSATLAS_CONFIG=/path/to/my_config.yaml python my_script.py
```

## Developer reference

Default configuration ships with the package in `fnirsatlas/configs/`. `FNIRSATLAS_CONFIG` overrides `config.yaml` only. Task definitions and parameter grids are changed by editing their files.

- `config.yaml`: main settings
- `classification_tasks.yaml`: label mappings per dataset and task
- `param_grids.yaml`: hyperparameter grids for model search
- `downloads.yaml`: dataset download sources, see `docs/downloads.md`
- `best_params_*.yaml`: selected hyperparameters, written by `scripts/best_params.py`
- `blocked_cv_gap*.yaml`: override files for blocked cross-validation
- `quick_run_snakemake.yaml`: Snakemake workflow settings

### config.yaml

- **paths**: data and output locations
  - `raw_datasets`, `processed_datasets`: dataset locations
  - `results_cache`: reusable computational results such as classifier outputs
  - `results`: generated plots and tables
- **datasets**: per-dataset settings
  - `subpath`: location inside `paths.raw_datasets`
  - `path`: full dataset location, used instead of `subpath`
  - `description`: short text
  - `original_data_type`: e.g., `hbo_hbr`, `fnirs_cw_amplitude`, `fnirs_od`, `fnirs_fd_phase__fnirs_fd_ac_amplitude`
  - Optional per-dataset overrides, e.g. `tmax`, `preprocessing`, task-specific subsections (e.g., `afim`)
- **preprocessing**: defaults applied during parsing and epoching
  - `sci_threshold`, `ch_interpolation`, `tddr`, `l_freq`, `h_freq`, `l_trans_bandwidth`, `h_trans_bandwidth`, `filter_length`, `filter_kwargs`
  - `bll_ppf`, `remove_short`, `short_ch_regression`, `tmin`, `tmax`, `reject_criteria`, `allow_overlaps`, `verbose`
- **parser**: `keep_stim_channel`, `montage_type`
- **cv**: `cross` and `within` splitters, as a fully qualified class and constructor arguments, with nested CV under `inner`
- **paper**: `comparison_models`
- **task_selection**: `drop_if_present`
- **dataloader**: PyTorch dataloader settings per device

### classification_tasks.yaml

```yaml
shin2019:
  classification_tasks:
    ma:
      - description: "BL|MA"
        mapping:
          Idle State: 0
          Mental Arithmetic: 1
```

### param_grids.yaml

Search spaces keyed by model name. Double underscores address pipeline steps.

```yaml
lda:
  extractor__n_windows: [3, 6]
  extractor__features: [["mean"], ["mean", "slope"]]
eegnet:
  clf__F1: [4, 8]
  clf__D: [2, 4]
```
