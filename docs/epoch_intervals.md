# Dataset Minimum Intervals

Minimum event intervals for supported datasets (important for setting `tmax` to avoid epoch overlap):

These are the minimum times between consecutive events, not epoch durations. They are used to choose a safe `tmax` per dataset. Datasets whose minimum interval is shorter than the default epoch use a dataset-specific `tmax` in `fnirsatlas/configs/config.yaml`. The epoch durations reported in the paper (Table 1, "Ep. dur.") are the resulting epoch lengths.

| Dataset      | Min Interval|
|-------------|--------------|
| bak2019     | 28.95s       |
| shin2016    | 27.40s       | 
| shin2019    | 29.25s       | 
| shin2018    | 28.20s       | 
| shin2018b   | 28.40s       | 
| luke2021    | 19.07s       | 
| luke2024a   | 14.98s       | 
| nemo        | 18.66s       | 
| bnci        | 20.00s       | 
| cockx2023   | 35.36s       | 
| shader2021  | 26.88s       | 
| herff2014   | 64.24s       | 
| ortega2020  | 37.92s       | 
| buccino2016 | 13.92s       | 
| gao2024     | 11.12s       | 
| luke2024b   | 7.06s        | 

luke2024b uses `tmax: 10` despite its 7.06s minimum: the few shorter-interval outliers are dropped.

**Default:** tmin = -2.0s, tmax = 15.0s (override in config if needed)

Override example in `config.yaml`:
```yaml
buccino2016:
  tmax: 11.8

luke2024a:
  tmax: 13
```

Check overlaps: `uv run python scripts/checks/check_overlap.py -d <dataset_name> -p`

#epoch, #length