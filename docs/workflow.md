# Benchmark workflow

## Configuration

`config/benchmark.yaml` holds the grid:

```yaml
pipelines: [pipelines/Windowed_LogReg.yml]
datasets: [Bnci2014FnirsMA]   # remove to run every dataset of each paradigm
evaluations: [CrossSubject]
seeds: [0]
results: null     # the MOABB results directory (MOABB_RESULTS)
output: null      # <results>/benchmark
threads_per_job: 1
```

Any key is overridable on the command line:

```bash
uv run snakemake -n --cores 4 --config datasets='["Bnci2014FnirsMA"]' seeds='[1,2]'
```

## Rules

- `warm_cache` — loads one dataset once per paradigm, so parallel cells only read the MOABB cache.
- `run_cell` — the default target, one `scripts/run_fnirs_benchmark.py` run per cell, writing `<results>/results/<Paradigm>/<Evaluation>Evaluation/results_<dataset>_<pipeline>_seed-<seed>.hdf5`.

## Run

```bash
uv run snakemake -n --cores 1          # validate the DAG
uv run snakemake --cores 4
```

Rule output goes to `logs/snakemake/`.

## Run on Slurm

```bash
bash scripts/slurm/submit.sh --cpus 8 --memory 16G --time 02:00:00 \
    --datasets "Bnci2014FnirsMA" --pipelines "pipelines/Windowed_RF.yml"
```

Job output goes to `logs/slurm/`.

The script sources `config/cluster.env` if it exists, so a checkout can carry its own settings. See `config/cluster.env.example`.
