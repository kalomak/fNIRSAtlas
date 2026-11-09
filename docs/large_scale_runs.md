# Large-scale runs

For runs over many pipelines, datasets, evaluations, and seeds, we recommend the [Snakemake](https://snakemake.readthedocs.io/) workflow in `Snakefile`. It splits a run into one job per pipeline, dataset, evaluation, and seed, and runs the jobs in parallel. Completed jobs are skipped when the workflow is run again, and a failed job does not stop the others.

For a single run, use [`fnirs_benchmark`](benchmark.md) instead.

Install the workflow extra:

```bash
uv sync --extra workflow
```

## Configuration

`config/benchmark.yaml` selects what to run:

```yaml
pipelines: [pipelines/Windowed_LogReg.yml]
datasets: [Bnci2014FnirsMA]   # remove to run every dataset of each paradigm
evaluations: [CrossSubject]
seeds: [0]
results: null     # the MOABB results directory (MOABB_RESULTS)
output: null      # <results>/benchmark
threads_per_job: 1
```

Each pipeline runs under the paradigms its definition declares. Any key can be overridden on the command line:

```bash
uv run snakemake -n --cores 4 --config datasets='["Bnci2014FnirsMA"]' seeds='[1,2]'
```

## Run locally

```bash
uv run snakemake -n --cores 1   # list the jobs without running them
uv run snakemake --cores 4
```

Each dataset is loaded once before its jobs start, so parallel jobs only read the cached data. Results are written to `<results>/results/<Paradigm>/<Evaluation>Evaluation/`. Job logs go to `logs/snakemake/`.

## Run on Slurm

`scripts/slurm/submit.sh` submits the workflow as one Slurm job that runs the jobs in parallel on the allocated CPUs:

```bash
bash scripts/slurm/submit.sh --cpus 8 --memory 16G --time 02:00:00 \
    --datasets "Bnci2014FnirsMA" --pipelines "pipelines/Windowed_RF.yml"
```

The options override `config/benchmark.yaml`. See `bash scripts/slurm/submit.sh --help`. Slurm output goes to `logs/slurm/`.

Set the Slurm account and partition in `config/cluster.env`, which the script sources if it exists. Copy `config/cluster.env.example` to start.
