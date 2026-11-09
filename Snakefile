"""Benchmark workflow: one job per paradigm, evaluation, dataset, pipeline and seed."""

from pathlib import Path

from fnirsatlas.benchmark import (
    PARADIGMS,
    load_pipeline_file,
    results_dir,
    select_datasets,
)
from fnirsatlas.results import result_path


configfile: "config/benchmark.yaml"


PIPELINES = {Path(path).stem: path for path in config["pipelines"]}
EVALUATIONS = config["evaluations"]
SEEDS = [str(seed) for seed in config["seeds"]]
RESULTS = Path(config["results"] or results_dir())
OUTPUT = Path(config["output"] or RESULTS / "benchmark")
THREADS = int(config["threads_per_job"])
# Declared pipeline parameters are fitted directly instead of searched.
NO_GRID = "--no-grid" if config.get("no_grid") else ""
LOG_DIR = Path("logs/snakemake")

CELL = str(
    result_path("{paradigm}", "{evaluation}", "{dataset}", "{pipeline}", "{seed}", RESULTS)
)
STAMP = str(RESULTS / "snakemake/{paradigm}/{dataset}.done")


def _pipeline_paradigms(pipeline, path):
    """Paradigms a pipeline definition declares, limited to the configured selection."""
    declared = {name for cfg in load_pipeline_file(path) for name in cfg["paradigms"]}
    selected = config.get("paradigms")
    if selected is None:
        return sorted(declared)
    supported = sorted(declared & set(selected))
    if not supported:
        raise ValueError(f"{pipeline} supports none of the selected paradigms")
    return supported


def _paradigm_datasets(paradigm):
    """Dataset codes of one paradigm, limited to the configured selection."""
    selected = select_datasets(PARADIGMS[paradigm](), config.get("datasets"))
    return [dataset.code for dataset in selected]


PIPELINE_PARADIGMS = {
    pipeline: _pipeline_paradigms(pipeline, path) for pipeline, path in PIPELINES.items()
}
PARADIGM_DATASETS = {
    paradigm: _paradigm_datasets(paradigm)
    for paradigms in PIPELINE_PARADIGMS.values()
    for paradigm in paradigms
}
if unknown := set(config.get("datasets") or ()) - {
    code for codes in PARADIGM_DATASETS.values() for code in codes
}:
    raise ValueError(f"Datasets in no selected paradigm: {sorted(unknown)}")

CELL_FILES = [
    CELL.format(
        paradigm=paradigm,
        evaluation=evaluation,
        dataset=dataset,
        pipeline=pipeline,
        seed=seed,
    )
    for pipeline, paradigms in PIPELINE_PARADIGMS.items()
    for paradigm in paradigms
    for dataset in PARADIGM_DATASETS[paradigm]
    for evaluation in EVALUATIONS
    for seed in SEEDS
]
if not CELL_FILES:
    raise ValueError("No evaluation cells selected")


wildcard_constraints:
    paradigm="|".join(PARADIGM_DATASETS),
    evaluation="|".join(EVALUATIONS),
    dataset="|".join({code for codes in PARADIGM_DATASETS.values() for code in codes}),
    pipeline="|".join(PIPELINES),
    seed=r"\d+",


rule all:
    input:
        CELL_FILES,


# Fill the MOABB epoch cache once per dataset and workflow run so parallel cells
# do not race on it. The stamp is temporary, so each run re-checks the cache.
rule warm_cache:
    output:
        temp(touch(STAMP)),
    threads: THREADS
    log:
        str(LOG_DIR / "warm__{paradigm}__{dataset}.log"),
    shell:
        """
        python scripts/warm_cache.py \
            --paradigm {wildcards.paradigm} \
            --dataset {wildcards.dataset} > {log} 2>&1
        """


rule run_cell:
    input:
        STAMP,
    output:
        CELL,
    params:
        pipeline=lambda wildcards: PIPELINES[wildcards.pipeline],
        output=lambda wildcards: str(
            OUTPUT
            / wildcards.paradigm
            / wildcards.evaluation
            / wildcards.dataset
            / wildcards.pipeline
            / f"seed-{wildcards.seed}"
        ),
    threads: THREADS
    log:
        str(LOG_DIR / "cell__{paradigm}__{evaluation}__{dataset}__{pipeline}__{seed}.log"),
    shell:
        """
        python scripts/run_fnirs_benchmark.py \
            --paradigms {wildcards.paradigm} \
            --evaluations {wildcards.evaluation} \
            --datasets {wildcards.dataset} \
            --pipelines {params.pipeline} \
            --seed {wildcards.seed} \
            --suffix {wildcards.dataset}_{wildcards.pipeline} \
            --results {RESULTS} \
            --output {params.output} \
            --n-jobs {threads} {NO_GRID} > {log} 2>&1
        """
