import os
from pathlib import Path
from fnirsatlas.config import get_results_cache_dir
from fnirsatlas.utils import get_processed_data_path, get_raw_data_path

configfile: "fnirsatlas/configs/quick_run_snakemake.yaml"

DATASETS = [dataset.lower() for dataset in config["datasets"]]
MODELS = config["models"]
SEEDS = config["seeds"]
# Classification setting -> search strategy.
SETTINGS = config["settings"]
N_JOBS = int(config.get("n_jobs", 1))
LOG_LEVEL = config.get("log_level", "INFO")
LOG_DIR = Path("logs/snakemake")
RAW_OUTPUTS = {
    dataset: str(get_raw_data_path(dataset)) for dataset in DATASETS
}
RAW_ROOT = Path(
    os.path.commonpath([str(Path(raw_output).parent) for raw_output in RAW_OUTPUTS.values()])
)
DATASET_SUBPATHS = {
    dataset: Path(get_raw_data_path(dataset)).relative_to(RAW_ROOT).as_posix()
    for dataset in DATASETS
}
SUBPATH_TO_DATASET = {subpath: dataset for dataset, subpath in DATASET_SUBPATHS.items()}
PROCESSED_DIR = Path(get_processed_data_path(DATASETS[0])).parent
RESULT_PATTERN = str(
    get_results_cache_dir() / "{setting}" / "{search_strategy}" / "{model}" / "{dataset}" / "{seed}" / "results.json"
)


rule all:
    input:
        expand(".snakemake/display/{setting}.txt", setting=SETTINGS)


rule download_dataset:
    output:
        raw_dataset=directory(str(RAW_ROOT / "{dataset_subpath}"))
    wildcard_constraints:
        dataset_subpath=".+"
    params:
        dataset=lambda wildcards: SUBPATH_TO_DATASET[wildcards.dataset_subpath],
        log_level=LOG_LEVEL
    log:
        str(LOG_DIR / "download__{dataset_subpath}.log")
    shell:
        "FNIRSATLAS_LOG_LEVEL={params.log_level} uv run python scripts/download_dataset.py -d {params.dataset} > {log} 2>&1"


rule process_dataset:
    input:
        raw_dataset=lambda wildcards: str(get_raw_data_path(wildcards.dataset))
    output:
        processed_dataset=str(PROCESSED_DIR / "{dataset}_processed.npz")
    params:
        log_level=LOG_LEVEL
    log:
        str(LOG_DIR / "process__{dataset}.log")
    shell:
        "FNIRSATLAS_LOG_LEVEL={params.log_level} uv run python scripts/process_dataset.py -d {wildcards.dataset} --override > {log} 2>&1"


rule classify_dataset:
    input:
        processed_dataset=lambda wildcards: str(get_processed_data_path(wildcards.dataset))
    output:
        RESULT_PATTERN
    params:
        log_level=LOG_LEVEL,
        n_jobs=N_JOBS
    log:
        str(LOG_DIR / "classify__{setting}__{search_strategy}__{model}__{dataset}__{seed}.log")
    shell:
        "FNIRSATLAS_LOG_LEVEL={params.log_level} uv run python scripts/classify_dataset.py -d {wildcards.dataset} --model {wildcards.model} --seed {wildcards.seed} --n_jobs {params.n_jobs} --no_table --classification_setting {wildcards.setting} --search-strategy {wildcards.search_strategy} --override > {log} 2>&1"


rule display_results:
    input:
        result_json=lambda wildcards: expand(
            RESULT_PATTERN,
            setting=wildcards.setting,
            search_strategy=SETTINGS[wildcards.setting],
            model=MODELS,
            dataset=DATASETS,
            seed=SEEDS,
        )
    output:
        ".snakemake/display/{setting}.txt"
    log:
        str(LOG_DIR / "display__{setting}.log")
    params:
        datasets=" ".join(DATASETS),
        models=" ".join(MODELS),
        seeds=" ".join(str(seed) for seed in SEEDS),
        search_strategy=lambda wildcards: SETTINGS[wildcards.setting]
    shell:
        "FNIRSATLAS_LOG_LEVEL=WARNING uv run python scripts/display_results.py -d {params.datasets} -m {params.models} --seeds {params.seeds} -c {wildcards.setting} --metric {wildcards.setting}_acc --search-strategy {params.search_strategy} > {output} 2> {log}"
