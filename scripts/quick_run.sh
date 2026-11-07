#!/usr/bin/env bash
set -euo pipefail

SMOKE_ROOT="/tmp/fnirsatlas-smoke"
LOG_PATH="$SMOKE_ROOT/smoke.log"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-$HOME/mne_data/fnirsatlas/datasets}"
PROCESSED_DATA_ROOT="${PROCESSED_DATA_ROOT:-$HOME/mne_data/fnirsatlas/processed_datasets}"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/mne_data/fnirsatlas/results}"
DELETE_RAW_DATA_ON_START="${DELETE_RAW_DATA_ON_START:-false}"
DELETE_PROCESSED_DATA_ON_START="${DELETE_PROCESSED_DATA_ON_START:-false}"
DELETE_RESULTS_ON_START="${DELETE_RESULTS_ON_START:-false}"
DATASETS=(bnci luke2021 bak2019)
RAW_DATA_DIRS=(BNCI Luke2021)
MODELS_STRING="${MODELS:-lda fnirsnet23}"
read -r -a MODELS <<< "$MODELS_STRING"

rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"
exec > >(tee "$LOG_PATH") 2>&1

export FNIRSATLAS_LOG_LEVEL=INFO

printf '%s\n' "Using smoke root: $SMOKE_ROOT"
printf '%s\n' "Models: ${MODELS[*]}"
printf '%s\n' "Delete raw data on start: $DELETE_RAW_DATA_ON_START"
printf '%s\n' "Delete processed data on start: $DELETE_PROCESSED_DATA_ON_START"
printf '%s\n' "Delete results on start: $DELETE_RESULTS_ON_START"

if command -v uv >/dev/null; then
    uv sync --extra process --extra classify --extra dl --extra cli --group dev
    PYTHON=(uv run python)
else
    PYTHON=(python)
fi

if [[ "$DELETE_RAW_DATA_ON_START" == "true" ]]; then
    for dataset_dir in "${RAW_DATA_DIRS[@]}"; do
        printf '%s\n' "Removing raw data: $RAW_DATA_ROOT/$dataset_dir"
        rm -rf "$RAW_DATA_ROOT/$dataset_dir"
    done
fi

if [[ "$DELETE_PROCESSED_DATA_ON_START" == "true" ]]; then
    for dataset in "${DATASETS[@]}"; do
        printf '%s\n' "Removing processed data: $PROCESSED_DATA_ROOT/${dataset}_processed.npz"
        rm -f "$PROCESSED_DATA_ROOT/${dataset}_processed.npz"
    done
fi

if [[ "$DELETE_RESULTS_ON_START" == "true" ]]; then
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            printf '%s\n' "Removing results: $RESULTS_ROOT/cross/none/$model/$dataset"
            rm -rf "$RESULTS_ROOT/cross/none/$model/$dataset"
        done
    done
fi

for model in "${MODELS[@]}"; do
    printf '%s\n' "Running model: $model"
    "${PYTHON[@]}" scripts/run_all.py \
        -d "${DATASETS[@]}" \
        --model "$model" \
        --n_jobs 1 \
        --no_table \
        --classification_setting cross \
        --search-strategy none
done

export FNIRSATLAS_LOG_LEVEL=WARNING

printf '%s\n' "Displaying results for models: ${MODELS[*]}"
"${PYTHON[@]}" scripts/display_results.py \
    -d "${DATASETS[@]}" \
    -m "${MODELS[@]}" \
    --seeds 42 \
    -c cross \
    --search-strategy none
