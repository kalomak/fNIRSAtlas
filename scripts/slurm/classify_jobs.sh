#!/bin/bash
# Usage: ./scripts/slurm/classify_jobs.sh -m <model> [-d datasets] [-c cross|within] [--start-seed N] [--stop-seed N] [--use-grid 0|1] [--override] [--dry-run]
c="cross"
start_seed=0
stop_seed=0
use_grid=1
override=0
dry_run=0
use_gpu=0
account=${FNIRSATLAS_SLURM_ACCOUNT:-${SLURM_ACCOUNT:-}}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model) model=$2; shift 2 ;;
        -d|--datasets) datasets_arg=$2; shift 2 ;;
        -c) c=$2; shift 2 ;;
        --start-seed) start_seed=$2; shift 2 ;;
        --stop-seed) stop_seed=$2; shift 2 ;;
        --use-grid) use_grid=1; shift ;;
        --override) override=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --use-gpu) use_gpu=1; shift ;;
        -a|--account) account=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
n_splits=5
run_separate_seeds=$([ "$use_grid" -eq 1 ] && [ "$c" = "cross" ] && echo 1 || echo 0)

IFS=',' read -r -a datasets <<< "$datasets_arg"

for dataset in "${datasets[@]}"; do
    for seed in $(seq $start_seed $stop_seed); do
        base_flags="-m $model -d $dataset --seed $seed -c $c --use-grid $use_grid"
        base_flags="$base_flags --account $account"
        [ "$override" -eq 1 ] && base_flags="$base_flags --override"
        [ "$dry_run" -eq 1 ] && base_flags="$base_flags --dry-run"
        [ "$use_gpu" -eq 1 ] && base_flags="$base_flags --use-gpu"
        if [ "$run_separate_seeds" -eq 1 ]; then
            for split_ix in $(seq 0 $((n_splits - 1))); do
                ./scripts/slurm/classify_single_sbatch.job.sh $base_flags --split-ix $split_ix
            done
        else
            ./scripts/slurm/classify_single_sbatch.job.sh $base_flags
        fi
    done
done
