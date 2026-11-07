#!/bin/bash
# Usage: bash scripts/slurm/learning_curve_jobs.sh [options]
set -euo pipefail

account=${FNIRSATLAS_SLURM_ACCOUNT:-${SLURM_ACCOUNT:-}}
dry_run=0
datasets_arg=bak2019,shin2016__ma,cockx2023,guevara2024,shin2016__mi,shin2018__nback,nemo__empe,shin2018__wg,huang2021
models_arg=lda,ann,cnn2a,lstm,eegnet,dcn,aeon_multirockethydra,fnirst,fnirsnet23,fnirsnet24
start_seed=0
stop_seed=4

usage() {
    cat <<'EOF'
Usage: bash scripts/slurm/learning_curve_jobs.sh [options]
  -d, --datasets NAMES    Comma-separated datasets
  -m, --models NAMES      Comma-separated models
  --start-seed N          First seed (default: 0)
  --stop-seed N           Last seed (default: 4)
  -a, --account ACCOUNT   Slurm account
  --dry-run               Print jobs without submitting
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--account) account=$2; shift 2 ;;
        -d|--datasets) datasets_arg=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -m|--models) models_arg=$2; shift 2 ;;
        --start-seed) start_seed=$2; shift 2 ;;
        --stop-seed) stop_seed=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

IFS=',' read -r -a datasets <<< "$datasets_arg"
IFS=',' read -r -a models <<< "$models_arg"
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for seed in $(seq "$start_seed" "$stop_seed"); do
            flags=(-d "$dataset" -m "$model" --seed "$seed" -a "$account")
            [[ $dry_run -eq 1 ]] && flags+=(--dry-run)
            bash scripts/slurm/learning_curve_single_sbatch.job.sh "${flags[@]}"
        done
    done
done
