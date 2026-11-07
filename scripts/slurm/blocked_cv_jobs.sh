#!/bin/bash
# Usage: bash scripts/slurm/blocked_cv_jobs.sh -d A,B [blocked_cv_single_sbatch.job.sh args]
# Submits one blocked_cv_single_sbatch.job.sh job per dataset.
set -euo pipefail
[[ ${1:-} == -d ]] || { echo "First arg must be -d A,B" >&2; exit 1; }
IFS=',' read -r -a datasets <<< "$2"
shift 2
for dataset in "${datasets[@]}"; do
    bash scripts/slurm/blocked_cv_single_sbatch.job.sh -d "$dataset" "$@"
done
