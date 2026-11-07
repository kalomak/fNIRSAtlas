#!/bin/bash
set -euo pipefail

account=${FNIRSATLAS_SLURM_ACCOUNT:-${SLURM_ACCOUNT:-}}
partition=${FNIRSATLAS_SLURM_PARTITION:-cpu}
dry_run=0

usage() {
    cat <<EOF
Usage: bash $0 -d DATASET -m MODEL --seed N [--dry-run]
Runs all outer folds in one job.
Example: bash $0 -d buccino2016 -m lda --seed 0 --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--account) account=$2; shift 2 ;;
        -d|--dataset) dataset=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -m|--model) model=$2; shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

: "${dataset:?Missing --dataset}"
: "${model:?Missing --model}"
: "${seed:?Missing --seed}"

declare -A model_minutes=([lda]=5 [ann]=20 [cnn2a]=5 [lstm]=20 [eegnet]=10 [dcn]=10 [aeon_multirockethydra]=10 [fnirst]=120 [fnirsnet23]=10 [fnirsnet24]=40)
declare -A model_mem=([ann]=2G [cnn2a]=2G [aeon_multirockethydra]=10G)
declare -A dataset_mult=([yi2025]=4 [luke2024b]=4 [gao2024]=4 [shin2016__ma]=2 [shin2016__mi]=2 [shin2018__wg]=2 [buccino2016]=2 [cockx2023]=2)
# Per-fold minutes times 5 outer folds.
minutes=$((${model_minutes[$model]} * ${dataset_mult[$dataset]:-1} * 5))
time=$(printf "%02d:%02d:00" "$((minutes / 60))" "$((minutes % 60))")
mem=${model_mem[$model]:-5G}
job_name="lc_${model}_${dataset}_${seed}"
output_path="logs/learning_curve/${model}/${dataset}/${seed}.out"
mkdir -p "$(dirname "$output_path")"

echo "Running ${job_name} for ${time} (1 CPU, ${mem})"
submit=sbatch
if [[ $dry_run -eq 1 ]]; then
    submit=cat
fi

$submit <<EOF
#!/bin/bash
#SBATCH --account=$account
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=$mem
#SBATCH --time=$time
#SBATCH --job-name=$job_name
#SBATCH --output=$output_path

module purge
if [ -n "${FNIRSATLAS_SLURM_MODULE:-}" ]; then
    module load "${FNIRSATLAS_SLURM_MODULE:-}"
fi
if [ -n "${FNIRSATLAS_SLURM_ACTIVATE:-}" ]; then
    source "${FNIRSATLAS_SLURM_ACTIVATE:-}"
else
    source .venv/bin/activate
fi
export FNIRSATLAS_LOG_LEVEL=INFO
time srun python scripts/learning_curve.py \
    --dataset "$dataset" --model "$model" --seed "$seed"
EOF
