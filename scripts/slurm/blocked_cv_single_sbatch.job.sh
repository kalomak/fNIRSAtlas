#!/bin/bash
set -euo pipefail

account=${FNIRSATLAS_SLURM_ACCOUNT:-${SLURM_ACCOUNT:-}}
partition=${FNIRSATLAS_SLURM_PARTITION:-cpu}
models=(lda)
datasets=(bnci)
seeds=(0)
cvs=(shuffled blocked_gap0 blocked_gap1)
submit=sbatch

usage() {
    cat <<EOF
Usage: bash $0 [options]
Submits one job running scripts/checks/blocked_cv.py for every model, dataset,
seed and CV variant.
  -m, --models A,B        Models (default: ${models[*]})
  -d, --datasets A,B      Datasets (default: ${datasets[*]})
  -s, --seeds 0,1         Seeds (default: ${seeds[*]})
  -c, --cvs A,B           CV variants (default: ${cvs[*]})
  -a, --account NAME      Slurm account (default: \$FNIRSATLAS_SLURM_ACCOUNT)
  -p, --partition NAME    Partition (default: $partition)
  --dry-run               Print the job instead of submitting
Example: bash $0 -m lda,ann -d bnci,yi2025 -s 0 --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models) IFS=',' read -r -a models <<< "$2"; shift 2 ;;
        -d|--datasets) IFS=',' read -r -a datasets <<< "$2"; shift 2 ;;
        -s|--seeds) IFS=',' read -r -a seeds <<< "$2"; shift 2 ;;
        -c|--cvs) IFS=',' read -r -a cvs <<< "$2"; shift 2 ;;
        -a|--account) account=$2; shift 2 ;;
        -p|--partition) partition=$2; shift 2 ;;
        --dry-run) submit=cat; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# Minutes per CV variant, about 1.5-2x seed 0 runtimes on all datasets.
declare -A model_minutes=([lda]=1 [ann]=1 [cnn2a]=4 [lstm]=10 [eegnet]=5 [dcn]=8 [aeon_multirockethydra]=1 [fnirst]=60 [fnirsnet23]=4 [fnirsnet24]=6)
declare -A dataset_mult=([yi2025]=4 [luke2024b]=4 [gao2024]=4 [shin2016__ma]=2 [shin2016__mi]=2 [shin2018__wg]=2 [buccino2016]=2 [cockx2023]=2)
minutes=0
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        minutes=$((minutes + ${model_minutes[$model]} * ${dataset_mult[$dataset]:-1} * ${#seeds[@]} * ${#cvs[@]}))
    done
done
time=$(printf "%02d:%02d:00" "$((minutes / 60))" "$((minutes % 60))")

# Single value as is, multiple as count: lda, 23d.
label() { local suffix=$1; shift; [[ $# -eq 1 ]] && echo "$1" || echo "$#$suffix"; }
job_name="bcv_$(label m "${models[@]}")_$(label d "${datasets[@]}")_$(label s "${seeds[@]/#/s}")"
mkdir -p logs/blocked_cv/jobs
echo "Running ${job_name} for ${time} (1 CPU, 3G)"

$submit <<EOF
#!/bin/bash
#SBATCH --account=$account
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=$time
#SBATCH --job-name=$job_name
#SBATCH --output=logs/blocked_cv/jobs/%x_%j.out

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
failed=()
for model in ${models[*]}; do
    for dataset in ${datasets[*]}; do
        for seed in ${seeds[*]}; do
            for cv in ${cvs[*]}; do
                log=logs/blocked_cv/\$model/\$dataset/\$seed/\$cv.out
                mkdir -p "\$(dirname "\$log")"
                echo "\$(date +%T) \$model \$dataset \$seed \$cv"
                srun python scripts/checks/blocked_cv.py \
                    -m "\$model" -d "\$dataset" --seeds "\$seed" --cv "\$cv" > "\$log" 2>&1 \
                    || failed+=("\$model/\$dataset/\$seed/\$cv")
            done
        done
    done
done
echo "\$(date +%T) done. Failed: \${failed[*]:-none}"
[[ \${#failed[@]} -eq 0 ]]
EOF
