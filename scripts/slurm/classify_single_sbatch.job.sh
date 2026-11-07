#!/bin/bash
seed=42
c="cross"
split_ix=""
use_grid=1
override=0
dry_run=0
use_gpu=0
n_jobs=0
account=${FNIRSATLAS_SLURM_ACCOUNT:-${SLURM_ACCOUNT:-}}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model) model=$2; shift 2 ;;
        -d|--dataset) dataset=$2; shift 2 ;;
        --seed) seed=$2; shift 2 ;;
        -c) c=$2; shift 2 ;;
        --split-ix) split_ix=$2; shift 2 ;;
        --use-grid) use_grid=$2; shift 2 ;;
        --override) override=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --use-gpu) use_gpu=1; shift ;;
        -j|--n-jobs) n_jobs=$2; shift 2 ;;
        -a|--account) account=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cap_time() {
    local t=$1
    local max_secs=259200
    IFS=: read -r h m s <<< "$t"
    local secs=$(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))
    [ "$secs" -gt "$max_secs" ] && secs=$max_secs
    printf "%02d:%02d:%02d" $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}

min_time() {
    local t=$1
    local min_secs=3600
    IFS=: read -r h m s <<< "$t"
    local secs=$(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))
    [ "$secs" -lt "$min_secs" ] && secs=$min_secs
    printf "%02d:%02d:%02d" $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}

scale_time() {
    local current_time=$1
    local numerator=$2
    local denominator=${3:-1}
    IFS=: read -r h m s <<< "$current_time"
    local total_seconds=$(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))
    local scaled=$(( total_seconds * numerator / denominator ))
    printf "%02d:%02d:%02d" $((scaled/3600)) $(((scaled%3600)/60)) $((scaled%60))
}

if [ "$n_jobs" -eq 0 ]; then
    declare -A default_jobs=([fnirst]=5 [simplet]=5 [dcn]=5 [fnirsnet24]=5 [lstm]=5)
    n_jobs=${default_jobs[$model]:-1}
fi
partition=${FNIRSATLAS_SLURM_PARTITION:-cpu}
cpus=$n_jobs
gres=""
declare -A model_time=([eegnet]="06:00:00" [dcn]="04:00:00" [cnn2a]="06:00:00" [lda]="00:30:00" [logreg]="00:30:00" [svm]="00:30:00" [rf]="00:30:00" [knn]="00:30:00" [aeon_multirockethydra]="01:00:00" [fnirst]="20:00:00" [simplet]="02:00:00" [fnirsnet23]="03:00:00" [fnirsnet24]="04:00:00" [lstm]="08:00:00")
 declare -A model_mem=([ann]="2G" [eegnet]="5G" [dcn]="5G" [lda]="5G" [logreg]="5G" [svm]="5G" [rf]="5G" [knn]="5G" [aeon_multirockethydra]="10G" [fnirst]="5G" [simplet]="5G" [fnirsnet23]="5G" [fnirsnet24]="5G" [lstm]="2G" [cnn2a]="2G")
declare -A dataset_mult=([yi2025]=4 [shin2018__wg]=2 [shin2016__ma]=2 [shin2016__mi]=2 [luke2024b]=4 [gao2024]=4 [buccino2016]=2 [cockx2023]=2)
declare -A c_mult=([cross]=1 [within]=2)
time=${model_time[$model]:-${model_time[eegnet]}}
mem=${model_mem[$model]:-5G}
if [ "$c" = "within" ] && [[ "$mem" =~ ^([0-9]+)G$ ]]; then
    mem_gb=$((BASH_REMATCH[1] * 2))
        [ "$mem_gb" -gt 8 ] && mem_gb=8
    mem="${mem_gb}G"
fi
time=$(scale_time "$time" "${dataset_mult[$dataset]:-1}")
time=$(scale_time "$time" "${c_mult[$c]:-1}")

if [ "$use_gpu" -eq 1 ]; then
    partition=${FNIRSATLAS_SLURM_GPU_PARTITION:-gpu}
    cpus=4
    mem="16G"
    gres="#SBATCH --gres=${FNIRSATLAS_SLURM_GPU_GRES:-gpu:1}"
    n_jobs=1
    time=$(scale_time "$time" 1 20)
    time=$(min_time "$time")
fi

time=$(cap_time "$time")

if [ "$use_grid" -eq 1 ]; then
    if [ "$c" = "within" ]; then
        search_strategy="block"
    else
        search_strategy="nested"
    fi
else
    search_strategy="none"
fi

job_name="${c}_${model}_${dataset}_${seed}"
output_path="logs/classify/${c}/${model}/${dataset}/${seed}"
if [ -n "$split_ix" ]; then
    job_name="${job_name}_split${split_ix}"
    output_path="${output_path}_split${split_ix}"
fi
output_path="${output_path}.out"

flags="-d ${dataset} -m ${model} -c ${c} --seed ${seed} --search-strategy ${search_strategy} -j ${n_jobs}"
if [ -n "$split_ix" ]; then
    flags="${flags} --split-ix ${split_ix}"
fi
if [ "$override" -eq 1 ]; then
    flags="${flags} --override"
fi
echo "Running scripts/classify_dataset.py ${flags} for ${time} (${cpus} cpus, ${mem}) as job ${job_name} with output ${output_path}"

submit_cmd=sbatch
if [ "$dry_run" -eq 1 ]; then
    echo "Dry run: printing job script without submitting"
    submit_cmd=cat
fi

$submit_cmd <<EOT
#!/bin/bash
#SBATCH --account=$account
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem=$mem
#SBATCH --time=$time
#SBATCH -J ${job_name}
#SBATCH --output=${output_path}
$gres

module purge
if [ -n "${FNIRSATLAS_SLURM_MODULE:-}" ]; then
    module load "$FNIRSATLAS_SLURM_MODULE"
fi
if [ -n "${FNIRSATLAS_SLURM_ACTIVATE:-}" ]; then
    source "$FNIRSATLAS_SLURM_ACTIVATE"
else
    source .venv/bin/activate
fi
export FNIRSATLAS_LOG_LEVEL=INFO

time srun python scripts/classify_dataset.py ${flags}

EOT
