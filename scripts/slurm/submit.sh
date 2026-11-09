#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit.sh [options]

Submit the benchmark workflow as one Slurm job that runs the cells in parallel.

Options:
  --account ACCOUNT         Slurm account (default: $SLURM_ACCOUNT)
  --cpus CPUS               CPUs and concurrent cells (default: 8)
  --datasets CODES          Space-separated dataset codes
  --evaluations NAMES       Space-separated evaluation names
  -h, --help                Show this help text
  --memory MEMORY           Slurm memory request (default: 16G)
  --no-grid                 Fit declared pipeline parameters instead of searching param_grid
  --paradigms NAMES         Space-separated paradigm names
  --partition PARTITION     Slurm partition (default: $SLURM_PARTITION)
  --pipelines FILES         Space-separated pipeline definition files
  --seeds SEEDS             Space-separated evaluation seeds
  --time TIME               Slurm time limit (default: 01:00:00)
EOF
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)
# Optional per-checkout settings, e.g. SLURM_ACCOUNT.
env_file="$repo_dir/config/cluster.env"
if [[ -f $env_file ]]; then
    source "$env_file"
fi

yaml_list() {
    local item entries=()
    for item in $1; do
        entries+=("\"$item\"")
    done
    local IFS=,
    printf '[%s]' "${entries[*]}"
}

account=${SLURM_ACCOUNT:-}
cpus=8
memory=16G
partition=${SLURM_PARTITION:-}
time=01:00:00
no_grid=false
declare -A overrides=()
while (($#)); do
    case "$1" in
        --account) account=${2:?Missing value for --account}; shift 2 ;;
        --cpus) cpus=${2:?Missing value for --cpus}; shift 2 ;;
        --datasets) overrides[datasets]=${2:?Missing value for --datasets}; shift 2 ;;
        --evaluations) overrides[evaluations]=${2:?Missing value for --evaluations}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --memory) memory=${2:?Missing value for --memory}; shift 2 ;;
        --no-grid) no_grid=true; shift ;;
        --paradigms) overrides[paradigms]=${2:?Missing value for --paradigms}; shift 2 ;;
        --partition) partition=${2:?Missing value for --partition}; shift 2 ;;
        --pipelines) overrides[pipelines]=${2:?Missing value for --pipelines}; shift 2 ;;
        --seeds) overrides[seeds]=${2:?Missing value for --seeds}; shift 2 ;;
        --time) time=${2:?Missing value for --time}; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 1 ;;
    esac
done

: "${account:?Set --account or SLURM_ACCOUNT}"
: "${partition:?Set --partition or SLURM_PARTITION}"

config_args=()
for key in "${!overrides[@]}"; do
    config_args+=("$key=$(yaml_list "${overrides[$key]}")")
done
if [[ $no_grid == true ]]; then
    config_args+=("no_grid=True")
fi
config_q=""
if ((${#config_args[@]})); then
    printf -v config_q '%q ' "${config_args[@]}"
fi

mkdir -p "$repo_dir/logs/slurm"
sbatch <<EOF
#!/bin/bash
#SBATCH --account=$account
#SBATCH --partition=$partition
#SBATCH --time=$time
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem=$memory
#SBATCH --job-name=fnirsatlas-benchmark
#SBATCH --output=$repo_dir/logs/slurm/benchmark-%j.out

set -euo pipefail
export PATH="$repo_dir/.venv/bin:\$PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$repo_dir"
config_args=($config_q)
snakemake --cores "$cpus" --printshellcmds --rerun-incomplete --keep-going \
    \${config_args[@]+--config "\${config_args[@]}"}
EOF
