#!/bin/bash
set -euo pipefail

account=${FNIRSATLAS_SLURM_ACCOUNT:-${SLURM_ACCOUNT:-}}
partition=${FNIRSATLAS_SLURM_PARTITION:-cpu}
# Defaults fit the small default run in fnirsatlas/configs/quick_run_snakemake.yaml.
cpus=16
mem=32G
time=00:15:00
gres=""
submit=sbatch

usage() {
    cat <<EOF
Usage: bash $0 [options]
Submits one job running the Snakemake workflow on one node.
Raise the resources for larger runs set in fnirsatlas/configs/quick_run_snakemake.yaml.
  --cpus N                CPUs, also Snakemake --cores (default: $cpus)
  --mem SIZE              Memory (default: $mem)
  --time HH:MM:SS         Time limit (default: $time)
  --use-gpu               Request a GPU (\$FNIRSATLAS_SLURM_GPU_GRES, default: gpu:1)
  -a, --account NAME      Slurm account (default: \$FNIRSATLAS_SLURM_ACCOUNT)
  -p, --partition NAME    Partition (default: $partition)
  --dry-run               Print the job instead of submitting
Example: bash $0 -a project_123 -p small --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpus) cpus=$2; shift 2 ;;
        --mem) mem=$2; shift 2 ;;
        --time) time=$2; shift 2 ;;
        --use-gpu) gres="#SBATCH --gres=${FNIRSATLAS_SLURM_GPU_GRES:-gpu:1}"; shift ;;
        -a|--account) account=$2; shift 2 ;;
        -p|--partition) partition=$2; shift 2 ;;
        --dry-run) submit=cat; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

mkdir -p logs/snakemake/jobs

$submit <<EOF
#!/bin/bash
#SBATCH --account=$account
#SBATCH --partition=$partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem=$mem
#SBATCH --time=$time
#SBATCH --job-name=fnirsatlas-snakemake
#SBATCH --output=logs/snakemake/jobs/%x_%j.out
$gres

set -euo pipefail
# --restart-times 3 retries download jobs that fail randomly
uv run --extra all snakemake --cores $cpus --rerun-incomplete --restart-times 3
for table in .snakemake/display/*.txt; do
    echo "== \$table"
    cat "\$table"
done
EOF
