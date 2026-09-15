#!/bin/bash
# Submit with: sbatch -A PROJECT --export=ALL,FRONTIER_STACK_CONFIG=/absolute/path/frontier_stack.conf scripts/frontier/preflight_stack.sbatch
#SBATCH --job-name=exaserve-preflight
#SBATCH --partition=extended
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --constraint=nvme
#SBATCH --network=disable_rdzv_get
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --output=exaserve-preflight-%j.out
#SBATCH --error=exaserve-preflight-%j.err

set -euo pipefail
if [[ -z "${FRONTIER_STACK_CONFIG:-}" ]]; then
    echo "FRONTIER_STACK_CONFIG is required" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${FRONTIER_STACK_CONFIG}"
export EXPECTED_GPU_COUNT=1

bash "${EXASERVE_SOURCE_DIR}/scripts/frontier/probe_stack.sh"
srun --ntasks=1 --gpus-per-task=1 --gpu-bind=closest \
    bash "${EXASERVE_SOURCE_DIR}/scripts/frontier/verify_stack.sh" "${FRONTIER_STACK_CONFIG}"
