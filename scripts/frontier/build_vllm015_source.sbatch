#!/usr/bin/env bash
#SBATCH -J vllm015-gfx90a
#SBATCH -p extended
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -C nvme
#SBATCH --gpus-per-node=1
#SBATCH --network=disable_rdzv_get
#SBATCH -o vllm015-build-%j.out
#SBATCH -e vllm015-build-%j.err

set -euo pipefail

: "${FRONTIER_STACK_CONFIG:?submit with --export=ALL,FRONTIER_STACK_CONFIG=/absolute/path/frontier_stack.conf}"
# shellcheck disable=SC1090
source "${FRONTIER_STACK_CONFIG}"
bash "${EXASERVE_SOURCE_DIR}/scripts/frontier/build_vllm015_source.sh" \
    "${FRONTIER_STACK_CONFIG}"
bash "${EXASERVE_SOURCE_DIR}/scripts/frontier/complete_vllm015_stack.sh" \
    "${FRONTIER_STACK_CONFIG}"
