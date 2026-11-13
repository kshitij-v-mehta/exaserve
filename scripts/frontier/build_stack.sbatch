#!/bin/bash -l
# Build the Frontier ExaServe runtime archive on one compute node.
#   sbatch -A <account> -p extended -q debug \
#     --export=ALL,FRONTIER_STACK_CONFIG=/abs/frontier_stack.conf \
#     scripts/frontier/build_stack.sbatch
#SBATCH -J exaserve-stack
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -C nvme
#SBATCH -o %x-%j.out
set -euo pipefail
: "${FRONTIER_STACK_CONFIG:?export FRONTIER_STACK_CONFIG=/abs/frontier_stack.conf}"
source "${FRONTIER_STACK_CONFIG}"
exec bash "${EXASERVE_SOURCE_DIR}/scripts/frontier/build_stack.sh" "${FRONTIER_STACK_CONFIG}"
