#!/bin/bash
set -euo pipefail

echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host=$(hostname)"
echo "slurm_job_id=${SLURM_JOB_ID:-none}"
echo "os=$(cat /etc/os-release | tr '\n' ' ')"
echo "glibc=$(getconf GNU_LIBC_VERSION)"
echo "kernel=$(uname -r)"

for name in PrgEnv-gnu cpe miniforge3 rocm rccl-net-plugin craype-accel-amd-gfx90a; do
    echo "--- module spider ${name}"
    module spider "${name}" 2>&1 || true
done

echo "--- loaded modules"
module -t list 2>&1 || true
for command_name in python3 cc hipcc rocminfo srun sbatch; do
    printf '%s=' "${command_name}"
    command -v "${command_name}" || true
done

if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -E 'Name:.*gfx|Marketing Name' | head -24 || true
fi
