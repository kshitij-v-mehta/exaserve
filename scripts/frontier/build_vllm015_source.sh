#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-${FRONTIER_STACK_CONFIG:-}}"
if [[ -z "${CONFIG}" || ! -f "${CONFIG}" ]]; then
    echo "Usage: $0 /absolute/path/frontier_stack.conf" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${CONFIG}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "This script must run in a Frontier compute-node allocation." >&2
    exit 2
fi

module reset
module load "PrgEnv-gnu/${EXASERVE_FRONTIER_PRGENV_VERSION}"
module load "cpe/${EXASERVE_FRONTIER_CPE_VERSION}"
module load "miniforge3/${EXASERVE_FRONTIER_MINIFORGE_VERSION}"
module load "rocm/${EXASERVE_FRONTIER_ROCM_VERSION}"
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

# Do not pass unrelated Python installations from the submitting shell into
# vLLM's CMake configuration.
unset PYTHONPATH

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${EXASERVE_FRONTIER_VENV}"

export UV_NO_MANAGED_PYTHON=1
export UV_NO_PYTHON_DOWNLOADS=1
export UV_OFFLINE=1
export PIP_NO_INDEX=1
export PYTORCH_ROCM_ARCH
export VLLM_TARGET_DEVICE
export TRITON_KERNELS_SRC_DIR
export MAX_JOBS
export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"
export CC=gcc
export CXX=g++
export CCACHE_NOHASHDIR=true
export CCACHE_DIR="/mnt/bb/${USER}/vllm-ccache"
export TMPDIR="/mnt/bb/${USER}/vllm-build-${SLURM_JOB_ID}/tmp"
export UV_CACHE_DIR="/mnt/bb/${USER}/vllm-build-${SLURM_JOB_ID}/uv-cache"
export MIOPEN_USER_DB_PATH="/mnt/bb/${USER}/vllm-build-${SLURM_JOB_ID}/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${TMPDIR}" "${UV_CACHE_DIR}" "${CCACHE_DIR}" \
    "${MIOPEN_USER_DB_PATH}" "${VLLM_BUILD_LOG_DIR}"

python3 -I - <<'PY'
import os
import torch

assert torch.__version__.split("+")[0] == os.environ["TORCH_VERSION"], torch.__version__
assert torch.version.hip and torch.version.hip.startswith(
    os.environ["EXPECTED_TORCH_ROCM_ABI"]
), torch.version.hip
assert torch.cuda.is_available(), "ROCm GPU is not visible to PyTorch"
name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
print(f"torch={torch.__version__} hip={torch.version.hip}")
print(f"gpu={name} gcnArchName={getattr(props, 'gcnArchName', 'unknown')}")
PY

test -d "${VLLM_SOURCE_DIR}/.git"
test "$(git -C "${VLLM_SOURCE_DIR}" describe --tags --exact-match)" = "v${VLLM_VERSION}"
test -f "${TRITON_KERNELS_SRC_DIR}/__init__.py"
test "$(git -C "${TRITON_KERNELS_REPO_DIR}" describe --tags --exact-match)" = "${TRITON_KERNELS_TAG}"

cd "${VLLM_SOURCE_DIR}"
"${EXASERVE_FRONTIER_VENV}/bin/uv" pip install \
    --python "${EXASERVE_FRONTIER_VENV}/bin/python3" \
    --offline --no-deps --no-build-isolation --reinstall .

python3 -I - <<'PY'
import importlib.util
import sysconfig
from importlib.metadata import version
from pathlib import Path

assert version("vllm").startswith("0.15."), version("vllm")
site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
spec = importlib.util.find_spec("vllm")
assert spec is not None and spec.origin is not None
module_path = Path(spec.origin).resolve()
assert site_packages in module_path.parents, (site_packages, module_path)
print(f"vllm={version('vllm')} module={module_path}")
print("vLLM is installed non-editably in site-packages")
PY

"${EXASERVE_FRONTIER_VENV}/bin/uv" pip check \
    --python "${EXASERVE_FRONTIER_VENV}/bin/python3"
echo "vLLM ${VLLM_VERSION} non-editable source build completed successfully."
