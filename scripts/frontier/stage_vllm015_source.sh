#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-${FRONTIER_STACK_CONFIG:-}}"
if [[ -z "${CONFIG}" || ! -f "${CONFIG}" ]]; then
    echo "Usage: $0 /absolute/path/frontier_stack.conf" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${CONFIG}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this staging script on a Frontier login node, not inside Slurm." >&2
    exit 2
fi

module reset
module load "PrgEnv-gnu/${EXASERVE_FRONTIER_PRGENV_VERSION}"
module load "cpe/${EXASERVE_FRONTIER_CPE_VERSION}"
module load "miniforge3/${EXASERVE_FRONTIER_MINIFORGE_VERSION}"
module load "rocm/${EXASERVE_FRONTIER_ROCM_VERSION}"
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

mkdir -p "${EXASERVE_STACK_ROOT}/src" "${VLLM_BUILD_LOG_DIR}"

mkdir -p "$(dirname "${HAPROXY_SOURCE_TARBALL}")"
if [[ ! -f "${HAPROXY_SOURCE_TARBALL}" ]]; then
    HAPROXY_MINOR="${HAPROXY_VERSION%.*}"
    curl --proto '=https' --tlsv1.2 -fsSL \
        -o "${HAPROXY_SOURCE_TARBALL}" \
        "https://www.haproxy.org/download/${HAPROXY_MINOR}/src/haproxy-${HAPROXY_VERSION}.tar.gz"
fi
echo "${HAPROXY_SOURCE_SHA256}  ${HAPROXY_SOURCE_TARBALL}" | sha256sum -c -

if [[ -e "${VLLM_SOURCE_DIR}" && ! -d "${VLLM_SOURCE_DIR}/.git" ]]; then
    echo "Refusing to overwrite non-git path: ${VLLM_SOURCE_DIR}" >&2
    exit 2
fi
if [[ ! -d "${VLLM_SOURCE_DIR}/.git" ]]; then
    git clone --branch "v${VLLM_VERSION}" --depth 1 --recurse-submodules \
        https://github.com/vllm-project/vllm.git "${VLLM_SOURCE_DIR}"
fi
git -C "${VLLM_SOURCE_DIR}" fetch --tags --force origin "v${VLLM_VERSION}"
git -C "${VLLM_SOURCE_DIR}" checkout --detach "v${VLLM_VERSION}"
git -C "${VLLM_SOURCE_DIR}" submodule sync --recursive
git -C "${VLLM_SOURCE_DIR}" submodule update --init --recursive

# vLLM's CMake configuration otherwise downloads this repository during the
# build. Frontier compute nodes cannot reach GitHub, so stage the exact tag now.
if [[ -e "${TRITON_KERNELS_REPO_DIR}" && ! -d "${TRITON_KERNELS_REPO_DIR}/.git" ]]; then
    echo "Refusing to overwrite non-git path: ${TRITON_KERNELS_REPO_DIR}" >&2
    exit 2
fi
if [[ ! -d "${TRITON_KERNELS_REPO_DIR}/.git" ]]; then
    git clone --branch "${TRITON_KERNELS_TAG}" --depth 1 \
        https://github.com/triton-lang/triton.git "${TRITON_KERNELS_REPO_DIR}"
fi
git -C "${TRITON_KERNELS_REPO_DIR}" fetch --tags --force origin "${TRITON_KERNELS_TAG}"
git -C "${TRITON_KERNELS_REPO_DIR}" checkout --detach "${TRITON_KERNELS_TAG}"
if [[ ! -f "${TRITON_KERNELS_SRC_DIR}/__init__.py" ]]; then
    echo "Staged Triton kernels directory is invalid: ${TRITON_KERNELS_SRC_DIR}" >&2
    exit 2
fi

if [[ ! -x "${EXASERVE_FRONTIER_VENV}/bin/python3" ]]; then
    conda create -y -p "${EXASERVE_FRONTIER_VENV}" \
        "python=${PYTHON_VERSION}" uv ccache -c conda-forge
fi
if [[ ! -x "${EXASERVE_FRONTIER_VENV}/bin/uv" ]]; then
    conda install -y -p "${EXASERVE_FRONTIER_VENV}" uv ccache -c conda-forge
fi

UV="${EXASERVE_FRONTIER_VENV}/bin/uv"
PYTHON="${EXASERVE_FRONTIER_VENV}/bin/python3"
export UV_NO_MANAGED_PYTHON=1
export UV_NO_PYTHON_DOWNLOADS=1

"${UV}" pip install --python "${PYTHON}" \
    --index-url "${TORCH_ROCM_INDEX}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"

"${PYTHON}" - <<'PY'
import os
import torch

assert torch.__version__.split("+")[0] == os.environ["TORCH_VERSION"], torch.__version__
assert torch.version.hip and torch.version.hip.startswith(
    os.environ["EXPECTED_TORCH_ROCM_ABI"]
), torch.version.hip
print(f"staged torch={torch.__version__} hip={torch.version.hip}")
PY

cd "${VLLM_SOURCE_DIR}"
"${PYTHON}" use_existing_torch.py

# Resolve and install all build/runtime dependencies while the login node still
# has network access. The compute-node step is forced offline.
"${UV}" pip install --python "${PYTHON}" -r requirements/build.txt
"${UV}" pip install --python "${PYTHON}" -r requirements/rocm.txt
"${UV}" pip install --python "${PYTHON}" \
    "ray[serve]==${RAY_VERSION}"
"${UV}" pip install --python "${PYTHON}" \
    --reinstall --no-deps \
    "fastapi==${EXPECTED_FASTAPI_VERSION:-0.136.0}"

"${PYTHON}" - <<'PY'
import os
import torch

assert torch.__version__.split("+")[0] == os.environ["TORCH_VERSION"], torch.__version__
assert torch.version.hip and torch.version.hip.startswith(
    os.environ["EXPECTED_TORCH_ROCM_ABI"]
), torch.version.hip
print(f"dependency resolution retained torch={torch.__version__} hip={torch.version.hip}")
PY

# ROCm ships AMD SMI as source under /opt, which is read-only. Building it in
# place fails when setuptools tries to create amdsmi.egg-info, so build an exact
# copy in a writable temporary directory instead.
if ! "${PYTHON}" -c 'import amdsmi' >/dev/null 2>&1; then
    AMD_SMI_SOURCE="${ROCM_PATH:-/opt/rocm}/share/amd_smi"
    if [[ ! -d "${AMD_SMI_SOURCE}" ]]; then
        echo "ROCm AMD SMI source not found: ${AMD_SMI_SOURCE}" >&2
        exit 2
    fi
    AMD_SMI_BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/exaserve-amdsmi.XXXXXX")"
    cleanup_amdsmi_build() {
        rm -rf -- "${AMD_SMI_BUILD_DIR}"
    }
    trap cleanup_amdsmi_build EXIT
    cp -a "${AMD_SMI_SOURCE}/." "${AMD_SMI_BUILD_DIR}/"
    "${UV}" pip install --python "${PYTHON}" "${AMD_SMI_BUILD_DIR}"
    cleanup_amdsmi_build
    trap - EXIT
fi

"${UV}" pip install --python "${PYTHON}" \
    --no-deps --no-build-isolation --reinstall "${EXASERVE_SOURCE_DIR}"
"${UV}" pip check --python "${PYTHON}"

"${PYTHON}" - <<'PY'
import amdsmi

print(f"amdsmi={amdsmi.__file__}")
PY

git -C "${VLLM_SOURCE_DIR}" status --short
echo "Staged Triton kernels from $(git -C "${TRITON_KERNELS_REPO_DIR}" describe --tags --exact-match)."
echo "Login-node staging complete. Submit scripts/frontier/build_vllm015_source.sbatch."
