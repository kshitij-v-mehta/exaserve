#!/bin/bash
# Source this file inside an allocated Frontier compute node.  The defaults
# match the qualified vLLM ROCm wheel path selected after the Frontier module
# probe. Override the *_MODULE_VERSION variables only as a coordinated stack
# change and retain the output of scripts/frontier/probe_stack.sh.

# Slurm may inherit an internally inconsistent Conda shell state.  Sanitize it
# before `module reset`, because unloading the login-node Miniforge module can
# otherwise invoke Conda's deactivate hook before the runtime setup begins.
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER
export CONDA_SHLVL=0

module reset
module load "PrgEnv-gnu/${EXASERVE_FRONTIER_PRGENV_VERSION:-8.7.0}"
module load "cpe/${EXASERVE_FRONTIER_CPE_VERSION:-26.03}"
module load "rocm/${EXASERVE_FRONTIER_ROCM_VERSION:-7.0.2}"
module load craype-accel-amd-gfx90a
module load rccl-net-plugin

# Slurm's AMD GPU binding may export ROCR_VISIBLE_DEVICES. Ray 2.53 rejects
# that variable and requires HIP_VISIBLE_DEVICES, although both masks describe
# the same visible GCD indices on Frontier.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    if [[ -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
        export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
    fi
    unset ROCR_VISIBLE_DEVICES
fi

if [[ -n "${CRAY_LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

if [[ -z "${EXASERVE_FRONTIER_VENV:-}" ]]; then
    echo "EXASERVE_FRONTIER_VENV must name the shared ExaServe virtual environment" >&2
    return 1 2>/dev/null || exit 1
fi
if [[ ! -x "${EXASERVE_FRONTIER_VENV}/bin/python3" ]]; then
    echo "ExaServe virtual environment is missing: ${EXASERVE_FRONTIER_VENV}" >&2
    return 1 2>/dev/null || exit 1
fi

# This prefix was created by conda, but runtime jobs need neither the Miniforge
# module nor conda's shell activation machinery.  In particular, Slurm can
# inherit CONDA_SHLVL without a matching CONDA_PREFIX from the submitting login
# shell; `conda activate` then crashes while trying to deactivate a nonexistent
# prior environment.
# Select the already-built interpreter directly and keep job startup entirely
# offline and independent of conda plugins/state.
export PATH="${EXASERVE_FRONTIER_VENV}/bin:${EXASERVE_STACK_ROOT}/bin:${PATH}"
hash -r

# A shared Python environment causes severe import-metadata latency on Orion.
# For the one-node Frontier validation deployment, publish site-packages from
# one sequential tar stream onto node-local NVMe before Python starts.
if [[ -n "${SLURM_JOB_ID:-}" && \
      "${EXASERVE_SKIP_AUTO_RUNTIME_STAGE:-0}" != "1" ]]; then
    if [[ -z "${EXASERVE_PYTHON_RUNTIME_ARCHIVE:-}" || \
          ! -s "${EXASERVE_PYTHON_RUNTIME_ARCHIVE}" ]]; then
        echo "ExaServe Python runtime archive is missing: ${EXASERVE_PYTHON_RUNTIME_ARCHIVE:-unset}" >&2
        return 1 2>/dev/null || exit 1
    fi

    EXASERVE_RUNTIME_NODE="${HOSTNAME:-unknown}"
    EXASERVE_RUNTIME_NODE="${EXASERVE_RUNTIME_NODE%%.*}"
    EXASERVE_RUNTIME_ROOT="/mnt/bb/${USER}/exaserve-python-runtime-${SLURM_JOB_ID}-${EXASERVE_RUNTIME_NODE}"
    EXASERVE_LOCAL_SITE_PACKAGES="${EXASERVE_RUNTIME_ROOT}/site-packages"
    EXASERVE_RUNTIME_ID="$(stat -c '%s:%Y' "${EXASERVE_PYTHON_RUNTIME_ARCHIVE}")"
    EXASERVE_RUNTIME_REUSE=0
    if [[ -f "${EXASERVE_RUNTIME_ROOT}/.archive-id" && \
          -f "${EXASERVE_LOCAL_SITE_PACKAGES}/vllm/__init__.py" && \
          -f "${EXASERVE_LOCAL_SITE_PACKAGES}/exaserve/__init__.py" ]]; then
        IFS= read -r EXASERVE_OBSERVED_RUNTIME_ID \
            < "${EXASERVE_RUNTIME_ROOT}/.archive-id" || true
        if [[ "${EXASERVE_OBSERVED_RUNTIME_ID}" == "${EXASERVE_RUNTIME_ID}" ]]; then
            EXASERVE_RUNTIME_REUSE=1
        fi
    fi

    if [[ "${EXASERVE_RUNTIME_REUSE}" != "1" ]]; then
        EXASERVE_RUNTIME_CANDIDATE="${EXASERVE_RUNTIME_ROOT}.tmp.${BASHPID}"
        rm -rf -- "${EXASERVE_RUNTIME_CANDIDATE}"
        mkdir -p "${EXASERVE_RUNTIME_CANDIDATE}/site-packages"
        EXASERVE_RUNTIME_STAGE_START="${SECONDS}"
        echo "Staging ExaServe Python packages onto node-local NVMe."
        if ! tar --extract \
            --file "${EXASERVE_PYTHON_RUNTIME_ARCHIVE}" \
            --directory "${EXASERVE_RUNTIME_CANDIDATE}/site-packages"; then
            rm -rf -- "${EXASERVE_RUNTIME_CANDIDATE}"
            echo "Failed to stage the ExaServe Python runtime" >&2
            return 1 2>/dev/null || exit 1
        fi
        printf '%s\n' "${EXASERVE_RUNTIME_ID}" \
            > "${EXASERVE_RUNTIME_CANDIDATE}/.archive-id"
        rm -rf -- "${EXASERVE_RUNTIME_ROOT}"
        mv "${EXASERVE_RUNTIME_CANDIDATE}" "${EXASERVE_RUNTIME_ROOT}"
        echo "ExaServe Python staging completed in $((SECONDS - EXASERVE_RUNTIME_STAGE_START)) seconds."
    else
        echo "Reusing node-local ExaServe Python runtime: ${EXASERVE_RUNTIME_ROOT}"
    fi

    export EXASERVE_LOCAL_SITE_PACKAGES
    export PYTHONPATH="${EXASERVE_LOCAL_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
    export PYTHONNOUSERSITE=1
    export PYTHONPYCACHEPREFIX="${EXASERVE_RUNTIME_ROOT}/pycache"
    mkdir -p "${PYTHONPYCACHEPREFIX}"
fi

# The ROCm wheel/source build can carry a local version suffix.  ExaServe's
# fail-closed compatibility profile must record the exact installed metadata,
# not assume that the public release number has no suffix.
export EXASERVE_FRONTIER_PYTHON_VERSION="$(python3 -c 'import platform; print(platform.python_version())')"
export EXASERVE_FRONTIER_RAY_VERSION="$(python3 -c 'from importlib.metadata import version; print(version("ray"))')"
export EXASERVE_FRONTIER_VLLM_VERSION="$(python3 -c 'from importlib.metadata import version; print(version("vllm"))')"
EXASERVE_OBSERVED_FASTAPI_VERSION="$(python3 -c 'from importlib.metadata import version; print(version("fastapi"))')"
if [[ "${EXASERVE_OBSERVED_FASTAPI_VERSION}" != "${EXPECTED_FASTAPI_VERSION:-0.136.0}" ]]; then
    echo "FastAPI ${EXASERVE_OBSERVED_FASTAPI_VERSION} is incompatible with Ray Serve 2.53; expected ${EXPECTED_FASTAPI_VERSION:-0.136.0}" >&2
    return 1 2>/dev/null || exit 1
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn0,hsn1,hsn2,hsn3}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-/tmp/${USER}/miopen-${SLURM_JOB_ID:-interactive}}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-${MIOPEN_USER_DB_PATH}}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
