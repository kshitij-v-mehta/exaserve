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
    echo "Run this smoke test inside a Frontier compute-node allocation." >&2
    exit 2
fi

if [[ "${EXASERVE_SMOLLM2_WORKER:-0}" != "1" ]]; then
    echo "Starting one-GCD offline SmolLM2 generation through Slurm."
    exec srun --overlap \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=7 \
        --gpus-per-task=1 \
        --gpu-bind=closest \
        env EXASERVE_SMOLLM2_WORKER=1 \
        bash "$0" "${CONFIG}"
fi

if [[ ! -f "${SMOLLM2_MODEL_PATH}/model.safetensors" ]]; then
    echo "Shared model is missing: ${SMOLLM2_MODEL_PATH}" >&2
    exit 2
fi
if [[ ! -s "${EXASERVE_PYTHON_RUNTIME_ARCHIVE}" ]]; then
    echo "Node-local Python runtime archive is missing:" >&2
    echo "  ${EXASERVE_PYTHON_RUNTIME_ARCHIVE}" >&2
    echo "Run scripts/frontier/pack_python_runtime.sh on a login node first." >&2
    exit 2
fi

export EXASERVE_SKIP_AUTO_RUNTIME_STAGE=1
# shellcheck disable=SC1090
source "${EXASERVE_SOURCE_DIR}/scripts/frontier/env_frontier.sh"
unset EXASERVE_SKIP_AUTO_RUNTIME_STAGE

STEP_TAG="${SLURM_STEP_ID:-manual}"
STEP_TAG="${STEP_TAG//[^A-Za-z0-9_.-]/_}"
LOCAL_ROOT="/mnt/bb/${USER}/exaserve-smollm2-${SLURM_JOB_ID}-${STEP_TAG}"
LOCAL_MODEL_PATH="${LOCAL_ROOT}/HuggingFaceTB--SmolLM2-360M-Instruct"
LOCAL_SITE_PACKAGES="${LOCAL_ROOT}/python/site-packages"
cleanup_local_smoke() {
    rm -rf -- "${LOCAL_ROOT}"
}
trap cleanup_local_smoke EXIT
mkdir -p "${LOCAL_MODEL_PATH}" "${LOCAL_SITE_PACKAGES}" "${LOCAL_ROOT}/cache"

STAGE_START="${SECONDS}"
echo "Staging Python packages onto node-local NVMe."
tar --extract \
    --file "${EXASERVE_PYTHON_RUNTIME_ARCHIVE}" \
    --directory "${LOCAL_SITE_PACKAGES}"
echo "Python package staging completed in $((SECONDS - STAGE_START)) seconds."

MODEL_STAGE_START="${SECONDS}"
echo "Staging model onto node-local NVMe."
cp --archive --sparse=always "${SMOLLM2_MODEL_PATH}/." "${LOCAL_MODEL_PATH}/"
echo "Model staging completed in $((SECONDS - MODEL_STAGE_START)) seconds."

export SMOLLM2_LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH}"
export EXASERVE_LOCAL_SITE_PACKAGES="${LOCAL_SITE_PACKAGES}"
export PYTHONPATH="${LOCAL_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="${LOCAL_ROOT}/cache/pycache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT="${LOCAL_ROOT}/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_ROOT}/cache/torchinductor"
export TRITON_CACHE_DIR="${LOCAL_ROOT}/cache/triton"
export XDG_CACHE_HOME="${LOCAL_ROOT}/cache/xdg"
mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}"

echo "Starting Python with node-local site-packages."
python3 "${EXASERVE_SOURCE_DIR}/scripts/frontier/smoke_vllm_smollm2.py"
