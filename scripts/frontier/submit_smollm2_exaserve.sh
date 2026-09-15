#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dry-run}"
CONFIG="${2:-${FRONTIER_STACK_CONFIG:-}}"
if [[ -z "${CONFIG}" || ! -f "${CONFIG}" ]]; then
    echo "Usage: FRONTIER_STACK_CONFIG=/absolute/path/frontier_stack.conf $0 [mode]" >&2
    echo "   or: $0 [mode] /absolute/path/frontier_stack.conf" >&2
    exit 2
fi

# shellcheck disable=SC1090
source "${CONFIG}"
DEPLOYMENT_CONFIG="${EXASERVE_DEPLOYMENT_CONFIG:-${EXASERVE_SOURCE_DIR}/examples/config.frontier.smollm2.yaml}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit ExaServe from a Frontier login node, not from inside an allocation." >&2
    exit 2
fi
if [[ ! -f "${SMOLLM2_MODEL_PATH}/.exaserve_complete.json" ]]; then
    echo "The staged model is incomplete: ${SMOLLM2_MODEL_PATH}" >&2
    echo "Run scripts/frontier/download_smollm2.sh on a login node first." >&2
    exit 2
fi
if [[ ! -s "${EXASERVE_PYTHON_RUNTIME_ARCHIVE}" ]]; then
    echo "The packed Python runtime is missing: ${EXASERVE_PYTHON_RUNTIME_ARCHIVE}" >&2
    echo "Run scripts/frontier/pack_python_runtime.sh on a login node first." >&2
    exit 2
fi

export EXASERVE_SITE_ID="olcf-frontier"
export EXASERVE_SCHEDULER="slurm"
export EXASERVE_PROJECT_ACCOUNT="${PROJECT_ID}"
export EXASERVE_PROJECT_ROOT="${WORK_ROOT}"
export EXASERVE_MODEL_STORAGE_PATH
export EXASERVE_LOCAL_STAGE_PATH
export EXASERVE_SOURCE_ENV_SCRIPT="${EXASERVE_SOURCE_DIR}/scripts/frontier/env_frontier.sh"
export EXASERVE_SLURM_CONSTRAINT="nvme"
export EXASERVE_SLURM_NETWORK="disable_rdzv_get"
export EXASERVE_SLURM_QOS="debug"
export EXASERVE_SUBMISSION_REGISTRY="${WORK_ROOT}/submission-registry"
export EXASERVE_DEFAULT_QUEUE="extended"
export EXASERVE_DEFAULT_WALLTIME="00:30:00"
export EXASERVE_FRONTIER_PYTHON_VERSION="3.12.14"
export EXASERVE_FRONTIER_RAY_VERSION="${RAY_VERSION}"
export EXASERVE_FRONTIER_VLLM_VERSION="${EXPECTED_VLLM_VERSION}"

SUBMIT="${EXASERVE_FRONTIER_VENV}/bin/exaserve-serve-submit"
COMMON=(
    "${DEPLOYMENT_CONFIG}"
    --project-account "${PROJECT_ID}"
    --queue "extended"
    --walltime "00:30:00"
    --job-name "sm"
    --job-id-file "${WORK_ROOT}/smollm2.jobid"
    --log-dir "${WORK_ROOT}/smollm2-logs"
)

case "${MODE}" in
    dry-run)
        exec "${SUBMIT}" "${COMMON[@]}" --dry-run
        ;;
    submit)
        exec "${SUBMIT}" "${COMMON[@]}"
        ;;
    submit-wait)
        exec "${SUBMIT}" "${COMMON[@]}" --wait --timeout 1800
        ;;
    submit-new)
        exec "${SUBMIT}" "${COMMON[@]}" --new-generation
        ;;
    submit-new-wait)
        exec "${SUBMIT}" "${COMMON[@]}" --new-generation --wait --timeout 1800
        ;;
    *)
        echo "Usage: $0 [dry-run|submit|submit-wait|submit-new|submit-new-wait]" >&2
        exit 2
        ;;
esac
