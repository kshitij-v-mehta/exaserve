#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-${FRONTIER_STACK_CONFIG:-}}"
if [[ -z "${CONFIG}" || ! -f "${CONFIG}" ]]; then
    echo "Usage: FRONTIER_STACK_CONFIG=/absolute/path/frontier_stack.conf $0" >&2
    echo "   or: $0 /absolute/path/frontier_stack.conf" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${CONFIG}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Run this endpoint test from a Frontier login node." >&2
    exit 2
fi

export EXASERVE_SITE_ID="olcf-frontier"
export EXASERVE_SCHEDULER="slurm"
export EXASERVE_PROJECT_ACCOUNT="${PROJECT_ID}"
export EXASERVE_PROJECT_ROOT="${WORK_ROOT}"
export EXASERVE_MODEL_STORAGE_PATH
export EXASERVE_LOCAL_STAGE_PATH
export EXASERVE_SUBMISSION_REGISTRY="${WORK_ROOT}/submission-registry"

JOB_ID_FILE="${WORK_ROOT}/granite33-8b.jobid"
if [[ ! -s "${JOB_ID_FILE}" ]]; then
    echo "Job ID file is missing: ${JOB_ID_FILE}" >&2
    exit 2
fi

URL="$("${EXASERVE_FRONTIER_VENV}/bin/exaserve-serve-url" \
    "${JOB_ID_FILE}" --wait --timeout 2400)"

exec "${EXASERVE_FRONTIER_VENV}/bin/python3" \
    "${EXASERVE_SOURCE_DIR}/scripts/frontier/test_granite33_8b_8n8g.py" \
    --url "${URL}" \
    --requests "${REQUESTS:-256}" \
    --concurrency "${CONCURRENCY:-64}" \
    --timeout "${REQUEST_TIMEOUT:-180}"
