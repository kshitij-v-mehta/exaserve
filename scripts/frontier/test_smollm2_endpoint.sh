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
    echo "Run this endpoint test on a Frontier login node." >&2
    exit 2
fi

export EXASERVE_SITE_ID="olcf-frontier"
export EXASERVE_SCHEDULER="slurm"
export EXASERVE_PROJECT_ACCOUNT="${PROJECT_ID}"
export EXASERVE_PROJECT_ROOT="${WORK_ROOT}"
export EXASERVE_MODEL_STORAGE_PATH
export EXASERVE_LOCAL_STAGE_PATH
export EXASERVE_SUBMISSION_REGISTRY="${WORK_ROOT}/submission-registry"

JOB_ID_FILE="${WORK_ROOT}/smollm2.jobid"
if [[ ! -s "${JOB_ID_FILE}" ]]; then
    echo "Job ID file is missing: ${JOB_ID_FILE}" >&2
    exit 2
fi

URL="$("${EXASERVE_FRONTIER_VENV}/bin/exaserve-serve-url" \
    "${JOB_ID_FILE}" --wait --timeout 1800)"
echo "Endpoint: ${URL}"

curl --fail-with-body --silent --show-error \
    "${URL%/}/chat/completions" \
    --header "Content-Type: application/json" \
    --data @- <<'JSON'
{
  "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "In one sentence, name the U.S. national laboratory that operates Frontier."
    }
  ],
  "temperature": 0,
  "max_tokens": 48,
  "stream": false
}
JSON
echo
