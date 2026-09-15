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
    echo "Run this downloader on a Frontier login node with network access." >&2
    exit 2
fi

MODEL_ID="ibm-granite/granite-3.3-8b-instruct"
MODEL_REVISION="51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"
MODEL_PATH="${EXASERVE_MODEL_STORAGE_PATH}/ibm-granite--granite-3.3-8b-instruct"
HF_BIN="${EXASERVE_FRONTIER_VENV}/bin/hf"

if [[ ! -x "${HF_BIN}" ]]; then
    echo "Hugging Face CLI is missing: ${HF_BIN}" >&2
    exit 2
fi
if [[ -f "${MODEL_PATH}/.exaserve_complete.json" ]]; then
    echo "Model is already staged: ${MODEL_PATH}"
    exit 0
fi
if [[ -e "${MODEL_PATH}" ]]; then
    echo "Refusing to overwrite incomplete model directory: ${MODEL_PATH}" >&2
    exit 2
fi

mkdir -p "${EXASERVE_MODEL_STORAGE_PATH}" "${EXASERVE_STACK_ROOT}/huggingface-cache"
DOWNLOAD_DIR="$(mktemp -d "${EXASERVE_MODEL_STORAGE_PATH}/.granite33-8b-download.XXXXXX")"
cleanup_download() {
    rm -rf -- "${DOWNLOAD_DIR}"
}
trap cleanup_download EXIT

export HF_HOME="${EXASERVE_STACK_ROOT}/huggingface-cache"
export HF_HUB_DISABLE_TELEMETRY=1
"${HF_BIN}" download "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --local-dir "${DOWNLOAD_DIR}"

if [[ ! -s "${DOWNLOAD_DIR}/config.json" || ! -s "${DOWNLOAD_DIR}/tokenizer.json" ]]; then
    echo "Downloaded model is missing its configuration or tokenizer." >&2
    exit 2
fi
if ! find "${DOWNLOAD_DIR}" -maxdepth 1 -type f -name '*.safetensors' -size +0c -print -quit | grep -q .; then
    echo "Downloaded model has no nonempty safetensors weights." >&2
    exit 2
fi

"${EXASERVE_FRONTIER_VENV}/bin/python3" - "${DOWNLOAD_DIR}" \
    "${MODEL_ID}" "${MODEL_REVISION}" <<'PY'
import sys
from pathlib import Path

from exaserve.model_staging import write_completion_marker

write_completion_marker(
    Path(sys.argv[1]),
    source_identity=f"hf:{sys.argv[2]}@{sys.argv[3]}",
)
PY

mv "${DOWNLOAD_DIR}" "${MODEL_PATH}"
trap - EXIT
echo "Model staged: ${MODEL_PATH}"
du -sh "${MODEL_PATH}"
