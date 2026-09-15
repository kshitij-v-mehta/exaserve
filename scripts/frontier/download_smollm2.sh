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

HF_BIN="${EXASERVE_FRONTIER_VENV}/bin/hf"
if [[ ! -x "${HF_BIN}" ]]; then
    echo "Hugging Face CLI is missing: ${HF_BIN}" >&2
    exit 2
fi

if [[ -f "${SMOLLM2_MODEL_PATH}/config.json" && \
      -f "${SMOLLM2_MODEL_PATH}/model.safetensors" && \
      -f "${SMOLLM2_MODEL_PATH}/tokenizer.json" ]]; then
    echo "Model is already staged: ${SMOLLM2_MODEL_PATH}"
    exit 0
fi
if [[ -e "${SMOLLM2_MODEL_PATH}" ]]; then
    echo "Refusing to overwrite incomplete model directory: ${SMOLLM2_MODEL_PATH}" >&2
    exit 2
fi

mkdir -p "${EXASERVE_MODEL_STORAGE_PATH}" "${EXASERVE_STACK_ROOT}/huggingface-cache"
DOWNLOAD_DIR="$(mktemp -d "${EXASERVE_MODEL_STORAGE_PATH}/.smollm2-download.XXXXXX")"
cleanup_download() {
    rm -rf -- "${DOWNLOAD_DIR}"
}
trap cleanup_download EXIT

export HF_HOME="${EXASERVE_STACK_ROOT}/huggingface-cache"
export HF_HUB_DISABLE_TELEMETRY=1
"${HF_BIN}" download "${SMOLLM2_MODEL_ID}" \
    config.json \
    generation_config.json \
    merges.txt \
    model.safetensors \
    special_tokens_map.json \
    tokenizer.json \
    tokenizer_config.json \
    vocab.json \
    --revision "${SMOLLM2_MODEL_REVISION}" \
    --local-dir "${DOWNLOAD_DIR}"

for required_file in config.json model.safetensors tokenizer.json tokenizer_config.json; do
    if [[ ! -s "${DOWNLOAD_DIR}/${required_file}" ]]; then
        echo "Downloaded model is missing ${required_file}" >&2
        exit 2
    fi
done

"${EXASERVE_FRONTIER_VENV}/bin/python3" - "${DOWNLOAD_DIR}" \
    "${SMOLLM2_MODEL_ID}" "${SMOLLM2_MODEL_REVISION}" <<'PY'
import sys
from pathlib import Path

from exaserve.model_staging import write_completion_marker

model_path = Path(sys.argv[1])
write_completion_marker(
    model_path,
    source_identity=f"hf:{sys.argv[2]}@{sys.argv[3]}",
)
PY

mv "${DOWNLOAD_DIR}" "${SMOLLM2_MODEL_PATH}"
trap - EXIT
echo "Model staged: ${SMOLLM2_MODEL_PATH}"
du -sh "${SMOLLM2_MODEL_PATH}"
