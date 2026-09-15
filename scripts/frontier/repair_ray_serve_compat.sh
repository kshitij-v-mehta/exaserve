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
    echo "Run this repair on a networked Frontier login node." >&2
    exit 2
fi

PYTHON="${EXASERVE_FRONTIER_VENV}/bin/python3"
UV="${EXASERVE_FRONTIER_VENV}/bin/uv"
if [[ ! -x "${PYTHON}" || ! -x "${UV}" ]]; then
    echo "Installed Frontier environment is incomplete: ${EXASERVE_FRONTIER_VENV}" >&2
    exit 2
fi

export UV_NO_MANAGED_PYTHON=1
export UV_NO_PYTHON_DOWNLOADS=1
export UV_LINK_MODE=copy
export UV_CACHE_DIR="${EXASERVE_STACK_ROOT}/uv-cache"
unset UV_OFFLINE PIP_NO_INDEX

"${UV}" pip install \
    --python "${PYTHON}" \
    --reinstall \
    "fastapi==${EXPECTED_FASTAPI_VERSION:-0.136.0}"
"${UV}" pip check --python "${PYTHON}"

"${PYTHON}" -I - <<'PY'
from importlib.metadata import version
from pathlib import Path

expected_ray = "2.53.0"
observed_ray = version("ray")
if observed_ray != expected_ray:
    raise SystemExit(f"Ray compatibility repair requires {expected_ray}, found {observed_ray}")

import ray.serve._private.config as serve_config

source = Path(serve_config.__file__).resolve()
text = source.read_text(encoding="utf-8")
legacy = "field.label == FieldDescriptor.LABEL_REPEATED"
replacement = "field.is_repeated"
legacy_count = text.count(legacy)
replacement_count = text.count(replacement)
if legacy_count == 1 and replacement_count == 0:
    source.write_text(text.replace(legacy, replacement), encoding="utf-8")
    print(f"Patched Ray Serve protobuf compatibility: {source}")
elif legacy_count == 0 and replacement_count == 1:
    print(f"Ray Serve protobuf compatibility is already patched: {source}")
else:
    raise SystemExit(
        "Refusing ambiguous Ray source repair: "
        f"legacy={legacy_count}, replacement={replacement_count}, source={source}"
    )
PY

bash "${EXASERVE_SOURCE_DIR}/scripts/frontier/reinstall_exaserve_noneditable.sh" "${CONFIG}"

"${PYTHON}" -I - <<'PY'
from importlib.metadata import version

from fastapi import FastAPI
from ray.cloudpickle import dumps
from ray.serve._private.config import DeploymentConfig

expected_fastapi = "0.136.0"
observed_fastapi = version("fastapi")
if observed_fastapi != expected_fastapi:
    raise SystemExit(
        f"FastAPI repair failed: expected {expected_fastapi}, found {observed_fastapi}"
    )
dumps(FastAPI())

config = DeploymentConfig()
round_trip = DeploymentConfig.from_proto(config.to_proto())
if not isinstance(round_trip, DeploymentConfig):
    raise SystemExit("Ray Serve DeploymentConfig protobuf round trip returned wrong type")
print(
    f"FastAPI {observed_fastapi} serialization and Ray Serve protobuf "
    f"round trip passed with protobuf {version('protobuf')}"
)
PY

bash "${EXASERVE_SOURCE_DIR}/scripts/frontier/pack_python_runtime.sh" "${CONFIG}"
echo "Ray Serve dependency repair, semantic preflight, and runtime repack completed"
