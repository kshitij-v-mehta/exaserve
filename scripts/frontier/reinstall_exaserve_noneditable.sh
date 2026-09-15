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
    echo "Run this repair on a Frontier login node." >&2
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
export UV_OFFLINE=1
export PIP_NO_INDEX=1

"${UV}" pip install \
    --python "${PYTHON}" \
    --offline \
    --no-deps \
    --no-build-isolation \
    --reinstall \
    "${EXASERVE_SOURCE_DIR}"

"${PYTHON}" -I - <<'PY'
import importlib.util
import sysconfig
from pathlib import Path

site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
spec = importlib.util.find_spec("exaserve")
if spec is None or spec.origin is None:
    raise SystemExit("ExaServe is not importable after installation")
module_path = Path(spec.origin).resolve()
if site_packages not in module_path.parents:
    raise SystemExit(f"ExaServe still resolves outside site-packages: {module_path}")
print(f"exaserve: {module_path}")
print("ExaServe is installed non-editably in site-packages")
PY
