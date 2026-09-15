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
    echo "Create the Python runtime archive on a Frontier login node." >&2
    exit 2
fi
if [[ ! -x "${EXASERVE_FRONTIER_VENV}/bin/python3" ]]; then
    echo "Python environment is missing: ${EXASERVE_FRONTIER_VENV}" >&2
    exit 2
fi

SITE_PACKAGES="$("${EXASERVE_FRONTIER_VENV}/bin/python3" -I -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
case "${SITE_PACKAGES}/" in
    "${EXASERVE_FRONTIER_VENV}/"*) ;;
    *)
        echo "Refusing to archive unexpected site-packages path: ${SITE_PACKAGES}" >&2
        exit 2
        ;;
esac
if [[ ! -d "${SITE_PACKAGES}" ]]; then
    echo "site-packages directory is missing: ${SITE_PACKAGES}" >&2
    exit 2
fi

"${EXASERVE_FRONTIER_VENV}/bin/python3" -I - "${SITE_PACKAGES}" <<'PY'
import importlib.util
import sys
from pathlib import Path

site_packages = Path(sys.argv[1]).resolve()
for package in ("vllm", "openai", "pydantic", "torch", "ray", "exaserve"):
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise SystemExit(f"installed package is not resolvable: {package}")
    origin = Path(spec.origin).resolve()
    if site_packages not in origin.parents:
        raise SystemExit(
            f"{package} resolves outside site-packages: {origin}; "
            "the runtime must be installed non-editably"
        )
    print(f"{package}: {origin}")

# Exercise the two Ray Serve compatibility boundaries that have broken this
# stack in released dependencies.  Version-only checks are insufficient: the
# protobuf source adaptation must work in the exact environment being packed.
from fastapi import FastAPI
from ray.cloudpickle import dumps
from ray.serve._private.config import DeploymentConfig

dumps(FastAPI())
round_trip = DeploymentConfig.from_proto(DeploymentConfig().to_proto())
if not isinstance(round_trip, DeploymentConfig):
    raise SystemExit("Ray Serve DeploymentConfig protobuf round trip returned wrong type")
print("Ray Serve FastAPI serialization and protobuf round trip passed")
PY

ARCHIVE_DIR="$(dirname "${EXASERVE_PYTHON_RUNTIME_ARCHIVE}")"
mkdir -p "${ARCHIVE_DIR}"
TEMP_ARCHIVE="$(mktemp "${ARCHIVE_DIR}/.python-runtime.XXXXXX")"
cleanup_archive() {
    rm -f -- "${TEMP_ARCHIVE}"
}
trap cleanup_archive EXIT

echo "Packing ${SITE_PACKAGES}"
echo "Source size: $(du -sh "${SITE_PACKAGES}" | awk '{print $1}')"
START_SECONDS="${SECONDS}"

# An uncompressed archive is intentional. Frontier reads one sequential file
# from Lustre and performs decompression-free extraction onto node-local NVMe.
tar --create \
    --file "${TEMP_ARCHIVE}" \
    --directory "${SITE_PACKAGES}" \
    .

mv "${TEMP_ARCHIVE}" "${EXASERVE_PYTHON_RUNTIME_ARCHIVE}"
trap - EXIT

echo "Runtime archive: ${EXASERVE_PYTHON_RUNTIME_ARCHIVE}"
echo "Archive size: $(du -sh "${EXASERVE_PYTHON_RUNTIME_ARCHIVE}" | awk '{print $1}')"
echo "Packing duration: $((SECONDS - START_SECONDS)) seconds"
