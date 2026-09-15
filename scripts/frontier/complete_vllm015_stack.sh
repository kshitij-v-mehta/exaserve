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
    echo "Run this script inside a Frontier compute-node allocation." >&2
    exit 2
fi

if [[ ! -f "${HAPROXY_SOURCE_TARBALL}" ]]; then
    echo "Staged HAProxy source is missing: ${HAPROXY_SOURCE_TARBALL}" >&2
    exit 2
fi
echo "${HAPROXY_SOURCE_SHA256}  ${HAPROXY_SOURCE_TARBALL}" | sha256sum -c -

if [[ ! -x "${EXASERVE_STACK_ROOT}/bin/haproxy" ]]; then
    module reset
    module load "PrgEnv-gnu/${EXASERVE_FRONTIER_PRGENV_VERSION}"
    module load "cpe/${EXASERVE_FRONTIER_CPE_VERSION}"

    LOCAL_TMP_ROOT="/mnt/bb/${USER}/exaserve-complete-${SLURM_JOB_ID}"
    mkdir -p "${LOCAL_TMP_ROOT}"
    cleanup_local_tmp() {
        rm -rf -- "${LOCAL_TMP_ROOT}"
    }
    trap cleanup_local_tmp EXIT

    TMPDIR="${LOCAL_TMP_ROOT}" \
    HAPROXY_PREFIX="${EXASERVE_STACK_ROOT}/haproxy-${HAPROXY_VERSION}" \
    HAPROXY_BIN_DIR="${EXASERVE_STACK_ROOT}/bin" \
    HAPROXY_SOURCE_SHA256="${HAPROXY_SOURCE_SHA256}" \
    HAPROXY_SOURCE_TARBALL="${HAPROXY_SOURCE_TARBALL}" \
    HAPROXY_BUILD_JOBS="${HAPROXY_BUILD_JOBS}" \
        bash "${EXASERVE_SOURCE_DIR}/scripts/build_haproxy.sh" "${HAPROXY_VERSION}"

    cleanup_local_tmp
    trap - EXIT
else
    "${EXASERVE_STACK_ROOT}/bin/haproxy" -v
fi

echo "Starting one-GCD verification through an isolated Slurm step."
srun --overlap \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=7 \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    env EXPECTED_GPU_COUNT=1 \
    bash "${EXASERVE_SOURCE_DIR}/scripts/frontier/verify_stack.sh" "${CONFIG}"

echo "Frontier ExaServe software stack and one-GCD hardware preflight passed."
