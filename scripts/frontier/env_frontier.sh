# shellcheck shell=bash
# Frontier site environment for ExaServe batch jobs (EXASERVE_ENV_SCRIPT).
#
# Sourced by the job body that `exaserve-serve-submit` renders, before the
# canonical launcher starts.  Its job is environment preparation only:
#
#   1. load the Cray PE / ROCm modules the stack was built against;
#   2. inside an allocation, unpack the immutable runtime archive at the same
#      node-local prefix (/tmp/exaserve-site/...) on every allocated node and
#      install this checkout's ExaServe wheel there, so no worker ever runs its
#      interpreter or imports a package from Orion or NFS/DVS;
#   3. drop ambient variables that name shared storage, which the launcher
#      would otherwise reject as a child-environment leak.
#
# It does not start daemons or implement readiness, lifecycle or cleanup.
# Requires FRONTIER_STACK_CONFIG (see frontier_stack.conf.example).

if [[ -z "${FRONTIER_STACK_CONFIG:-}" || ! -f "${FRONTIER_STACK_CONFIG}" ]]; then
    echo "[env_frontier] FRONTIER_STACK_CONFIG must name the stack configuration" >&2
    return 2 2>/dev/null || exit 2
fi
# shellcheck source=/dev/null
source "${FRONTIER_STACK_CONFIG}"

# The submitting shell's checkout PYTHONPATH (sbatch exports it) must not shadow
# the ExaServe wheel installed into the node-local runtime below.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
module reset >/dev/null 2>&1
# Default-loaded tools that inject shared-storage paths or LD_PRELOAD hooks.
module unload darshan-runtime hsi lfs-wrapper DefApps >/dev/null 2>&1 || true
module load "PrgEnv-gnu/${EXASERVE_FRONTIER_PRGENV_VERSION}"
module load "rocm/${EXASERVE_FRONTIER_ROCM_VERSION}"
module load cray-mpich
# No craype-accel module at run time: it enables GPU-aware MPICH, and ExaServe's
# MPI helpers are not linked with the GTL library (GPU traffic uses RCCL).
export MPICH_GPU_SUPPORT_ENABLED=0
if [[ -n "${CRAY_LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
unset LD_PRELOAD

_es_python="${EXASERVE_RUNTIME_PREFIX}/bin/python3"

_es_stage_runtime() {
    local nodes="${SLURM_JOB_NUM_NODES:?}"
    local digest marker archive_local wheel_dir wheel wheel_local
    if [[ ! -f "${EXASERVE_RUNTIME_ARCHIVE}" || ! -f "${EXASERVE_RUNTIME_ARCHIVE}.sha256" ]]; then
        echo "[env_frontier] runtime archive missing: ${EXASERVE_RUNTIME_ARCHIVE}" >&2
        return 2
    fi
    digest="$(awk '{print $1}' "${EXASERVE_RUNTIME_ARCHIVE}.sha256")"
    marker="${EXASERVE_RUNTIME_PREFIX}/.exaserve-archive-sha256"
    archive_local="/tmp/exaserve-runtime-${SLURM_JOB_ID}.tar.zst"

    # One wheel of the current checkout, built on the head from shared source.
    wheel_dir="$(mktemp -d /tmp/exaserve-wheel.XXXXXX)"
    if [[ ! -x "${_es_python}" || "$(cat "${marker}" 2>/dev/null)" != "${digest}" ]]; then
        echo "[env_frontier] staging $(basename "${EXASERVE_RUNTIME_ARCHIVE}") onto ${nodes} node(s)"
        sbcast --force "${EXASERVE_RUNTIME_ARCHIVE}" "${archive_local}" || return 3
        srun --overlap --nodes="${nodes}" --ntasks-per-node=1 --cpu-bind=none bash -c '
            set -euo pipefail
            echo "'"${digest}"'  '"${archive_local}"'" | sha256sum -c --quiet -
            if [[ -d "'"${EXASERVE_RUNTIME_PREFIX}"'" ]]; then
                chmod -R u+w "'"${EXASERVE_RUNTIME_PREFIX}"'"
                rm -rf -- "'"${EXASERVE_RUNTIME_PREFIX}"'"
            fi
            mkdir -p "$(dirname "'"${EXASERVE_RUNTIME_PREFIX}"'")"
            tar -C / -I "zstd -T8" -xf "'"${archive_local}"'"
            rm -f -- "'"${archive_local}"'"
            echo "'"${digest}"'" > "'"${marker}"'"
        ' || return 3
    fi
    (cd /tmp && PYTHONNOUSERSITE=1 "${_es_python}" -m pip wheel -q --no-deps \
        --no-build-isolation --no-index -w "${wheel_dir}" "${EXASERVE_SOURCE_DIR}") || return 4
    wheel="$(ls "${wheel_dir}"/exaserve-*.whl)"
    wheel_local="/tmp/$(basename "${wheel}")"
    sbcast --force "${wheel}" "${wheel_local}" || return 4
    srun --overlap --nodes="${nodes}" --ntasks-per-node=1 --cpu-bind=none bash -c '
        set -euo pipefail
        prefix="'"${EXASERVE_RUNTIME_PREFIX}"'"
        chmod -R u+w "${prefix}"
        PYTHONNOUSERSITE=1 "${prefix}/bin/python3" -m pip install -q --no-deps --no-index \
            --force-reinstall --disable-pip-version-check "'"${wheel_local}"'"
        rm -f -- "'"${wheel_local}"'"
        find "${prefix}" -name __pycache__ -prune -exec rm -rf {} +
        chmod -R a-w "${prefix}"
    ' || return 4
    rm -rf -- "${wheel_dir}"
    echo "[env_frontier] runtime ready at ${EXASERVE_RUNTIME_PREFIX} on ${nodes} node(s)"
}

if [[ -n "${SLURM_JOB_ID:-}" && "${EXASERVE_SKIP_RUNTIME_STAGE:-0}" != "1" ]]; then
    _es_stage_runtime || { echo "[env_frontier] runtime staging failed" >&2; return 3 2>/dev/null || exit 3; }
fi
unset -f _es_stage_runtime

export PATH="${EXASERVE_RUNTIME_PREFIX}/bin:${PATH}"
hash -r
export EXASERVE_SITE_ID=olcf-frontier
export EXASERVE_SCHEDULER=slurm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
# Ray assigns GPUs through HIP_VISIBLE_DEVICES and rejects a ROCr mask.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

# Drop ambient variables that name shared storage.  Search paths are cleaned by
# ExaServe itself; ExaServe/Slurm/module bookkeeping is discarded or validated
# there.  Anything else here (e.g. *_ROOT/*_DIR exported by /sw modules) is not
# an input the serving runtime uses.
while IFS='=' read -r _es_name _es_value; do
    case "${_es_name}" in
        PATH|LD_LIBRARY_PATH|HOME|PWD|OLDPWD|EXASERVE_*|SLURM_*|FRONTIER_STACK_CONFIG) continue ;;
        LMOD_*|__LMOD_*|_ModuleTable*|MODULEPATH*|MODULESHOME|LOADEDMODULES|_LMFILES_|BASH_FUNC_*) continue ;;
    esac
    case "${_es_value}" in
        */lustre/orion*|*/ccs/home*|*/ccs/proj*|*/autofs/nccs*|*/sw/frontier*|*/sw/sources*)
            unset "${_es_name}" ;;
    esac
done < <(env | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' )
unset _es_name _es_value _es_python
# Stack-configuration paths are submission inputs, not runtime inputs.
unset FRONTIER_STACK_CONFIG EXASERVE_STACK_ROOT EXASERVE_SOURCE_DIR EXASERVE_RUNTIME_ARCHIVE
