#!/bin/bash
# Build the Frontier ExaServe serving runtime (Python, ROCm PyTorch, vLLM, Ray,
# mpi4py, HAProxy) at its final node-local prefix and publish it as one archive.
#
# Runs inside a one-node Slurm allocation (see build_stack.sbatch).  The stack
# is built directly at EXASERVE_RUNTIME_PREFIX (under /tmp/exaserve-site) so
# that every node can unpack the archive at the identical path without
# relocation; the interpreter therefore never runs from Orion or NFS/DVS.
# Downloads use the OLCF compute-node proxy.  ExaServe itself is NOT part of
# the archive: env_frontier.sh installs the checkout's wheel at job start.
set -euo pipefail

CONFIG="${1:?usage: build_stack.sh /abs/frontier_stack.conf}"
# shellcheck source=/dev/null
source "${CONFIG}"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "build_stack.sh must run inside a Frontier compute allocation" >&2
    exit 2
fi
case "${EXASERVE_RUNTIME_PREFIX}" in
    /tmp/exaserve-site/*) ;;
    *) echo "EXASERVE_RUNTIME_PREFIX must be below /tmp/exaserve-site" >&2; exit 2 ;;
esac
if [[ -e "${EXASERVE_RUNTIME_ARCHIVE}" ]]; then
    echo "Refusing to overwrite existing archive ${EXASERVE_RUNTIME_ARCHIVE}" >&2
    exit 2
fi

LOG_DIR="${EXASERVE_STACK_ROOT}/logs/build-${SLURM_JOB_ID}"
WORK="/mnt/bb/${USER}/exaserve-build-${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}" "${WORK}" "$(dirname "${EXASERVE_RUNTIME_ARCHIVE}")"
trap 'rm -rf -- "${WORK}"' EXIT
echo "[stack] job=${SLURM_JOB_ID} node=$(hostname) prefix=${EXASERVE_RUNTIME_PREFIX}"
echo "[stack] logs=${LOG_DIR}"

# --- Toolchain -----------------------------------------------------------------
module reset >/dev/null 2>&1
module load "PrgEnv-gnu/${EXASERVE_FRONTIER_PRGENV_VERSION}"
module load "rocm/${EXASERVE_FRONTIER_ROCM_VERSION}"
module load craype-accel-amd-gfx90a
module load cray-mpich
module load "miniforge3/${EXASERVE_FRONTIER_MINIFORGE_VERSION}"
module list 2>&1 | tee "${LOG_DIR}/modules.txt"

export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
curl -fsS -m 30 -o /dev/null https://pypi.org/simple/pip/ \
    || { echo "[stack] compute-node proxy unavailable" >&2; exit 3; }

export TMPDIR="${WORK}/tmp"
export PIP_CACHE_DIR="${WORK}/pip-cache"
export CONDA_PKGS_DIRS="${WORK}/conda-pkgs"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
# The batch shell carries a ROCr mask; Ray 2.53 refuses to import with
# ROCR_VISIBLE_DEVICES set (observed in build job 5919113).
echo "[stack] inherited masks: ROCR=${ROCR_VISIBLE_DEVICES:-unset} HIP=${HIP_VISIBLE_DEVICES:-unset}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
mkdir -p "${TMPDIR}"

# --- Interpreter ---------------------------------------------------------------
rm -rf -- "${EXASERVE_RUNTIME_PREFIX}"
mkdir -p "$(dirname "${EXASERVE_RUNTIME_PREFIX}")"
conda create -y -q -p "${EXASERVE_RUNTIME_PREFIX}" --override-channels -c conda-forge \
    "python=${PYTHON_VERSION}" pip 2>&1 | tee "${LOG_DIR}/conda.log"
PY="${EXASERVE_RUNTIME_PREFIX}/bin/python3"
PIP=("${PY}" -m pip --disable-pip-version-check)
"${PY}" -c "import platform,sys; assert platform.python_version()=='${PYTHON_VERSION}'; print(sys.executable)"

CONSTRAINTS="${WORK}/constraints.txt"
cat > "${CONSTRAINTS}" <<EOC
torch==${TORCH_VERSION}
torchvision==${TORCHVISION_VERSION}
torchaudio==${TORCHAUDIO_VERSION}
ray==${RAY_VERSION}
transformers==${TRANSFORMERS_VERSION}
fastapi==${FASTAPI_VERSION}
grpcio-tools==${GRPCIO_TOOLS_VERSION}
protobuf<7
setuptools<80
EOC

# --- PyTorch (ROCm) ------------------------------------------------------------
"${PIP[@]}" install --index-url "${TORCH_ROCM_INDEX}" --extra-index-url https://pypi.org/simple \
    -c "${CONSTRAINTS}" torch torchvision torchaudio 2>&1 | tee "${LOG_DIR}/torch.log"
"${PY}" - <<'PY'
import torch
print("torch", torch.__version__, "hip", torch.version.hip)
assert torch.version.hip and torch.cuda.is_available(), "ROCm GPU not visible to torch"
print("gpu0", torch.cuda.get_device_name(0), torch.cuda.device_count(), "devices")
PY

# --- vLLM from the pinned tag ----------------------------------------------------
SRC="${WORK}/vllm"
git clone -q --branch "${VLLM_GIT_TAG}" --depth 1 https://github.com/vllm-project/vllm.git "${SRC}"
git -C "${SRC}" rev-parse HEAD | tee "${LOG_DIR}/vllm-commit.txt"
cd "${SRC}"
"${PY}" use_existing_torch.py
"${PIP[@]}" install -c "${CONSTRAINTS}" -r requirements/build.txt 2>&1 | tee "${LOG_DIR}/vllm-build-reqs.log"
"${PIP[@]}" install -c "${CONSTRAINTS}" -r requirements/rocm.txt "ray[serve,cgraph]==${RAY_VERSION}" \
    2>&1 | tee "${LOG_DIR}/vllm-reqs.log"
# xgrammar declares an unpinned "triton" dependency, so pip installs the upstream
# triton wheel over torch's ROCm build (distribution triton-rocm) in the same
# triton/ package directory.  Restore the exact ROCm Triton torch was built with;
# pip check will then report xgrammar's "triton" requirement as unmet, which is
# expected for ROCm stacks.
ROCM_TRITON="$("${PY}" -c "from importlib.metadata import requires
print(next(r.split(';')[0].replace(' ', '') for r in requires('torch') if r.startswith('triton-rocm')))")"
"${PIP[@]}" uninstall -y triton 2>&1 | tee "${LOG_DIR}/triton-repair.log"
"${PIP[@]}" install --force-reinstall --no-deps --index-url "${TORCH_ROCM_INDEX}" "${ROCM_TRITON}" \
    2>&1 | tee -a "${LOG_DIR}/triton-repair.log"
"${PY}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import triton
try:
    version("triton")
    raise SystemExit("upstream triton distribution is still installed")
except PackageNotFoundError:
    pass
print("triton", triton.__version__, "from triton-rocm", version("triton-rocm"))
PY
if ! "${PY}" -c "import amdsmi" >/dev/null 2>&1; then
    cp -a "${ROCM_PATH}/share/amd_smi" "${WORK}/amd_smi"
    "${PIP[@]}" install "${WORK}/amd_smi" 2>&1 | tee "${LOG_DIR}/amdsmi.log"
fi
# Building from the exact tag: pin the scm version so the local tree changes made
# by use_existing_torch.py cannot turn 0.15.0 into a ".dev+dirty" version.
export SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_GIT_TAG#v}"
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH
export MAX_JOBS="${MAX_JOBS:-48}"
export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"
export CC=gcc CXX=g++
"${PIP[@]}" install -v --no-build-isolation --no-deps . > "${LOG_DIR}/vllm-compile.log" 2>&1 \
    || { tail -50 "${LOG_DIR}/vllm-compile.log"; exit 4; }
cd /
"${PY}" - <<PY
from importlib.metadata import version
v = version("vllm")
print("vllm", v)
assert v == "${EXPECTED_VLLM_VERSION}", f"unexpected vLLM version {v}"
PY

# --- mpi4py against Cray MPICH ---------------------------------------------------
MPICC="cc -shared" "${PIP[@]}" install --no-cache-dir --no-binary=mpi4py \
    "mpi4py==${MPI4PY_VERSION}" 2>&1 | tee "${LOG_DIR}/mpi4py.log"

# --- HAProxy into the prefix -------------------------------------------------------
TMPDIR="${WORK}/tmp" HAPROXY_PREFIX="${EXASERVE_RUNTIME_PREFIX}/opt/haproxy-${HAPROXY_VERSION}" \
    HAPROXY_BIN_DIR="${EXASERVE_RUNTIME_PREFIX}/bin" HAPROXY_SOURCE_SHA256="${HAPROXY_SOURCE_SHA256}" \
    bash "${EXASERVE_SOURCE_DIR}/scripts/build_haproxy.sh" "${HAPROXY_VERSION}" \
    2>&1 | tee "${LOG_DIR}/haproxy.log"
# build_haproxy.sh links; replace the link with the file so the prefix has no
# symlinked executables.
cp --remove-destination "${EXASERVE_RUNTIME_PREFIX}/opt/haproxy-${HAPROXY_VERSION}/sbin/haproxy" \
    "${EXASERVE_RUNTIME_PREFIX}/bin/haproxy"

# --- Verification -----------------------------------------------------------------
"${PIP[@]}" check 2>&1 | tee "${LOG_DIR}/pip-check.txt" || true
"${PIP[@]}" freeze > "${LOG_DIR}/pip-freeze.txt"
"${PY}" -I - <<'PY' 2>&1 | tee "${LOG_DIR}/verify.txt"
import platform
from importlib.metadata import version
for name in ("torch", "vllm", "ray", "transformers", "mpi4py", "fastapi", "protobuf"):
    print(f"{name}=={version(name)}")
print("python", platform.python_version())
import ray.serve  # noqa: F401  (protobuf/FastAPI compatibility)
from ray.serve._private.config import DeploymentConfig
DeploymentConfig.from_proto_bytes(DeploymentConfig().to_proto_bytes())
import mpi4py.MPI as MPI
print("mpi", MPI.Get_library_version().splitlines()[0])
import torch
x = torch.ones(1024, device="cuda")
print("torch device sum", float(x.sum()))
PY
"${EXASERVE_RUNTIME_PREFIX}/bin/haproxy" -v | head -1

# --- Publish -----------------------------------------------------------------------
find "${EXASERVE_RUNTIME_PREFIX}" -name '__pycache__' -prune -exec rm -rf {} +
chmod -R a-w "${EXASERVE_RUNTIME_PREFIX}"
CANDIDATE="${EXASERVE_RUNTIME_ARCHIVE}.partial.${SLURM_JOB_ID}"
tar -C / -I 'zstd -T16 -3' -cf "${CANDIDATE}" "${EXASERVE_RUNTIME_PREFIX#/}"
sha256sum "${CANDIDATE}" | sed "s|${CANDIDATE}|$(basename "${EXASERVE_RUNTIME_ARCHIVE}")|" \
    > "${EXASERVE_RUNTIME_ARCHIVE}.sha256"
cp "${LOG_DIR}/pip-freeze.txt" "${EXASERVE_RUNTIME_ARCHIVE}.freeze.txt"
mv -n "${CANDIDATE}" "${EXASERVE_RUNTIME_ARCHIVE}"
chmod a-w "${EXASERVE_RUNTIME_ARCHIVE}"
ls -l "${EXASERVE_RUNTIME_ARCHIVE}"
echo "[stack] DONE ${EXASERVE_RUNTIME_ARCHIVE}"
