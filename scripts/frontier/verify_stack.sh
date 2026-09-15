#!/bin/bash
set -euo pipefail

SOFTWARE_ONLY=0
if [[ "${1:-}" == "--software-only" ]]; then
    SOFTWARE_ONLY=1
    shift
fi
if [[ $# -ne 1 ]]; then
    echo "usage: bash scripts/frontier/verify_stack.sh [--software-only] /absolute/path/frontier_stack.conf" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$1"
export EXASERVE_VERIFY_SOFTWARE_ONLY="${SOFTWARE_ONLY}"

# shellcheck disable=SC1091
source "${EXASERVE_SOURCE_DIR}/scripts/frontier/env_frontier.sh"
export PATH="${EXASERVE_STACK_ROOT}/bin:${PATH}"

python3 - <<'PY'
import json
import os
import platform
from importlib.metadata import version

import ray
import torch
import vllm

assert torch.version.hip, "PyTorch is not a ROCm build"
expected_torch = os.environ.get("EXPECTED_TORCH_VERSION")
if expected_torch:
    assert version("torch") == expected_torch, version("torch")
assert torch.version.hip.startswith(os.environ.get("EXPECTED_TORCH_ROCM_ABI", "7.0")), torch.version.hip
assert version("ray") == os.environ.get("RAY_VERSION", "2.53.0"), version("ray")
installed_vllm = version("vllm")
expected_vllm = os.environ.get("EXPECTED_VLLM_VERSION")
if expected_vllm:
    assert installed_vllm == expected_vllm, installed_vllm
else:
    requested_vllm = os.environ.get("VLLM_VERSION", "0.14.1")
    assert installed_vllm.split("+")[0] == requested_vllm, installed_vllm
expected_gpu_count = os.environ.get("EXPECTED_GPU_COUNT")
software_only = os.environ.get("EXASERVE_VERIFY_SOFTWARE_ONLY") == "1"
if software_only and expected_gpu_count is not None:
    raise AssertionError("EXPECTED_GPU_COUNT is invalid with --software-only")
if not software_only and expected_gpu_count is not None:
    assert torch.cuda.device_count() == int(expected_gpu_count), torch.cuda.device_count()
gpu_arches = [] if software_only else [
    getattr(torch.cuda.get_device_properties(i), "gcnArchName", "")
    for i in range(torch.cuda.device_count())
]
if not software_only:
    assert all(arch.startswith("gfx90a") for arch in gpu_arches), gpu_arches
report = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_hip": torch.version.hip,
    "ray": ray.__version__,
    "vllm": installed_vllm,
    "verification": "software-only" if software_only else "hardware",
    "gpu_count": None if software_only else torch.cuda.device_count(),
    "gpu_names": [] if software_only else [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "gpu_arches": gpu_arches,
}
print(json.dumps(report, indent=2, sort_keys=True))
PY

for executable in exaserve-serve-submit exaserve-status; do
    command -v "${executable}" >/dev/null
    "${executable}" --help >/dev/null
done

if command -v haproxy >/dev/null 2>&1; then
    haproxy -v
else
    echo "HAProxy is missing; set and audit HAPROXY_SOURCE_SHA256, then rerun the installer" >&2
    exit 1
fi

if [[ "${SOFTWARE_ONLY}" -eq 1 ]]; then
    echo "Frontier software verification passed; GPU verification is pending"
else
    echo "Frontier stack verification passed"
fi
