"""AMD ROCm vendor (OLCF Frontier MI250X; also NCSA Delta MI100, LUMI).

Notes:
- Device isolation uses ``HIP_VISIBLE_DEVICES``.  Ray's AMD accelerator
  manager (2.53) assigns GPUs through that variable and raises if
  ``ROCR_VISIBLE_DEVICES`` is set without it, so a ROCr-level mask would break
  every Ray actor that requests GPUs.  Any inherited ``ROCR_VISIBLE_DEVICES``
  is removed so the two masks cannot compose into an unexpected subset; sites
  also list it in ``SiteProfile.environment_unset``.
- PyTorch-ROCm reports the device as ``cuda`` (HIP masquerades as CUDA), so
  ``torch_device()`` returns ``cuda`` and vLLM's ROCm build auto-detects.
- MI250X exposes two GCDs per package; ``device_ids`` are GCD indices (0-7 on
  a Frontier node).
"""

from __future__ import annotations

import os
from typing import List

from .base import VendorBackend


class ROCmVendor(VendorBackend):
    name = "rocm"
    per_device_compile_cache = True

    def isolate_devices(self, device_ids: List[int], engine_name: str = "vllm") -> None:
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        if device_ids:
            os.environ["HIP_VISIBLE_DEVICES"] = ",".join(str(g) for g in device_ids)
        else:
            os.environ.pop("HIP_VISIBLE_DEVICES", None)

    def torch_device(self) -> str:
        return "cuda"  # PyTorch-ROCm / HIP masquerades as cuda

    def default_gpus_per_node(self) -> int:
        return 8  # Frontier / Delta: 8 devices per node (site config may override)

    # engine_env / distributed_env: vLLM's ROCm build auto-detects the device and
    # its Ray path works without the XPU compiled-DAG workaround, so no vendor env
    # is set here — ROCm PATH / HSA_* belong in the site env script.
