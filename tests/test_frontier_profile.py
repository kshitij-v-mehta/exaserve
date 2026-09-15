from __future__ import annotations

import os
from pathlib import Path

from exaserve.compat.profile import default_profile
from exaserve.compat import generated_overlay
from exaserve.plan.compiler import compile_deployment_plan
from exaserve.schedulers.base import JobSpec, default_queue_and_walltime
from exaserve.schedulers.slurm import SlurmScheduler
from exaserve.site import FRONTIER_SITE_ID, default_site_profile
from exaserve.vendors.rocm import ROCmVendor


def test_frontier_profile_is_slurm_rocm_and_validation_only(tmp_path, monkeypatch):
    default_site_profile.cache_clear()
    default_profile.cache_clear()
    monkeypatch.setenv("EXASERVE_PROJECT_ROOT", str(tmp_path))
    site = default_site_profile(FRONTIER_SITE_ID)
    assert site.scheduler_types == ("slurm",)
    assert site.vendors == ("rocm",)
    assert site.gpus_per_node == 8
    assert site.cpus_per_node == 56
    assert site.local_stage_path.endswith("/exaserve")
    assert site.scale_envelopes == ()
    assert site.readiness.serve_start_proxy_timeout_s == 60.0
    assert dict(site.prepared_environment)["RAY_raylet_start_wait_time_s"] == "120"

    raw = {
        "validation_mode": True,
        "vendor": "rocm",
        "accelerator": "mi250x-gcd",
        "models": [{"model_id": "test/model", "size": 1, "num_replicas": 1}],
        "exposure": {"mode": "DIRECT_VALIDATION"},
    }
    plan = compile_deployment_plan(raw, site=site)
    assert plan.validation_mode is True
    assert plan.scale_envelope.scheduler_type == "slurm"


def test_frontier_tp1_auto_replica_count_scales_with_declared_gcd_capacity(
    tmp_path, monkeypatch
):
    default_site_profile.cache_clear()
    default_profile.cache_clear()
    monkeypatch.setenv("EXASERVE_PROJECT_ROOT", str(tmp_path))
    site = default_site_profile(FRONTIER_SITE_ID)
    for nodes, gcds in ((1, 1), (1, 8), (8, 8), (256, 8)):
        raw = {
            "validation_mode": True,
            "num_nodes": nodes,
            "num_gpus_per_node": gcds,
            "node_cpus": 56,
            "vendor": "rocm",
            "accelerator": "mi250x-gcd",
            "models": [
                {
                    "model_id": "test/model",
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "num_cpus_per_replica": 4,
                }
            ],
            "exposure": {"mode": "DIRECT_VALIDATION"},
        }
        plan = compile_deployment_plan(raw, site=site)
        model = plan.models[0]
        assert model.num_replicas == nodes * gcds
        slots = {
            (replica.planned_ranks[0], replica.planned_device_ids[0][0])
            for replica in model.replicas
        }
        assert len(slots) == nodes * gcds


def test_frontier_rocm_profile_does_not_inherit_xpu_sources(monkeypatch):
    default_profile.cache_clear()
    monkeypatch.setenv("EXASERVE_FRONTIER_PYTHON_VERSION", "3.12.12")
    profile = default_profile("rocm")
    assert profile.vendor == "rocm"
    assert profile.vllm == "0.14.1"
    assert profile.patches == ()
    assert "+xpu" not in profile.name


def test_frontier_patch_free_overlay_is_valid(tmp_path, monkeypatch):
    default_profile.cache_clear()
    monkeypatch.setenv("EXASERVE_FRONTIER_PYTHON_VERSION", "3.12.14")
    monkeypatch.setenv("EXASERVE_FRONTIER_RAY_VERSION", "2.53.0")
    monkeypatch.setenv("EXASERVE_FRONTIER_VLLM_VERSION", "0.15.0+rocm")
    profile = default_profile("rocm")
    manifest = generated_overlay.materialize(profile, tmp_path / "overlay")
    assert manifest["entries"] == []
    assert generated_overlay.load_manifest(profile, tmp_path / "overlay") == manifest


def test_frontier_rocm_isolation_uses_ray_compatible_hip_mask(monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "7")
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    ROCmVendor().isolate_devices([2, 3])
    assert "ROCR_VISIBLE_DEVICES" not in os.environ
    assert os.environ["HIP_VISIBLE_DEVICES"] == "2,3"


def test_frontier_slurm_render_includes_gpu_and_nvme_contract():
    spec = JobSpec(
        job_name="es-test",
        num_nodes=2,
        walltime="01:00:00",
        account="ABC123",
        stdout_dir=Path("/tmp/logs"),
        stderr_dir=Path("/tmp/logs"),
        queue="batch",
        qos="debug",
        command_argv=("python", "-m", "exaserve.launcher", "/tmp/plan.json"),
        gpus_per_node=8,
        constraint="nvme",
        network="disable_rdzv_get",
    )
    script = SlurmScheduler().render_job(spec)
    assert "#SBATCH --partition=batch" in script
    assert "#SBATCH --qos=debug" in script
    assert "#SBATCH --gpus-per-node=8" in script
    assert "#SBATCH --constraint=nvme" in script
    assert "#SBATCH --network=disable_rdzv_get" in script
    assert "#SBATCH --ntasks-per-node=1" in script
    assert default_queue_and_walltime(2, scheduler="slurm") == ("batch", "01:00:00")
