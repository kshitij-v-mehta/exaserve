"""Hermetic checks for the Frontier SiteProfile scaffold (Slurm + ROCm + MI250X).

The profile is deliberately unqualified.  These tests pin that it stays
fail-closed and that Aurora's profile is unaffected by its existence.
"""

from __future__ import annotations

import pytest

from exaserve.plan.compiler import compile_deployment_plan, compile_run_plan
from exaserve.plan.contracts import PlanError, SchedulerPlan
from exaserve.plan.io import load_site_profile, write_site_profile
from exaserve.site import (
    AURORA_SITE_ID,
    FRONTIER_SITE_ID,
    apply_site_profile_environment,
    default_site_profile,
    frontier_site_profile,
    is_production_qualified,
)

_INTEL_PREFIXES = ("ZE_", "ONEAPI_", "CCL_", "RAY_EXPERIMENTAL_NOSET_ONEAPI")


@pytest.fixture(autouse=True)
def _frontier_env(monkeypatch):
    for name in (
        "EXASERVE_SITE_ID",
        "EXASERVE_SITE_MAX_NODES",
        "EXASERVE_SITE_GPUS_PER_NODE",
        "EXASERVE_SITE_CPUS_PER_NODE",
        "EXASERVE_PROJECT_ROOT",
        "EXASERVE_LOCAL_STAGE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EXASERVE_MODEL_STORAGE_PATH", "/lustre/orion/prj000/models")
    default_site_profile.cache_clear()
    yield
    default_site_profile.cache_clear()


def _raw(**kw):
    base = {
        "num_nodes": 2,
        "num_gpus_per_node": 8,
        "validation_mode": True,
        "vendor": "rocm",
        "models": [
            {
                "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
                "tensor_parallel_size": 1,
                "max_model_len": 4096,
                "size": 8,
            }
        ],
        "gateway": {"kind": "haproxy", "port": 4001},
    }
    base.update(kw)
    return base


def test_frontier_profile_declares_slurm_rocm_mi250x_only():
    site = frontier_site_profile()
    assert site.site_id == FRONTIER_SITE_ID == "olcf-frontier"
    assert site.scheduler_types == ("slurm",)
    assert site.vendors == ("rocm",)
    assert site.engines == ("vllm",)
    assert site.accelerator_inventory == ("mi250x",)
    assert site.gpus_per_node == 8  # 4 MI250X x 2 GCDs
    assert "srun" in site.launcher_capabilities
    assert "mpi" not in site.launcher_capabilities
    assert "pbs" not in site.launcher_capabilities


def test_frontier_profile_is_never_production_qualified():
    site = frontier_site_profile()
    assert site.scale_envelopes == ()
    qualified, reason = is_production_qualified(site)
    assert qualified is False
    assert "unmeasured" in reason
    assert site.environment_profile_ref == ""  # no ROCm compat profile yet


def test_frontier_prepared_environment_has_no_intel_or_allocation_state():
    site = frontier_site_profile()
    names = {key for key, _ in site.prepared_environment}
    assert names, "common Ray/thread environment must still be applied"
    assert not any(name.startswith(_INTEL_PREFIXES) for name in names)
    assert "OMP_NUM_THREADS" in names and "RAY_enable_metrics_collection" in names


def test_default_site_profile_dispatches_on_site_id(monkeypatch):
    assert default_site_profile().site_id == AURORA_SITE_ID
    assert default_site_profile(FRONTIER_SITE_ID) == frontier_site_profile()
    default_site_profile.cache_clear()
    monkeypatch.setenv("EXASERVE_SITE_ID", FRONTIER_SITE_ID)
    assert default_site_profile().site_id == FRONTIER_SITE_ID


def test_aurora_profile_unchanged_by_frontier_scaffold():
    aurora = default_site_profile()
    assert aurora.scheduler_types == ("pbs",)
    assert aurora.vendors == ("xpu",)
    names = dict(aurora.prepared_environment)
    assert names["ZE_FLAT_DEVICE_HIERARCHY"] == "FLAT"
    assert names["CCL_PROCESS_LAUNCHER"] == "hydra"
    assert "ONEAPI_DEVICE_SELECTOR" in aurora.environment_unset
    assert aurora.site_profile_hash != frontier_site_profile().site_profile_hash


def test_frontier_hash_tracks_environment_overrides(monkeypatch):
    base = frontier_site_profile().site_profile_hash
    monkeypatch.setenv("EXASERVE_SITE_GPUS_PER_NODE", "4")
    assert frontier_site_profile().site_profile_hash != base


def test_frontier_model_path_requires_explicit_project(monkeypatch):
    monkeypatch.delenv("EXASERVE_MODEL_STORAGE_PATH")
    with pytest.raises(RuntimeError, match="no default model path"):
        frontier_site_profile()
    monkeypatch.setenv("EXASERVE_PROJECT_ROOT", "relative/root")
    with pytest.raises(RuntimeError, match="absolute"):
        frontier_site_profile()
    monkeypatch.setenv("EXASERVE_PROJECT_ROOT", "/lustre/orion/prj000/proj-shared")
    monkeypatch.setattr("exaserve.site.local_account_name", lambda: "alice")
    assert (
        frontier_site_profile().model_storage_path
        == "/lustre/orion/prj000/proj-shared/alice/models"
    )


def test_frontier_profile_round_trips_through_artifact_io(tmp_path):
    site = frontier_site_profile()
    path = tmp_path / "site.profile.json"
    write_site_profile(str(path), site)
    assert load_site_profile(str(path)) == site


def test_apply_environment_overwrites_ambient_values_without_touching_intel_names():
    env = {"OMP_NUM_THREADS": "64", "ZE_AFFINITY_MASK": "0"}
    apply_site_profile_environment(frontier_site_profile(), environment=env)
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["ZE_AFFINITY_MASK"] == "0"  # not Frontier's business


def test_slurm_rocm_validation_plan_gets_unqualified_generic_envelope():
    site = frontier_site_profile()
    plan = compile_deployment_plan(_raw(), site=site, deployment_id="frontier-v")
    assert plan.site_profile_id == FRONTIER_SITE_ID
    assert plan.vendor == "rocm"
    assert plan.validation_mode is True
    assert plan.scale_envelope.envelope_id == "olcf-frontier-unqualified-generic"
    assert plan.scale_envelope.scheduler_type == "slurm"
    assert plan.scale_envelope.accelerator == "mi250x"


def test_frontier_rejects_aurora_stack_choices():
    site = frontier_site_profile()
    with pytest.raises(PlanError):
        compile_deployment_plan(_raw(vendor="xpu"), site=site, deployment_id="x")
    with pytest.raises(PlanError, match="unsupported by site"):
        compile_run_plan(
            _raw(),
            site=site,
            run_id="r",
            deployment_id="d",
            scheduler=SchedulerPlan(type="pbs", nodes=2),
        )


def test_slurm_run_plan_compiles_against_frontier():
    run = compile_run_plan(
        _raw(),
        site=frontier_site_profile(),
        run_id="r",
        deployment_id="d",
        scheduler=SchedulerPlan(type="slurm", nodes=2, queue="batch", account="prj000"),
    )
    assert run.scheduler.type == "slurm"
    assert run.deployment.site_profile_id == FRONTIER_SITE_ID


def test_frontier_node_ceiling_is_enforced(monkeypatch):
    monkeypatch.setenv("EXASERVE_SITE_MAX_NODES", "4")
    site = frontier_site_profile()
    with pytest.raises(PlanError, match="exceeds site"):
        compile_deployment_plan(_raw(num_nodes=5), site=site, deployment_id="x")


# -- per-site address resolution and queue defaults --------------------------


def test_aurora_hsn_hostnames_are_scoped_to_aurora():
    from exaserve.site import node_address_candidates

    assert node_address_candidates("x4000c0s1b0n0.hostmgmt.cm", site_id=AURORA_SITE_ID) == (
        "x4000c0s1b0n0.hsn.cm.aurora.alcf.anl.gov",
        "x4000c0s1b0n0.hostmgmt.cm",
    )
    assert node_address_candidates("frontier00123", site_id=FRONTIER_SITE_ID) == (
        "frontier00123",
    )
    assert node_address_candidates("nid001234", site_id="nersc-perlmutter") == ("nid001234",)


def test_resolution_uses_only_the_requested_sites_candidates(monkeypatch):
    import socket

    from exaserve.site import resolve_allocation_node_address, resolve_local_node_address

    asked = []

    def fake_getaddrinfo(host, *_a, **_k):
        asked.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert resolve_allocation_node_address("frontier00123", site_id=FRONTIER_SITE_ID) == "10.1.2.3"
    assert asked == ["frontier00123"]

    asked.clear()
    resolve_allocation_node_address("x4000c0s1b0n0", site_id=AURORA_SITE_ID)
    assert asked == ["x4000c0s1b0n0.hsn.cm.aurora.alcf.anl.gov"]

    names = []
    monkeypatch.setattr(socket, "gethostname", lambda: "frontier00124")
    monkeypatch.setattr(socket, "gethostbyname", lambda h: names.append(h) or "10.9.9.9")
    assert resolve_local_node_address(site_id=FRONTIER_SITE_ID) == "10.9.9.9"
    assert names == ["frontier00124"]


def test_aurora_resolution_falls_back_to_plain_name(monkeypatch):
    import socket

    from exaserve.site import resolve_allocation_node_address

    def fake_getaddrinfo(host, *_a, **_k):
        if "hsn" in host:
            raise OSError("nxdomain")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.5.5.5", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert resolve_allocation_node_address("x1", site_id=AURORA_SITE_ID) == "10.5.5.5"


def test_ray_node_ip_resolves_workers_for_the_given_site(monkeypatch):
    from exaserve.control import ray_runtime

    seen = []
    monkeypatch.setattr(ray_runtime, "_local_hsn_ip", lambda site_id: seen.append(site_id) or "10.0.0.7")
    cluster = ray_runtime.RayClusterConfig(head_ip="10.0.0.1", port=6379, node_cpus=64)
    assert ray_runtime.ray_node_ip(cluster, 0, site_id=FRONTIER_SITE_ID) == "10.0.0.1"
    assert ray_runtime.ray_node_ip(cluster, 3, site_id=FRONTIER_SITE_ID) == "10.0.0.7"
    assert seen == [FRONTIER_SITE_ID]


def test_queue_defaults_are_site_keyed_and_fail_closed():
    from exaserve.schedulers import default_queue_and_walltime

    assert default_queue_and_walltime(1, site_id=AURORA_SITE_ID) == ("capacity", "01:00:00")
    assert default_queue_and_walltime(256, site_id=AURORA_SITE_ID) == ("prod", "02:00:00")
    assert default_queue_and_walltime(2) == ("capacity", "01:00:00")  # legacy default
    for nodes in (1, 64, 9408):
        assert default_queue_and_walltime(nodes, site_id=FRONTIER_SITE_ID) == (
            "batch",
            "01:00:00",
        )
    with pytest.raises(ValueError, match="no default queue/walltime for site"):
        default_queue_and_walltime(2, site_id="nersc-perlmutter")
    with pytest.raises(ValueError, match="positive integer"):
        default_queue_and_walltime(0, site_id=FRONTIER_SITE_ID)


def test_every_known_site_has_queue_defaults():
    from exaserve.schedulers.base import _QUEUE_DEFAULTS

    assert {AURORA_SITE_ID, FRONTIER_SITE_ID} <= set(_QUEUE_DEFAULTS)
