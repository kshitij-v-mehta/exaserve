# ExaServe demonstrated scale and release status

> **Current verdict: no production-qualified artifact.** This tree is
> development and research source. Production submission fails closed with
> `production execution is not qualified`. Hardware work needs authorized,
> predeclared `validation_mode: true`, which records `production_qualified=false`
> and cannot establish support.

## Demonstrated system scale

ExaServe has been evaluated from 1 to 256 Aurora nodes. The
[paper](https://arxiv.org/abs/2609.10812) reports 27.1k non-streaming requests/s
at 256 nodes, and pipeline-parallel Llama-3.1-405B evaluations also reached 256
nodes. These measurements were made on earlier snapshots of this code; the
current tree has not been re-measured at that scale, and they do not qualify it.
Workloads, gateway-specific limits and measurement practice are in the
[operations handbook](../operations_handbook.md).

Later maintenance checks have their own coverage in the
[changelog](../../CHANGELOG.md#validation-coverage). A smaller regression test
does not change this demonstrated scale.

## Production-release admission in the current tree

- **Production serving:** not qualified at any node count. The default Aurora
  `SiteProfile` has no measured control and readiness limits, and its proposed
  64-node envelope has no product-owner approval.
- **Compiled ceiling:** production plans compile to at most two nodes
  (`supported_max_nodes=2`, the boundary the `final43` package measured): a
  non-validation plan of 3–64 nodes fails with `exceeds evidence-backed
  supported maximum 2`, a larger one with `exceeds candidate qualification
  target 64`, and a plan that compiles stops at submission with `production
  execution is not qualified`. Validation-mode plans in the reference HAProxy,
  completion, non-streaming envelope compile up to its 64-node qualification
  target (`EXASERVE_QUALIFICATION_TARGET_NODES`); plans that differ in gateway,
  exposure, request or streaming mode compile, in validation mode only, into the
  generic envelope up to the site maximum (`EXASERVE_SITE_MAX_NODES`, 256 by
  default); other schedulers, vendors and engines are rejected in either mode.
  These are admission limits, not the measured scale of the system.
- **Portable tests:** they verify code and package behaviour for the named
  snapshot; they are not hardware qualification.
- **Target dimensions for the proposed release:** Aurora PBS; Intel PVC/XPU, 12 tiles per
  node; vLLM under the pinned frameworks profile; HAProxy on the allocation head;
  non-streaming completions; trusted allocation-internal network. Slurm,
  CUDA/ROCm, SGLang, streaming, public or direct exposure and other gateways are
  unsupported or validation-only.

- **OLCF Frontier:** an unqualified `olcf-frontier` profile (Slurm, ROCm
  MI250X, vLLM) exists for validation-mode bring-up only; it has no scale
  envelope and no hardware evidence in this tree. See
  [deploy_frontier.md](../deploy_frontier.md).

## What a new qualification requires

1. One clean candidate: sdist and wheel built in an isolated environment; the
   exact wheel installed into a clean environment; the full suite, resource
   checks, typed-core mypy, Ruff, compileall, Go checks and findings validator
   run outside the source tree on a compute node.
2. The non-head shared-filesystem gates in section 3.2.1 of the
   [execution plan](../PRODUCTION_HARDENING_EXECUTION_PLAN.md), including a
   two-node syscall trace from managed-process creation that shows zero
   non-head opens under declared shared roots, and per-rank shared-open
   telemetry at every claimed scale.
3. Baseline hardware cells from that wheel: null and real-engine lifecycles,
   pipeline-parallel placement, drain, gateway death, worker loss,
   duplicate port, partial readiness, Ray-head and supervisor faults, and
   cleanup with zero exact-generation survivors.
4. Measured `SiteProfile` control and readiness limits.
5. A recorded product-owner decision on the release ceiling
   ([ADR-000](decisions/ADR-000-production-envelope.md) proposes 64 nodes), the
   predeclared ladder for it (4, 16 and 64 nodes for 64) through
   `scripts/hardening/run_scale_qualification.py`, and a new candidate review
   binding every receipt.
6. For logical subsets inside a larger allocation: a new four-physical,
   two-logical-node proof of the current allocation controller.

## Recorded candidates and superseded evidence

| Evidence | What it covered | Why it does not qualify this tree |
|---|---|---|
| `final43` package | one and two Aurora nodes, validation mode | built before the non-head shared-filesystem cutover changed runtime, harness, model-manifest and result identities |
| `release-v0.4.0-rc2` package | clean package gate only | same cutover; no hardware ladder ran |
| Later campaign runs | null-compute start-up at 32 and 64 nodes, 405B PP at 4, 8 and 256 nodes, the allocation-subset proof | an earlier snapshot in validation mode; the runtime and allocation controller changed afterwards |
| Paper evaluations and earlier research runs | the 1–256-node serving and bring-up measurements; 4-, 16- and 64-node previews | earlier code; they are the demonstrated research scale and regression baselines, not release evidence |

Wheel SHA-256: `final43` `1041be53eb5b5875d198d5ee6c6664718b4085775dcba99107873dd3d1fcdff2`;
`release-v0.4.0-rc2` `75bab1c97948d29237000d5fa087f3ed1f1d18e5ef3233fe2c759b4faf0e32f8`.
Scale evaluations, package gates and fix-specific checks answer different
questions: the [validation coverage](../../CHANGELOG.md#validation-coverage)
records later checks, and the handbook records the scale measurements. Do not
update old hashes to match new code, relabel receipts, or join runs from
different snapshots into one scaling curve. Earlier revisions of this page and
the retired run records remain in the repository history.

## Ledger

The [findings ledger](FINDINGS.yaml) validates against its exact schema. Its
open records (eight `IN_PROGRESS`, one `EXTERNAL_BLOCKER`) and the work that
would close them are listed in [open work](../TODO.md).

See also the [compatibility matrix](COMPATIBILITY_MATRIX.md),
[known issues](../KNOWN_ISSUES.md) and the
[operations handbook](../operations_handbook.md), which records measured
behaviour, limits and lessons from earlier campaigns.
