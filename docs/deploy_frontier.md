# Frontier (OLCF) validation deployment

Status: **validation only.** The `olcf-frontier` `SiteProfile` has no measured
control/readiness limits and no scale envelope, so every Frontier plan must set
`validation_mode: true` and runs with `production_qualified=false`. Nothing on
this page is a support claim; see [hardening status](hardening/STATUS.md).

## What differs from Aurora

| Concern | Aurora | Frontier |
|---|---|---|
| Scheduler / launcher | PBS, PALS `mpiexec` | Slurm, `srun` (finite steps use `--overlap`) |
| Executable transfer | `mpiexec --transfer` | `srun --bcast=/tmp/` |
| Accelerator | 12 PVC tiles, `ZE_AFFINITY_MASK` | 8 MI250X GCDs, `HIP_VISIBLE_DEVICES` (Ray-owned) |
| Compatibility profile | `aurora-frameworks-2025.3.1-xpu` | `olcf-frontier-rocm711` (same upstream Ray 2.53.0 / vLLM v0.15.0 sources, XPU-only patches SC-09..SC-12 dropped) |
| Serving runtime | `/opt/aurora` module image | archive unpacked to `/tmp/exaserve-site/<stack>` on every node by the job environment script |
| Node-local storage | `/tmp` | `/mnt/bb/$USER` (NVMe, needs `--constraint=nvme`) for models, `/tmp` for the runtime |
| Head address | `<node>.hsn.cm.aurora.alcf.anl.gov` | plain hostname (resolves to `hsn0`) |

The Frontier filesystem facts in `exaserve.site._FRONTIER_FILESYSTEM_SEMANTICS`
were recorded from compute nodes: Orion is Lustre, home/project/software trees
are Cray DVS (NFS on login nodes), `/opt/cray` and `/opt/rocm-*` are
node-local overlays.

## 1. Build the runtime archive (once per stack)

Copy and edit the stack configuration outside the checkout:

```bash
cp scripts/frontier/frontier_stack.conf.example /lustre/orion/<proj>/.../frontier_stack.conf
# set EXASERVE_STACK_ROOT, EXASERVE_SOURCE_DIR, EXASERVE_FRONTIER_ACCOUNT
```

Build on one compute node (about an hour; downloads go through the OLCF proxy):

```bash
sbatch -A <account> -p extended -q debug \
  --export=ALL,FRONTIER_STACK_CONFIG=/abs/frontier_stack.conf \
  scripts/frontier/build_stack.sbatch
```

[`build_stack.sh`](../scripts/frontier/build_stack.sh) creates a conda-forge
Python 3.12.12 at `/tmp/exaserve-site/<stack>`, installs ROCm PyTorch 2.10.0,
builds vLLM from the `v0.15.0` tag for `gfx90a` (version `0.15.0+rocm711`),
installs Ray 2.53.0, mpi4py 4.1.1 against Cray MPICH and HAProxy 3.1.6, checks
them on the GPU, and writes `<stack>.tar.zst` plus its `.sha256` and pip freeze
under `EXASERVE_STACK_ROOT/runtime`. It refuses to overwrite an existing
archive; a changed stack needs a new `EXASERVE_STACK_NAME` and, if any pinned
version changes, a new compatibility profile.

## 2. Prepare models

Models live under `EXASERVE_MODEL_STORAGE_PATH` as `<org>--<name>` directories
and are staged by ExaServe onto `/mnt/bb/$USER`. Only architectures with
reviewed vLLM model-info seeds are accepted (currently `LlamaForCausalLM` and
`GptOssForCausalLM`). Download on a login node; compute jobs run with
`HF_HUB_OFFLINE=1`.

## 3. Submit

Create two small files next to the stack configuration:

- `env.sh` (the job's `EXASERVE_ENV_SCRIPT`):

  ```bash
  export FRONTIER_STACK_CONFIG=/abs/frontier_stack.conf
  source /abs/exaserve/scripts/frontier/env_frontier.sh
  ```

- a submission environment that sets `EXASERVE_SITE_ID=olcf-frontier`,
  `EXASERVE_SCHEDULER=slurm`, `EXASERVE_PROJECT_ACCOUNT`,
  `EXASERVE_MODEL_STORAGE_PATH`, `EXASERVE_ENV_SCRIPT=/abs/env.sh` and
  `EXASERVE_JOB_PYTHON=/tmp/exaserve-site/<stack>/bin/python3`, and puts the
  checkout's `src` on `PYTHONPATH` for `cray-python/3.12.12`.

Then, from a login node:

```bash
exaserve-serve-submit examples/config.frontier.smollm2.yaml \
  --queue extended --qos debug --walltime 00:30:00
```

`--qos` (or `EXASERVE_DEFAULT_QOS`) is new for Slurm. The Frontier job policy
adds `--constraint=nvme` and does not request GPU GRES: whole nodes are
allocated, all GCDs are visible to steps, and a GRES request would make Slurm
inject a ROCr mask that conflicts with Ray's device assignment.

[`env_frontier.sh`](../scripts/frontier/env_frontier.sh) loads the Cray PE and
ROCm modules, `sbcast`s the archive and unpacks it on every node, builds one
wheel of `EXASERVE_SOURCE_DIR` on the head and installs it on every node, then
removes ambient variables that name shared storage. It does not own readiness,
lifecycle or cleanup.

## Known gaps before any qualification

- Multi-node tensor/pipeline parallelism (RCCL over Slingshot, the
  `aws-ofi-rccl` plugin and `NCCL_SOCKET_IFNAME`) is not configured or tested.
- The runtime site root is a writable tmpfs: its immutability is an operator
  contract, recorded as `readonly=false` in the profile.
- Every item in [deploy_slurm_amd.md](deploy_slurm_amd.md) still applies.
