# ExaServe on Frontier

This directory contains the Frontier/ROCm validation port of ExaServe.  It
builds a native vLLM runtime for MI250X (`gfx90a`), starts a multi-node Ray
cluster under Slurm, creates one or more Ray Serve replicas, and publishes one
OpenAI-compatible endpoint through HAProxy.

The port is validation-only.  It does not claim a production-qualified scale
envelope yet, but the validated path has completed:

- one node and one GCD with SmolLM2-360M;
- one node and multiple GCDs;
- eight nodes and eight GCDs per node;
- 64 independent Granite 3.3 8B replicas; and
- 256 concurrent chat-completion requests with no failures after isolating the
  Triton caches.

The final scale topology was:

```text
client -> HAProxy -> 8 Ray Serve proxies -> 64 Granite replicas
                                      8 nodes x 8 MI250X GCDs
```

## Validated software stack

| Component | Validated value |
|---|---|
| Frontier programming environment | PrgEnv-gnu 8.7.0, CPE 26.03 |
| Python | 3.12.14 |
| ROCm | 7.1.1 |
| PyTorch | 2.10.0+rocm7.1 |
| vLLM | 0.15 source build; observed `0.15.1.dev0+gf17644344.d20260828.rocm711` |
| Ray / Ray Serve | 2.53.0 |
| FastAPI | 0.136.0 |
| Protobuf | 7.36.0 with the Ray source adaptation described below |
| HAProxy | 3.1.6 |
| Granite model | `ibm-granite/granite-3.3-8b-instruct` |
| Granite revision | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` |

The model is ungated and Apache 2.0 licensed.  Compute nodes remain offline;
source, packages, model weights, and HAProxy are staged from a networked login
node before deployment.

## Is scaling configuration-only?

For independent TP=1 replicas, yes.  The checked-in Granite configuration
omits `num_replicas`, causing the plan compiler to fill every declared GCD.
Only these two fields control the topology:

```yaml
num_nodes: 8
num_gpus_per_node: 8
```

The resulting replica count is:

```text
num_nodes * num_gpus_per_node
```

The compiler was exercised for all of the following configurations:

| Nodes | GCDs per node | Resulting replicas |
|---:|---:|---:|
| 1 | 1 | 1 |
| 1 | 2 | 2 |
| 1 | 8 | 8 |
| 2 | 8 | 16 |
| 8 | 8 | 64 |
| 16 | 8 | 128 |
| 32 | 8 | 256 |
| 64 | 8 | 512 |
| 128 | 8 | 1024 |
| 256 | 8 | 2048 |

Every compiled replica receives a unique `(Slurm rank, GCD index)` placement.
The same values are projected into `#SBATCH --nodes`,
`#SBATCH --gpus-per-node`, exact Ray membership/resource checks, model/source
staging, per-node proxy anchors, and the HAProxy backend list.

This statement has important boundaries:

- The current Frontier site profile allows at most 256 nodes and remains a
  validation profile.  Raising that maximum requires a separately reviewed
  site envelope and new evidence, not merely YAML editing.
- TP or PP changes alter how many GCDs and nodes each replica consumes.  Set
  `tensor_parallel_size`, `pipeline_parallel_size`, and optionally
  `num_replicas` deliberately for those topologies.
- The Slurm partition, QoS, account, and walltime are submission policy rather
  than model topology.  The validated wrapper requests partition `extended`,
  QoS `debug`, and a 30-minute walltime.
- Scale-dependent load generation is controlled independently through
  `REQUESTS` and `CONCURRENCY`.

## 1. Create the local stack configuration

Copy the template outside the checkout so project paths do not enter Git:

```bash
cd /path/to/exaserve
cp scripts/frontier/frontier_stack.conf.example /path/to/frontier_stack.conf
```

Edit `PROJECT`, `USER`, and every path.  Then make the path available to the
Frontier scripts:

```bash
export FRONTIER_STACK_CONFIG=/absolute/path/frontier_stack.conf
source "${FRONTIER_STACK_CONFIG}"
```

The source checkout, Python environment, build sources, model repository, and
logs should live on Orion.  Node-local runtime data uses `/mnt/bb`.

## 2. Stage and build vLLM

Run dependency and source staging on a networked login node:

```bash
bash scripts/frontier/stage_vllm015_source.sh "${FRONTIER_STACK_CONFIG}"
```

Submit the one-node build and hardware-preflight job:

```bash
sbatch -A "${PROJECT_ID}" \
  --export=ALL,FRONTIER_STACK_CONFIG="${FRONTIER_STACK_CONFIG}" \
  scripts/frontier/build_vllm015_source.sbatch
```

The build is offline inside the allocation.  It uses the staged vLLM 0.15 and
Triton sources, compiles for `gfx90a`, installs vLLM non-editably, builds
HAProxy from the checksum-verified staged tarball, and runs a one-GCD hardware
preflight.

## 3. Apply the Ray Serve dependency repair

On the login node:

```bash
bash scripts/frontier/repair_ray_serve_compat.sh "${FRONTIER_STACK_CONFIG}"
```

This command:

1. installs `fastapi==0.136.0` with its compatible dependency closure;
2. verifies Ray is exactly 2.53.0;
3. replaces Ray Serve's obsolete
   `field.label == FieldDescriptor.LABEL_REPEATED` expression with
   `field.is_repeated`, refusing unexpected or ambiguous source;
4. verifies FastAPI cloudpickle serialization;
5. performs a `DeploymentConfig` protobuf round trip;
6. reinstalls ExaServe non-editably; and
7. repacks the Python runtime archive for node-local staging.

The repair deliberately validates behavior rather than requiring a particular
protobuf version.  Protobuf 7.36.0 works after the Ray source adaptation.

Expected final output:

```text
Ray Serve dependency repair, semantic preflight, and runtime repack completed
```

`pack_python_runtime.sh` repeats the semantic checks and refuses to publish an
archive if either compatibility boundary regresses.

## 4. Download the Granite model

Run on a networked login node:

```bash
bash scripts/frontier/download_granite33_8b.sh "${FRONTIER_STACK_CONFIG}"
```

The downloader uses the pinned Hugging Face revision, validates configuration,
tokenizer, and safetensors files, writes ExaServe's content marker, and
atomically publishes the model under `EXASERVE_MODEL_STORAGE_PATH`.

## 5. Configure the deployment

The qualified example is:

```text
examples/config.frontier.granite33-8b.8n8g.yaml
```

It requests eight nodes and eight GCDs per node.  To use four nodes:

```yaml
num_nodes: 4
num_gpus_per_node: 8
```

To run two replicas on one node:

```yaml
num_nodes: 1
num_gpus_per_node: 2
```

Do not add `num_replicas` when the goal is one TP=1 replica per declared GCD.
An explicit value is a promise; compilation fails rather than silently
reducing it when the requested placement does not fit.

To use a copied configuration outside the checkout:

```bash
export EXASERVE_DEPLOYMENT_CONFIG=/absolute/path/granite.yaml
```

## 6. Render and submit

Always inspect the generated Slurm job before submission:

```bash
bash scripts/frontier/submit_granite33_8b_8n8g.sh \
    dry-run "${FRONTIER_STACK_CONFIG}"
```

The header must contain the intended values, including:

```text
#SBATCH --partition=extended
#SBATCH --qos=debug
#SBATCH --nodes=<configured nodes>
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=<configured GCDs>
#SBATCH --constraint=nvme
#SBATCH --network=disable_rdzv_get
```

Submit a fresh generation and wait for READY:

```bash
bash scripts/frontier/submit_granite33_8b_8n8g.sh \
    submit-new-wait "${FRONTIER_STACK_CONFIG}"
```

The batch allocation contains one long-lived ExaServe/NodeSupervisor task per
node.  Short finite staging steps also use one task per node with seven CPUs:

```text
srun --nodes=N --ntasks-per-node=1 --cpus-per-task=7
```

They intentionally do not set `--cpu-bind` or `--threads-per-core`.

## 7. Test concurrent requests

```bash
bash scripts/frontier/test_granite33_8b_8n8g.sh \
    "${FRONTIER_STACK_CONFIG}"
```

Override the workload without editing files:

```bash
REQUESTS=1024 CONCURRENCY=256 \
  bash scripts/frontier/test_granite33_8b_8n8g.sh \
    "${FRONTIER_STACK_CONFIG}"
```

The client reports success/failure counts, requests per second, completion
tokens, mean latency, p50, p95, maximum latency, and one response.  It exits
nonzero if any request fails.

## Frontier fixes implemented by this port

### Site, scheduler, and topology

- Added the `olcf-frontier` site profile with Slurm, ROCm, MI250X GCDs, 56 CPU
  cores per node, eight GCDs per node, Orion shared storage, and NVMe local
  staging.
- Added native Slurm job rendering, observation, cancellation, allocation
  discovery, GPU directives, NVMe constraint, Frontier network setting, and
  explicit QoS rendering.
- Made the deployment plan's vendor select the compatibility profile; a ROCm
  plan can no longer activate Aurora/XPU patches accidentally.
- Changed Ray node-address discovery so the Aurora `.hsn.cm.aurora...` suffix
  is never applied on Frontier.
- Added a 120-second Ray raylet startup wait.  The upstream default was too
  short when eight workers joined simultaneously.

### ROCm and vLLM

- Uses `HIP_VISIBLE_DEVICES` for replica isolation.  Ray 2.53 rejects an
  inherited `ROCR_VISIBLE_DEVICES`; the Frontier environment translates and
  removes it before Ray imports.
- Stages vLLM and Triton sources on the login node and builds vLLM offline on a
  compute node against PyTorch/ROCm 7.1.
- Builds AMD SMI from a writable copy because ROCm's copy under `/opt` is
  read-only and setuptools attempts to create metadata beside it.
- Requires non-editable installs so compute-node Python imports come from the
  packed node-local `site-packages`, not metadata-heavy shared source paths.
- Packs the Python environment into one sequential, uncompressed archive and
  extracts it independently on every allocated node's NVMe.

### Source and model staging

- The one-node path bypasses redundant `srun` boundaries while preserving
  manifests, atomic publication, generation identity, node identity, and
  receipts.
- Multi-node finite `srun` steps receive a bounded five-second descendant-exit
  grace.  Slurm may retain a helper briefly after the leader exits; longer
  lived descendants still fail closed and are terminated.
- Multi-node source/model staging uses one MPI task per node and seven CPUs per
  task, with no explicit CPU/thread binding.  This avoids Cray MPICH's
  `NIC_POLICY=NUMA` failure when an unconfined rank spans NUMA nodes.
- Source and model distribution remain manifest-verified and atomically
  published on every rank.

### Ray Serve compatibility

- Pins FastAPI 0.136.0.  FastAPI 0.139.2 embeds a thread lock that Ray Serve
  2.53 cannot serialize through `@serve.ingress`.
- Accepts an empty compatibility-overlay manifest for Frontier's deliberately
  patch-free ROCm profile.
- Uses Ray's native 60-second `HTTP_PROXY_TIMEOUT` on Frontier while retaining
  Aurora/XPU's patched 3600-second contract.
- Adapts Ray Serve's removed protobuf descriptor API from `field.label` to
  `field.is_repeated` and validates the exact protobuf conversion before
  runtime packing.

### Concurrent Triton compilation

- Assigns each Serve replica a separate Triton cache below ExaServe's
  generation- and rank-scoped node-local runtime tree.
- The vLLM EngineCore inherits `TRITON_CACHE_DIR` before importing Triton.
- The cache is included in ExaServe's existing owned-runtime cleanup.

Without this isolation, eight replicas on one node could compile identical
kernels into one shared cache.  One EngineCore lost the publish/read race and
raised `FileNotFoundError`; all requests subsequently routed to that replica
returned HTTP 500.  Per-replica node-local caches eliminated the failure.

## Failure chronology

| Failure | First cause | Resolution |
|---|---|---|
| One-node staging rejected `srun` descendants | Redundant one-node `srun` helper outlived the leader briefly | Direct local staging for one node |
| Ray Serve ingress could not be serialized | FastAPI introduced an internal lock | Pin FastAPI 0.136.0 and serialize in preflight |
| Proxy timing contract failed | Frontier inherited Aurora's patched 3600-second requirement | Verify Ray's native 60-second timeout on ROCm |
| Serve controller failed on `FieldDescriptor.label` | Protobuf 7 removed the legacy API used by Ray | Patch Ray to `field.is_repeated`; semantic round trip |
| Multi-node staging reported finite `srun` descendants | Slurm helper teardown lag | Bounded five-second descendant grace |
| Cray MPICH rejected `NIC_POLICY=NUMA` | Staging ranks were not confined to one NUMA domain | One task/node, seven CPUs/task, no CPU/thread binding flags |
| Ray workers missed startup deadline | Eight-node simultaneous startup exceeded the default wait | `RAY_raylet_start_wait_time_s=120` |
| 10 of 256 requests returned HTTP 500 | One of 64 EngineCores died in a shared Triton cache race | Per-replica node-local `TRITON_CACHE_DIR` |

## Troubleshooting without `rg`

Frontier does not need ripgrep.  Use `find` and `grep`:

```bash
JOBID=1234567
find "${WORK_ROOT}/granite33-8b-logs" -maxdepth 1 -type f \
    -name "*${JOBID}*" -print -exec tail -n 300 {} \;
```

Extract likely first causes:

```bash
grep -inE -B 10 -A 20 \
  'traceback|exception|error|out of memory|engine.*dead|HTTP 500|internal server' \
  "${WORK_ROOT}/granite33-8b-logs/"*"${JOBID}"*
```

Inspect scheduler state:

```bash
squeue -j "${JOBID}"
sacct -j "${JOBID}" --format=JobID,State,ExitCode,Elapsed,NodeList
```

## Qualification and scale limits

The current site profile intentionally has no production envelope.  A
production claim should retain evidence at increasing tiers and add a reviewed
`ScaleEnvelope` only after correctness, failure propagation, cleanup, and
performance are qualified.  Suggested next tiers are 16, 32, 64, 128, and 256
nodes, followed by TP and PP qualification as separate topology families.

Scaling beyond 256 nodes requires all of the following:

1. a larger site maximum and approved scale envelope;
2. Slurm policy/allocation approval for the requested partition and QoS;
3. Ray control-plane and HAProxy qualification at the larger backend count;
4. model/source broadcast and cleanup evidence at that scale; and
5. a concurrent workload sized to exercise every replica.

It must not be enabled only by increasing `EXASERVE_SITE_MAX_NODES`.

## Repository hygiene

Do not commit:

- project- or username-specific stack configuration files;
- Slurm logs, job IDs, generated plans, run directories, or submission
  registries;
- model weights, Python environments, runtime archives, build trees, or
  wheelhouses;
- delivery tarballs, IDE metadata, `nohup.out`, or macOS `._*` files; or
- Hugging Face credentials or tokens.

Commit the portable example configuration, scripts, tests, source changes, and
this README.  Keep the edited `frontier_stack.conf` outside the checkout.

## Upstream compatibility references

- Ray Serve / FastAPI serialization incompatibility:
  <https://github.com/ray-project/ray/issues/64939>
- Ray Serve / protobuf 7 incompatibility:
  <https://github.com/ray-project/ray/issues/64710>
- Triton concurrent shared-cache race:
  <https://github.com/triton-lang/triton/issues/11512>
