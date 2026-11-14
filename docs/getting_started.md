# Getting started

Choose the environment for the work you intend to do. The portable development
environment is not an installation recipe for Aurora's serving stack.

## Portable development

Use a Linux workstation or a suitable CI runner with Git, Make, Python, and
`uv==0.13.4` (install it with `python3 -m pip install uv==0.13.4`).
Package metadata accepts Python 3.10 or newer; current portable CI targets
Python 3.12, matching Aurora's current Python 3.12.12 runtime. Use Python 3.12
for development. Other interpreter versions are outside the current CI scope,
and portable CI does not qualify any GPU runtime.

The committed uv lock resolves Linux Python 3.11–3.13; the existing optional
vLLM dependency excludes Python 3.14, and the test group's Pillow 13.0.0 requires
Python 3.11. Use a CI interpreter version above for development rather than
assuming every version allowed by package metadata is covered by the lock or
tested by CI.

```bash
git clone https://github.com/AI-ModCon/exaserve.git
cd exaserve
make install-dev
```

On Aurora, the stack loaded in [Prepare the runtime](#prepare-the-runtime)
puts its own uv 0.10.1 first on `PATH`, which this project refuses. Install the
pinned uv for your user with `python3 -m pip install --user uv==0.13.4` and
pass it to make, for example
`make UV="$(python3 -m site --user-base)/bin/uv" install-dev`.

`make install-dev` synchronizes the portable development and test dependency
groups from [uv.lock](../uv.lock). The dependency declarations and tool settings
live in [pyproject.toml](../pyproject.toml). Ray, vLLM, MPI, and accelerator
drivers are not required for the portable control-plane test environment.

From the repository root:

```bash
make lint
make format-check
make type-check
make test
make test-cov
make build
```

These targets provide lint/format checks, the selected typed-core check, tests,
coverage, and distribution builds. The typed-core check is not a claim that the
entire repository is statically checked. Use locked tool invocations for
individual checks, for example:

```bash
uv run --locked ruff check src/exaserve/config.py
uv run --locked exaserve-status --help
```

The filesystem/staging tests exercise the real node-local guards and require
`/tmp` to be a separate writable tmpfs from `/home` and other declared shared
roots. Aurora compute nodes already provide that layout. GitHub-hosted test and
installed-wheel jobs mount a temporary tmpfs on their disposable runners before
installing test dependencies. On another workstation, use a suitable isolated
VM/container with that layout; do not disable the locality checks to run tests.
Tests of the optional Ray/FastAPI serving API skip explicitly when those
dependencies are absent. The portable plan, gateway-routing, trace, and
filesystem tests still run without the serving stack.

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before changing dependencies, runtime
contracts, or generated compatibility files. Do not regenerate the lockfile as
a routine response to a failing test; first establish whether the interpreter
and dependency profile are correct.

## Aurora development and serving

This section is the operator setup guide, not a requirement to install a
maintainer's personal scripts. Use the site's scheduler and module system.
Portable development dependencies and the serving runtime are separate.

### Obtain compute resources

Login nodes are for editing, inspection, configuration preparation, submission
and brief lightweight checks. Full suites, package/build gates, benchmarks,
evaluations, MPI and GPU work belong on allocated compute nodes.

For ad-hoc work, reuse an authorized allocation or request an interactive PBS
session using the site's published instructions and your project's account,
queue, node count and walltime. ExaServe does not supply or require a personal
lease manager. Do not guess allocation settings or enter a compute node through
plain SSH and assume that creates a valid scheduler session.

Before using an interactive session, verify:

- `PBS_JOBID` is set and `PBS_NODEFILE` is readable.
- The current hostname belongs to the nodefile and the node count is correct.
- The shell was entered through the scheduler's interactive workflow
  (`PBS_ENVIRONMENT=PBS_INTERACTIVE` for native PBS), not fabricated exports.
- The remaining allocation walltime is sufficient; inspect
  `qstat -f "$PBS_JOBID"`. If your site offers an approved lease service, check
  its deadline too and preserve its real scheduler state.

For normal serving/evaluation, the [canonical submission commands](usage.md)
create native PBS batch jobs; a separate interactive allocation is not required
to submit them. Those jobs execute in their own allocation. Do not set
interactive/lease markers in a batch job to bypass checks. SSH from an allocated
head is limited to nodes actually assigned to that session.

### Prepare the runtime

The serving profile is Aurora's `frameworks/2025.3.1` stack: Python 3.12.12,
Ray 2.53.0 and vLLM 0.15.0+xpu. Since Aurora's default image changed to
26.181.0 in late September 2026, the unversioned `frameworks` module is
2026.1.0 (Python 3.12, Ray 2.56.1, vLLM 0.26), which ExaServe's compatibility
check rejects, and `module load frameworks/2025.3.1` fails from the default
module tree. The validation runs loaded the profiled stack from the 26.26.0
image's module tree, as below. Load it this way in every fresh session; brief
permitted login-node Python/Go checks also need it rather than the old system
Python:

```bash
module purge
_old=/opt/aurora/26.26.0/spack/unified/1.1.1/install/modulefiles
_mpich=$_old/mpich/5.0.0.aurora_test.87e2045-po72xy6
export MODULEPATH=$_mpich/Core:$_mpich/intel-oneapi-compilers/2025.3.1:$_old/Core:$_old/intel-oneapi-compilers/2025.3.1:/usr/share/lmod/modulefiles/Linux:/usr/share/lmod/modulefiles/Core:/usr/share/lmod/lmod/modulefiles/Core:/opt/cray/pals/lmod/modulefiles/core:/opt/cray/modulefiles:/opt/aurora/26.26.0/modulefiles
unset _old _mpich
module load frameworks/2025.3.1 cray-pals cray-libpals
module load go
unset ONEAPI_DEVICE_SELECTOR
```

The recipe uses the 26.26.0 image's MPICH and oneAPI 2025.3.1 module trees, and
`cray-pals` provides `mpiexec`. It works only while ALCF keeps the 26.26.0
tree. Compare the loaded stack with the
[site dependency record](../requirements/aurora-frameworks-2025.3.1.lock) and
[compatibility matrix](hardening/COMPATIBILITY_MATRIX.md). A version mismatch
requires an approved runtime update, not bypassing compatibility checks. These
are site-built distributions, not a public-PyPI installation recipe.

Use the qualified module-provided interpreter. Canonical launch requires its
executable to be non-shared and inside a declared read-only site root
(`/opt/aurora` in the current profile). A home-directory virtualenv can fail this
check even if its Python symlinks to the module interpreter. This is stricter
than merely being able to import Ray.

The operator must provision the exact approved ExaServe artifact and dependencies
for that interpreter and its batch bootstrap. Do not modify the shared module
installation yourself. The [README preflight](../README.md#install-on-aurora)
checks imports, not installation or qualification. The setup must provide:

- ExaServe and its control-plane dependencies to that Python with
  `PYTHONNOUSERSITE=1` and `PYTHONSAFEPATH=1`. A `pip install --user` alone is
  insufficient. Regular eval supplies its immutable source snapshot explicitly;
  this is distinct from installed-wheel serving.
- The compatible Ray/vLLM/XPU/MPI libraries, MPI compiler/launcher and required
  shared libraries.
- HAProxy on the batch runtime's `PATH`, or the configured gateway executable.
  The repo's [build helper](../scripts/build_haproxy.sh) requires an audited
  `HAPROXY_SOURCE_SHA256`; compile on compute, not a login node.
- Readable model inputs and writable run/scratch paths with sufficient space.
- Site-approved proxy/network settings when downloads require them, without
  routing allocation-internal serving traffic through an external proxy.

Do not install the `server` extra over the site stack or use
`uv sync --all-extras` to prepare it. Keep the portable `.venv` separate.
An isolated LiteLLM gateway uses its own configured Python; the parent stays
in the Ray/vLLM runtime. Use `ZE_AFFINITY_MASK` for Aurora device visibility,
never `ONEAPI_DEVICE_SELECTOR`.

Before launching, check the selected interpreter and tools without starting
Ray or importing the GPU stack:

```bash
python3 --version
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 python3 -c 'import sys, exaserve; print(sys.executable); print(exaserve.__file__)'
command -v mpiexec
command -v mpicc
command -v haproxy
```

Check the printed paths against the intended interpreter and artifact; an
inherited checkout `PYTHONPATH` must not masquerade as an installed-wheel check.
Successful command lookup is a preflight check, not hardware qualification.

### Configure evaluation environment preparation

Serving submissions and ordinary Ray evaluation jobs both source a setup file
that you select; there is no home-directory default. Save a site-reviewed shell
file at an absolute path readable by the job head. Its job is only environment
preparation; it must not launch daemons, implement readiness/cleanup, or stage
data per node. For example, using the recipe from
[Prepare the runtime](#prepare-the-runtime) and adapting the paths to your
installation:

```bash
# Contents of your operator-owned environment.sh; sourced by the job head.
module purge
_old=/opt/aurora/26.26.0/spack/unified/1.1.1/install/modulefiles
_mpich=$_old/mpich/5.0.0.aurora_test.87e2045-po72xy6
export MODULEPATH=$_mpich/Core:$_mpich/intel-oneapi-compilers/2025.3.1:$_old/Core:$_old/intel-oneapi-compilers/2025.3.1:/usr/share/lmod/modulefiles/Linux:/usr/share/lmod/modulefiles/Core:/usr/share/lmod/lmod/modulefiles/Core:/opt/cray/pals/lmod/modulefiles/core:/opt/cray/modulefiles:/opt/aurora/26.26.0/modulefiles
unset _old _mpich
module load frameworks/2025.3.1 cray-pals cray-libpals || return 1
module load go || return 1
unset ONEAPI_DEVICE_SELECTOR
export PATH="/absolute/path/to/haproxy/bin:$PATH"
```

Keep the module-provided `python3`; do not prepend a home virtualenv's Python.
Select the setup file and the site variables before submitting a deployment or
materializing a run:

```bash
export EXASERVE_ENV_SCRIPT_AURORA=/absolute/path/to/environment.sh
export EXASERVE_PROJECT_ROOT=/lus/flare/projects/YOUR_PROJECT
export EXASERVE_PROJECT_ACCOUNT=YOUR_PROJECT
PYTHONPATH="$PWD:$PWD/src" python3 -m eval.site_config get env_script_aurora
```

Eval and ClientLab run from the checkout and import ExaServe from its `src/`
tree, as their materialized jobs do: run them from the repository root with
`PYTHONPATH="$PWD:$PWD/src"`, as in the check above. Do not set it in the
shell you use for the installed-wheel check or for `exaserve-serve-submit`.

None of these variables has a built-in default. `exaserve-serve-submit` needs
the environment script, the project root (or `EXASERVE_MODEL_STORAGE_PATH`)
and an account (`--project-account` or `EXASERVE_PROJECT_ACCOUNT`). Its job
sources `EXASERVE_ENV_SCRIPT_AURORA`, which must name an existing file when you
submit. The site profile derives the site's
model-storage root from `EXASERVE_PROJECT_ROOT` (or takes
`EXASERVE_MODEL_STORAGE_PATH`), even when the YAML sets `model_storage_path`;
a YAML launch with `exaserve-launch-cluster` needs it too. The PBS account
comes from `--project-account` or `EXASERVE_PROJECT_ACCOUNT`. Evaluation
derives storage paths that are not set explicitly from `EXASERVE_PROJECT_ROOT`
and charges PBS jobs to `EXASERVE_PROJECT_ACCOUNT` unless a spec sets
`scheduler.project`; when neither storage variable is exported, `run
materialize` takes the storage path from the local site configuration below
(serving does not). Commands that need a value fail with an error that names
it. These variables configure planning and submission on the head only: a
batch job does not need them, and ExaServe drops the path-valued ones from
every rank's environment (the account name is not a path), so exporting them
in a shell profile is safe.

Set model/data/output paths in your
[local site configuration](../eval/site_config_local.example.py).
`EXASERVE_SITE_CONFIG_LOCAL` can select another override file.
A per-spec `backend.args.ray.launch.env_script` takes precedence over the
environment variable, which takes precedence over local `SITE_OVERRIDES`;
use an absolute path in the spec too. The field names are existing API names,
not required filenames. If using the validation-only LiteLLM gateway, configure
`EXASERVE_LITELLM_PYTHON_PATH` (or its per-spec Python override) separately.

### Current bootstrap limitations

Environment preparation is not yet uniform across all entry points:

- **Serving CLI:** captures the submitting interpreter's `sys.executable`.
  Its job bootstrap sources the script that `EXASERVE_ENV_SCRIPT_AURORA` names
  at submission (an absolute path). The variable is required: without it,
  submission fails before writing anything, with an error that names it; no
  module is loaded by default. A retry that reuses a prepared or rejected
  submission (same `--log-dir`) is refused if the stored job sources a
  different script; use a fresh `--log-dir` or remove the intent record that
  the error names. Ensure the recorded interpreter, required libraries and
  gateway are available after that bootstrap; inspect the rendered job with
  `--dry-run` before submission. Loading a different module in the job does
  not change the recorded interpreter.
- **Allocation campaigns:** take the PBS account, the site environment script
  and the PBS output directory from `--account`, `--env-script` and
  `--pbs-spool` (or `EXASERVE_PROJECT_ACCOUNT` and `EXASERVE_ENV_SCRIPT_AURORA`)
  and record them in the immutable campaign identity (schema 2). The WP12 proof
  has not been re-run for this bootstrap; do not edit a generated campaign.
- **Historical hardening helpers:** some still assume the maintainer's
  allocation-lease helper. They are not the general installation interface.

Use the ordinary explicit-config Ray evaluation path where applicable; do not
advertise the allocation controller as portable until its new bootstrap is
qualified.

### Before the first run

Read [usage](usage.md) and [hardening status](hardening/STATUS.md).
Production admission is fail-closed; `production execution is not qualified`
is a qualification gate, not a request to bypass checks. Only predeclared,
authorized validation work may enable `validation_mode`.

Verify the working directory, configs, inputs, environment, output paths and
walltime. Use unique output directories and preserve prior artifacts. Monitor
stdout/stderr and canonical status; diagnose early errors and safely stop
clearly broken work. Report the exact command, session and evidence paths;
distinguish scheduler walltime expiry from application failure. Cleanup belongs
to the canonical executor, not a second shell lifecycle.

## Frontier development and serving

Frontier support is validation-only. Building the serving runtime, staging
models and submitting deployments are covered in
[Frontier validation deployment](deploy_frontier.md).

Run the portable suite on a compute node, not a login node, and work from
node-local storage: on Orion, test collection alone takes longer than 30
minutes. Clone the checkout and put the uv environment under `/tmp`. Inside a
`-C nvme` allocation, load the toolchain the tests need: the default `g++`
lacks C++20, and one test needs Go.

```bash
module load PrgEnv-gnu cray-python/3.12.12 go/1.26.1
export GOCACHE=/tmp/gocache GOMODCACHE=/tmp/gomod GOPATH=/tmp/gopath
export UV_PROJECT_ENVIRONMENT=/tmp/es-devvenv UV_PYTHON="$(command -v python3)"
export UV_CACHE_DIR=/mnt/bb/$USER/uv-cache
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES PYTHONPATH
git clone --branch <branch> /path/to/checkout /tmp/es-src && cd /tmp/es-src
make UV=/path/to/uv install-dev test lint format-check type-check
```

Use the pinned `uv==0.13.4`. Downloads go through the OLCF proxy
(`http_proxy`/`https_proxy=http://proxy.ccs.ornl.gov:3128/`).
