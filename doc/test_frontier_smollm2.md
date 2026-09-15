# Frontier SmolLM2 smoke test

This smoke test uses `HuggingFaceTB/SmolLM2-360M-Instruct`, pinned to commit
`a10cc1512eabd3dde888204e902eca88bddb4951`. Downloads run on a Frontier login
node; compute-node execution is offline.

Set the portable stack configuration first:

```bash
export FRONTIER_STACK_CONFIG=/absolute/path/frontier_stack.conf
source "${FRONTIER_STACK_CONFIG}"
```

Download and verify the model:

```bash
bash scripts/frontier/download_smollm2.sh "${FRONTIER_STACK_CONFIG}"
```

After installing or changing ExaServe, create the node-local runtime archive:

```bash
bash scripts/frontier/reinstall_exaserve_noneditable.sh "${FRONTIER_STACK_CONFIG}"
bash scripts/frontier/pack_python_runtime.sh "${FRONTIER_STACK_CONFIG}"
```

Inspect and submit the one-node/one-GCD deployment:

```bash
bash scripts/frontier/submit_smollm2_exaserve.sh \
    dry-run "${FRONTIER_STACK_CONFIG}"
bash scripts/frontier/submit_smollm2_exaserve.sh \
    submit-new-wait "${FRONTIER_STACK_CONFIG}"
```

Send an OpenAI-compatible chat request through HAProxy:

```bash
bash scripts/frontier/test_smollm2_endpoint.sh "${FRONTIER_STACK_CONFIG}"
```

The complete Frontier installation, compatibility, scale, and troubleshooting
guide is in [`scripts/frontier/README.md`](../scripts/frontier/README.md).
