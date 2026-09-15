from __future__ import annotations

import faulthandler
import os
import time
from pathlib import Path


def main() -> None:
    model_path = Path(os.environ["SMOLLM2_LOCAL_MODEL_PATH"])
    required = ("config.json", "model.safetensors", "tokenizer.json")
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete local model at {model_path}: missing {missing}")

    local_site_packages = Path(os.environ["EXASERVE_LOCAL_SITE_PACKAGES"]).resolve()
    faulthandler.dump_traceback_later(300, repeat=True)

    import_start = time.monotonic()
    print("Importing vLLM from node-local site-packages.", flush=True)
    import vllm
    from vllm import LLM, SamplingParams

    import_seconds = time.monotonic() - import_start
    vllm_path = Path(vllm.__file__).resolve()
    if local_site_packages not in vllm_path.parents:
        raise RuntimeError(
            f"vLLM resolved to {vllm_path}, not node-local {local_site_packages}"
        )
    print(
        f"vLLM import completed in {import_seconds:.1f} seconds from {vllm_path}.",
        flush=True,
    )

    engine_start = time.monotonic()
    print("Creating the one-GCD vLLM engine.", flush=True)
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
        trust_remote_code=False,
    )
    print(
        f"vLLM engine initialized in {time.monotonic() - engine_start:.1f} seconds.",
        flush=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=48)
    outputs = llm.chat(
        [
            {
                "role": "user",
                "content": (
                    "In one sentence, name the U.S. national laboratory that "
                    "operates the Frontier supercomputer."
                ),
            }
        ],
        sampling_params=sampling,
    )
    text = outputs[0].outputs[0].text.strip()
    if not text:
        raise RuntimeError("vLLM returned an empty response")

    print(f"model={os.environ['SMOLLM2_MODEL_ID']}")
    print(f"response={text}")
    print("One-GCD offline vLLM generation passed.")
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
