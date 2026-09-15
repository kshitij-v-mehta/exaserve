#!/usr/bin/env python3
"""Issue bounded concurrent chat requests to the Granite scale deployment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


MODEL_ID = "ibm-granite/granite-3.3-8b-instruct"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def endpoint_from_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


def request_once(endpoint: str, index: int, timeout: float) -> dict:
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Request {index}: In one sentence, state which U.S. national "
                    "laboratory operates the Frontier supercomputer."
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
        "stream": False,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        latency = time.perf_counter() - started
        usage = body.get("usage") or {}
        choices = body.get("choices") or []
        text = ""
        if choices:
            text = ((choices[0].get("message") or {}).get("content") or "").strip()
        return {
            "ok": True,
            "latency": latency,
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "text": text,
        }
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        return {
            "ok": False,
            "latency": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("requests and concurrency must be positive")

    endpoint = endpoint_from_url(args.url)
    print(f"Endpoint: {endpoint}")
    print(f"Requests: {args.requests}; concurrency: {args.concurrency}")
    started = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(request_once, endpoint, index, args.timeout)
            for index in range(args.requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - started

    successes = [item for item in results if item["ok"]]
    failures = [item for item in results if not item["ok"]]
    latencies = [float(item["latency"]) for item in successes]
    completion_tokens = sum(int(item["completion_tokens"]) for item in successes)

    print(f"Successful: {len(successes)}/{args.requests}")
    print(f"Failed: {len(failures)}/{args.requests}")
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"Throughput: {len(successes) / elapsed:.3f} requests/s")
    print(f"Completion tokens: {completion_tokens}")
    if latencies:
        print(f"Mean latency: {statistics.fmean(latencies):.3f} s")
        print(f"p50 latency: {percentile(latencies, 0.50):.3f} s")
        print(f"p95 latency: {percentile(latencies, 0.95):.3f} s")
        print(f"Max latency: {max(latencies):.3f} s")
        print(f"Sample response: {successes[0]['text']}")
    for item in failures[:10]:
        print(f"Failure: {item['error']}")
    return 0 if not failures and len(successes) == args.requests else 1


if __name__ == "__main__":
    raise SystemExit(main())
