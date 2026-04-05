#!/usr/bin/env python3
"""
Load test a serving /predict endpoint from inside a container on Chameleon (or locally for dev).

Outputs JSON with p50/p95 latency (seconds), throughput (req/s), error rate, counts.
Use results to fill SERVING_OPTIONS.md table.

Example (on Chameleon VM, same Docker network as serving):
  python scripts/evaluate_serving.py --base-url http://model-serve:8080 --concurrency 8 --requests 400
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Any

import httpx


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


async def run_worker(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    n: int,
    latencies: list[float],
    errors: list[bool],
    sem: asyncio.Semaphore,
) -> None:
    for _ in range(n):
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.post(url, json=payload, timeout=60.0)
                ok = r.status_code == 200
            except Exception:
                ok = False
            dt = time.perf_counter() - t0
            latencies.append(dt)
            errors.append(not ok)


async def main_async() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:18080")
    p.add_argument("--path", default="/predict")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--requests", type=int, default=200)
    p.add_argument(
        "--payload-json",
        default='{"inputs":[[5.1,3.5,1.4,0.2]]}',
        help='JSON body (Iris-sized input by default)',
    )
    p.add_argument("--output-json", default="", help="Write summary to this path")
    args = p.parse_args()

    payload = json.loads(args.payload_json)
    url = args.base_url.rstrip("/") + args.path

    per_worker = max(1, args.requests // args.concurrency)
    total = per_worker * args.concurrency

    latencies: list[float] = []
    errors: list[bool] = []
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient() as client:
        wall0 = time.perf_counter()
        tasks = [
            asyncio.create_task(
                run_worker(client, url, payload, per_worker, latencies, errors, sem)
            )
            for _ in range(args.concurrency)
        ]
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - wall0

    latencies.sort()
    err_n = sum(1 for e in errors if e)
    ok_n = total - err_n
    summary = {
        "base_url": args.base_url,
        "path": args.path,
        "concurrency": args.concurrency,
        "requests_planned": total,
        "requests_recorded": len(latencies),
        "errors": err_n,
        "error_rate": err_n / total if total else 0.0,
        "wall_seconds": wall,
        "throughput_rps": ok_n / wall if wall > 0 else 0.0,
        "latency_p50_ms": percentile(latencies, 50) * 1000,
        "latency_p95_ms": percentile(latencies, 95) * 1000,
        "latency_mean_ms": (statistics.mean(latencies) * 1000) if latencies else 0.0,
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
