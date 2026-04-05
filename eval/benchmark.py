"""
eval/benchmark.py : Latency and throughput benchmark for Mealie ML serving.
Targets serving stack (/predict endpoint with float feature vectors).
Measures p50/p95/p99 latency and throughput for each serving configuration.

Usage:
    # Baseline (port 18080)
    python eval/benchmark.py --url http://<IP>:18080/predict --n 200 --concurrency 1 --option baseline_http

    # ONNX optimized (port 18081)
    python eval/benchmark.py --url http://<IP>:18081/predict --n 200 --concurrency 4 --option onnx_ort_all

    # Multi-worker (port 18082)
    python eval/benchmark.py --url http://<IP>:18082/predict --n 200 --concurrency 8 --option multi_worker

    # GPU (port 18083)
    python eval/benchmark.py --url http://<IP>:18083/predict --n 200 --concurrency 8 --option gpu_cuda

    # Save results
    python eval/benchmark.py --url http://<IP>:18080/predict --n 200 --concurrency 1 --out results/baseline_c1.json

"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


# Test payload ::
# /predict expects {"inputs": [[float, ...]]}
# Using 4-feature Iris-style vectors matching the toy ONNX model in MLflow.
# When Shubham's food classifier is registered, update this to match its input shape.
DUMMY_PAYLOAD = {
    "inputs": [[5.1, 3.5, 1.4, 0.2]]
}


def send_one(url: str, payload: dict, timeout: int = 30):
    """Send one POST request. Returns (latency_ms, status_code)."""
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        return (time.perf_counter() - t0) * 1000, resp.status_code
    except requests.exceptions.RequestException as e:
        print(f"  Request error: {e}", file=sys.stderr)
        return (time.perf_counter() - t0) * 1000, 0


def check_health(base_url: str) -> bool:
    """Check /healthz and /readyz before benchmarking."""
    try:
        base = base_url.rsplit("/predict", 1)[0]
        h = requests.get(f"{base}/healthz", timeout=5)
        r = requests.get(f"{base}/readyz", timeout=5)
        if h.status_code == 200 and r.status_code == 200:
            print(f"  Health OK | readyz OK")
            return True
        print(f"  Health: {h.status_code} | readyz: {r.status_code}")
        return False
    except Exception as e:
        print(f"  Health check failed: {e}", file=sys.stderr)
        return False


def get_metadata(base_url: str) -> dict:
    """Fetch /metadata to record model version and serving config."""
    try:
        base = base_url.rsplit("/predict", 1)[0]
        resp = requests.get(f"{base}/metadata", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def warmup(url: str, payload: dict, n: int = 5):
    print(f"  Warming up ({n} requests)...")
    for _ in range(n):
        send_one(url, payload)
    print("  Warmup complete.")


def run_benchmark(url: str, n_requests: int, concurrency: int, payload: dict) -> dict:
    latencies, errors = [], 0
    status_counts: dict = {}
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_one, url, payload) for _ in range(n_requests)]
        for fut in as_completed(futures):
            lat, status = fut.result()
            latencies.append(lat)
            status_counts[status] = status_counts.get(status, 0) + 1
            if status != 200:
                errors += 1

    total_s = time.perf_counter() - t_start
    latencies.sort()
    n = len(latencies)

    def pct(p):
        return round(latencies[min(int(p / 100 * n), n - 1)], 2)

    return {
        "n_requests":     n_requests,
        "concurrency":    concurrency,
        "total_wall_s":   round(total_s, 3),
        "throughput_rps": round(n_requests / total_s, 2),
        "p50_ms":         pct(50),
        "p95_ms":         pct(95),
        "p99_ms":         pct(99),
        "mean_ms":        round(statistics.mean(latencies), 2),
        "min_ms":         round(latencies[0], 2),
        "max_ms":         round(latencies[-1], 2),
        "errors":         errors,
        "error_rate":     round(errors / n_requests, 4),
        "status_counts":  status_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Mealie ML serving benchmark")
    parser.add_argument("--url",         default="http://localhost:18080/predict")
    parser.add_argument("--n",           type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup",      type=int, default=5)
    parser.add_argument("--option",      default="baseline_http",
                        help="Serving option ID label (for results file)")
    parser.add_argument("--out",         default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Mealie ML Serving Benchmark")
    print("=" * 60)
    print(f"  Option      : {args.option}")
    print(f"  Endpoint    : {args.url}")
    print(f"  Requests    : {args.n}")
    print(f"  Concurrency : {args.concurrency}")
    print()

    print("Checking service health...")
    if not check_health(args.url):
        print("WARNING: Service not healthy. Proceeding anyway.")
    print()

    meta = get_metadata(args.url)
    if meta:
        print("Metadata:")
        print(f"  model_name    : {meta.get('model_name', 'unknown')}")
        print(f"  model_version : {meta.get('model_version', 'unknown')}")
        print(f"  model_alias   : {meta.get('model_alias', 'unknown')}")
        print(f"  provider      : {meta.get('provider', 'unknown')}")
        print(f"  build_sha     : {meta.get('build_sha', 'unknown')}")
        print(f"  serving_option: {meta.get('serving_option_id', 'unknown')}")
        print()

    warmup(args.url, DUMMY_PAYLOAD, n=args.warmup)
    print()
    print(f"Running {args.n} requests at concurrency={args.concurrency}...")
    results = run_benchmark(args.url, args.n, args.concurrency, DUMMY_PAYLOAD)

    results["endpoint_url"] = args.url
    results["serving_option_id"] = args.option
    results["metadata"] = meta

    print()
    print("Results:")
    print(f"  p50 latency   : {results['p50_ms']} ms")
    print(f"  p95 latency   : {results['p95_ms']} ms")
    print(f"  p99 latency   : {results['p99_ms']} ms")
    print(f"  Mean latency  : {results['mean_ms']} ms")
    print(f"  Throughput    : {results['throughput_rps']} req/s")
    print(f"  Error rate    : {results['error_rate']*100:.1f}%")
    print(f"  Status counts : {results['status_counts']}")

    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.out}")

    return results


if __name__ == "__main__":
    main()
