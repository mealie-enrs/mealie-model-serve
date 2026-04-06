"""
eval/benchmark.py

Latency and throughput benchmark for Mealie ML serving endpoints.
Targets serving-rahil/ FastAPI apps accepting base64 food images.
Measures p50/p95/p99 latency and throughput for each serving configuration.

All experiments must run on Chameleon Cloud inside a container.
Local runs do not count for credit per course rubric.

Usage:
    # Baseline (port 8000)
    python eval/benchmark.py --url http://localhost:8000/predict --option baseline_pytorch --n 200 --concurrency 1

    # ONNX graph optimized (port 8001)
    python eval/benchmark.py --url http://localhost:8001/predict --option onnx_graph_opt --n 200 --concurrency 4

    # ONNX quantized (port 8002)
    python eval/benchmark.py --url http://localhost:8002/predict --option onnx_quantized --n 200 --concurrency 4

    # ONNX CUDA GPU (port 8003)
    python eval/benchmark.py --url http://localhost:8003/predict --option onnx_cuda --n 200 --concurrency 8

    # Save results to JSON
    python eval/benchmark.py --url http://localhost:8000/predict --option baseline_pytorch --n 200 --concurrency 1 --out eval/results/baseline_c1.json

"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image


# Test image generation 

def make_test_payload(size: int = 224) -> dict:
    """
    Generate a representative food-colored image encoded as base64 JPEG.
    Uses orange-brown tones to simulate a food photo.
    BYPASS_INFERENCE_GATE=1 must be set in the serving container so this
    synthetic image passes the food class gate and reaches full inference.

    Returns the complete request payload matching PredictRequest schema.
    """
    img = Image.new("RGB", (size, size), color=(210, 140, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {
        "request_id": "benchmark-001",
        "image": b64,
        "image_filename": "benchmark_test.jpg",
    }


# Single request 

def send_one(url: str, payload: dict, timeout: int = 30) -> tuple[float, int]:
    """Send one POST request. Returns (latency_ms, status_code)."""
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        return (time.perf_counter() - t0) * 1000, resp.status_code
    except requests.exceptions.RequestException as e:
        print(f"  Request error: {e}", file=sys.stderr)
        return (time.perf_counter() - t0) * 1000, 0


# Health check 

def check_health(predict_url: str) -> bool:
    """Check /healthz and /readyz before benchmarking."""
    base = predict_url.rsplit("/predict", 1)[0]
    try:
        h = requests.get(f"{base}/healthz", timeout=5)
        r = requests.get(f"{base}/readyz", timeout=5)
        if h.status_code == 200 and r.status_code == 200:
            print("  Health OK | readyz OK")
            return True
        print(f"  Health: {h.status_code} | readyz: {r.status_code}")
        return False
    except Exception as e:
        print(f"  Health check failed: {e}", file=sys.stderr)
        return False


#  Metadata

def get_metadata(predict_url: str) -> dict:
    """Fetch serving metadata for recording in results."""
    base = predict_url.rsplit("/predict", 1)[0]
    try:
        resp = requests.get(f"{base}/healthz", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


# Warmup 

def warmup(url: str, payload: dict, n: int = 5):
    """Send warmup requests to ensure model is loaded and JIT-warmed."""
    print(f"  Warming up ({n} requests)...")
    for _ in range(n):
        send_one(url, payload)
    print("  Warmup complete.")


# Benchmark 

def run_benchmark(
    url: str,
    n_requests: int,
    concurrency: int,
    payload: dict,
) -> dict:
    """
    Run benchmark with ThreadPoolExecutor for concurrent requests.
    Returns dict with latency percentiles, throughput, and error rate.

    Note: 422 responses from the inference gate (non-food image detected)
    are counted as errors. Run with BYPASS_INFERENCE_GATE=1 in the container
    to ensure synthetic images pass through to full inference.
    """
    latencies = []
    errors = 0
    status_counts: dict[int, int] = {}

    t_wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(send_one, url, payload)
            for _ in range(n_requests)
        ]
        for fut in as_completed(futures):
            lat, status = fut.result()
            latencies.append(lat)
            status_counts[status] = status_counts.get(status, 0) + 1
            if status != 200:
                errors += 1

    total_wall_s = time.perf_counter() - t_wall_start
    latencies.sort()
    n = len(latencies)

    def pct(p: float) -> float:
        idx = min(int(p / 100 * n), n - 1)
        return round(latencies[idx], 2)

    return {
        "endpoint_url":   url,
        "n_requests":     n_requests,
        "concurrency":    concurrency,
        "total_wall_s":   round(total_wall_s, 3),
        "throughput_rps": round(n_requests / total_wall_s, 2),
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


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Mealie ML serving benchmark — p50/p95/p99 latency and throughput"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/predict",
        help="Serving endpoint URL (default: http://localhost:8000/predict)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Total number of requests (default: 200)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent threads (default: 1)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup requests before measuring (default: 5)",
    )
    parser.add_argument(
        "--option",
        default="baseline_pytorch",
        help="Serving option ID label for results (default: baseline_pytorch)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to save results JSON",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Synthetic test image size in pixels (default: 224)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Mealie ML Serving Benchmark")
    print("=" * 60)
    print(f"  Option      : {args.option}")
    print(f"  Endpoint    : {args.url}")
    print(f"  Requests    : {args.n}")
    print(f"  Concurrency : {args.concurrency}")
    print()

    # Health check
    print("Checking service health...")
    if not check_health(args.url):
        print("WARNING: Service not healthy. Proceeding anyway.")
    print()

    # Metadata
    meta = get_metadata(args.url)
    if meta:
        print("Service info:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print()

    # Build payload once — same image for all requests
    # (isolates inference latency from image generation overhead)
    payload = make_test_payload(args.image_size)

    # Warmup
    warmup(args.url, payload, n=args.warmup)
    print()

    # Benchmark
    print(f"Running {args.n} requests at concurrency={args.concurrency}...")
    results = run_benchmark(args.url, args.n, args.concurrency, payload)
    results["serving_option_id"] = args.option
    results["metadata"] = meta

    # Print results
    print()
    print("Results:")
    print(f"  p50 latency   : {results['p50_ms']} ms")
    print(f"  p95 latency   : {results['p95_ms']} ms")
    print(f"  p99 latency   : {results['p99_ms']} ms")
    print(f"  Mean latency  : {results['mean_ms']} ms")
    print(f"  Throughput    : {results['throughput_rps']} req/s")
    print(f"  Error rate    : {results['error_rate'] * 100:.1f}%")
    print(f"  Status counts : {results['status_counts']}")
    print(f"  Wall time     : {results['total_wall_s']} s")
    print()

    if args.out:
        import os
        os.makedirs(
            os.path.dirname(args.out) if os.path.dirname(args.out) else ".",
            exist_ok=True,
        )
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.out}")

    return results


if __name__ == "__main__":
    main()