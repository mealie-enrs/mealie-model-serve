#!/usr/bin/env bash
# eval/run_all_benchmarks.sh
#
# Runs benchmarks for all 4 serving configurations on Chameleon.
# Must be executed on the Chameleon GPU instance (192.5.86.170)
# after all 4 Docker containers are running.
#
# Prerequisites:
#   1. SSH into serving-proj26 (192.5.86.170)
#   2. docker compose -f serving-rahil/docker-compose.yml up -d
#   3. Wait for all 4 containers to show healthy
#   4. Run this script
#
# Usage:
#   bash eval/run_all_benchmarks.sh
#   bash eval/run_all_benchmarks.sh --concurrency 4   # override concurrency
#
# Output: results saved to eval/results/ as JSON files


set -euo pipefail

CONCURRENCY=${1:-1}
N_REQUESTS=200
WARMUP=10
RESULTS_DIR="eval/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Mealie ML Serving — Full Benchmark Suite"
echo "Chameleon nc33 | RTX 6000 | $(date)"
echo "Requests: $N_REQUESTS | Concurrency: $CONCURRENCY | Warmup: $WARMUP"
echo "============================================================"
echo ""

# Helper: wait for a container to be ready
wait_for_ready() {
    local url=$1
    local name=$2
    local max_wait=60
    local waited=0

    echo "Waiting for $name to be ready..."
    while ! curl -sf "$url" > /dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [ $waited -ge $max_wait ]; then
            echo "ERROR: $name not ready after ${max_wait}s"
            exit 1
        fi
    done
    echo "$name is ready."
}

# Wait for all 4 containers
wait_for_ready "http://localhost:8000/healthz" "baseline_pytorch (port 8000)"
wait_for_ready "http://localhost:8001/healthz" "onnx_graph_opt (port 8001)"
wait_for_ready "http://localhost:8002/healthz" "onnx_quantized (port 8002)"
wait_for_ready "http://localhost:8003/healthz" "onnx_cuda (port 8003)"

echo ""
echo "All containers healthy. Starting benchmarks..."
echo ""

# Config 1: baseline_pytorch 
echo "------------------------------------------------------------"
echo "Config 1/4: baseline_pytorch (port 8000)"
echo "------------------------------------------------------------"
python eval/benchmark.py \
    --url "http://localhost:8000/predict" \
    --option "baseline_pytorch" \
    --n $N_REQUESTS \
    --concurrency $CONCURRENCY \
    --warmup $WARMUP \
    --out "$RESULTS_DIR/baseline_pytorch_c${CONCURRENCY}_${TIMESTAMP}.json"
echo ""

#  Config 2: onnx_graph_opt 
echo "------------------------------------------------------------"
echo "Config 2/4: onnx_graph_opt (port 8001)"
echo "------------------------------------------------------------"
python eval/benchmark.py \
    --url "http://localhost:8001/predict" \
    --option "onnx_graph_opt" \
    --n $N_REQUESTS \
    --concurrency $CONCURRENCY \
    --warmup $WARMUP \
    --out "$RESULTS_DIR/onnx_graph_opt_c${CONCURRENCY}_${TIMESTAMP}.json"
echo ""

#  Config 3: onnx_quantized 
echo "------------------------------------------------------------"
echo "Config 3/4: onnx_quantized (port 8002)"
echo "------------------------------------------------------------"
python eval/benchmark.py \
    --url "http://localhost:8002/predict" \
    --option "onnx_quantized" \
    --n $N_REQUESTS \
    --concurrency $CONCURRENCY \
    --warmup $WARMUP \
    --out "$RESULTS_DIR/onnx_quantized_c${CONCURRENCY}_${TIMESTAMP}.json"
echo ""

# Config 4: onnx_cuda 
echo "------------------------------------------------------------"
echo "Config 4/4: onnx_cuda (port 8003) ⭐ Most Promising"
echo "------------------------------------------------------------"
python eval/benchmark.py \
    --url "http://localhost:8003/predict" \
    --option "onnx_cuda" \
    --n $N_REQUESTS \
    --concurrency $CONCURRENCY \
    --warmup $WARMUP \
    --out "$RESULTS_DIR/onnx_cuda_c${CONCURRENCY}_${TIMESTAMP}.json"
echo ""

# Summary
echo "============================================================"
echo "Benchmark complete. Results saved to $RESULTS_DIR/"
echo ""
echo "Summary (p50 / p95 latency):"
for f in "$RESULTS_DIR"/*_${TIMESTAMP}.json; do
    option=$(python3 -c "import json,sys; d=json.load(open('$f')); print(d['serving_option_id'])")
    p50=$(python3 -c "import json,sys; d=json.load(open('$f')); print(d['p50_ms'])")
    p95=$(python3 -c "import json,sys; d=json.load(open('$f')); print(d['p95_ms'])")
    rps=$(python3 -c "import json,sys; d=json.load(open('$f')); print(d['throughput_rps'])")
    echo "  $option: p50=${p50}ms p95=${p95}ms throughput=${rps}req/s"
done
echo "============================================================"
