# Serving options (team deliverable — fill measurements on Chameleon)

This document satisfies the **serving team** rubric: compare **baseline vs optimized** deployments across **model-level**, **system-level**, and **infrastructure-level** changes, with tradeoffs (fast / good / cheap).

**Important:** Populate **p50 / p95 / throughput / error rate** from runs on a **Chameleon compute instance**, **inside Docker**, using `scripts/evaluate_serving.py` (see `docs/CHAMELEON_BENCHMARK.md`). Local numbers are for debugging only.

## How to run each option (same image, different env)

Image: `mealie-model-serve-api:local` (build via `infra/docker-compose.serving.yml`).

| Option ID | Docker env overrides (examples) |
|-----------|-----------------------------------|
| `baseline_http` | `SERVING_OPTION_ID=baseline_http` `ORT_GRAPH_OPTIMIZATION_LEVEL=disable` `UVICORN_WORKERS=1` |
| `onnx_ort_all` | `SERVING_OPTION_ID=onnx_ort_all` `ORT_GRAPH_OPTIMIZATION_LEVEL=all` `UVICORN_WORKERS=1` |
| `onnx_threads_tuned` | `SERVING_OPTION_ID=onnx_threads_tuned` `ORT_GRAPH_OPTIMIZATION_LEVEL=all` `ORT_INTRA_OP_NUM_THREADS=4` `ORT_INTER_OP_NUM_THREADS=2` `UVICORN_WORKERS=1` |
| `multi_worker_http` | `SERVING_OPTION_ID=multi_worker_http` `ORT_GRAPH_OPTIMIZATION_LEVEL=all` `UVICORN_WORKERS=4` — **4× model RAM** |
| `triton_dynamic_batch` | Separate stack: Triton (`infra/triton/README.md`); not FastAPI — infrastructure-level |

**GPU row (optional):** set `MODEL_PROVIDER=cuda` on a Chameleon GPU node and add a row.

## Rollout lanes

For the deployed k3s stack, use separate rollout lanes:

| Lane | URL | Registry alias | Purpose |
|------|-----|----------------|---------|
| production | `http://<ip>:30608` | `production` | Stable endpoint for app traffic |
| canary | `http://<ip>:30609` | `canary` | Candidate validation before promotion |

This is endpoint-level canarying, not weighted traffic splitting yet.

## Results table (copy to your report after Chameleon runs)

⭐ = best for that priority column (latency / cost / simplicity).

| Option | Endpoint URL | Model version | Code version (git SHA) | Hardware | p50 latency | p95 latency | Throughput (req/s) | Error rate | Concurrency tested | Compute instance type | Notes |
|--------|--------------|---------------|-------------------------|----------|------------|------------|--------------------|------------|--------------------|-----------------------|-------|
| **baseline_http** ⭐ simplicity | `http://<ip>:18080` | *(from `/metadata`)* | `BUILD_SHA` from `/metadata` | CPU | *fill* | *fill* | *fill* | *fill* | e.g. 4 | e.g. `m1.large` KVM | ORT graph opts **off**, 1 uvicorn worker |
| **onnx_ort_all** ⭐ cost/latency (CPU) | same | same | same | CPU | *fill* | *fill* | *fill* | *fill* | e.g. 4 | same | **Model-level**: ORT `all` graph optimizations |
| **onnx_threads_tuned** | same | same | same | CPU | *fill* | *fill* | *fill* | *fill* | e.g. 8 | same | **Model + system**: threads tuned to vCPU count |
| **multi_worker_http** ⭐ peak RPS (maybe) | same | same | same | CPU | *fill* | *fill* | *fill* | *fill* | e.g. 16 | same | **System-level**: `UVICORN_WORKERS>1`; watch RSS×N |
| **triton_dynamic_batch** ⭐ infra batching | `http://<ip>:8000` Triton | same | Triton image tag | CPU or GPU | *fill* | *fill* | *fill* | *fill* | *fill* | GPU optional | **Infrastructure-level**: Triton dynamic batching / concurrency |
| **combined_best** | same | same | same | CPU | *fill* | *fill* | *fill* | *fill* | *fill* | same | e.g. ORT `all` + tuned threads + workers=2 — **combined** |

## Placeholder model (no training teammate required)

```bash
python trainer/toy_random_onnx.py --register --alias staging --model-name serving-bench-model
```

Set serving `MODEL_URI=models:/serving-bench-model@staging` for apples-to-apples benchmarks. When the real model arrives, re-run the same matrix and add **task-quality** metrics (accuracy, ECE, etc.).

## Repository map (deliverables)

| Artifact | Purpose |
|----------|---------|
| `Dockerfile.serving` | Serving container (FastAPI + ONNX Runtime + entrypoint workers) |
| `scripts/docker-entrypoint-serving.sh` | **System-level**: `UVICORN_WORKERS` |
| `serving/ort_session.py` | **Model-level**: ORT graph / thread / execution mode |
| `serving/app.py` | `/predict`, `/metadata` (includes `serving_option_id`, ORT flags), `/reload` |
| `scripts/evaluate_serving.py` | Load generator → JSON summary for the table |
| `trainer/toy_random_onnx.py` | Random-equivalent ONNX + optional registry publish |
| `infra/triton/` | **Infrastructure-level** Triton layout + README |
| `docs/CHAMELEON_BENCHMARK.md` | Where/how to run for credit |
| `docs/DEMO_VIDEO.md` | Sped-up demo storyboard |
| `docs/RAY_SERVE_ALTERNATIVE.md` | Bonus: alternative framework justification |

## Example benchmark command (after containers up)

```bash
python scripts/evaluate_serving.py \
  --base-url http://127.0.0.1:18080 \
  --concurrency 8 --requests 500 \
  --output-json eval/run01_baseline.json
```

## `/metadata` fields for the report

Call `GET /metadata` and record `serving_option_id`, `model_version`, `build_sha`, `ort_graph_optimization_level`, and `provider` in your write-up.
