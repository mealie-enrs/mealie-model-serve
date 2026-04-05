# Running serving benchmarks on Chameleon (for credit)

Course requirement: experiments should run **on Chameleon**, **from inside a container** on a compute instance (local-only runs are for dev).

## 1) VM + Docker

- Use your project KVM instance (or bare metal if allocated).
- Install Docker (or use a base image that includes it).
- Clone this repo and build images on the VM (or pull from your registry).

## 2) Start control plane + serving on the VM

```bash
cd mealie-model-serve
docker compose -f infra/docker-compose.control-plane.yml up -d --build
# Register a model (trained or toy placeholder)
docker run --rm --network mealie-model-serve-net \
  -v "$(pwd)":/app -w /app \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -e MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
  -e AWS_ACCESS_KEY_ID=mmsminio \
  -e AWS_SECRET_ACCESS_KEY=mmsminio-dev-secret \
  python:3.11-slim bash -c "pip install -q -r requirements.txt && python trainer/toy_random_onnx.py --register --alias staging"

docker compose -f infra/docker-compose.serving.yml up -d --build
```

Point serving `MODEL_URI` at the registered name/version (edit compose env or export before `up`).

## 3) Benchmark container (same network)

```bash
docker run --rm --network mealie-model-serve-net \
  -v "$(pwd)":/app -w /app \
  python:3.11-slim bash -c \
  "pip install -q httpx && python scripts/evaluate_serving.py \
    --base-url http://model-serve:8080 \
    --concurrency 4 --requests 400 \
    --output-json /app/eval/chameleon_run_baseline.json"
```

Repeat for each **serving option** (different container env / image), saving JSON per run.

## 4) Right-sizing (CPU / memory / GPU)

While load runs, on the host:

```bash
docker stats --no-stream
```

Record peak **CPU %**, **MEM USAGE / LIMIT**, and **GPU** (if `MODEL_PROVIDER=cuda`) for the serving container. Paste into `SERVING_OPTIONS.md` **Notes** column.

## 5) Fill the table

Copy numbers into `SERVING_OPTIONS.md` (p50/p95 ms, throughput, error rate, concurrency, instance flavor).
