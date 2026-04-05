# Mealie Model Serve

Build-ready **v1** layout for a Chameleon-friendly model platform: **MLflow + PostgreSQL + MinIO**, training → **Model Registry** with **aliases** (`staging`, `canary`, `champion`), and a **FastAPI + ONNX Runtime** serving plane with **local cache**, **`/reload`**, and **Prometheus** metrics.

## Repo layout (spec §14)

```text
infra/
  docker-compose.control-plane.yml   # Postgres, MinIO, MLflow
  docker-compose.serving.yml         # Serving API (joins same Docker network)
trainer/
  train.py           # Iris MVP → ONNX → register + @staging
  export_onnx.py     # Optional pickle → ONNX
  register_model.py  # Register existing run
serving/
  app.py, resolver.py, loader.py, cache.py, metrics.py
ops/
  promote_alias.py, rollback_alias.py, benchmark_candidate.py (stub)
infra/k8s/          # k3s / Kubernetes (same stack as Docker Compose)
```

## Kubernetes on Chameleon (k3s VM, e.g. after SSH to your server)

See **`infra/k8s/DEPLOY.md`**: build/push images, apply secrets, then `kubectl apply -k infra/k8s/` (in-cluster MinIO) or **`infra/k8s/overlays/chameleon-s3`** for Swift / S3 artifacts (`serve/` prefix in your container). Pushes to **`main`** / **`master`** run **`.github/workflows/deploy.yml`**: build → GHCR → SSH apply kustomize → roll out images on the VM.

Provisioning a **dedicated MMS VM** (FIP + SG + k3s cloud-init): **`infra/terraform/`** in this package (`terraform init` / `apply` there). The root **`mealie/infra/terraform`** stack is the DMS-oriented variant. Workloads are still **kubectl/kustomize**, not Terraform, unless you add a Kubernetes provider.

## Phase 1 — control plane

From repo root:

```bash
docker compose -f infra/docker-compose.control-plane.yml up -d --build
```

- **MLflow UI:** `http://localhost:15001`
- **MinIO console:** `http://localhost:9021` (user `mmsminio` / password `mmsminio-dev-secret` — change for real deployments)
- **Postgres:** host port `55435`

Creates Docker network **`mealie-model-serve-net`** (used by serving compose).

## Phase 2 — train + register (host)

Point at MLflow on localhost (mapped port):

```bash
cd /Users/mudrex/Desktop/mealie/mealie-model-serve   # your path
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export MLFLOW_TRACKING_URI=http://127.0.0.1:15001
export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9020
export AWS_ACCESS_KEY_ID=mmsminio
export AWS_SECRET_ACCESS_KEY=mmsminio-dev-secret
export AWS_DEFAULT_REGION=us-east-1

export MODEL_NAME=food-classifier
python trainer/train.py
```

This registers **`food-classifier`** version **1**, sets tags (spec §7), and assigns **`@staging`**.

Train again (no code change) to get **version 2**; only the latest run gets `@staging` if you re-run `train.py` as-is (script sets alias to new version).

## Phase 3 — serving

After control plane is up:

```bash
docker compose -f infra/docker-compose.serving.yml up -d --build
```

- **API:** `http://localhost:18080`
- Default **`MODEL_URI=models:/food-classifier@staging`**

### Endpoints (spec §8 + FR-5/9)

| Method | Path | Notes |
|--------|------|--------|
| GET | `/healthz` | Process up |
| GET | `/readyz` | 200 if model loaded, 503 otherwise |
| GET | `/metadata` | model name, version, alias, provider, `build_sha` |
| POST | `/predict` | `{"inputs": [[float,...]]}` — Iris MVP uses **4** features |
| POST | `/reload` | `{"model_uri": "models:/food-classifier@canary"}` |
| GET | `/metrics` | Prometheus |

Example:

```bash
curl -s http://127.0.0.1:18080/metadata | jq .
curl -s http://127.0.0.1:18080/predict -H 'Content-Type: application/json' \
  -d '{"inputs": [[5.1,3.5,1.4,0.2]]}' | jq .
```

## Phase 4 — promotion / rollback (one-liners)

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:15001
PYTHONPATH=. python ops/promote_alias.py --model food-classifier --alias champion --version 1
PYTHONPATH=. python ops/rollback_alias.py --model food-classifier --alias staging --version 1
PYTHONPATH=. python ops/benchmark_candidate.py --model food-classifier --version 1 --status passed
```

Then reload serving:

```bash
curl -s -X POST http://127.0.0.1:18080/reload \
  -H 'Content-Type: application/json' \
  -d '{"model_uri":"models:/food-classifier@champion"}' | jq .
```

## Acceptance checklist (spec §11)

1. Run `trainer/train.py` → registry **v1**, `@staging`.
2. Start serving → `/metadata` shows **v1** (via `@staging`).
3. Run `train.py` again → **v2**, script sets `@staging` → **v2**.
4. `POST /reload` or restart container → `/metadata` shows **v2**.
5. `ops/rollback_alias.py` (or `promote_alias`) to move `@staging` to **v1**, then `/reload` → rollback.

## Environment (spec §9)

**Serving (compose already sets many):** `MODEL_URI`, `MODEL_CACHE_DIR`, `MLFLOW_*`, `AWS_*`, `MODEL_PROVIDER` (`cpu` or `cuda`).

**Trainer:** `MODEL_NAME`, `DATASET_VERSION`, `REGISTER_MODEL`, `MLFLOW_TRACKING_URI`, plus S3-compatible env for artifacts.

## Security note

Default MinIO credentials and ports are for **local dev only**. For Chameleon: private subnets, secrets, persistent volumes (`ceph-ssd`), and no public Postgres/MinIO (spec §10).

## Serving team rubric (course deliverable)

- **`SERVING_OPTIONS.md`** — comparison table + how to mark “most promising” options by priority.
- **`docs/CHAMELEON_BENCHMARK.md`** — run benchmarks **on Chameleon in Docker** (required for credit).
- **`docs/DEMO_VIDEO.md`** — storyboard for sped-up demo video.
- **`scripts/evaluate_serving.py`** — measure p50/p95, throughput, error rate.
- **`trainer/toy_random_onnx.py`** — benchmark without waiting for final trained weights.
- **`docs/RAY_SERVE_ALTERNATIVE.md`** — optional bonus: Ray Serve vs FastAPI rationale.
- **Triton path:** `infra/triton/` — infrastructure-level option vs FastAPI.

## Cut line (spec §13)

In repo now: control plane, trainer MVP, registry + aliases, serving + reload + Prometheus metrics, ops scripts (+ benchmark stub).

Later: canary traffic split, Triton/TensorRT, Octavia LB, automated promotion gates.
