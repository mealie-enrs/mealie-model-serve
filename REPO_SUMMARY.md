# mealie-model-serve — standalone repo summary

Use this when splitting **`mealie-model-serve/`** out of the parent **mealie** monorepo. The package is **self-contained**: no Python imports from the rest of mealie; only docs/README reference the parent Terraform variant for comparison.

---

## What it is

A **v1 model platform** for research / Chameleon-style deployments:

- **Control plane:** PostgreSQL, MinIO (S3-compatible), MLflow (tracking + model registry + aliases).
- **Training:** Python scripts that train (Iris MVP), export **ONNX**, register models, set aliases (`staging`, `canary`, `champion`).
- **Serving:** **FastAPI** + **ONNX Runtime**, resolves `models:/name@alias` from MLflow, local disk cache, **`/reload`**, Prometheus **`/metrics`**.
- **Ops:** promote / rollback aliases, benchmark helper (stub), evaluation script.

---

## Layout

| Area | Contents |
|------|----------|
| **`serving/`** | `app.py`, `resolver.py`, `loader.py`, `cache.py`, `metrics.py`, `ort_session.py`, `config.py` |
| **`trainer/`** | `train.py`, `export_onnx.py`, `register_model.py`, `toy_random_onnx.py` (quick ONNX for benchmarks) |
| **`ops/`** | `promote_alias.py`, `rollback_alias.py`, `benchmark_candidate.py` |
| **`scripts/`** | `docker-entrypoint-serving.sh`, `mlflow-entrypoint.sh`, `ensure_minio_bucket.py`, `evaluate_serving.py` |
| **`infra/docker-compose.*.yml`** | Control plane + serving stacks; shared Docker network `mealie-model-serve-net` |
| **`infra/k8s/`** | Kustomize: namespace **`mms`**, ConfigMap, Postgres/MinIO/MLflow/model-serve manifests, `secrets.example.yaml`, **`DEPLOY.md`** |
| **`infra/terraform/`** | OpenStack + `local-exec` VM create, cloud-init (**Docker + k3s**), SG rules **22 / 30601 / 30608**, FIP |
| **`infra/triton/`** | Optional Triton `model_repository` sample (`config.pbtxt` + README) |
| **`docs/`** | `CHAMELEON_BENCHMARK.md`, `DEMO_VIDEO.md`, `RAY_SERVE_ALTERNATIVE.md` |
| **Dockerfiles** | `Dockerfile.mlflow`, `Dockerfile.serving` |
| **Root** | `README.md`, `requirements.txt`, `SERVING_OPTIONS.md`, `.gitignore` |

---

## Kubernetes (k3s)

- **Namespace:** `mms`
- **Images (override in `infra/k8s/kustomization.yaml`):**  
  `ghcr.io/your-org/mealie-model-serve-mlflow`, `ghcr.io/your-org/mealie-model-serve-api`
- **NodePorts (as in manifests):** MLflow **30601**, model API **30608** (align SG with Terraform or Horizon).
- **Secrets:** copy `infra/k8s/secrets.example.yaml` → apply as `mms-secrets` (real file is gitignored).

Full flow: **`infra/k8s/DEPLOY.md`**.

---

## Terraform (Chameleon / OpenStack)

- **`infra/terraform/`** — same pattern as mealie DMS stack: `null` + OpenStack providers, security group, floating IP, `openstack server create` with **`cloud-init.yaml.tftpl`**.
- **`terraform.tfvars.example`** — template; real **`terraform.tfvars`** must stay local / secret and is gitignored.
- Do **not** commit **`.terraform/`** or **`*.tfstate`** (gitignored); **do** commit **`.terraform.lock.hcl`**.

---

## Local dev (Docker Compose)

1. `infra/docker-compose.control-plane.yml` — Postgres, MinIO, MLflow (documented ports in `README.md`).
2. `infra/docker-compose.serving.yml` — serving API on **18080**, default `MODEL_URI=models:/food-classifier@staging`.

---

## Python dependencies

See **`requirements.txt`**: MLflow, Postgres driver, boto3, sklearn, skl2onnx/onnx, FastAPI, uvicorn, onnxruntime, prometheus-client, httpx, etc.

---

## Splitting checklist (new Git repo)

1. Copy only the **`mealie-model-serve/`** tree (or make it the repo root).
2. Confirm **`.gitignore`** excludes: `infra/k8s/mms-secrets.yaml`, `infra/terraform/.terraform/`, `infra/terraform/*.tfstate*`, `infra/terraform/terraform.tfvars`, venvs, `__pycache__`, etc.
3. Remove or reword any **absolute paths** in local notes pointing at `/Users/.../mealie/...`.
4. In **`README.md`**, you can drop the sentence comparing to **`mealie/infra/terraform`** if the parent repo no longer exists for consumers.
5. Initialize git, add remote, push; CI optional (build/push images to GHCR, etc.).

---

## Current operational status (context)

- **k3s** has been installed manually on **`serving-proj26`**; cluster shows CoreDNS, **local-path-provisioner**, Traefik, etc. Next step for “full stack on k8s” is build/push or import images, apply secrets, then **`kubectl apply -k infra/k8s/`** per **`DEPLOY.md`**.

---

## Not in this folder

- Main **mealie** app (API/worker/scheduler), **DMS** k8s manifests, or root **`mealie/infra/terraform`** — those stay in the parent repo unless you copy them separately.
