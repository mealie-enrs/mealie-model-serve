# Bonus: Ray Serve vs FastAPI + ONNX Runtime (design note)

## Why consider Ray Serve?

**Ray Serve** adds a **deployment graph**, **replicas**, **autoscaling**, and **request batching** as first-class concepts on a Ray cluster. It fits when:

- You have **multiple models** or **pipelines** (preprocess → model → postprocess) that must scale together.
- You need **dynamic scaling** on a pool of Chameleon workers without hand-rolling a load balancer + N× containers.
- You want **built-in batching** with deadlines (similar to Triton’s dynamic batcher) but in Python.

## Why we still shipped FastAPI first

- **Smaller blast radius** for MVP: one container, one `MODEL_URI`, clear `/metadata` and `/reload` for registry aliases.
- **Fewer moving parts** on a single KVM node (no Ray head/worker cluster to operate for class deadline).
- **ONNX Runtime** sits directly in-process — lowest overhead for a **single-model** HTTP service.

## Concrete example where Ray would help

A **two-stage** food pipeline: (1) image decode + resize ONNX, (2) classifier ONNX, with **shared GPU** and **adaptive batching** on stage 2. FastAPI can do this with custom code; Ray Serve’s `@serve.batch` and `DeploymentHandle` reduce glue code when stages scale independently.

## If you implement Ray for bonus credit

- Keep the **same** MLflow resolver + MinIO artifact path logic; swap only the **HTTP front** and **replica** layer.
- Re-run `scripts/evaluate_serving.py` against Ray’s HTTP port and add a row to `SERVING_OPTIONS.md` comparing **p95** and **operational complexity** vs FastAPI.
