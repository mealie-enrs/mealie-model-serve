# Sped-up demo video (checklist)

Record on **Chameleon** (screen capture of browser + terminal). Speed up silent sections in your editor.

## Script (~2–3 min raw → ~45–60 s sped up)

1. **Show instance** — Horizon or `openstack server show` snippet with floating IP (2–3 s).
2. **SSH** — `ssh cc@<floating-ip>` (2 s).
3. **Containers up** — `docker compose -f infra/docker-compose.control-plane.yml ps` and `docker compose -f infra/docker-compose.serving.yml ps` (5 s).
4. **MLflow UI** — browser `http://<ip>:15001` → Registered model + alias (10 s).
5. **Metadata** — `curl -s http://<ip>:18080/metadata | jq .` showing `serving_option_id`, `model_version`, `build_sha` (5 s).
6. **Predict** — agreed example request:
   ```bash
   curl -s http://<ip>:18080/predict -H 'Content-Type: application/json' \
     -d '{"inputs":[[5.1,3.5,1.4,0.2]]}' | jq .
   ```
7. **Optional: reload** — `POST /reload` with `model_uri` to show alias swap (10 s).
8. **Benchmark flash** — one line running `scripts/evaluate_serving.py` and showing JSON summary `p50_ms` / `p95_ms` / `throughput_rps` (10 s).

## Audio (optional)

Short voiceover: “FastAPI + ONNX Runtime on Chameleon; baseline vs optimized configs compared in SERVING_OPTIONS.md.”

## Submit

Upload per course instructions; link **endpoint URL**, **model version**, **git SHA** in your report table (`SERVING_OPTIONS.md`).
