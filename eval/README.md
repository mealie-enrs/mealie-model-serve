# Evaluation artifacts

- **`scripts/evaluate_serving.py`** — primary load generator (async `httpx`, p50/p95, throughput, errors). Run **on Chameleon inside Docker** per `docs/CHAMELEON_BENCHMARK.md`.
- **Output JSON** — store under `eval/` (gitignored patterns optional) or attach to submission.
- **Notebook (optional)** — import subprocess to run the script with different `--concurrency` / `--requests` and plot bars in Jupyter on the Chameleon VM.

Example loop for sweeps:

```bash
for c in 1 2 4 8 16; do
  python scripts/evaluate_serving.py --base-url http://model-serve:8080 \
    --concurrency "$c" --requests 400 --output-json "eval/sweep_c${c}.json"
done
```

Fill **`../SERVING_OPTIONS.md`** from the printed / saved JSON.
