# Training Plan

This repo now has a containerized, config-driven training path intended to run on Chameleon compute and log every run to MLflow.

## First implementation

- Framework: `scikit-learn`
- Single training entrypoint: `trainer/train.py`
- Single tuning entrypoint: `trainer/tune.py`
- Config format: JSON under `trainer/configs/`
- Container entrypoint: `Dockerfile.train`

## Manager choices

| Candidate | Role | Quality | Training Cost | Serving Cost | Notes |
|---|---|---:|---:|---:|---|
| Logistic Regression | Simple baseline | Low-Medium | Very low | Very low | Best first sanity check |
| Random Forest | Strong tabular baseline | Medium | Low | Low-Medium | Good quality without tuning pain |
| Extra Trees | Higher-variance ensemble | Medium-High | Low-Medium | Low-Medium | Good tradeoff for embedding features |

## What each run logs

- Config parameters
- Validation metrics (`val_accuracy`, `val_macro_f1`, optional `val_log_loss`)
- Cost metrics (`fit_time_sec`, `inference_time_sec`)
- Environment metadata (`hostname`, `python_version`, `git_sha`, `nvidia_smi`)

## Chameleon execution

Run training from inside a container on the Chameleon instance:

```bash
AWS_ACCESS_KEY_ID=... \
AWS_SECRET_ACCESS_KEY=... \
MLFLOW_TRACKING_URI=http://127.0.0.1:30601 \
./scripts/run_train_chameleon.sh trainer/configs/iris-logreg.json
```

Run tuning:

```bash
AWS_ACCESS_KEY_ID=... \
AWS_SECRET_ACCESS_KEY=... \
MLFLOW_TRACKING_URI=http://127.0.0.1:30601 \
./scripts/run_tune_chameleon.sh trainer/configs/iris-optuna.json
```

The baseline configs use sklearn builtin datasets so the first end-to-end run can be validated before swapping in Recipe1M-derived embeddings via `npz_classification`.
