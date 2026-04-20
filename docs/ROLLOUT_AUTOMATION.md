# Rollout Automation

This repo now has four rollout layers:

1. `mms-model-serve` — direct production lane (`@production`)
2. `mms-model-serve-canary` — direct canary lane (`@canary`)
3. `mms-rollout-router` — weighted entrypoint that can split traffic between them
4. `mms-rollout-controller` — CronJob that benchmarks, promotes, or rolls back based on policy

## Weighted traffic split

The stable weighted endpoint is `http://<ip>:30610`.

It routes to:

- production backend: `mms-model-serve`
- canary backend: `mms-model-serve-canary`

The split is controlled by `ROUTER_DEFAULT_CANARY_WEIGHT` and can be adjusted live through the controller.

## Promotion gates

The controller policy lives at `ops/rollout_policy.json`.

On each monitor cycle it:

1. benchmarks production and canary
2. checks whether the live canary service is still serving `models:/<model>@canary`
3. reloads the canary service automatically if the MLflow `@canary` alias moved or the service drifted
4. compares canary latency/error rate against production
5. promotes `production` to the canary model version if thresholds pass
6. resets router canary traffic back to the stable weight after promotion

## Automatic rollback triggers

The controller stores:

- `last_good_production_version`
- `current_production_version`
- `rollout.last_gate`
- `rollout.last_gate_details`

If production health degrades after a promotion and the current version differs from `last_good_production_version`, the controller moves the `production` alias back and reloads the production deployment.

## Feedback and retraining demo

Use `scripts/emit_production_traffic.py` to generate realistic traffic against the router and `POST /feedback` events.

Feedback logs are persisted under `/var/lib/mms-feedback`.

Use `scripts/build_feedback_dataset.py` to convert those logs into an `npz_classification` dataset, then run a retraining cycle and register the next candidate as `@canary`.

That gives you a demo loop:

1. weighted production traffic
2. user feedback captured
3. feedback saved for retraining
4. retrain to canary
5. controller promotes or rolls back automatically

## Packaged operator flow

Use `scripts/run_operator_flow_chameleon.sh` when you want one reproducible operator command that:

1. generates live feedback against the rollout router
2. copies the raw JSONL feedback log from the running cluster
3. builds a curated `npz_classification` dataset
4. retrains inside the training container on Chameleon
5. registers the new model in MLflow with a chosen alias
6. optionally runs a manual rollout-controller check when targeting `@canary`

Safe demo command:

```bash
bash scripts/run_operator_flow_chameleon.sh \
  --host 192.5.87.188 \
  --alias retrain-demo \
  --requests 12
```

Canary-refresh demo command:

```bash
bash scripts/run_operator_flow_chameleon.sh \
  --host 192.5.87.188 \
  --alias canary \
  --requests 12 \
  --controller-check
```

The script prints a JSON summary with:

- raw feedback path and row count
- curated dataset path and size
- MLflow run id and public run URL
- registered alias version
- rollout-controller logs when `--controller-check` is used
