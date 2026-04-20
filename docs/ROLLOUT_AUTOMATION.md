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

For the Mealie-integrated flow, the same builder can consume the curated Airflow artifact:

- source artifact: `feedback_labels.parquet`
- produced by: DMS `training_data_assembly -> write_feedback_labels`
- training bridge: `scripts/run_mealie_feedback_retraining.sh`
- object-store handoff path example:
  `swift://proj26-obj-store/airflow-fallback/recipe_feedback_labels/snapshot_date=YYYY-MM-DD/feedback_labels.parquet`

That path lets you demo:

1. feedback captured in Mealie/DMS
2. Airflow compiles a curated feedback artifact
3. model-serve converts that artifact into `npz_classification`
4. standard MLflow training + registration runs on Chameleon

Important note:

- the curated artifact must contain at least two distinct labels for the chosen training target
- if the artifact only contains one approved recipe class, dataset build succeeds but classifier training should be treated as blocked by insufficient supervision rather than forced through

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
- rollout-controller exit code and logs when `--controller-check` is used

When `--alias canary --controller-check` is used, the controller may do one of two valid things:

- refresh and promote the canary if the gate passes
- refresh and reject the canary if it regresses on the configured gate

Both outcomes are useful for the demo because they show the promotion logic is active rather than blindly advancing every new model.
