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
2. compares canary latency/error rate against production
3. promotes `production` to the canary model version if thresholds pass
4. resets router canary traffic back to the stable weight after promotion

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
