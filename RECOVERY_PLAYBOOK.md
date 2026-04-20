# Model Serve Recovery Playbook

Last updated: April 20, 2026

This document explains how to preserve and restore the model-serving side of the project after Chameleon instances are terminated and recreated.

This repo owns:

- MLflow tracking and model registry
- production and canary model-serving deployments
- weighted rollout router
- feedback capture for retraining
- training and retraining scripts

## Current live topology

As of April 20, 2026:

- GPU host IP: `192.5.87.188`
- MLflow: `http://192.5.87.188:30601`
- production model serve: `http://192.5.87.188:30608`
- canary model serve: `http://192.5.87.188:30609`
- rollout router: `http://192.5.87.188:30610`
- runtime platform: `k3s` on a Chameleon GPU node
- namespace: `mms`

## What is durable vs not durable

### Durable

The good news:

- this Git repository is durable
- MLflow artifact storage is configured to Chameleon object storage, not a local disk

Current live config in `mms-config`:

- `MLFLOW_ARTIFACTS_DESTINATION=s3://proj26-obj-store/serve/mlflow-artifacts`
- `MLFLOW_S3_ENDPOINT_URL=https://chi.uc.chameleoncloud.org:7480`

That means model artifact files stored by MLflow should survive node deletion.

### Not durable by default

Two important pieces are currently using node-local storage:

1. MLflow Postgres metadata
2. rollout feedback PVC

Current Kubernetes storage:

- MLflow Postgres StatefulSet uses a `local-path` PVC
- rollout feedback uses `mms-feedback-pvc`, also on `local-path`

On the live cluster, `local-path` has:

- `provisioner: rancher.io/local-path`
- `reclaimPolicy: Delete`

That means if the node is deleted, assume these are lost unless backed up first.

## What this means in practice

### Likely to survive

- model artifacts in object storage
- files already copied out of the cluster
- code and manifests in Git

### At risk

- MLflow experiment metadata
- MLflow run rows
- model registry metadata
- aliases such as `@production` and `@canary`
- rollout feedback JSONL logs under `/var/lib/mms-feedback`

This is the single most important distinction for this repo:

- the **model files** may survive
- the **registry state and run history** may not

## Minimum backup set before shutdown

If time is short, do these in order:

1. dump MLflow Postgres
2. copy raw rollout feedback logs
3. save a snapshot of current model aliases and versions
4. verify MLflow artifact root in object storage

## Backup commands

All commands below are intended to be run from your local machine.

### 1. Back up MLflow Postgres metadata

Create a local backup directory:

```bash
mkdir -p ~/mealie-backups/2026-04-20
```

Dump the MLflow Postgres database from the k3s cluster:

```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@192.5.87.188 '
DB_PASS=$(sudo k3s kubectl get secret -n mms mms-secrets -o jsonpath="{.data.DATABASE_PASSWORD}" | base64 -d)
sudo k3s kubectl exec -n mms postgres-0 -- sh -lc "PGPASSWORD=$DB_PASS pg_dump -U mlflow -d mlflow -Fc"
' > ~/mealie-backups/2026-04-20/mlflow-postgres.dump
```

Verify:

```bash
ls -lh ~/mealie-backups/2026-04-20/mlflow-postgres.dump
file ~/mealie-backups/2026-04-20/mlflow-postgres.dump
```

### 2. Back up rollout feedback logs

The rollout router writes feedback to `/var/lib/mms-feedback`, backed by `mms-feedback-pvc`.

Copy the raw JSONL logs from the running router pod:

```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@192.5.87.188 '
ROUTER_POD=$(sudo k3s kubectl get pods -n mms -l app=mms-rollout-router -o jsonpath="{.items[0].metadata.name}")
sudo k3s kubectl exec -n mms "$ROUTER_POD" -- sh -lc "tar czf - -C /var/lib/mms-feedback ."
' > ~/mealie-backups/2026-04-20/mms-feedback-logs.tar.gz
```

Verify:

```bash
ls -lh ~/mealie-backups/2026-04-20/mms-feedback-logs.tar.gz
tar tzf ~/mealie-backups/2026-04-20/mms-feedback-logs.tar.gz | head
```

### 3. Save model registry state as JSON

This does not replace the DB dump, but it gives a readable record of what was live.

```bash
curl -s http://192.5.87.188:30608/metadata > ~/mealie-backups/2026-04-20/model-serve-production-metadata.json
curl -s http://192.5.87.188:30609/metadata > ~/mealie-backups/2026-04-20/model-serve-canary-metadata.json
curl -s http://192.5.87.188:30610/metadata > ~/mealie-backups/2026-04-20/model-serve-router-metadata.json
```

Also export the MLflow model version list if possible:

```bash
python trainer/register_model.py --help >/dev/null
```

If you have an MLflow client environment available, also save:

- model name
- version numbers
- aliases
- source run IDs

### 4. Verify artifact storage still exists

The MLflow artifact root currently points to:

- bucket or Swift container: `proj26-obj-store`
- prefix: `serve/mlflow-artifacts`

Use your S3-compatible or Swift client to confirm that prefix still exists and contains run artifacts and model artifacts.

## Restore on a new reservation

### 1. Recreate the GPU node and k3s environment

From this repo:

```bash
cd infra/terraform
terraform init
terraform plan -out tfplan
terraform apply tfplan
```

Then reapply the correct overlay:

```bash
kubectl kustomize infra/k8s/overlays/chameleon-s3-gpu --load-restrictor LoadRestrictionsNone | kubectl apply -f -
```

### 2. Restore secrets

You will need at minimum:

- Chameleon object store credentials
- `mms-secrets`
- database password
- any GitHub Actions deployment secrets if CI/CD is used

### 3. Restore MLflow Postgres metadata

Copy the dump to the new host:

```bash
scp -i ~/.ssh/id_rsa_chameleon \
  ~/mealie-backups/2026-04-20/mlflow-postgres.dump \
  cc@<NEW_GPU_IP>:/home/cc/mlflow-postgres.dump
```

Restore it into the new cluster:

```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@<NEW_GPU_IP> '
DB_PASS=$(sudo k3s kubectl get secret -n mms mms-secrets -o jsonpath="{.data.DATABASE_PASSWORD}" | base64 -d)
sudo k3s kubectl cp /home/cc/mlflow-postgres.dump mms/postgres-0:/tmp/mlflow-postgres.dump
sudo k3s kubectl exec -n mms postgres-0 -- sh -lc "PGPASSWORD=$DB_PASS pg_restore -U mlflow -d mlflow --clean --if-exists /tmp/mlflow-postgres.dump"
'
```

### 4. Restore feedback logs

```bash
scp -i ~/.ssh/id_rsa_chameleon \
  ~/mealie-backups/2026-04-20/mms-feedback-logs.tar.gz \
  cc@<NEW_GPU_IP>:/home/cc/mms-feedback-logs.tar.gz

ssh -i ~/.ssh/id_rsa_chameleon cc@<NEW_GPU_IP> '
ROUTER_POD=$(sudo k3s kubectl get pods -n mms -l app=mms-rollout-router -o jsonpath="{.items[0].metadata.name}")
sudo k3s kubectl cp /home/cc/mms-feedback-logs.tar.gz mms/$ROUTER_POD:/tmp/mms-feedback-logs.tar.gz
sudo k3s kubectl exec -n mms "$ROUTER_POD" -- sh -lc "mkdir -p /var/lib/mms-feedback && tar xzf /tmp/mms-feedback-logs.tar.gz -C /var/lib/mms-feedback"
'
```

### 5. Validate aliases and serving state

After the cluster is back:

```bash
curl http://<NEW_GPU_IP>:30601
curl http://<NEW_GPU_IP>:30608/metadata
curl http://<NEW_GPU_IP>:30609/metadata
curl http://<NEW_GPU_IP>:30610/metadata
```

Confirm:

- MLflow loads
- production alias resolves
- canary alias resolves
- router metadata shows expected production and canary lanes
- provider is expected, for example `CUDAExecutionProvider`

## Recovery if the MLflow DB was lost but artifacts survived

This is the likely worst-case scenario.

If the MLflow Postgres metadata is gone but the artifact bucket still exists:

1. bring the stack back up
2. inspect the artifact bucket for the latest good model
3. re-register the model into MLflow
4. re-apply aliases like `@production` and `@canary`
5. redeploy or reload serving pods

The repo already includes helper scripts that are useful here:

- `trainer/register_model.py`
- `ops/promote_alias.py`
- `ops/rollback_alias.py`

## Smoke test checklist

After restore, verify:

1. MLflow UI opens
2. old runs are visible if DB restore succeeded
3. model aliases exist
4. production and canary endpoints respond
5. router endpoint responds
6. router feedback logging works
7. feedback-to-dataset scripts can still run

## Important note for future hardening

The current setup is workable for a demo, but the right long-term fix is to move all critical state off node-local storage:

- MLflow backend DB on a real persistent external DB or attached durable volume
- feedback logs copied to object storage automatically
- registry snapshots exported automatically during promotion

Until that is done, always assume the cluster must be backed up explicitly before any instance teardown.
