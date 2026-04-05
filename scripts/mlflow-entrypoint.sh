#!/usr/bin/env sh
set -eu
: "${MLFLOW_BACKEND_STORE_URI:?}"
ART_ROOT="${MLFLOW_ARTIFACTS_DESTINATION:-s3://mlflow-artifacts}"
if [ -n "${MLFLOW_S3_ENDPOINT_URL:-}" ] && [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
  python /app/scripts/ensure_minio_bucket.py
fi
exec mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
  --artifacts-destination "$ART_ROOT" \
  --serve-artifacts
