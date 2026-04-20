#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 trainer/configs/<config>.json [extra train args...]"
  exit 1
fi

CONFIG_PATH="$1"
shift || true

: "${MLFLOW_TRACKING_URI:=http://127.0.0.1:30601}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"
: "${TRAIN_IMAGE:=mealie-model-serve-train:latest}"

docker build -f Dockerfile.train -t "${TRAIN_IMAGE}" .

docker run --rm \
  --network host \
  -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
  -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  -e AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" \
  -v "$(pwd)/trainer:/app/trainer" \
  "${TRAIN_IMAGE}" \
  --config "${CONFIG_PATH}" "$@"
