#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <curated-feedback.parquet> <dataset-npz> <config-json> [extra train args...]"
  exit 1
fi

CURATED_FEEDBACK="$1"
DATASET_NPZ="$2"
CONFIG_JSON="$3"
shift 3 || true

: "${TRAIN_IMAGE:=mealie-model-serve-train:latest}"
: "${MODEL_NAME:=food-classifier-feedback}"
: "${MODEL_ALIAS:=feedback-demo}"
: "${MLFLOW_TRACKING_URI:=http://127.0.0.1:30601}"
: "${LABEL_FIELD:=auto}"
: "${SWIFT_AUTH_URL:=}"
: "${SWIFT_APP_CREDENTIAL_ID:=}"
: "${SWIFT_APP_CREDENTIAL_SECRET:=}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_ABS="$(cd "$(dirname "${DATASET_NPZ}")" && pwd)/$(basename "${DATASET_NPZ}")"
CONFIG_ABS="$(cd "$(dirname "${CONFIG_JSON}")" && pwd)/$(basename "${CONFIG_JSON}")"

if [[ "${CURATED_FEEDBACK}" == swift://* ]]; then
  swift_target="${CURATED_FEEDBACK#swift://}"
  swift_container="${swift_target%%/*}"
  swift_key="${swift_target#*/}"
  if [[ -z "${swift_container}" || -z "${swift_key}" || "${swift_container}" == "${swift_key}" ]]; then
    echo "Invalid swift path: ${CURATED_FEEDBACK}" >&2
    exit 1
  fi
  if [[ -z "${SWIFT_AUTH_URL}" || -z "${SWIFT_APP_CREDENTIAL_ID}" || -z "${SWIFT_APP_CREDENTIAL_SECRET}" ]]; then
    echo "Swift credentials are required for swift:// inputs." >&2
    exit 1
  fi
  CURATED_ABS="${ROOT_DIR}/tmp/$(basename "${swift_key}")"
else
  CURATED_ABS="$(cd "$(dirname "${CURATED_FEEDBACK}")" && pwd)/$(basename "${CURATED_FEEDBACK}")"
fi

case "${CURATED_ABS}" in
  "${ROOT_DIR}"/*) CURATED_REL="${CURATED_ABS#${ROOT_DIR}/}" ;;
  *)
    echo "Curated feedback file must live under ${ROOT_DIR} so it can be mounted into the train container." >&2
    exit 1
    ;;
esac

case "${DATASET_ABS}" in
  "${ROOT_DIR}"/*) DATASET_REL="${DATASET_ABS#${ROOT_DIR}/}" ;;
  *)
    echo "Dataset output must live under ${ROOT_DIR}." >&2
    exit 1
    ;;
esac

case "${CONFIG_ABS}" in
  "${ROOT_DIR}"/*) CONFIG_REL="${CONFIG_ABS#${ROOT_DIR}/}" ;;
  *)
    echo "Config output must live under ${ROOT_DIR}." >&2
    exit 1
    ;;
esac

mkdir -p "$(dirname "${DATASET_NPZ}")" "$(dirname "${CONFIG_JSON}")"

cat > "${CONFIG_ABS}" <<EOF
{
  "dataset": {
    "source": "npz_classification",
    "name": "$(basename "${DATASET_ABS}" .npz)",
    "path": "/app/${DATASET_REL}"
  },
  "candidate": {
    "name": "logistic_regression",
    "params": {
      "C": 0.5,
      "max_iter": 1000
    }
  },
  "mlflow": {
    "tracking_uri": "${MLFLOW_TRACKING_URI}",
    "experiment_name": "mealie-model-serve-training",
    "run_name": "mealie-feedback-$(date -u +%Y%m%dT%H%M%SZ)",
    "registered_model_name": "${MODEL_NAME}",
    "register_model": true,
    "register_alias": "${MODEL_ALIAS}"
  }
}
EOF

docker build -f "${ROOT_DIR}/Dockerfile.train" -t "${TRAIN_IMAGE}" "${ROOT_DIR}"

if [[ "${CURATED_FEEDBACK}" == swift://* ]]; then
  docker run --rm \
    -e SWIFT_AUTH_URL="${SWIFT_AUTH_URL}" \
    -e SWIFT_APP_CREDENTIAL_ID="${SWIFT_APP_CREDENTIAL_ID}" \
    -e SWIFT_APP_CREDENTIAL_SECRET="${SWIFT_APP_CREDENTIAL_SECRET}" \
    -v "${ROOT_DIR}:/app" \
    --entrypoint python \
    "${TRAIN_IMAGE}" \
    /app/scripts/fetch_swift_object.py \
    --container "${swift_container}" \
    --key "${swift_key}" \
    --output "/app/${CURATED_REL}" \
    --auth-url "${SWIFT_AUTH_URL}" \
    --app-credential-id "${SWIFT_APP_CREDENTIAL_ID}" \
    --app-credential-secret "${SWIFT_APP_CREDENTIAL_SECRET}"
fi

docker run --rm \
  -v "${ROOT_DIR}:/app" \
  -v "${ROOT_DIR}/trainer:/app/trainer" \
  -v "${ROOT_DIR}/scripts:/app/scripts" \
  --entrypoint python \
  "${TRAIN_IMAGE}" \
  /app/scripts/build_feedback_dataset.py \
  --input "/app/${CURATED_REL}" \
  --output "/app/${DATASET_REL}" \
  --label-field "${LABEL_FIELD}"

bash "${ROOT_DIR}/scripts/run_train_chameleon.sh" "${CONFIG_ABS}" --register-model --alias "${MODEL_ALIAS}" "$@"
