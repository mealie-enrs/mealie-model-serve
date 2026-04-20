#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_operator_flow_chameleon.sh [options]

Packages and runs the full operator flow on the Chameleon node:
  feedback generation -> raw feedback copy -> curated dataset build ->
  containerized training -> MLflow registration -> optional canary refresh check

Options:
  --host <host>              SSH host (default: 192.5.87.188)
  --user <user>              SSH user (default: cc)
  --ssh-key <path>           SSH private key (default: ~/.ssh/id_rsa_chameleon)
  --router-url <url>         Router URL reachable from the Chameleon node
                             (default: http://127.0.0.1:30610)
  --requests <n>             Feedback events to generate (default: 24)
  --alias <alias>            Model alias to assign (default: retrain-demo)
  --model-name <name>        Registered model name (default: food-classifier)
  --controller-check         If alias=canary, create a manual controller job after training
  --workspace <path>         Remote workspace root (default: ~/mms-operator-flow)
  --help                     Show this message
EOF
}

HOST="192.5.87.188"
SSH_USER="cc"
SSH_KEY="${HOME}/.ssh/id_rsa_chameleon"
ROUTER_URL="http://127.0.0.1:30610"
REQUESTS=24
MODEL_ALIAS="retrain-demo"
MODEL_NAME="food-classifier"
CONTROLLER_CHECK=0
WORKSPACE_ROOT="~/mms-operator-flow"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --user)
      SSH_USER="$2"
      shift 2
      ;;
    --ssh-key)
      SSH_KEY="$2"
      shift 2
      ;;
    --router-url)
      ROUTER_URL="$2"
      shift 2
      ;;
    --requests)
      REQUESTS="$2"
      shift 2
      ;;
    --alias)
      MODEL_ALIAS="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --controller-check)
      CONTROLLER_CHECK=1
      shift
      ;;
    --workspace)
      WORKSPACE_ROOT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_WORKSPACE="${WORKSPACE_ROOT}/${STAMP}"
PUBLIC_MLFLOW_URL="http://${HOST}:30601"

echo "Uploading operator-flow workspace to ${SSH_USER}@${HOST}:${REMOTE_WORKSPACE}" >&2
tar czf - \
  -C "${ROOT_DIR}" \
  --exclude='__pycache__' \
  Dockerfile.train \
  requirements-train.txt \
  trainer \
  scripts/build_feedback_dataset.py | \
  ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new "${SSH_USER}@${HOST}" \
    "mkdir -p ${REMOTE_WORKSPACE} && tar xzf - -C ${REMOTE_WORKSPACE}"

echo "Running full operator flow on ${HOST}" >&2
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new "${SSH_USER}@${HOST}" \
  env \
    REMOTE_WORKSPACE="${REMOTE_WORKSPACE}" \
    ROUTER_URL="${ROUTER_URL}" \
    REQUESTS="${REQUESTS}" \
    MODEL_ALIAS="${MODEL_ALIAS}" \
    MODEL_NAME="${MODEL_NAME}" \
    CONTROLLER_CHECK="${CONTROLLER_CHECK}" \
    PUBLIC_MLFLOW_URL="${PUBLIC_MLFLOW_URL}" \
    GIT_SHA="$(git -C "${ROOT_DIR}" rev-parse HEAD)" \
  'bash -s' <<'REMOTE_SCRIPT'
set -euo pipefail

cd "${REMOTE_WORKSPACE}"
mkdir -p feedback trainer/artifacts trainer/configs

python3 - <<'PY'
import json
import os
import urllib.request

samples = [
    ([5.1, 3.5, 1.4, 0.2], 0),
    ([4.9, 3.0, 1.4, 0.2], 0),
    ([5.8, 2.7, 5.1, 1.9], 2),
    ([6.3, 2.9, 5.6, 1.8], 2),
    ([6.0, 2.2, 4.0, 1.0], 1),
    ([5.6, 2.9, 3.6, 1.3], 1),
]
router_url = os.environ["ROUTER_URL"].rstrip("/")
requests = int(os.environ["REQUESTS"])
generated = []
for idx in range(requests):
    row, label = samples[idx % len(samples)]
    body = json.dumps({"inputs": [row], "source": "operator-flow"}).encode()
    req = urllib.request.Request(
        f"{router_url}/predict",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        pred = json.loads(resp.read().decode())
    feedback = {
        "request_id": pred["request_id"],
        "inputs": row,
        "predicted_output": pred["predictions"][0],
        "approved_output": label,
        "backend_lane": pred["served_by_lane"],
        "source": "operator-flow",
        "accepted_prediction": False,
        "notes": f"operator-flow-{idx}",
    }
    freq = urllib.request.Request(
        f"{router_url}/feedback",
        data=json.dumps(feedback).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(freq, timeout=20):
        pass
    generated.append(feedback)
print(json.dumps({"generated_feedback_events": len(generated)}, indent=2))
PY

feedback_date="$(date -u +%Y%m%d)"
raw_feedback="feedback/feedback-${feedback_date}.jsonl"
dataset_npz="trainer/artifacts/retrain-feedback-${feedback_date}.npz"
config_json="trainer/configs/retrain-feedback-${feedback_date}.json"

router_pod="$(sudo k3s kubectl get pod -n mms -l app=mms-rollout-router -o jsonpath="{.items[0].metadata.name}")"
sudo k3s kubectl exec -n mms "${router_pod}" -- cat "/var/lib/mms-feedback/feedback-${feedback_date}.jsonl" > "${raw_feedback}"

python3 - <<'PY'
import json
import os
from pathlib import Path

feedback_date = os.popen("date -u +%Y%m%d").read().strip()
cfg = {
    "dataset": {
        "source": "npz_classification",
        "name": f"feedback-{feedback_date}",
        "path": f"/app/trainer/artifacts/retrain-feedback-{feedback_date}.npz",
    },
    "candidate": {
        "name": "logistic_regression",
        "params": {
            "C": 0.5,
            "max_iter": 1000,
        },
    },
    "mlflow": {
        "tracking_uri": "http://127.0.0.1:30601",
        "experiment_name": "mealie-model-serve-training",
        "run_name": f"feedback-retrain-logreg-{feedback_date}",
        "registered_model_name": os.environ["MODEL_NAME"],
        "register_model": True,
        "register_alias": os.environ["MODEL_ALIAS"],
    },
}
Path(f"trainer/configs/retrain-feedback-{feedback_date}.json").write_text(
    json.dumps(cfg, indent=2),
    encoding="utf-8",
)
PY

eval "$(python3 - <<'PY'
import base64
import json
import subprocess
secret = json.loads(
    subprocess.check_output(
        ["sudo", "k3s", "kubectl", "get", "secret", "-n", "mms", "mms-secrets", "-o", "json"]
    ).decode()
)
print("AWS_ACCESS_KEY_ID=" + base64.b64decode(secret["data"]["MINIO_ROOT_USER"]).decode())
print("AWS_SECRET_ACCESS_KEY=" + base64.b64decode(secret["data"]["MINIO_ROOT_PASSWORD"]).decode())
PY
)"

TRAIN_IMAGE="mealie-model-serve-train:latest"
MLFLOW_TRACKING_URI="http://127.0.0.1:30601"
MLFLOW_S3_ENDPOINT_URL="$(sudo k3s kubectl get configmap -n mms mms-config -o jsonpath="{.data.MLFLOW_S3_ENDPOINT_URL}")"

docker build -f Dockerfile.train -t "${TRAIN_IMAGE}" .

docker run --rm \
  -v "${PWD}/trainer:/app/trainer" \
  -v "${PWD}/scripts:/app/scripts" \
  -v "${PWD}/feedback:/app/feedback" \
  --entrypoint python \
  "${TRAIN_IMAGE}" \
  /app/scripts/build_feedback_dataset.py \
  --input "/app/${raw_feedback}" \
  --output "/app/${dataset_npz}"

train_output="$(
  docker run --rm \
    --network host \
    -e GIT_SHA="${GIT_SHA}" \
    -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
    -e MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL}" \
    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    -e AWS_DEFAULT_REGION="us-east-1" \
    -v "${PWD}/trainer:/app/trainer" \
    "${TRAIN_IMAGE}" \
    --config "/app/${config_json}" \
    --register-model \
    --alias "${MODEL_ALIAS}"
)"
printf "%s\n" "${train_output}"

run_id="$(printf "%s\n" "${train_output}" | sed -n "s/.*run_id=\([a-f0-9]\{32\}\).*/\1/p" | tail -n 1)"
alias_version="$(
  docker run --rm --network host --entrypoint python "${TRAIN_IMAGE}" -c \
    "from mlflow.tracking import MlflowClient; mv = MlflowClient(tracking_uri='http://127.0.0.1:30601').get_model_version_by_alias('${MODEL_NAME}', '${MODEL_ALIAS}'); print(mv.version)"
)"

controller_log=""
if [[ "${CONTROLLER_CHECK}" = "1" && "${MODEL_ALIAS}" = "canary" ]]; then
  job_name="mms-rollout-controller-manual-$(date +%s)"
  sudo k3s kubectl create job -n mms --from=cronjob/mms-rollout-controller "${job_name}" >/dev/null
  for _ in $(seq 1 30); do
    succeeded="$(sudo k3s kubectl get job -n mms "${job_name}" -o jsonpath="{.status.succeeded}" 2>/dev/null || true)"
    if [[ "${succeeded}" = "1" ]]; then
      break
    fi
    sleep 2
  done
  controller_log="$(sudo k3s kubectl logs -n mms "job/${job_name}" --tail=200)"
fi

python3 - <<PY
import json
import os
import subprocess
from pathlib import Path

feedback_path = Path("${raw_feedback}")
dataset_path = Path("${dataset_npz}")
summary = {
    "workspace": os.environ["REMOTE_WORKSPACE"],
    "raw_feedback_path": str(feedback_path),
    "raw_feedback_rows": sum(1 for line in feedback_path.read_text().splitlines() if line.strip()),
    "dataset_path": str(dataset_path),
    "dataset_bytes": dataset_path.stat().st_size,
    "model_name": os.environ["MODEL_NAME"],
    "alias": os.environ["MODEL_ALIAS"],
    "alias_version": "${alias_version}".strip(),
    "run_id": "${run_id}".strip(),
    "mlflow_run_url": f"{os.environ['PUBLIC_MLFLOW_URL']}/#/experiments/2/runs/${run_id}".strip(),
    "controller_check": ${CONTROLLER_CHECK},
}
if """${controller_log}""".strip():
    summary["controller_log"] = """${controller_log}"""
print(json.dumps(summary, indent=2))
PY
REMOTE_SCRIPT
