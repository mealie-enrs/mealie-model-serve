#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <feedback-jsonl> <dataset-npz> <config-json>"
  exit 1
fi

FEEDBACK_JSONL="$1"
DATASET_NPZ="$2"
CONFIG_JSON="$3"

python scripts/build_feedback_dataset.py --input "${FEEDBACK_JSONL}" --output "${DATASET_NPZ}"
python -m trainer.train --config "${CONFIG_JSON}" --register-model --alias canary
