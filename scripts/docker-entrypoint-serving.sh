#!/usr/bin/env sh
set -eu
# System-level: UVICORN_WORKERS>1 uses multiple processes (each loads the model → N× RAM).
WORKERS="${UVICORN_WORKERS:-1}"
exec uvicorn serving.app:app --host 0.0.0.0 --port "${PORT:-8080}" --workers "$WORKERS"
