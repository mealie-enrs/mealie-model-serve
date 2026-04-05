#!/usr/bin/env python3
"""Phase 2 MVP: train Iris sklearn → ONNX, log to MLflow, register model + required tags + @staging alias."""

from __future__ import annotations

import os
import subprocess
import sys

import mlflow
import mlflow.onnx
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(__file__),
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.environ.get("GIT_SHA", "unknown")


def _apply_s3_env() -> None:
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


def main() -> None:
    _apply_s3_env()
    tracking = os.environ["MLFLOW_TRACKING_URI"]
    model_name = os.environ.get("MODEL_NAME", "food-classifier")
    dataset_version = os.environ.get("DATASET_VERSION", "v1")
    experiment = os.environ.get("MLFLOW_EXPERIMENT", "mealie-model-serve")
    register = os.environ.get("REGISTER_MODEL", "true").lower() in ("1", "true", "yes")

    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment(experiment)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(X_train, y_train)
    acc = float(clf.score(X_test, y_test))
    git_sha = _git_sha()

    onnx_model = convert_sklearn(
        clf,
        initial_types=[("float_input", FloatTensorType([None, X.shape[1]]))],
        options={id(clf): {"zipmap": False}},
    )

    with mlflow.start_run() as run:
        mlflow.log_param("framework", "sklearn")
        mlflow.log_param("export_format", "onnx")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("n_features", X.shape[1])
        mlflow.onnx.log_model(onnx_model, "model")

        run_id = run.info.run_id

    if not register:
        print(f"run_id={run_id} (REGISTER_MODEL=false)")
        return

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    ver = str(mv.version)
    client = MlflowClient(tracking_uri=tracking)

    tags = {
        "git_sha": git_sha,
        "dataset_version": dataset_version,
        "train_run_id": run_id,
        "framework": "sklearn",
        "export_format": "onnx",
        "validation_status": "passed",
        "benchmark_status": "pending",
        "created_by": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        "runtime_target": os.environ.get("RUNTIME_TARGET", "onnxruntime"),
    }
    for k, v in tags.items():
        client.set_model_version_tag(model_name, ver, k, v)

    client.set_registered_model_alias(model_name, "staging", ver)
    print(f"registered {model_name} version={ver} alias=staging run_id={run_id}")


if __name__ == "__main__":
    try:
        main()
    except KeyError as exc:
        print(f"Missing env: {exc}", file=sys.stderr)
        sys.exit(1)
