#!/usr/bin/env python3
"""Config-driven sklearn training entrypoint with MLflow tracking."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.onnx
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, log_loss
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from trainer.utils.config import load_json_config
from trainer.utils.data import load_dataset
from trainer.utils.env import collect_environment_info, flatten_dict
from trainer.utils.modeling import build_estimator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON configuration file under trainer/configs/ or elsewhere.",
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register the resulting ONNX model in MLflow Model Registry.",
    )
    parser.add_argument(
        "--alias",
        default=None,
        help="Optional MLflow alias to assign when registering a model.",
    )
    return parser.parse_args()


def _log_environment_info(env_info: dict[str, Any]) -> None:
    for key, value in flatten_dict(env_info, prefix="env").items():
        mlflow.log_param(key, value)


def _export_and_log_model(estimator: Any, input_dim: int) -> None:
    onnx_model = convert_sklearn(
        estimator,
        initial_types=[("float_input", FloatTensorType([None, input_dim]))],
        options={id(estimator): {"zipmap": False}},
    )
    mlflow.onnx.log_model(onnx_model, "model")


def _register_model(
    *,
    tracking_uri: str,
    run_id: str,
    model_name: str,
    alias: str | None,
    metadata: dict[str, Any],
) -> tuple[str, str]:
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(mv.version)
    client = MlflowClient(tracking_uri=tracking_uri)
    for key, value in metadata.items():
        client.set_model_version_tag(model_name, version, key, str(value))
    if alias:
        client.set_registered_model_alias(model_name, alias, version)
    return model_name, version


def main() -> None:
    args = _parse_args()
    config = load_json_config(Path(args.config))

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    experiment_name = config["mlflow"]["experiment_name"]
    run_name = config["mlflow"].get("run_name", config["candidate"]["name"])
    alias = args.alias if args.alias is not None else config["mlflow"].get("register_alias")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    dataset = load_dataset(config["dataset"])
    estimator = build_estimator(
        candidate_name=config["candidate"]["name"],
        params=config["candidate"].get("params", {}),
    )
    env_info = collect_environment_info()

    with mlflow.start_run(run_name=run_name) as run:
        flattened = flatten_dict(config)
        for key, value in flattened.items():
            mlflow.log_param(key, value)
        _log_environment_info(env_info)

        start_time = time.perf_counter()
        estimator.fit(dataset.X_train, dataset.y_train)
        fit_time_sec = time.perf_counter() - start_time

        infer_start = time.perf_counter()
        y_pred = estimator.predict(dataset.X_val)
        infer_time_sec = time.perf_counter() - infer_start

        metrics = {
            "val_accuracy": float(accuracy_score(dataset.y_val, y_pred)),
            "val_macro_f1": float(f1_score(dataset.y_val, y_pred, average="macro")),
            "fit_time_sec": fit_time_sec,
            "inference_time_sec": infer_time_sec,
            "train_rows": float(dataset.X_train.shape[0]),
            "val_rows": float(dataset.X_val.shape[0]),
            "feature_dim": float(dataset.X_train.shape[1]),
        }
        if hasattr(estimator, "predict_proba"):
            metrics["val_log_loss"] = float(
                log_loss(dataset.y_val, estimator.predict_proba(dataset.X_val))
            )
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            resolved_path = Path(tmp_dir) / "resolved_config.json"
            resolved_path.write_text(json.dumps(config, indent=2))
            mlflow.log_artifact(str(resolved_path), artifact_path="config")

            env_path = Path(tmp_dir) / "environment.json"
            env_path.write_text(json.dumps(env_info, indent=2))
            mlflow.log_artifact(str(env_path), artifact_path="environment")

        _export_and_log_model(estimator, dataset.X_train.shape[1])
        run_id = run.info.run_id

    if args.register_model or config["mlflow"].get("register_model", False):
        metadata = {
            "candidate_name": config["candidate"]["name"],
            "dataset_source": config["dataset"]["source"],
            "dataset_name": config["dataset"]["name"],
            "train_run_id": run_id,
            "validation_status": "passed",
            "runtime_target": "onnxruntime",
            "fit_time_sec": fit_time_sec,
            "val_accuracy": metrics["val_accuracy"],
            "val_macro_f1": metrics["val_macro_f1"],
            "build_sha": env_info["git_sha"],
        }
        model_name, version = _register_model(
            tracking_uri=tracking_uri,
            run_id=run_id,
            model_name=config["mlflow"]["registered_model_name"],
            alias=alias,
            metadata=metadata,
        )
        print(
            f"registered {model_name} version={version} alias={alias or 'none'} run_id={run_id}"
        )
        return

    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
