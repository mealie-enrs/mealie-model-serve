#!/usr/bin/env python3
"""
Placeholder model for serving benchmarks (no waiting on full training).
Exports a tiny linear ONNX model with fixed random weights (reproducible seed).
Optionally logs + registers to MLflow like train.py.
"""

from __future__ import annotations

import argparse
import os

import mlflow
import mlflow.onnx
import numpy as np
import onnx
from mlflow.tracking import MlflowClient
from onnx import TensorProto, helper


def build_toy_onnx(n_in: int = 4, n_out: int = 3, seed: int = 42) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n_in, n_out)).astype(np.float32)
    b = rng.standard_normal((n_out,)).astype(np.float32)
    w_init = onnx.numpy_helper.from_array(w, name="W")
    b_init = onnx.numpy_helper.from_array(b, name="B")
    x = helper.make_tensor_value_info("float_input", TensorProto.FLOAT, [None, n_in])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, n_out])
    matmul = helper.make_node("MatMul", ["float_input", "W"], ["pre"])
    add = helper.make_node("Add", ["pre", "B"], ["logits"])
    graph = helper.make_graph(
        [matmul, add],
        "toy_linear",
        [x],
        [y],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-name",
        default="serving-bench-model",
        help="Use a dedicated name to avoid clashing with real food-classifier",
    )
    p.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    p.add_argument("--experiment", default="mealie-model-serve")
    p.add_argument("--register", action="store_true")
    p.add_argument("--alias", default="staging", choices=("staging", "canary", "champion"))
    args = p.parse_args()
    if not args.tracking_uri:
        raise SystemExit("Set MLFLOW_TRACKING_URI")

    model = build_toy_onnx()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="toy-random-onnx") as run:
        mlflow.log_param("placeholder", True)
        mlflow.log_param("export_format", "onnx")
        mlflow.onnx.log_model(model, "model")
        rid = run.info.run_id

    if args.register:
        mv = mlflow.register_model(f"runs:/{rid}/model", args.model_name)
        ver = str(mv.version)
        client = MlflowClient(tracking_uri=args.tracking_uri)
        for k, v in {
            "git_sha": os.environ.get("GIT_SHA", "n/a"),
            "dataset_version": "placeholder",
            "train_run_id": rid,
            "framework": "onnx-toy",
            "export_format": "onnx",
            "validation_status": "skipped",
            "benchmark_status": "pending",
            "created_by": "toy_random_onnx",
        }.items():
            client.set_model_version_tag(args.model_name, ver, k, v)
        client.set_registered_model_alias(args.model_name, args.alias, ver)
        print(f"registered {args.model_name} v{ver} @{args.alias} run={rid}")
    else:
        print(f"run_id={rid} (add --register to push to registry)")


if __name__ == "__main__":
    main()
