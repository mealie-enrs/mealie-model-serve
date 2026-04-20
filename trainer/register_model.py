#!/usr/bin/env python3
"""Register an existing run's model artifact as a new registry version and apply tags."""

from __future__ import annotations

import argparse
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--run-id", required=True, help="MLflow run id containing /model artifact")
    p.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    p.add_argument(
        "--set-alias",
        default=None,
        choices=("staging", "canary", "champion", "production"),
        help="Optional alias to assign to the new version",
    )
    args = p.parse_args()
    if not args.tracking_uri:
        print("Set MLFLOW_TRACKING_URI or pass --tracking-uri", file=sys.stderr)
        sys.exit(1)

    mlflow.set_tracking_uri(args.tracking_uri)
    uri = f"runs:/{args.run_id}/model"
    mv = mlflow.register_model(model_uri=uri, name=args.model_name)
    ver = str(mv.version)
    print(f"registered {args.model_name} version={ver}")

    client = MlflowClient(tracking_uri=args.tracking_uri)
    if args.set_alias:
        client.set_registered_model_alias(args.model_name, args.set_alias, ver)
        print(f"alias {args.set_alias} -> {ver}")


if __name__ == "__main__":
    main()
