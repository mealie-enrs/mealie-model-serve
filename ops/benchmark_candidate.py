#!/usr/bin/env python3
"""Phase 4 stub: mark benchmark_status on a model version (extend with real latency/accuracy gates)."""

from __future__ import annotations

import argparse
import os
import sys

from mlflow.tracking import MlflowClient


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--status", default="passed", choices=("passed", "failed", "pending"))
    p.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    args = p.parse_args()
    if not args.tracking_uri:
        print("MLFLOW_TRACKING_URI required", file=sys.stderr)
        sys.exit(1)

    client = MlflowClient(tracking_uri=args.tracking_uri)
    client.set_model_version_tag(args.model, args.version, "benchmark_status", args.status)
    print(f"{args.model} v{args.version} benchmark_status={args.status}")


if __name__ == "__main__":
    main()
