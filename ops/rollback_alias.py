#!/usr/bin/env python3
"""Rollback: point an alias at an older version (same API as promotion)."""

from __future__ import annotations

import argparse
import os
import sys

from mlflow.tracking import MlflowClient

ALIASES = ("staging", "canary", "champion", "production")


def main() -> None:
    p = argparse.ArgumentParser(description="Move alias back to a previous version")
    p.add_argument("--model", required=True)
    p.add_argument("--alias", required=True, choices=ALIASES)
    p.add_argument("--version", required=True, help="Older version to point the alias at")
    p.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    args = p.parse_args()
    if not args.tracking_uri:
        print("MLFLOW_TRACKING_URI required", file=sys.stderr)
        sys.exit(1)

    client = MlflowClient(tracking_uri=args.tracking_uri)
    client.set_registered_model_alias(args.model, args.alias, args.version)
    print(f"rollback: {args.model}@{args.alias} -> version {args.version}")


if __name__ == "__main__":
    main()
