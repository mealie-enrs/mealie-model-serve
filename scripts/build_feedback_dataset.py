#!/usr/bin/env python3
"""Build an NPZ retraining dataset from rollout-router feedback logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def _load_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        features.append([float(value) for value in row["inputs"]])
        labels.append(int(row["approved_output"]))
    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to feedback-YYYYMMDD.jsonl")
    parser.add_argument("--output", required=True, help="Destination NPZ path")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    X, y = _load_rows(Path(args.input))
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if len(set(y.tolist())) > 1 else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
    summary = {
        "input": args.input,
        "output": args.output,
        "train_rows": int(X_train.shape[0]),
        "val_rows": int(X_val.shape[0]),
        "feature_dim": int(X_train.shape[1]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
