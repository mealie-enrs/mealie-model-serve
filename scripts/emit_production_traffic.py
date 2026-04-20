#!/usr/bin/env python3
"""Generate weighted production-like traffic and optional user feedback against the rollout router."""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

import httpx
from sklearn.datasets import load_iris


def _sample_inputs(count: int) -> list[list[float]]:
    X, y = load_iris(return_X_y=True)
    pairs = list(zip(X.tolist(), y.tolist(), strict=False))
    random.shuffle(pairs)
    rows = [row for row, _ in pairs[:count]]
    labels = [label for _, label in pairs[:count]]
    return rows, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url", default="http://127.0.0.1:30610")
    parser.add_argument("--requests", type=int, default=25)
    parser.add_argument("--feedback-rate", type=float, default=0.7)
    parser.add_argument("--accept-rate", type=float, default=0.5)
    args = parser.parse_args()

    inputs, labels = _sample_inputs(args.requests)
    summary: dict[str, Any] = {"requests": 0, "feedback_events": 0, "lanes": {"production": 0, "canary": 0}}

    with httpx.Client(timeout=20.0) as client:
        for row, approved_label in zip(inputs, labels, strict=False):
            response = client.post(
                f"{args.router_url.rstrip('/')}/predict",
                json={"inputs": [row], "source": "production-demo"},
            )
            response.raise_for_status()
            payload = response.json()
            lane = payload["served_by_lane"]
            summary["requests"] += 1
            summary["lanes"][lane] += 1

            if random.random() > args.feedback_rate:
                continue

            predictions = payload.get("predictions", [[]])[0]
            accepted = random.random() < args.accept_rate
            feedback = {
                "request_id": payload["request_id"],
                "inputs": row,
                "predicted_output": predictions,
                "approved_output": int(approved_label if not accepted else max(range(len(predictions)), key=lambda idx: predictions[idx])),
                "backend_lane": lane,
                "accepted_prediction": accepted,
                "source": "production-demo",
            }
            feedback_response = client.post(
                f"{args.router_url.rstrip('/')}/feedback",
                json=feedback,
            )
            feedback_response.raise_for_status()
            summary["feedback_events"] += 1

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
