#!/usr/bin/env python3
"""Automate canary gating, promotion, and rollback using MLflow aliases and live endpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from mlflow.tracking import MlflowClient


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


async def _benchmark(base_url: str, requests: int, concurrency: int) -> dict[str, float]:
    payload = {"inputs": [[5.1, 3.5, 1.4, 0.2]], "source": "rollout-controller"}
    latencies: list[float] = []
    errors = 0
    sem = asyncio.Semaphore(concurrency)

    async def _one(client: httpx.AsyncClient) -> None:
        nonlocal errors
        async with sem:
            start = time.perf_counter()
            try:
                response = await client.post(f"{base_url.rstrip('/')}/predict", json=payload, timeout=30.0)
                ok = response.status_code == 200
            except Exception:
                ok = False
            latencies.append(time.perf_counter() - start)
            if not ok:
                errors += 1

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[_one(client) for _ in range(requests)])

    latencies.sort()
    successes = requests - errors
    return {
        "requests": requests,
        "successes": successes,
        "error_rate": errors / requests if requests else 0.0,
        "p50_ms": _percentile(latencies, 50) * 1000,
        "p95_ms": _percentile(latencies, 95) * 1000,
    }


def _load_policy(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _client(tracking_uri: str) -> MlflowClient:
    return MlflowClient(tracking_uri=tracking_uri)


def _get_registered_tags(client: MlflowClient, model: str) -> dict[str, str]:
    registered = client.get_registered_model(model)
    return {tag.key: tag.value for tag in registered.tags}


def _set_registered_tag(client: MlflowClient, model: str, key: str, value: str) -> None:
    client.set_registered_model_tag(model, key, value)


def _set_alias(client: MlflowClient, model: str, alias: str, version: str) -> None:
    client.set_registered_model_alias(model, alias, version)


def _get_alias_version(client: MlflowClient, model: str, alias: str) -> str | None:
    try:
        mv = client.get_model_version_by_alias(model, alias)
    except Exception:  # noqa: BLE001
        return None
    return str(mv.version)


def _fetch_json(url: str) -> dict[str, Any]:
    response = httpx.get(url, timeout=20.0)
    response.raise_for_status()
    return response.json()


def _reload(url: str, model_uri: str) -> dict[str, Any]:
    response = httpx.post(f"{url.rstrip('/')}/reload", json={"model_uri": model_uri}, timeout=20.0)
    response.raise_for_status()
    return response.json()


def _set_router_weight(router_url: str, weight: float) -> dict[str, Any]:
    response = httpx.post(
        f"{router_url.rstrip('/')}/admin/weights",
        json={"canary_weight": weight},
        timeout=20.0,
    )
    response.raise_for_status()
    return response.json()


def _gate_passed(prod: dict[str, float], canary: dict[str, float], policy: dict[str, Any]) -> bool:
    gate = policy["gate"]
    if canary["successes"] < int(gate["min_successful_requests"]):
        return False
    if canary["error_rate"] > float(gate["max_error_rate"]):
        return False
    if prod["p50_ms"] > 0 and canary["p50_ms"] > prod["p50_ms"] * float(gate["max_p50_regression_ratio"]):
        return False
    if prod["p95_ms"] > 0 and canary["p95_ms"] > prod["p95_ms"] * float(gate["max_p95_regression_ratio"]):
        return False
    return True


def _promote(args: argparse.Namespace, policy: dict[str, Any]) -> int:
    prod_meta = _fetch_json(f"{args.production_url.rstrip('/')}/metadata")
    canary_meta = _fetch_json(f"{args.canary_url.rstrip('/')}/metadata")
    prod_bench = asyncio.run(
        _benchmark(args.production_url, policy["gate"]["benchmark_requests"], policy["gate"]["benchmark_concurrency"])
    )
    canary_bench = asyncio.run(
        _benchmark(args.canary_url, policy["gate"]["benchmark_requests"], policy["gate"]["benchmark_concurrency"])
    )
    passed = _gate_passed(prod_bench, canary_bench, policy)

    client = _client(args.tracking_uri)
    _set_registered_tag(client, args.model, "rollout.last_gate", "passed" if passed else "failed")
    _set_registered_tag(client, args.model, "rollout.last_gate_details", json.dumps({"production": prod_bench, "canary": canary_bench}))

    if not passed:
        print(json.dumps({"ok": False, "action": "rejected", "production": prod_bench, "canary": canary_bench}, indent=2))
        return 1

    old_prod = str(prod_meta["model_version"])
    new_prod = str(canary_meta["model_version"])
    _set_registered_tag(client, args.model, "last_good_production_version", old_prod)
    _set_registered_tag(client, args.model, "current_production_version", new_prod)
    _set_alias(client, args.model, "production", new_prod)
    _reload(args.production_url, f"models:/{args.model}@production")
    _set_router_weight(args.router_url, float(policy["router"]["stable_canary_weight"]))
    print(json.dumps({"ok": True, "action": "promoted", "from": old_prod, "to": new_prod, "production": prod_bench, "canary": canary_bench}, indent=2))
    return 0


def _rollback(args: argparse.Namespace, policy: dict[str, Any]) -> int:
    client = _client(args.tracking_uri)
    tags = _get_registered_tags(client, args.model)
    target = tags.get("last_good_production_version")
    if not target:
        print("No last_good_production_version tag set", file=sys.stderr)
        return 1
    _set_alias(client, args.model, "production", target)
    _reload(args.production_url, f"models:/{args.model}@production")
    _set_registered_tag(client, args.model, "current_production_version", target)
    _set_router_weight(args.router_url, float(policy["router"]["stable_canary_weight"]))
    print(json.dumps({"ok": True, "action": "rolled_back", "version": target}, indent=2))
    return 0


def _monitor(args: argparse.Namespace, policy: dict[str, Any]) -> int:
    client = _client(args.tracking_uri)
    prod_meta = _fetch_json(f"{args.production_url.rstrip('/')}/metadata")
    canary_meta = _fetch_json(f"{args.canary_url.rstrip('/')}/metadata")
    tags = _get_registered_tags(client, args.model)
    last_good = tags.get("last_good_production_version")
    target_canary_model_uri = f"models:/{args.model}@canary"
    target_canary_version = _get_alias_version(client, args.model, "canary")

    canary_is_stale = (
        canary_meta.get("model_alias") != "canary"
        or canary_meta.get("model_uri") != target_canary_model_uri
        or (
            target_canary_version is not None
            and str(canary_meta.get("model_version")) != target_canary_version
        )
    )

    if canary_is_stale:
        reload_result = _reload(args.canary_url, target_canary_model_uri)
        canary_meta = _fetch_json(f"{args.canary_url.rstrip('/')}/metadata")
        if canary_meta.get("model_alias") != "canary" or canary_meta.get("model_uri") != target_canary_model_uri:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "action": "canary_refresh_failed",
                        "target_model_uri": target_canary_model_uri,
                        "target_version": target_canary_version,
                        "reload_result": reload_result,
                        "canary_metadata": canary_meta,
                    },
                    indent=2,
                )
            )
            return 1
        print(
            json.dumps(
                {
                    "ok": True,
                    "action": "canary_refreshed",
                    "target_model_uri": target_canary_model_uri,
                    "target_version": target_canary_version,
                    "reload_result": reload_result,
                    "canary_metadata": canary_meta,
                },
                indent=2,
            )
        )

    if str(prod_meta["model_version"]) != str(canary_meta["model_version"]):
        _set_router_weight(args.router_url, float(policy["router"]["trial_canary_weight"]))
        if policy["rollout"]["auto_promote"]:
            return _promote(args, policy)
        return 0

    prod_bench = asyncio.run(
        _benchmark(args.production_url, policy["gate"]["benchmark_requests"], policy["gate"]["benchmark_concurrency"])
    )
    prod_healthy = (
        prod_bench["successes"] >= int(policy["gate"]["min_successful_requests"])
        and prod_bench["error_rate"] <= float(policy["gate"]["max_error_rate"])
    )
    if prod_healthy or not policy["rollout"]["auto_rollback"]:
        _set_router_weight(args.router_url, float(policy["router"]["stable_canary_weight"]))
        print(json.dumps({"ok": True, "action": "monitor_only", "production": prod_bench}, indent=2))
        return 0

    if last_good and last_good != str(prod_meta["model_version"]):
        return _rollback(args, policy)

    print(json.dumps({"ok": False, "action": "unhealthy_but_no_rollback_target", "production": prod_bench}, indent=2))
    return 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("promote", "rollback", "monitor"), required=True)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--model", default=os.environ.get("ROLLOUT_MODEL_NAME", "food-classifier"))
    parser.add_argument("--production-url", default=os.environ.get("PRODUCTION_URL", "http://mms-model-serve:8080"))
    parser.add_argument("--canary-url", default=os.environ.get("CANARY_URL", "http://mms-model-serve-canary:8080"))
    parser.add_argument("--router-url", default=os.environ.get("ROUTER_URL", "http://mms-rollout-router:8082"))
    parser.add_argument(
        "--policy",
        default=os.environ.get("ROLLOUT_POLICY_PATH", "/app/ops/rollout_policy.json"),
    )
    args = parser.parse_args()
    if not args.tracking_uri:
        print("MLFLOW_TRACKING_URI required", file=sys.stderr)
        sys.exit(1)
    return args


def main() -> None:
    args = _parse_args()
    policy = _load_policy(Path(args.policy))
    if args.mode == "promote":
        sys.exit(_promote(args, policy))
    if args.mode == "rollback":
        sys.exit(_rollback(args, policy))
    sys.exit(_monitor(args, policy))


if __name__ == "__main__":
    main()
