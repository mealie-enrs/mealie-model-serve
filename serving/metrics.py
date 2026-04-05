"""Prometheus metrics for requests, latency, errors, active model."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest

REQUESTS = Counter(
    "mms_serving_requests_total",
    "HTTP requests",
    ["method", "path", "status"],
)
LATENCY = Histogram(
    "mms_serving_request_latency_seconds",
    "Request latency",
    ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
ERRORS = Counter("mms_serving_errors_total", "Inference / handler errors", ["kind"])

ACTIVE_MODEL = Info("mms_serving_active_model", "Currently loaded model identity")
LAST_LOAD_UNIX = Gauge("mms_serving_last_load_unixtime", "Unix time of last successful model load")


def set_active_model(name: str, version: str, alias: str | None) -> None:
    ACTIVE_MODEL.info(
        {
            "model_name": name,
            "model_version": version,
            "model_alias": alias or "",
        }
    )


def metrics_payload() -> bytes:
    return generate_latest()
