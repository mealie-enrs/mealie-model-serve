"""Weighted rollout router for production/canary traffic and feedback capture."""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter as PromCounter, Gauge, generate_latest

from rollout.config import router_settings

_client: httpx.AsyncClient | None = None
_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "canary_weight": float(router_settings.default_canary_weight),
    "last_route_unix": None,
}

ROUTED_REQUESTS = PromCounter(
    "mms_rollout_router_requests_total",
    "Requests proxied by rollout router",
    ["lane", "status"],
)
FEEDBACK_WRITES = PromCounter(
    "mms_rollout_feedback_events_total",
    "Feedback events captured by rollout router",
    ["decision"],
)
CANARY_WEIGHT = Gauge(
    "mms_rollout_canary_weight_ratio",
    "Current canary routing weight",
)


def _feedback_path() -> Path:
    router_settings.feedback_dir.mkdir(parents=True, exist_ok=True)
    return router_settings.feedback_dir / f"feedback-{datetime.now(UTC):%Y%m%d}.jsonl"


def _read_recent_feedback(limit: int = 5000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted(router_settings.feedback_dir.glob("feedback-*.jsonl"))
    for path in files[-7:]:
        for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def _append_feedback(row: dict[str, Any]) -> None:
    payload = json.dumps(row, sort_keys=True)
    with _feedback_path().open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")


def _route_lane(force_lane: str | None) -> str:
    if force_lane in {"production", "canary"}:
        return force_lane
    with _state_lock:
        weight = float(_state["canary_weight"])
    return "canary" if random.random() < weight else "production"


def _lane_base_url(lane: str) -> str:
    if lane == "production":
        return router_settings.production_url.rstrip("/")
    if lane == "canary":
        return router_settings.canary_url.rstrip("/")
    raise ValueError(f"Unsupported lane: {lane}")


async def _fetch_json(url: str) -> dict[str, Any]:
    assert _client is not None
    response = await _client.get(url, timeout=router_settings.timeout_sec)
    response.raise_for_status()
    return response.json()


async def _proxy_predict(lane: str, payload: dict[str, Any]) -> dict[str, Any]:
    assert _client is not None
    response = await _client.post(
        f"{_lane_base_url(lane)}/predict",
        json=payload,
        timeout=router_settings.timeout_sec,
    )
    ROUTED_REQUESTS.labels(lane, str(response.status_code)).inc()
    response.raise_for_status()
    return response.json()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client  # noqa: PLW0603
    _client = httpx.AsyncClient()
    CANARY_WEIGHT.set(router_settings.default_canary_weight)
    yield
    await _client.aclose()
    _client = None


app = FastAPI(title="MMS Rollout Router", version="0.1.0", lifespan=lifespan)


class PredictBody(BaseModel):
    inputs: list[list[float]]
    source: str = Field(default="demo")


class FeedbackBody(BaseModel):
    request_id: str
    inputs: list[float]
    predicted_output: list[float] | int | float | str | None = None
    approved_output: int | float | str
    backend_lane: str
    source: str = Field(default="demo")
    accepted_prediction: bool = Field(default=False)
    notes: str | None = None


class WeightBody(BaseModel):
    canary_weight: float = Field(ge=0.0, le=1.0)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, str]:
    try:
        prod, canary = await asyncio.gather(
            _fetch_json(f"{router_settings.production_url.rstrip('/')}/metadata"),
            _fetch_json(f"{router_settings.canary_url.rstrip('/')}/metadata"),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not prod.get("model_version") or not canary.get("model_version"):
        raise HTTPException(status_code=503, detail="backend metadata incomplete")
    return {"status": "ok"}


@app.get("/metadata")
async def metadata() -> dict[str, Any]:
    prod, canary = await asyncio.gather(
        _fetch_json(f"{router_settings.production_url.rstrip('/')}/metadata"),
        _fetch_json(f"{router_settings.canary_url.rstrip('/')}/metadata"),
    )
    with _state_lock:
        weight = float(_state["canary_weight"])
        last_route_unix = _state["last_route_unix"]
    return {
        "service": "rollout-router",
        "production_backend": prod,
        "canary_backend": canary,
        "canary_weight": weight,
        "last_route_unix": last_route_unix,
        "feedback_dir": str(router_settings.feedback_dir),
    }


@app.post("/predict")
async def predict(body: PredictBody, x_mms_lane: str | None = Header(default=None)) -> dict[str, Any]:
    lane = _route_lane(x_mms_lane)
    request_id = str(uuid.uuid4())
    try:
        proxied = await _proxy_predict(lane, body.model_dump())
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    with _state_lock:
        _state["last_route_unix"] = time.time()
        weight = float(_state["canary_weight"])

    return {
        **proxied,
        "request_id": request_id,
        "served_by_lane": lane,
        "router_canary_weight": weight,
    }


@app.post("/feedback")
def feedback(body: FeedbackBody) -> dict[str, Any]:
    row = {
        **body.model_dump(),
        "event_time": datetime.now(UTC).isoformat(),
    }
    _append_feedback(row)
    FEEDBACK_WRITES.labels("accepted" if body.accepted_prediction else "corrected").inc()
    return {"ok": True, "feedback_path": str(_feedback_path()), "request_id": body.request_id}


@app.get("/feedback/summary")
def feedback_summary() -> dict[str, Any]:
    rows = _read_recent_feedback()
    lane_counts = Counter(row.get("backend_lane", "unknown") for row in rows)
    accepted = sum(1 for row in rows if row.get("accepted_prediction"))
    corrected = len(rows) - accepted
    return {
        "events": len(rows),
        "accepted": accepted,
        "corrected": corrected,
        "lanes": dict(lane_counts),
        "feedback_dir": str(router_settings.feedback_dir),
    }


@app.post("/admin/weights")
def set_weights(body: WeightBody) -> dict[str, Any]:
    with _state_lock:
        _state["canary_weight"] = float(body.canary_weight)
        weight = float(_state["canary_weight"])
    CANARY_WEIGHT.set(weight)
    return {"ok": True, "canary_weight": weight}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
