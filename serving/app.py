"""FastAPI serving: health, metadata, predict, reload, Prometheus metrics."""

from __future__ import annotations

import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from serving import cache as cache_mod
from serving import metrics as prom
from serving.config import settings
from serving.loader import OnnxModel, find_onnx_file
from serving.resolver import ModelUriError, resolve

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "loaded": False,
    "model_uri": "",
    "model_name": "",
    "model_version": "",
    "model_alias": None,
    "onnx": None,
    "provider_line": "",
    "last_load_unix": None,
    "load_error": None,
}


def _apply_mlflow_env() -> None:
    if settings.mlflow_s3_endpoint_url:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.mlflow_s3_endpoint_url
    if settings.aws_access_key_id:
        os.environ["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
    if settings.aws_secret_access_key:
        os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
    os.environ.setdefault("AWS_DEFAULT_REGION", settings.aws_default_region)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


def _mlflow_client() -> MlflowClient:
    uri = settings.mlflow_registry_uri or settings.mlflow_tracking_uri
    return MlflowClient(tracking_uri=uri)


def load_model_uri(uri: str) -> tuple[str, str, str | None]:
    """Download artifacts to cache, build ONNX session, swap global state."""
    _apply_mlflow_env()
    client = _mlflow_client()
    resolved = resolve(client, uri)
    cache_root = Path(settings.model_cache_dir)
    target = cache_mod.version_dir(cache_root, resolved.name, resolved.version)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    mlflow.artifacts.download_artifacts(
        artifact_uri=resolved.download_uri,
        tracking_uri=settings.mlflow_tracking_uri,
        dst_path=str(target),
    )
    onnx_path = find_onnx_file(target)
    session = OnnxModel(onnx_path, settings.model_provider, settings)
    provider_line = ",".join(session.providers_in_use)

    with _state_lock:
        old = _state.get("onnx")
        _state["onnx"] = session
        _state["loaded"] = True
        _state["model_uri"] = uri
        _state["model_name"] = resolved.name
        _state["model_version"] = resolved.version
        _state["model_alias"] = resolved.alias
        _state["provider_line"] = provider_line
        _state["last_load_unix"] = time.time()
        _state["load_error"] = None
        del old  # noqa: WPS420

    prom.set_active_model(resolved.name, resolved.version, resolved.alias)
    prom.LAST_LOAD_UNIX.set(_state["last_load_unix"])
    cache_mod.prune_old_versions(cache_root, resolved.name, keep_versions=3)
    return resolved.name, resolved.version, resolved.alias


@asynccontextmanager
async def lifespan(app: FastAPI):
    _apply_mlflow_env()
    if os.environ.get("SKIP_MODEL_LOAD", "").lower() not in ("1", "true", "yes"):
        try:
            load_model_uri(settings.model_uri)
        except Exception as exc:  # noqa: BLE001
            _state["load_error"] = str(exc)
            _state["loaded"] = False
    yield


app = FastAPI(title="Mealie Model Serve", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    path = request.url.path
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = str(response.status_code)
        prom.REQUESTS.labels(request.method, path, status).inc()
        return response
    except Exception:
        prom.REQUESTS.labels(request.method, path, "500").inc()
        prom.ERRORS.labels("handler").inc()
        raise
    finally:
        prom.LATENCY.labels(path).observe(time.perf_counter() - start)


class PredictBody(BaseModel):
    inputs: list[list[float]] = Field(..., description="Batch of feature vectors (float)")


class ReloadBody(BaseModel):
    model_uri: str


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> Response:
    if not _state.get("loaded"):
        detail = _state.get("load_error") or "model not loaded"
        return Response(status_code=503, content=detail, media_type="text/plain")
    return Response(status_code=200, content="ok", media_type="text/plain")


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    with _state_lock:
        return {
            "service": settings.service_name,
            "serving_option_id": settings.serving_option_id,
            "model_name": _state.get("model_name") or "",
            "model_version": _state.get("model_version") or "",
            "model_alias": _state.get("model_alias"),
            "provider": _state.get("provider_line") or "",
            "build_sha": settings.build_sha,
            "model_uri": _state.get("model_uri") or "",
            "last_load_unix": _state.get("last_load_unix"),
            "ort_graph_optimization_level": settings.ort_graph_optimization_level,
            "ort_intra_op_num_threads": settings.ort_intra_op_num_threads,
            "ort_inter_op_num_threads": settings.ort_inter_op_num_threads,
            "ort_execution_mode": settings.ort_execution_mode,
        }


@app.post("/predict")
def predict(body: PredictBody) -> dict[str, Any]:
    with _state_lock:
        if not _state.get("loaded") or _state.get("onnx") is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        m = _state["onnx"]
        name = _state["model_name"]
        version = _state["model_version"]
        alias = _state["model_alias"]
    try:
        out = m.predict(body.inputs)
        preds = out.tolist() if hasattr(out, "tolist") else out
    except Exception as exc:  # noqa: BLE001
        prom.ERRORS.labels("inference").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "predictions": preds,
        "model_name": name,
        "model_version": version,
        "model_alias": alias,
    }


@app.post("/reload")
def reload_model(body: ReloadBody) -> dict[str, Any]:
    with _state_lock:
        old_v = _state.get("model_version")
        old_a = _state.get("model_alias")
    try:
        new_name, new_ver, new_alias = load_model_uri(body.model_uri.strip())
    except ModelUriError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        prom.ERRORS.labels("reload").inc()
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "ok": True,
        "previous_version": old_v,
        "previous_alias": old_a,
        "model_name": new_name,
        "model_version": new_ver,
        "model_alias": new_alias,
        "model_uri": body.model_uri,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=prom.metrics_payload(), media_type=CONTENT_TYPE_LATEST)
