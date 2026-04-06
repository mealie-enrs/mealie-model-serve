"""
serving-rahil/app_onnx_cuda.py

Serving Configuration: onnx_cuda
Hardware: GPU — Quadro RTX 6000 (24GB VRAM) on Chameleon CHI@UC nc33
Optimization: Combined — ONNX + CUDAExecutionProvider + graph optimization
Framework: FastAPI + ONNXRuntime CUDAExecutionProvider

CUDAExecutionProvider swaps generic CPU kernels for NVIDIA GPU-specific
implementations, enabling GPU parallelism for all tensor operations.
Combined with ORT_ENABLE_EXTENDED graph optimization.

This is the most promising configuration for latency-sensitive production.

Tradeoffs vs onnx_quantized:
  + Much lower latency for single requests (GPU parallelism)
  + Much higher batch throughput
  - Requires GPU hardware (higher infrastructure cost)
  - Cold start: CUDA context initialization adds ~2-5s at startup

Fallback: if CUDAExecutionProvider unavailable, falls back to CPU.

"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
import uuid
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from recipe_assembler import FOOD_CLASS_MAP, assemble_draft

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config 
MODEL_PATH      = os.getenv("MODEL_PATH", "efficientnet_b0.onnx")
MODEL_VERSION   = os.getenv("MODEL_VERSION", "efficientnet-b0-onnx-cuda-v1")
SERVING_OPTION  = os.getenv("SERVING_OPTION_ID", "onnx_cuda")
BYPASS_GATE     = os.getenv("BYPASS_INFERENCE_GATE", "0") == "1"
CONFIDENCE_THR  = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

#App
app = FastAPI(
    title="Mealie ML Serving — ONNX CUDA",
    description=(
        "EfficientNet-B0, GPU (RTX 6000), "
        "ONNX + CUDAExecutionProvider + graph optimization."
    ),
)

session: Optional[ort.InferenceSession] = None
active_provider: str = "CPUExecutionProvider"

# Preprocessing 
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((256, 256), Image.BILINEAR)
    left = (256 - 224) // 2
    image = image.crop((left, left, left + 224, left + 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# Schemas 
class PredictRequest(BaseModel):
    request_id: Optional[str] = None
    image: str
    image_filename: Optional[str] = "upload.jpg"
    timestamp: Optional[str] = None

class PredictResponse(BaseModel):
    request_id: str
    status: str
    confidence: float
    confidence_flag: bool
    model_version: str
    serving_option: str
    execution_provider: str
    draft_recipe: dict
    safeguarding: dict
    latency_ms: float


def load_cuda_session(model_path: str) -> tuple[ort.InferenceSession, str]:
    """
    Load ONNX model preferring CUDAExecutionProvider.

    Provider selection:
      1. CUDAExecutionProvider — if nvidia-container-toolkit + CUDA available
      2. CPUExecutionProvider  — fallback if GPU not accessible

    On Chameleon nc33 (Quadro RTX 6000, 24GB VRAM, CUDA 12.8):
      Expected provider: CUDAExecutionProvider

    Graph optimization level: ORT_ENABLE_EXTENDED
    (applied before CUDA execution — fusions reduce GPU kernel launches)
    """
    logger.info(f"Loading ONNX session with GPU preference: {model_path}")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    available = ort.get_available_providers()
    logger.info(f"Available ORT providers: {available}")

    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_used = "CUDAExecutionProvider"
        logger.info("CUDA provider available — using GPU")
    else:
        providers = ["CPUExecutionProvider"]
        provider_used = "CPUExecutionProvider"
        logger.warning("CUDA provider NOT available — falling back to CPU")

    sess = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    logger.info(f"Session ready | active providers={sess.get_providers()}")
    return sess, provider_used


# Lifecycle 
@app.on_event("startup")
def startup():
    global session, active_provider
    session, active_provider = load_cuda_session(MODEL_PATH)
    logger.info(
        f"Serving ready | option={SERVING_OPTION} | "
        f"version={MODEL_VERSION} | provider={active_provider} | "
        f"bypass_gate={BYPASS_GATE}"
    )


# Routes 
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "serving_option": SERVING_OPTION,
        "model_version": MODEL_VERSION,
        "execution_provider": active_provider,
        "session_ready": session is not None,
        "available_providers": ort.get_available_providers(),
    }


@app.get("/readyz")
def readyz():
    if session is None:
        raise HTTPException(status_code=503, detail="Session not ready")
    return {"status": "ready", "provider": active_provider}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t_start = time.perf_counter()
    request_id = req.request_id or str(uuid.uuid4())

    if session is None:
        raise HTTPException(status_code=503, detail="Session not ready")
    if not req.image:
        raise HTTPException(status_code=400, detail="image field is required")

    try:
        img_bytes = base64.b64decode(req.image)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    if not BYPASS_GATE and (image.width < 32 or image.height < 32):
        raise HTTPException(status_code=422, detail="Image too small.")

    x = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    outputs = session.run(["output"], {input_name: x})
    logits = outputs[0]
    probs = softmax(logits)[0]
    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])

    is_food = BYPASS_GATE or (class_idx in FOOD_CLASS_MAP)
    if not is_food:
        raise HTTPException(status_code=422, detail="No recognizable food detected.")

    confidence_flag = confidence < CONFIDENCE_THR
    category, draft_recipe = assemble_draft(class_idx, confidence, BYPASS_GATE)
    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return PredictResponse(
        request_id=request_id,
        status="success",
        confidence=round(confidence, 4),
        confidence_flag=confidence_flag,
        model_version=MODEL_VERSION,
        serving_option=SERVING_OPTION,
        execution_provider=active_provider,
        draft_recipe=draft_recipe,
        safeguarding={
            "is_food": is_food,
            "is_nsfw": False,
            "confidence_flag": confidence_flag,
            "confidence_threshold": CONFIDENCE_THR,
            "requires_user_confirmation": True,
            "disclaimer": (
                "AI-generated draft. Review before saving. "
                "Verify allergens independently."
            ),
        },
        latency_ms=latency_ms,
    )
