"""
serving-rahil/app_onnx_graph.py

Serving Configuration: onnx_graph_opt
Hardware: CPU
Optimization: Model-level — ONNX export + ORT graph optimization (ORT_ENABLE_EXTENDED)
Framework: FastAPI + ONNXRuntime CPUExecutionProvider

Graph optimization fuses operations (Conv+BatchNorm, Conv+ReLU, etc.),
eliminates no-ops (Dropout, Identity nodes), and applies constant folding.
This reduces memory round-trips and kernel launch overhead vs baseline.

Lab reference: serve-model-chi sections 2-3
  (ONNX export + ORT_ENABLE_EXTENDED graph optimization on CPU)

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
MODEL_VERSION   = os.getenv("MODEL_VERSION", "efficientnet-b0-onnx-graph-opt-v1")
SERVING_OPTION  = os.getenv("SERVING_OPTION_ID", "onnx_graph_opt")
BYPASS_GATE     = os.getenv("BYPASS_INFERENCE_GATE", "0") == "1"
CONFIDENCE_THR  = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# App 
app = FastAPI(
    title="Mealie ML Serving — ONNX Graph Optimized",
    description="EfficientNet-B0, CPU, ONNX + ORT_ENABLE_EXTENDED graph optimization.",
)

session: Optional[ort.InferenceSession] = None

# Preprocessing (numpy — no PyTorch dependency at runtime)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, center crop, normalize to ImageNet stats. Returns [1,3,224,224]."""
    image = image.resize((256, 256), Image.BILINEAR)
    left = (256 - 224) // 2
    image = image.crop((left, left, left + 224, left + 224))
    arr = np.array(image, dtype=np.float32) / 255.0   # [H, W, 3]
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)                       # [3, H, W]
    return np.expand_dims(arr, axis=0)                 # [1, 3, H, W]


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


def load_onnx_session(model_path: str) -> ort.InferenceSession:
    """
    Load ONNX model with ORT_ENABLE_EXTENDED graph optimization.

    ORT_ENABLE_EXTENDED applies:
      - Constant folding
      - Identity/no-op elimination
      - Conv+BatchNorm fusion
      - Conv+Add/Mul fusion
      - Slice/Reshape elimination
      - Dropout elimination (inference graphs)

    This is a model-level optimization — same model artifact,
    faster execution due to reduced kernel launches and memory traffic.
    """
    logger.info(f"Loading ONNX session: {model_path}")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    opts.intra_op_num_threads = 0   # ORT default (auto)
    opts.inter_op_num_threads = 0   # ORT default (auto)

    sess = ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    logger.info(f"ONNX session ready | providers={sess.get_providers()}")
    return sess


# Lifecycle 
@app.on_event("startup")
def startup():
    global session
    session = load_onnx_session(MODEL_PATH)
    logger.info(
        f"Serving ready | option={SERVING_OPTION} | "
        f"version={MODEL_VERSION} | bypass_gate={BYPASS_GATE}"
    )


# Routes 
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "serving_option": SERVING_OPTION,
        "model_version": MODEL_VERSION,
        "session_ready": session is not None,
        "providers": session.get_providers() if session else [],
    }


@app.get("/readyz")
def readyz():
    if session is None:
        raise HTTPException(status_code=503, detail="ONNX session not ready")
    return {"status": "ready"}


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

    # Preprocess → numpy (no PyTorch at inference time)
    x = preprocess_image(image)   # [1, 3, 224, 224] float32

    # ONNX inference with graph-optimized session
    input_name = session.get_inputs()[0].name
    outputs = session.run(["output"], {input_name: x})
    logits = outputs[0]                     # [1, 1000]
    probs = softmax(logits)[0]              # [1000]
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
        execution_provider="CPUExecutionProvider",
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
