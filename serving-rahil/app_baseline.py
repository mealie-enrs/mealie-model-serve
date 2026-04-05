"""
serving-rahil/app_baseline.py

Serving Configuration: baseline_pytorch
Hardware: CPU
Optimization: None — raw PyTorch EfficientNet-B0 in eager mode
Framework: FastAPI + PyTorch

This is the simplest reference configuration.
No graph compilation, no ONNX export, no quantization.
Establishes the latency/throughput baseline that all other
configurations are measured against.

"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
import uuid

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from typing import Optional

from recipe_assembler import FOOD_CLASS_MAP, assemble_draft

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
MODEL_VERSION   = os.getenv("MODEL_VERSION", "efficientnet-b0-imagenet-v1")
SERVING_OPTION  = os.getenv("SERVING_OPTION_ID", "baseline_pytorch")
BYPASS_GATE     = os.getenv("BYPASS_INFERENCE_GATE", "0") == "1"
CONFIDENCE_THR  = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# App 
app = FastAPI(
    title="Mealie ML Serving — Baseline PyTorch",
    description="EfficientNet-B0, CPU, no optimization. Reference baseline.",
)

# Global model state
model: Optional[torch.nn.Module] = None

# ImageNet preprocessing (standard)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Schemas 
class PredictRequest(BaseModel):
    request_id: Optional[str] = None
    image: str                          # base64-encoded JPEG/PNG
    image_filename: Optional[str] = "upload.jpg"
    timestamp: Optional[str] = None

class PredictResponse(BaseModel):
    request_id: str
    status: str
    confidence: float
    confidence_flag: bool
    model_version: str
    serving_option: str
    draft_recipe: dict
    safeguarding: dict
    latency_ms: float


def load_model() -> torch.nn.Module:
    """
    Load EfficientNet-B0 with ImageNet pretrained weights.

    Production: download checkpoint from MinIO object store.
        MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
        MODEL_BUCKET   = os.getenv("MODEL_BUCKET", "proj26-models")
        MODEL_KEY      = os.getenv("MODEL_KEY", "efficientnet_b0_v1.pt")

    Current: load ImageNet pretrained weights from torchvision (mocked MinIO).
    """
    logger.info("Loading EfficientNet-B0 [ImageNet pretrained — mocked MinIO load]")
    m = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    m.eval()
    logger.info("Model loaded successfully")
    return m


# Lifecycle
@app.on_event("startup")
def startup():
    global model
    model = load_model()
    logger.info(
        f"Serving ready | option={SERVING_OPTION} | "
        f"version={MODEL_VERSION} | bypass_gate={BYPASS_GATE}"
    )


#  Routes 
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "serving_option": SERVING_OPTION,
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None,
    }


@app.get("/readyz")
def readyz():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t_start = time.perf_counter()
    request_id = req.request_id or str(uuid.uuid4())

    # Guard: model must be loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.image:
        raise HTTPException(status_code=400, detail="image field is required")

    # Decode image
    try:
        img_bytes = base64.b64decode(req.image)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {e}")

    # Inference gate: reject tiny/ambiguous images
    if not BYPASS_GATE and (image.width < 32 or image.height < 32):
        raise HTTPException(
            status_code=422,
            detail="Image too small or ambiguous. Please upload a clear food photo.",
        )

    # Preprocess
    x = preprocess(image).unsqueeze(0)  # [1, 3, 224, 224]

    # Inference — PyTorch eager mode (no compilation, no ONNX)
    with torch.inference_mode():
        logits = model(x)                          # [1, 1000]
        probs = torch.softmax(logits, dim=1)[0]    # [1000]
        class_idx = int(torch.argmax(probs).item())
        confidence = float(probs[class_idx].item())

    # Inference gate: food class check
    is_food = BYPASS_GATE or (class_idx in FOOD_CLASS_MAP)
    if not is_food:
        raise HTTPException(
            status_code=422,
            detail=(
                "No recognizable food detected in image. "
                "Please upload a clear photo of a dish."
            ),
        )

    # Confidence flagging (Unit 10 safeguard — threshold 0.6)
    confidence_flag = confidence < CONFIDENCE_THR

    # Assemble Mealie draft recipe
    category, draft_recipe = assemble_draft(class_idx, confidence, BYPASS_GATE)

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return PredictResponse(
        request_id=request_id,
        status="success",
        confidence=round(confidence, 4),
        confidence_flag=confidence_flag,
        model_version=MODEL_VERSION,
        serving_option=SERVING_OPTION,
        draft_recipe=draft_recipe,
        safeguarding={
            "is_food": is_food,
            "is_nsfw": False,
            "confidence_flag": confidence_flag,
            "confidence_threshold": CONFIDENCE_THR,
            "requires_user_confirmation": True,
            "disclaimer": (
                "This is an AI-generated draft recipe. "
                "Review all ingredients and steps carefully before saving. "
                "Allergen information may be incomplete — verify independently."
            ),
        },
        latency_ms=latency_ms,
    )
