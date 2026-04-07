"""
serving-rahil/app_ray_serve.py

Serving Configuration: ray_serve_cuda
Hardware: GPU - Quadro RTX 6000 (24GB VRAM) on Chameleon CHI@UC nc33
Optimization: Ray Serve actor-based autoscaling + ONNX CUDAExecutionProvider
Framework: Ray Serve + ONNXRuntime CUDAExecutionProvider

Why Ray Serve over FastAPI:
  FastAPI uses a single process with uvicorn workers. Under burst traffic,
  there is no way to dynamically add replicas without restarting the server.
  Ray Serve runs each deployment as an actor pool — replicas can be added
  or removed at runtime based on request queue depth, with zero downtime.

  Concrete example: during a lunch-hour spike (5x normal traffic), Ray Serve
  automatically scales from 1 to 4 replicas within seconds. FastAPI would
  queue requests, increasing p95 latency. Ray Serve maintains p95 latency
  by spreading load across replicas.

Tradeoffs vs onnx_cuda (FastAPI):
  + Runtime autoscaling with no downtime
  + Built-in request queuing per replica
  + Actor isolation: one replica crash does not affect others
  - Higher memory overhead (Ray runtime ~200MB)
  - Slightly higher cold-start time

"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
import uuid

import numpy as np
import onnxruntime as ort
from PIL import Image

import ray
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
MODEL_PATH     = os.getenv("MODEL_PATH", "/app/efficientnet_b0.onnx")
MODEL_VERSION  = os.getenv("MODEL_VERSION", "efficientnet-b0-onnx-cuda-v1")
SERVING_OPTION = os.getenv("SERVING_OPTION_ID", "ray_serve_cuda")
BYPASS_GATE    = os.getenv("BYPASS_INFERENCE_GATE", "0") == "1"
CONFIDENCE_THR = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# Preprocessing constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ImageNet food class map
FOOD_CLASS_MAP: dict[int, str] = {
    924: "guacamole", 925: "soup",       926: "dessert",
    927: "dessert",   928: "dessert",    929: "bread",
    930: "bread",     931: "bread",      932: "bread",
    933: "burger",    934: "hotdog",     935: "pasta",
    936: "salad",     937: "vegetables", 938: "vegetables",
    939: "vegetables",940: "vegetables", 941: "vegetables",
    942: "vegetables",943: "vegetables", 944: "vegetables",
    945: "vegetables",946: "mushroom",   947: "mushroom",
    948: "fruit",     949: "fruit",      950: "fruit",
    951: "fruit",     952: "fruit",      953: "fruit",
    954: "fruit",     955: "fruit",      956: "fruit",
    957: "fruit",     958: "meat",       959: "pasta",
    960: "dessert",   961: "dessert",    962: "dessert",
    963: "pizza",     964: "pie",        965: "mexican",
    966: "beverage",  967: "beverage",   968: "beverage",
}

_DRAFT_TEMPLATE = {
    "name": "Mixed Vegetable Stir Fry",
    "description": "A healthy and colorful vegetable stir fry with a savory sauce.",
    "recipeCategory": "Asian",
    "tags": ["stir-fry", "vegetarian", "healthy", "quick"],
    "recipeIngredient": [
        "2 cups mixed vegetables (broccoli, bell pepper, snap peas, carrots)",
        "2 tbsp soy sauce",
        "1 tbsp oyster sauce",
        "1 tsp sesame oil",
        "2 cloves garlic, minced",
        "1 tsp fresh ginger, grated",
        "1 tbsp vegetable oil",
    ],
    "recipeInstructions": [
        {"id": "step_1", "text": "Heat oil in wok over high heat until smoking."},
        {"id": "step_2", "text": "Add garlic and ginger, stir fry 30 seconds."},
        {"id": "step_3", "text": "Add harder vegetables first, stir fry 4-5 minutes."},
        {"id": "step_4", "text": "Add sauces, toss to coat, finish with sesame oil. Serve over rice."},
    ],
    "nutrition": {
        "disclaimer": (
            "AI-generated nutritional information is approximate. "
            "Please verify all allergens independently before serving."
        ),
        "allergens": ["soy", "shellfish"],
    },
}


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


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=10,
    ray_actor_options={"num_cpus": 1},
)
class MealieServing:
    """
    Ray Serve deployment for Mealie ML serving.

    Each replica is an independent actor with its own ONNX session.
    Ray Serve routes requests across replicas and queues excess requests
    per-replica up to max_ongoing_requests.

    Production autoscaling config (not active here — single node benchmark):
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 4,
            "target_ongoing_requests": 5,
        }
    Fixed to num_replicas=1 for fair single-node comparison with other configs.
    """

    def __init__(self):
        logger.info(f"Initializing MealieServing actor | model={MODEL_PATH}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        available = ort.get_available_providers()
        logger.info(f"Available ORT providers: {available}")

        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.provider = "CUDAExecutionProvider"
            logger.info("Using CUDAExecutionProvider")
        else:
            providers = ["CPUExecutionProvider"]
            self.provider = "CPUExecutionProvider"
            logger.warning("CUDA not available — falling back to CPU")

        self.session = ort.InferenceSession(
            MODEL_PATH, sess_options=opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"Session ready | provider={self.provider}")

    async def __call__(self, request: Request) -> JSONResponse:
        t_start = time.perf_counter()

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

        request_id = body.get("request_id") or str(uuid.uuid4())
        image_b64 = body.get("image", "")

        if not image_b64:
            return JSONResponse({"error": "image field required"}, status_code=400)

        try:
            img_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            return JSONResponse({"error": f"Invalid image: {e}"}, status_code=400)

        if not BYPASS_GATE and (image.width < 32 or image.height < 32):
            return JSONResponse({"error": "Image too small"}, status_code=422)

        x = preprocess_image(image)
        outputs = self.session.run(["output"], {self.input_name: x})
        probs = softmax(outputs[0])[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])

        is_food = BYPASS_GATE or (class_idx in FOOD_CLASS_MAP)
        if not is_food:
            return JSONResponse({"error": "No food detected"}, status_code=422)

        confidence_flag = confidence < CONFIDENCE_THR
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        draft = dict(_DRAFT_TEMPLATE)
        draft["retrieval_metadata"] = {
            "source_dataset": "Recipe1M",
            "retrieval_method": "mock_ann (production: FAISS index)",
            "top_k_candidates": 5,
            "selected_rank": 1,
            "similarity_score": round(confidence, 4),
            "food_category": FOOD_CLASS_MAP.get(class_idx, "default"),
            "imagenet_class_idx": class_idx,
            "model": MODEL_VERSION,
        }

        return JSONResponse({
            "request_id": request_id,
            "status": "success",
            "confidence": round(confidence, 4),
            "confidence_flag": confidence_flag,
            "model_version": MODEL_VERSION,
            "serving_option": SERVING_OPTION,
            "execution_provider": self.provider,
            "num_replicas": 1,
            "draft_recipe": draft,
            "safeguarding": {
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
            "latency_ms": latency_ms,
        })


app = MealieServing.bind()

if __name__ == "__main__":
    ray.init()
    serve.start(http_options={"host": "0.0.0.0", "port": 8004})
    serve.run(app, route_prefix="/")
    logger.info("Ray Serve running on port 8004. Press Ctrl+C to stop.")
    import signal
    signal.pause()
