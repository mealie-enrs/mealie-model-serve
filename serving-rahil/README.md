# Mealie ML Serving — Rahil Shaikh
**Role:** Serving | **Team:** ENRS | **Project:** proj26 | **Course:** ECE-GY 9183

## Overview

This directory contains the complete serving layer for the Mealie ML food-to-recipe feature. It defines four serving configurations evaluated on Chameleon Cloud (CHI@UC, nc33 node, Quadro RTX 6000).

## Serving Configurations

| Config | Port | Optimization | Hardware |
|---|---|---|---|
| `baseline_pytorch` | 8000 | None — PyTorch eager mode | CPU |
| `onnx_graph_opt` | 8001 | ONNX + ORT graph optimization | CPU |
| `onnx_quantized` | 8002 | ONNX + INT8 dynamic quantization | CPU |
| `onnx_cuda` ⭐ | 8003 | ONNX + CUDAExecutionProvider | GPU (RTX 6000) |

⭐ = Most promising for production (lowest latency)

## Design Decisions

**Why EfficientNet-B0?**
5.3M parameters, designed for fast inference. Hits our <3s latency target even on CPU baseline. Will be replaced by fine-tuned checkpoint when ready.

**Why ONNX Runtime?**
Single serving framework that supports CPU, GPU, and quantized execution via different execution providers. No separate Triton setup needed for this model size.

**Why dynamic quantization for Config 3?**
No calibration dataset required at quantization time. When Recipe1M image pipeline is complete, we will switch to static quantization with calibration data for better accuracy preservation (as done in the course lab).

**Why CUDAExecutionProvider for Config 4?**
RTX 6000 on nc33 has 24GB VRAM, far more than EfficientNet-B0 needs. CUDAExecutionProvider swaps all tensor ops to GPU kernels, eliminating CPU bottleneck entirely.

## File Structure

```
serving-rahil/
  recipe_assembler.py    # Shared: mock ANN retrieval → Mealie JSON draft
  app_baseline.py        # Config 1: PyTorch CPU baseline
  app_onnx_graph.py      # Config 2: ONNX + graph optimization
  app_onnx_quantized.py  # Config 3: ONNX + INT8 quantization
  app_onnx_cuda.py       # Config 4: ONNX + CUDAExecutionProvider
  convert_model.py       # PyTorch → ONNX → INT8 conversion script
  Dockerfile.baseline
  Dockerfile.onnx_graph
  Dockerfile.onnx_quantized
  Dockerfile.onnx_cuda
  docker-compose.yml     # All 4 configs, ports 8000-8003
  input_sample.json      # Agreed API input schema 
  output_sample.json     # Agreed API output schema 
```

## Running on Chameleon

```bash
# SSH into GPU instance
ssh -i ~/.ssh/id_rsa_chameleon cc@192.5.86.170

# Clone repo and checkout serving branch
git clone https://github.com/mealie-enrs/mealie-model-serve.git
cd mealie-model-serve
git checkout rahil/serving-benchmarks

# Build and start all 4 configs
docker compose -f serving-rahil/docker-compose.yml up --build -d

# Verify all healthy
docker compose -f serving-rahil/docker-compose.yml ps

# Run full benchmark suite
bash eval/run_all_benchmarks.sh

# Results saved to eval/results/
```

## API Contract

**Input:** `POST /predict`
```json
{
  "request_id": "req_abc123",
  "image": "<base64_encoded_jpeg>",
  "image_filename": "dish.jpg",
  "timestamp": "2026-04-06T00:00:00Z"
}
```

**Output:**
```json
{
  "request_id": "req_abc123",
  "status": "success",
  "confidence": 0.87,
  "confidence_flag": false,
  "model_version": "efficientnet-b0-imagenet-v1",
  "serving_option": "baseline_pytorch",
  "draft_recipe": { ... },
  "safeguarding": { "requires_user_confirmation": true, ... },
  "latency_ms": 45.2
}
```

## Retraining Integration

When Shubham's trained EfficientNet-B0 checkpoint is registered in MLflow:
1. Update `MODEL_PATH` env var in `docker-compose.yml`
2. Update `load_model()` in `app_baseline.py` to pull from MinIO
3. Re-run `convert_model.py` to generate new ONNX artifacts
4. Rebuild containers and re-run benchmarks

No code changes required — model swap is config-only.
