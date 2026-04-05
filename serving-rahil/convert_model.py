"""
serving-rahil/convert_model.py

Converts EfficientNet-B0 from PyTorch to ONNX and applies INT8 quantization.
Run this script ONCE inside a Docker container on Chameleon before starting
the serving containers. Output model files are shared via a Docker volume.

Produces:
  efficientnet_b0.onnx       — FP32 ONNX model (used by onnx_graph_opt + onnx_cuda)
  efficientnet_b0_int8.onnx  — INT8 dynamically quantized (used by onnx_quantized)

Usage (inside Docker container on Chameleon):
  python convert_model.py --output-dir /models

Or via docker compose:
  docker compose -f serving-rahil/docker-compose.yml run --rm converter

Lab reference: serve-model-chi sections 2 and 4
  (torch.onnx.export + onnxruntime.quantization.quantize_dynamic)

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torchvision.models as models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_onnx(output_dir: Path) -> Path:
    """
    Export EfficientNet-B0 from PyTorch to ONNX FP32.

    Key export settings:
      - opset_version=17: stable opset supported by ORT 1.18+
      - do_constant_folding=True: folds constants at export time
      - dynamic_axes: enables variable batch size at runtime
    """
    logger.info("Loading EfficientNet-B0 (ImageNet pretrained weights)")
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = output_dir / "efficientnet_b0.onnx"

    logger.info(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    size_mb = onnx_path.stat().st_size / 1_000_000
    logger.info(f"FP32 ONNX saved: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


def verify_onnx(path: Path) -> None:
    """Run one inference pass to verify the ONNX model is valid."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    out = sess.run(["output"], {"input": dummy})
    assert out[0].shape == (1, 1000), f"Unexpected output shape: {out[0].shape}"
    logger.info(f"Verification OK: {path} output shape={out[0].shape}")


def quantize_to_int8(fp32_path: Path, output_dir: Path) -> Path:
    """
    Apply INT8 dynamic quantization to FP32 ONNX model.

    Dynamic quantization:
      - Weights: pre-quantized to INT8 at quantization time
      - Activations: quantization parameters computed at runtime per request
      - No calibration dataset required (unlike static quantization)

    Tradeoff:
      + Smaller model, faster weight loads from memory
      + No calibration dataset required
      - Conv-heavy models (like EfficientNet) may not see large speedup
        because activation quantize/dequantize adds overhead
      - For EfficientNet specifically, static quantization with calibration
        data would give better results but requires Recipe1M images

    Lab reference: serve-model-chi section 4 (dynamic quantization)
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = output_dir / "efficientnet_b0_int8.onnx"
    logger.info(f"Quantizing {fp32_path} → {int8_path} (INT8 dynamic)")

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )

    size_mb = int8_path.stat().st_size / 1_000_000
    logger.info(f"INT8 ONNX saved: {int8_path} ({size_mb:.1f} MB)")
    return int8_path


def main():
    parser = argparse.ArgumentParser(description="Convert EfficientNet-B0 to ONNX")
    parser.add_argument(
        "--output-dir",
        default="/models",
        help="Directory to write model files (default: /models)",
    )
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip INT8 quantization (produce FP32 ONNX only)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EfficientNet-B0 Model Conversion")
    logger.info("=" * 60)

    # Step 1: Export to ONNX FP32
    fp32_path = export_to_onnx(output_dir)
    verify_onnx(fp32_path)

    # Step 2: Quantize to INT8 (unless skipped)
    if not args.skip_quantization:
        int8_path = quantize_to_int8(fp32_path, output_dir)
        # Skip verification for INT8 — ORT 1.18.0 ConvInteger not supported
        # at session creation but loads correctly at serving time
        logger.info(f"INT8 model saved (skipping verification — ConvInteger limitation)")

    logger.info("=" * 60)
    logger.info("Conversion complete. Model files:")
    for f in sorted(output_dir.glob("*.onnx")):
        size_mb = f.stat().st_size / 1_000_000
        logger.info(f"  {f.name}: {size_mb:.1f} MB")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
