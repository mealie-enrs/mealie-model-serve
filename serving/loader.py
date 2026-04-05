"""ONNX Runtime session from a directory containing a .onnx file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from serving.config import Settings
from serving.ort_session import session_options_from_settings


def select_providers(preferred: str) -> list[str]:
    p = preferred.lower().strip()
    if p in ("cuda", "gpu"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def find_onnx_file(root: Path) -> Path:
    direct = root / "model.onnx"
    if direct.is_file():
        return direct
    onnx_files = list(root.rglob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx file under {root}")
    return sorted(onnx_files, key=lambda p: len(str(p)))[0]


class OnnxModel:
    def __init__(
        self,
        onnx_path: Path,
        provider_pref: str,
        settings: Settings | None = None,
    ) -> None:
        providers = select_providers(provider_pref)
        sess_opts = session_options_from_settings(settings) if settings else ort.SessionOptions()
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.providers_in_use = self.session.get_providers()

    def predict(self, inputs: list[list[float]]) -> np.ndarray:
        x = np.asarray(inputs, dtype=np.float32)
        out = self.session.run(self.output_names, {self.input_name: x})
        # Return first output (e.g. probabilities or class scores)
        return out[0]
