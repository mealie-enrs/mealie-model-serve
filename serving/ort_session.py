"""Build ONNX Runtime SessionOptions from environment (model-level optimizations)."""

from __future__ import annotations

import onnxruntime as ort

from serving.config import Settings


def session_options_from_settings(s: Settings) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    level = s.ort_graph_optimization_level.lower().strip()
    mapping = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    opts.graph_optimization_level = mapping.get(level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

    if s.ort_intra_op_num_threads > 0:
        opts.intra_op_num_threads = s.ort_intra_op_num_threads
    if s.ort_inter_op_num_threads > 0:
        opts.inter_op_num_threads = s.ort_inter_op_num_threads

    mode = s.ort_execution_mode.lower().strip()
    if mode == "sequential":
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    else:
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL

    return opts
