#!/usr/bin/env python3
"""Optional: export an sklearn model pickle to ONNX on disk (for custom training pipelines)."""

from __future__ import annotations

import argparse
import pickle

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pickle-path", required=True)
    p.add_argument("--output", required=True, help="Path to write model.onnx")
    p.add_argument("--n-features", type=int, required=True)
    args = p.parse_args()

    with open(args.pickle_path, "rb") as f:
        clf = pickle.load(f)

    onnx_model = convert_sklearn(
        clf,
        initial_types=[("float_input", FloatTensorType([None, args.n_features]))],
        options={id(clf): {"zipmap": False}},
    )
    with open(args.output, "wb") as out:
        out.write(onnx_model.SerializeToString())
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
