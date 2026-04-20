from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split


@dataclass
class ClassificationDataset:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray


def _load_builtin(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "iris":
        return load_iris(return_X_y=True)
    if name == "wine":
        return load_wine(return_X_y=True)
    raise ValueError(f"Unsupported builtin dataset: {name}")


def load_dataset(config: dict[str, Any]) -> ClassificationDataset:
    source = config["source"]
    if source == "sklearn_builtin":
        X, y = _load_builtin(config["name"])
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=float(config.get("test_size", 0.2)),
            random_state=int(config.get("random_state", 42)),
            stratify=y,
        )
        return ClassificationDataset(
            X_train=np.asarray(X_train, dtype=np.float32),
            X_val=np.asarray(X_val, dtype=np.float32),
            y_train=np.asarray(y_train),
            y_val=np.asarray(y_val),
        )

    if source == "npz_classification":
        payload = np.load(Path(config["path"]))
        return ClassificationDataset(
            X_train=np.asarray(payload["X_train"], dtype=np.float32),
            X_val=np.asarray(payload["X_val"], dtype=np.float32),
            y_train=np.asarray(payload["y_train"]),
            y_val=np.asarray(payload["y_val"]),
        )

    raise ValueError(f"Unsupported dataset source: {source}")
