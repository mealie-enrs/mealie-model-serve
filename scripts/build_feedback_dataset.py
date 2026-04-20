#!/usr/bin/env python3
"""Build an NPZ retraining dataset from rollout-router JSONL or curated feedback parquet."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_json_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [_normalize_text(item) for item in raw if _normalize_text(item)]
    text = _normalize_text(raw)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [_normalize_text(item) for item in parsed if _normalize_text(item)]
    return [_normalize_text(parsed)] if _normalize_text(parsed) else []


def _load_router_rows(path: Path) -> tuple[np.ndarray, np.ndarray, dict[int, str], str]:
    features: list[list[float]] = []
    labels: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        features.append([float(value) for value in row["inputs"]])
        labels.append(int(row["approved_output"]))
    unique_labels = sorted(set(labels))
    label_map = {label: f"class_{label}" for label in unique_labels}
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
        label_map,
        "approved_output",
    )


def _label_values_from_rows(rows: list[dict], label_field: str) -> tuple[list[str], str]:
    if label_field != "auto":
        return [_normalize_text(row.get(label_field)) for row in rows], label_field

    action_values = [_normalize_text(row.get("action")) for row in rows]
    action_classes = {value for value in action_values if value}
    if len(action_classes) > 1:
        return action_values, "action"

    slug_values = [_normalize_text(row.get("mealie_recipe_slug")) for row in rows]
    slug_classes = {value for value in slug_values if value}
    if len(slug_classes) > 1:
        return slug_values, "mealie_recipe_slug"

    title_values = [_normalize_text(row.get("recipe_title")) for row in rows]
    title_classes = {value for value in title_values if value}
    if len(title_classes) > 1:
        return title_values, "recipe_title"

    raise ValueError(
        "Could not infer a useful label field from the curated feedback rows. "
        "Pass --label-field explicitly or collect more than one approved recipe."
    )


def _load_curated_feedback_rows(
    path: Path,
    *,
    label_field: str,
) -> tuple[np.ndarray, np.ndarray, dict[int, str], str]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"No rows found in parquet input: {path}")

    label_values, resolved_label_field = _label_values_from_rows(rows, label_field)
    normalized_labels = []
    for row, label_value in zip(rows, label_values, strict=True):
        fallback = _normalize_text(row.get("draft_id")) or "unknown"
        normalized_labels.append(label_value or fallback)

    distinct_labels = sorted(set(normalized_labels))
    if len(distinct_labels) < 2:
        raise ValueError(
            "Need at least two distinct labels to train a classifier from curated feedback."
        )
    label_to_index = {label: idx for idx, label in enumerate(distinct_labels)}

    features: list[list[float]] = []
    labels: list[int] = []
    for row, label_value in zip(rows, normalized_labels, strict=True):
        title = _normalize_text(row.get("recipe_title"))
        ingredients = _parse_json_list(row.get("recipe_ingredients_json"))
        steps = _parse_json_list(row.get("recipe_steps_json"))
        edit_distance = float(row.get("edit_distance") or 0.0)

        ingredient_chars = sum(len(item) for item in ingredients)
        step_chars = sum(len(item) for item in steps)
        title_tokens = len([token for token in title.split() if token])

        features.append([
            float(len(title)),
            float(title_tokens),
            float(len(ingredients)),
            float(len(steps)),
            float(ingredient_chars),
            float(step_chars),
            float(ingredient_chars / max(len(ingredients), 1)),
            float(step_chars / max(len(steps), 1)),
            edit_distance,
        ])
        labels.append(label_to_index[label_value])

    inverse_label_map = {index: label for label, index in label_to_index.items()}
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
        inverse_label_map,
        resolved_label_field,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to feedback-YYYYMMDD.jsonl")
    parser.add_argument("--output", required=True, help="Destination NPZ path")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--label-field",
        default="auto",
        help=(
            "Label field for curated feedback parquet inputs. "
            "Use 'auto' to prefer action, then mealie_recipe_slug, then recipe_title."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        X, y, label_map, resolved_label_field = _load_router_rows(input_path)
    elif suffix == ".parquet":
        X, y, label_map, resolved_label_field = _load_curated_feedback_rows(
            input_path,
            label_field=args.label_field,
        )
    else:
        raise ValueError(f"Unsupported input format for {input_path}. Expected .jsonl or .parquet")

    unique_labels, counts = np.unique(y, return_counts=True)
    requested_val_rows = (
        math.ceil(X.shape[0] * args.test_size)
        if 0.0 < args.test_size < 1.0
        else int(args.test_size)
    )
    stratify_labels = y
    if (
        len(unique_labels) < 2
        or counts.min() < 2
        or requested_val_rows < len(unique_labels)
    ):
        stratify_labels = None

    train_rows = X.shape[0] - requested_val_rows
    if train_rows < len(unique_labels):
        X_train = X.copy()
        X_val = X.copy()
        y_train = y.copy()
        y_val = y.copy()
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify_labels,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
    label_map_path = output.with_suffix(".labels.json")
    label_map_path.write_text(
        json.dumps(
            {
                "input": args.input,
                "label_field": resolved_label_field,
                "labels": {str(index): label for index, label in label_map.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "input": args.input,
        "input_format": suffix.lstrip("."),
        "output": args.output,
        "train_rows": int(X_train.shape[0]),
        "val_rows": int(X_val.shape[0]),
        "feature_dim": int(X_train.shape[1]),
        "label_field": resolved_label_field,
        "label_map_path": str(label_map_path),
        "class_count": len(label_map),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
