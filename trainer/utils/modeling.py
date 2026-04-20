from __future__ import annotations

from typing import Any

import optuna
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_estimator(candidate_name: str, params: dict[str, Any]) -> Any:
    if candidate_name == "logistic_regression":
        defaults = {
            "max_iter": 1000,
            "solver": "lbfgs",
            "multi_class": "auto",
        }
        defaults.update(params)
        return LogisticRegression(**defaults)
    if candidate_name == "random_forest":
        defaults = {"n_estimators": 200, "random_state": 42}
        defaults.update(params)
        return RandomForestClassifier(**defaults)
    if candidate_name == "extra_trees":
        defaults = {"n_estimators": 300, "random_state": 42}
        defaults.update(params)
        return ExtraTreesClassifier(**defaults)
    raise ValueError(f"Unsupported candidate_name: {candidate_name}")


def suggest_candidate_and_params(
    trial: optuna.Trial, search_config: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    candidate_name = trial.suggest_categorical(
        "candidate_name", search_config["candidates"]
    )
    if candidate_name == "logistic_regression":
        params = {
            "C": trial.suggest_float("logreg_C", 1e-3, 10.0, log=True),
            "max_iter": 1000,
        }
    elif candidate_name == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("rf_n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("rf_max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
            "random_state": 42,
        }
    elif candidate_name == "extra_trees":
        params = {
            "n_estimators": trial.suggest_int("et_n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("et_max_depth", 2, 24),
            "min_samples_split": trial.suggest_int("et_min_samples_split", 2, 10),
            "random_state": 42,
        }
    else:
        raise ValueError(f"Unsupported candidate_name: {candidate_name}")
    return candidate_name, params
