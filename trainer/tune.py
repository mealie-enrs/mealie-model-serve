#!/usr/bin/env python3
"""Optuna-driven hyperparameter tuning with MLflow tracking."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import optuna
from sklearn.metrics import accuracy_score, f1_score

from trainer.utils.config import load_json_config
from trainer.utils.data import load_dataset
from trainer.utils.env import collect_environment_info, flatten_dict
from trainer.utils.modeling import build_estimator, suggest_candidate_and_params


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON tuning config path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_json_config(Path(args.config))
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    experiment_name = config["mlflow"]["experiment_name"]
    dataset = load_dataset(config["dataset"])
    env_info = collect_environment_info()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    direction = config["search"].get("direction", "maximize")
    metric_name = config["search"].get("metric_name", "val_macro_f1")
    study = optuna.create_study(direction=direction)

    with mlflow.start_run(run_name=config["mlflow"].get("run_name", "optuna-search")):
        for key, value in flatten_dict(config, prefix="search_config").items():
            mlflow.log_param(key, value)
        for key, value in flatten_dict(env_info, prefix="env").items():
            mlflow.log_param(key, value)

        def objective(trial: optuna.Trial) -> float:
            candidate_name, params = suggest_candidate_and_params(trial, config["search"])
            estimator = build_estimator(candidate_name=candidate_name, params=params)

            with mlflow.start_run(
                run_name=f"trial-{trial.number}-{candidate_name}",
                nested=True,
            ):
                mlflow.log_param("candidate_name", candidate_name)
                for key, value in params.items():
                    mlflow.log_param(f"candidate_param.{key}", value)
                start_time = time.perf_counter()
                estimator.fit(dataset.X_train, dataset.y_train)
                fit_time_sec = time.perf_counter() - start_time
                y_pred = estimator.predict(dataset.X_val)
                score = float(f1_score(dataset.y_val, y_pred, average="macro"))
                acc = float(accuracy_score(dataset.y_val, y_pred))
                mlflow.log_metrics(
                    {
                        "val_macro_f1": score,
                        "val_accuracy": acc,
                        "fit_time_sec": fit_time_sec,
                    }
                )
                with tempfile.TemporaryDirectory() as tmp_dir:
                    trial_path = Path(tmp_dir) / "trial.json"
                    trial_path.write_text(
                        json.dumps(
                            {
                                "trial_number": trial.number,
                                "candidate_name": candidate_name,
                                "params": params,
                                "score": score,
                                "accuracy": acc,
                            },
                            indent=2,
                        )
                    )
                    mlflow.log_artifact(str(trial_path), artifact_path="trials")
                trial.set_user_attr("candidate_name", candidate_name)
                trial.set_user_attr("params", params)
                return score if metric_name == "val_macro_f1" else acc

        study.optimize(objective, n_trials=int(config["search"]["n_trials"]))
        best = study.best_trial
        mlflow.log_metric("best_score", float(best.value))
        mlflow.log_param("best_candidate_name", best.user_attrs["candidate_name"])
        for key, value in best.user_attrs["params"].items():
            mlflow.log_param(f"best_param.{key}", value)
        print(
            json.dumps(
                {
                    "best_trial": best.number,
                    "best_value": best.value,
                    "best_candidate_name": best.user_attrs["candidate_name"],
                    "best_params": best.user_attrs["params"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
