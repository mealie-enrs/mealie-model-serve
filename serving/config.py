import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        validation_alias=AliasChoices("MLFLOW_TRACKING_URI"),
    )
    mlflow_registry_uri: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MLFLOW_REGISTRY_URI"),
    )

    mlflow_s3_endpoint_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MLFLOW_S3_ENDPOINT_URL"),
    )
    aws_access_key_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AWS_ACCESS_KEY_ID"),
    )
    aws_secret_access_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AWS_SECRET_ACCESS_KEY"),
    )
    aws_default_region: str = Field(
        default="us-east-1",
        validation_alias=AliasChoices("AWS_DEFAULT_REGION"),
    )

    model_uri: str = Field(
        default="models:/food-classifier@champion",
        validation_alias=AliasChoices("MODEL_URI"),
    )
    model_cache_dir: str = Field(
        default="/var/lib/model-cache",
        validation_alias=AliasChoices("MODEL_CACHE_DIR"),
    )
    model_provider: str = Field(
        default="cpu",
        validation_alias=AliasChoices("MODEL_PROVIDER"),
    )
    service_name: str = Field(
        default="model-serve",
        validation_alias=AliasChoices("SERVICE_NAME"),
    )
    build_sha: str = Field(
        default_factory=lambda: os.environ.get("BUILD_SHA", "unknown"),
    )

    # Serving rubric: label this deployment in /metadata (e.g. baseline_http, onnx_ort_all, multi_worker)
    serving_option_id: str = Field(
        default="baseline_http",
        validation_alias=AliasChoices("SERVING_OPTION_ID"),
    )

    # Model-level: ONNX Runtime session options
    ort_graph_optimization_level: str = Field(
        default="all",
        validation_alias=AliasChoices("ORT_GRAPH_OPTIMIZATION_LEVEL"),
        description="disable | basic | extended | all",
    )
    ort_intra_op_num_threads: int = Field(
        default=0,
        validation_alias=AliasChoices("ORT_INTRA_OP_NUM_THREADS"),
        description="0 = ORT default",
    )
    ort_inter_op_num_threads: int = Field(
        default=0,
        validation_alias=AliasChoices("ORT_INTER_OP_NUM_THREADS"),
        description="0 = ORT default",
    )
    ort_execution_mode: str = Field(
        default="parallel",
        validation_alias=AliasChoices("ORT_EXECUTION_MODE"),
        description="parallel | sequential",
    )


settings = Settings()
