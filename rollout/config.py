from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RouterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    production_url: str = Field(
        default="http://mms-model-serve:8080",
        validation_alias=AliasChoices("ROUTER_PRODUCTION_URL"),
    )
    canary_url: str = Field(
        default="http://mms-model-serve-canary:8080",
        validation_alias=AliasChoices("ROUTER_CANARY_URL"),
    )
    default_canary_weight: float = Field(
        default=0.1,
        validation_alias=AliasChoices("ROUTER_DEFAULT_CANARY_WEIGHT"),
    )
    timeout_sec: float = Field(
        default=15.0,
        validation_alias=AliasChoices("ROUTER_TIMEOUT_SEC"),
    )
    feedback_dir: Path = Field(
        default=Path("/var/lib/mms-feedback"),
        validation_alias=AliasChoices("FEEDBACK_LOG_DIR"),
    )


router_settings = RouterSettings()
