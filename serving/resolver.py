"""Parse models:/name/version or models:/name@alias and resolve via MLflow Registry."""

from __future__ import annotations

import re
from dataclasses import dataclass

from mlflow.tracking import MlflowClient

# Allowed registry aliases (spec §7)
ALIASES = frozenset({"staging", "canary", "champion", "production"})

MODEL_URI_RE = re.compile(r"^models:/([^/@]+)(?:/(\d+)|@([\w-]+))$")


class ModelUriError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedModel:
    name: str
    version: str
    alias: str | None
    """Artifact URI suitable for mlflow.artifacts.download_artifacts (models:/name/version)."""
    download_uri: str


def parse_model_uri(uri: str) -> tuple[str, str | None, str | None]:
    m = MODEL_URI_RE.match(uri.strip())
    if not m:
        raise ModelUriError(
            f"Invalid MODEL_URI {uri!r}; expected models:/<name>/<version> or models:/<name>@<alias>"
        )
    name, version, alias = m.group(1), m.group(2), m.group(3)
    if alias and alias not in ALIASES:
        raise ModelUriError(f"Alias {alias!r} not in allowed set {sorted(ALIASES)}")
    return name, version, alias


def resolve(client: MlflowClient, uri: str) -> ResolvedModel:
    name, version, alias = parse_model_uri(uri)
    if alias:
        mv = client.get_model_version_by_alias(name, alias)
        ver_str = str(mv.version)
        return ResolvedModel(
            name=name,
            version=ver_str,
            alias=alias,
            download_uri=f"models:/{name}/{ver_str}",
        )
    assert version is not None
    client.get_model_version(name, version)
    return ResolvedModel(
        name=name,
        version=str(version),
        alias=None,
        download_uri=f"models:/{name}/{version}",
    )
