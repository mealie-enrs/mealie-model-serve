"""Wait for S3-compatible storage and ensure the artifact bucket (container) exists."""

from __future__ import annotations

import os
from typing import Any
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


def _bucket_name() -> str:
    raw = os.environ.get("MLFLOW_ARTIFACTS_DESTINATION", "s3://mlflow-artifacts")
    if raw.startswith("s3://"):
        raw = raw[5:]
    return raw.split("/", 1)[0] or "mlflow-artifacts"


def _use_minio_healthcheck(endpoint: str) -> bool:
    mode = os.environ.get("ARTIFACT_STORE_WAIT_MODE", "").strip().lower()
    if mode == "minio":
        return True
    if mode == "s3":
        return False
    el = endpoint.lower()
    # Chameleon / RGW S3 — never use MinIO's /minio/health/live endpoint.
    if ":7480" in el or "chameleoncloud.org" in el:
        return False
    host = (urlparse(endpoint).hostname or "").lower()
    return host in ("minio", "127.0.0.1", "localhost")


def _wait_minio_ready(endpoint: str, deadline: float) -> None:
    health = f"{endpoint.rstrip('/')}/minio/health/live"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=3) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    print("ensure_minio_bucket: MinIO not ready", file=sys.stderr)
    sys.exit(1)


def _wait_s3_endpoint_and_bucket(client: Any, bucket: str, deadline: float) -> None:
    """Wait until the S3 API answers; allow missing bucket (we create it next)."""
    while time.monotonic() < deadline:
        try:
            client.head_bucket(Bucket=bucket)
            return
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchBucket", "NotFound"):
                return
        except OSError:
            pass
        time.sleep(1)
    print("ensure_minio_bucket: S3 endpoint not reachable", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000").rstrip("/")
    bucket = os.environ.get("MINIO_ARTIFACT_BUCKET") or _bucket_name()
    access = os.environ["AWS_ACCESS_KEY_ID"]
    secret = os.environ["AWS_SECRET_ACCESS_KEY"]
    deadline = time.monotonic() + 120.0

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    if _use_minio_healthcheck(endpoint):
        _wait_minio_ready(endpoint, deadline)
    else:
        _wait_s3_endpoint_and_bucket(client, bucket, deadline)
    try:
        client.create_bucket(Bucket=bucket)
        print(f"ensure_minio_bucket: created {bucket!r}")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"ensure_minio_bucket: bucket {bucket!r} exists")
        else:
            raise


if __name__ == "__main__":
    main()
