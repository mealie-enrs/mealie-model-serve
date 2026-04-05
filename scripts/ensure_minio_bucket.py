"""Wait for MinIO and create artifact bucket (used before MLflow server starts)."""

from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request

import boto3
from botocore.exceptions import ClientError


def _bucket() -> str:
    raw = os.environ.get("MLFLOW_ARTIFACTS_DESTINATION", "s3://mlflow-artifacts")
    if raw.startswith("s3://"):
        raw = raw[5:]
    return raw.split("/", 1)[0] or "mlflow-artifacts"


def main() -> None:
    endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000").rstrip("/")
    bucket = os.environ.get("MINIO_ARTIFACT_BUCKET") or _bucket()
    access = os.environ["AWS_ACCESS_KEY_ID"]
    secret = os.environ["AWS_SECRET_ACCESS_KEY"]

    health = f"{endpoint}/minio/health/live"
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=3) as resp:  # noqa: S310
                if resp.status == 200:
                    break
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    else:
        print("ensure_minio_bucket: MinIO not ready", file=sys.stderr)
        sys.exit(1)

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
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
