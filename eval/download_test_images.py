"""
eval/download_test_images.py

Downloads a sample of food images from the Chameleon Swift object store
(proj26-training-data) to use as representative benchmark inputs.

Images source: proj26-training-data → recipe1m → kaggle-food-images → Food Images

PLACEHOLDER: Swift credentials must be set via environment variables
before running on Chameleon. See comments below.

Usage (on Chameleon GPU instance, after sourcing openrc):
    source ~/openrc
    python eval/download_test_images.py --output-dir /test-images --n 20

Or inside Docker:
    docker run --rm \
        -e OS_STORAGE_URL=$OS_STORAGE_URL \
        -e OS_AUTH_TOKEN=$OS_AUTH_TOKEN \
        -v /test-images:/test-images \
        mealie-serving-baseline \
        python download_test_images.py --output-dir /test-images --n 20

"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Swift object store config 
# PLACEHOLDER: These are read from environment variables at runtime.
# On Chameleon, source ~/openrc first, then run this script.
# DO NOT hardcode credentials here.

SWIFT_STORAGE_URL = os.getenv(
    "OS_STORAGE_URL",
    # PLACEHOLDER: replace with actual URL from ~/openrc on Chameleon
    "https://chi.uc.chameleoncloud.org:7480/swift/v1/AUTH_<project_id>",
)
SWIFT_AUTH_TOKEN = os.getenv(
    "OS_AUTH_TOKEN",
    # PLACEHOLDER: replace by sourcing ~/openrc on Chameleon
    "",
)

CONTAINER = "proj26-training-data"
IMAGE_PREFIX = "recipe1m/kaggle-food-images/Food Images/"


def list_images(n: int) -> list[str]:
    """List first N image objects in the Swift container."""
    import urllib.request
    import json

    url = (
        f"{SWIFT_STORAGE_URL}/{CONTAINER}"
        f"?prefix={IMAGE_PREFIX.replace(' ', '%20')}"
        f"&format=json&limit={n}"
    )
    req = urllib.request.Request(
        url,
        headers={"X-Auth-Token": SWIFT_AUTH_TOKEN},
    )
    with urllib.request.urlopen(req) as resp:
        objects = json.loads(resp.read())
    return [obj["name"] for obj in objects if obj["name"].endswith(".jpg")]


def download_image(object_name: str, output_dir: Path) -> Path:
    """Download one image from Swift to local disk."""
    import urllib.request

    filename = Path(object_name).name
    output_path = output_dir / filename
    if output_path.exists():
        logger.info(f"Already exists, skipping: {filename}")
        return output_path

    url = f"{SWIFT_STORAGE_URL}/{CONTAINER}/{object_name.replace(' ', '%20')}"
    req = urllib.request.Request(url, headers={"X-Auth-Token": SWIFT_AUTH_TOKEN})

    with urllib.request.urlopen(req) as resp:
        output_path.write_bytes(resp.read())

    logger.info(f"Downloaded: {filename} ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download test images from Swift")
    parser.add_argument("--output-dir", default="/test-images")
    parser.add_argument("--n", type=int, default=20, help="Number of images to download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate credentials
    if not SWIFT_AUTH_TOKEN:
        logger.error(
            "OS_AUTH_TOKEN not set. "
            "Run 'source ~/openrc' on Chameleon before executing this script."
        )
        raise SystemExit(1)

    logger.info(f"Listing {args.n} images from {CONTAINER}/{IMAGE_PREFIX}")
    image_names = list_images(args.n)
    logger.info(f"Found {len(image_names)} images")

    for name in image_names:
        try:
            download_image(name, output_dir)
        except Exception as e:
            logger.warning(f"Failed to download {name}: {e}")

    downloaded = list(output_dir.glob("*.jpg"))
    logger.info(f"Downloaded {len(downloaded)} images to {output_dir}")


if __name__ == "__main__":
    main()
