"""
Entry point — Hawk_Aerospace 2D -> 3D prototype.

Usage:
    python main.py --image data/input/photo.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys

from config import config
from src.depth_estimator import DepthEstimator, ModelLoadError
from src.pipeline import run_pipeline


def configure_logging() -> None:
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a 2D image into a relative 3D point cloud."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image (e.g. data/input/photo.jpg)",
    )
    return parser.parse_args()


def main() -> int:
    configure_logging()
    logger = logging.getLogger(__name__)

    config.ensure_directories()
    args = parse_args()

    try:
        estimator = DepthEstimator()
    except ModelLoadError as exc:
        logger.error("Could not load the depth model: %s", exc)
        return 1

    try:
        output_dir = run_pipeline(args.image, estimator)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1

    print(f"\nDone. Output saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())