"""
Central configuration for Hawk_Aerospace.

Phase 1 (Environment) responsibility: define paths, device selection,
and shared constants used by every later module. No model loading or
inference logic lives here.
"""

from pathlib import Path
import logging

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
INPUT_DIR: Path = DATA_DIR / "input"
OUTPUT_DIR: Path = DATA_DIR / "output"

MODELS_DIR: Path = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
# Use GPU automatically if available, otherwise fall back to CPU.
# This laptop is CPU-only, so this will resolve to "cpu" here, but the
# code is written to pick up a GPU transparently if run elsewhere.
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Depth Anything V2 - Small, via Hugging Face transformers.
# Selected for CPU-friendliness, reliable pip install, and Apache-2.0
# licensing (the only Depth Anything V2 scale with a permissive license).
MODEL_NAME: str = "depth-anything/Depth-Anything-V2-Small-hf"

# Hugging Face will cache downloaded weights here instead of the
# default user-home cache, keeping the project self-contained.
HF_CACHE_DIR: Path = MODELS_DIR

# ---------------------------------------------------------------------------
# Image input
# ---------------------------------------------------------------------------
SUPPORTED_IMAGE_FORMATS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}

# Cap the longest image edge before inference, to keep runtime and memory
# reasonable on a normal laptop. Set to None to disable downscaling.
# (Section 15: "Allow configurable inference resolution.")
MAX_INFERENCE_DIMENSION: int | None = 1024

# ---------------------------------------------------------------------------
# Depth conversion (Section 5)
# ---------------------------------------------------------------------------
# Depth Anything V2 (relative models) output convention: LARGER raw values
# mean CLOSER to the camera. depth_processor.py inverts this so that our
# project's Z axis follows the conventional camera convention instead:
# larger Z = farther from the camera.
#
# The inverted, relative "distance" is rescaled into this range rather than
# left as [0, 1], because a distance of exactly 0 is degenerate for pinhole
# projection (X = (u-cx)*Z/fx collapses every pixel to the origin at Z=0).
# These bounds are an arbitrary relative scale, NOT metric units.
MIN_RELATIVE_DEPTH: float = 0.1
MAX_RELATIVE_DEPTH: float = 1.0

# ---------------------------------------------------------------------------
# 3D projection (Section 6) — pseudo camera intrinsics
# ---------------------------------------------------------------------------
# No real camera calibration is available in this version. fx/fy are
# approximated from image dimensions (a common placeholder assumption
# roughly corresponding to a ~53° horizontal field of view), and cx/cy
# are the image center. This is explicitly NOT real calibration — see
# README limitations. Designed so real intrinsics can replace this later.
FOCAL_LENGTH_FACTOR: float = 1.0  # fx = fy = max(width, height) * this factor

# ---------------------------------------------------------------------------
# Point cloud density (Section 7 / Section 15 performance)
# ---------------------------------------------------------------------------
# Project every Nth pixel instead of every pixel, to keep point counts and
# .ply file size reasonable on a normal laptop. 1 = full density.
POINT_CLOUD_STRIDE: int = 2

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
POINT_CLOUD_FILENAME: str = "scene.ply"
DEPTH_MAP_FILENAME: str = "depth.png"
ORIGINAL_IMAGE_FILENAME: str = "original.png"
METADATA_FILENAME: str = "metadata.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def ensure_directories() -> None:
    """Create required directories on first run if they don't exist yet."""
    for directory in (INPUT_DIR, OUTPUT_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)