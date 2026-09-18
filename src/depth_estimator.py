"""
Depth estimation module — Phase 2.

Loads the pretrained Depth Anything V2 (Small) monocular depth model
via Hugging Face transformers and runs inference on a single image.

Does NOT implement 3D reconstruction — that's depth_processor.py /
point_cloud.py (Phase 4). This module's only job is:
    RGB image (PIL.Image) -> normalized relative depth map (np.ndarray)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from config import config

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the depth model or its processor fails to load."""


@dataclass
class DepthEstimationResult:
    """Container for a single inference run's output."""

    depth_map: np.ndarray       # float32, normalized 0-1, shape (H, W)
    inference_seconds: float
    original_size: tuple[int, int]  # (width, height)


class DepthEstimator:
    """
    Wraps the pretrained Depth Anything V2 (Small) model.

    The model is loaded once per instance (Section 15: "Load the model
    once per execution") and reused across calls to estimate().
    """

    def __init__(self, device: str | None = None) -> None:
        self.device = device or config.DEVICE
        self.model_name = config.MODEL_NAME

        logger.info("Loading model '%s' on device '%s'...", self.model_name, self.device)

        try:
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                cache_dir=str(config.HF_CACHE_DIR),
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(
                self.model_name,
                cache_dir=str(config.HF_CACHE_DIR),
            )
        except Exception as exc:
            # Covers: no internet on first download, corrupted cache,
            # Hugging Face Hub unavailable, disk full, etc.
            raise ModelLoadError(
                f"Failed to load depth model '{self.model_name}'. "
                f"Check your internet connection (first run needs to "
                f"download the model) and available disk space. "
                f"Original error: {exc}"
            ) from exc

        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully.")

    def estimate(self, image: Image.Image) -> DepthEstimationResult:
        """
        Run monocular depth estimation on a single RGB image.

        Returns a DepthEstimationResult with depth normalized to [0, 1],
        where the convention (which end means "closer") is documented
        in depth_processor.py — do NOT assume raw model output is
        directly usable as physical distance.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        original_size = image.size  # (width, height)

        start = time.perf_counter()
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            post_processed = self.processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(original_size[1], original_size[0])],  # (H, W)
            )
            predicted_depth = post_processed[0]["predicted_depth"]

        except Exception as exc:
            raise RuntimeError(
                f"Depth inference failed for image of size {original_size}: {exc}"
            ) from exc

        elapsed = time.perf_counter() - start

        depth_np = predicted_depth.detach().cpu().numpy().astype(np.float32)

        depth_min, depth_max = depth_np.min(), depth_np.max()
        if depth_max - depth_min < 1e-8:
            raise RuntimeError(
                "Depth model produced a flat/invalid depth map "
                "(no variation across the image)."
            )
        normalized = (depth_np - depth_min) / (depth_max - depth_min)

        logger.info("Depth inference completed in %.2f s", elapsed)

        return DepthEstimationResult(
            depth_map=normalized,
            inference_seconds=elapsed,
            original_size=original_size,
        )