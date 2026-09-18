"""
Depth processing module — Phase 4 (part 1).

Converts the normalized depth map produced by DepthEstimator into a
"relative distance" map with a documented, consistent convention:
LARGER value = FARTHER from the camera.

This is the module responsible for the Section 5 requirement to
explicitly document whether larger values mean closer or farther.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from config import config

logger = logging.getLogger(__name__)


class InvalidDepthMapError(ValueError):
    """Raised when the input depth map is malformed or out of expected range."""


@dataclass
class ProcessedDepth:
    """Relative distance map ready for 3D projection."""

    distance: np.ndarray  # float32, shape (H, W); larger = farther. Range:
                           # [config.MIN_RELATIVE_DEPTH, config.MAX_RELATIVE_DEPTH]


def process_depth(normalized_depth: np.ndarray) -> ProcessedDepth:
    """
    Convert a normalized [0, 1] depth map (raw model convention: larger =
    closer) into a relative distance map (project convention: larger =
    farther), rescaled to [MIN_RELATIVE_DEPTH, MAX_RELATIVE_DEPTH].

    Raises InvalidDepthMapError if the input isn't a valid 2D array in
    the expected [0, 1] range.
    """
    if normalized_depth.ndim != 2:
        raise InvalidDepthMapError(
            f"Expected a 2D depth map, got shape {normalized_depth.shape}"
        )

    d_min, d_max = float(normalized_depth.min()), float(normalized_depth.max())
    if d_min < -1e-6 or d_max > 1 + 1e-6:
        raise InvalidDepthMapError(
            f"Depth map values out of expected [0, 1] range: "
            f"min={d_min:.4f}, max={d_max:.4f}"
        )

    # Invert convention: raw larger-is-closer -> project's larger-is-farther.
    inverted = 1.0 - normalized_depth

    # Rescale from [0, 1] into [MIN_RELATIVE_DEPTH, MAX_RELATIVE_DEPTH] so
    # no point ends up at distance 0 (degenerate for projection).
    span = config.MAX_RELATIVE_DEPTH - config.MIN_RELATIVE_DEPTH
    distance = config.MIN_RELATIVE_DEPTH + inverted * span

    logger.info(
        "Processed depth map: distance range [%.3f, %.3f] (larger = farther)",
        float(distance.min()), float(distance.max()),
    )

    return ProcessedDepth(distance=distance.astype(np.float32))