"""
Visualization module — Phase 5.

Saves the original image and a colorized depth map visualization to
disk, and optionally opens an interactive Open3D point cloud viewer.

Note on convention: this module visualizes the depth map using the
RAW model convention (brighter = closer), which is the standard way
Depth Anything results are typically displayed. This is independent
of, and NOT the same convention as, the inverted "larger = farther"
distance map used for 3D projection in depth_processor.py /
point_cloud.py. Both conventions are documented at their point of use
so they're never confused with each other downstream.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl   
import matplotlib.cm as cm
import numpy as np
import open3d as o3d
from PIL import Image

from src.point_cloud import PointCloudResult

logger = logging.getLogger(__name__)


class VisualizationError(RuntimeError):
    """Raised when saving or displaying a visualization fails."""


def save_original_image(image: Image.Image, output_path: str | Path) -> Path:
    """Save a copy of the (already-validated) input image to output_path."""
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
    except Exception as exc:
        raise VisualizationError(
            f"Failed to save original image to '{output_path}': {exc}"
        ) from exc

    logger.info("Saved original image to '%s'", output_path)
    return output_path


def save_depth_visualization(
    normalized_depth: np.ndarray,
    output_path: str | Path,
    colormap: str = "inferno",
) -> Path:
    """
    Save a colorized depth map image.

    normalized_depth is expected in [0, 1] using the RAW model
    convention (larger = closer) — i.e. the direct output of
    DepthEstimator.estimate(), before depth_processor.py's inversion.
    """
    if normalized_depth.ndim != 2:
        raise VisualizationError(
            f"Expected a 2D depth map, got shape {normalized_depth.shape}"
        )

    output_path = Path(output_path)

    try:
        cmap = mpl.colormaps.get_cmap(colormap)
        colored = cmap(normalized_depth)  # (H, W, 4) RGBA floats in [0, 1]
        rgb_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)

        depth_image = Image.fromarray(rgb_uint8, mode="RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        depth_image.save(output_path)
    except Exception as exc:
        raise VisualizationError(
            f"Failed to save depth visualization to '{output_path}': {exc}"
        ) from exc

    logger.info("Saved depth visualization ('%s' colormap) to '%s'", colormap, output_path)
    return output_path


def visualize_point_cloud_interactive(result: PointCloudResult) -> None:
    """
    Open an interactive Open3D window to inspect the point cloud.

    Optional/manual step — not run automatically by the pipeline, since
    it requires a display and blocks until the window is closed. Safe
    to skip entirely on headless machines or CI.
    """
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(result.points.astype(np.float64))
        if result.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(result.colors.astype(np.float64))

        logger.info("Opening interactive point cloud viewer (close window to continue)...")
        o3d.visualization.draw_geometries([pcd])
    except Exception as exc:
        raise VisualizationError(
            f"Failed to open interactive point cloud viewer: {exc}"
        ) from exc