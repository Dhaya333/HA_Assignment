"""
3D reconstruction module — Phase 4 (part 2).

Projects a relative distance map (from depth_processor.py) into a 3D
point cloud using pseudo camera intrinsics, and exports it as .ply.

IMPORTANT: fx, fy, cx, cy below are assumed placeholder values, not a
real camera calibration. See config.py and the README limitations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from config import config

logger = logging.getLogger(__name__)


class PointCloudGenerationError(RuntimeError):
    """Raised when point cloud construction or export fails."""


@dataclass
class CameraIntrinsics:
    """Assumed pseudo camera intrinsics — NOT real calibration."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_image_size(cls, width: int, height: int) -> "CameraIntrinsics":
        focal = max(width, height) * config.FOCAL_LENGTH_FACTOR
        return cls(fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0)


@dataclass
class PointCloudResult:
    points: np.ndarray       # float32, shape (N, 3) -> X, Y, Z
    colors: np.ndarray | None  # float32, shape (N, 3) in [0, 1], or None
    intrinsics: CameraIntrinsics
    point_count: int


def build_point_cloud(
    distance: np.ndarray,
    rgb_image: Image.Image | None = None,
    stride: int | None = None,
) -> PointCloudResult:
    """
    Project a (H, W) relative distance map into a 3D point cloud.

    For each sampled pixel (u, v):
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = distance[v, u]

    If rgb_image is provided (same dimensions as distance), per-point
    RGB color is attached for visualization.
    """
    if distance.ndim != 2:
        raise PointCloudGenerationError(
            f"Expected a 2D distance map, got shape {distance.shape}"
        )

    height, width = distance.shape
    stride = stride if stride is not None else config.POINT_CLOUD_STRIDE
    if stride < 1:
        raise PointCloudGenerationError(f"stride must be >= 1, got {stride}")

    intrinsics = CameraIntrinsics.from_image_size(width, height)

    color_array = None
    if rgb_image is not None:
        if rgb_image.size != (width, height):
            raise PointCloudGenerationError(
                f"RGB image size {rgb_image.size} doesn't match depth map "
                f"size {(width, height)}"
            )
        color_array = np.asarray(rgb_image.convert("RGB"), dtype=np.float32) / 255.0

    try:
        v_coords, u_coords = np.mgrid[0:height:stride, 0:width:stride]
        z = distance[v_coords, u_coords]

        x = (u_coords - intrinsics.cx) * z / intrinsics.fx
        y = (v_coords - intrinsics.cy) * z / intrinsics.fy

        points = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)

        colors = None
        if color_array is not None:
            colors = color_array[v_coords, u_coords].reshape(-1, 3)

    except Exception as exc:
        raise PointCloudGenerationError(f"Failed to project points: {exc}") from exc

    if points.shape[0] == 0:
        raise PointCloudGenerationError("Projection produced zero points.")

    logger.info(
        "Built point cloud: %d points (stride=%d, fx=fy=%.1f)",
        points.shape[0], stride, intrinsics.fx,
    )

    return PointCloudResult(
        points=points,
        colors=colors,
        intrinsics=intrinsics,
        point_count=points.shape[0],
    )


def save_point_cloud(result: PointCloudResult, output_path: str | Path) -> Path:
    """Save a PointCloudResult as a .ply file using Open3D."""
    output_path = Path(output_path)

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(result.points.astype(np.float64))
        if result.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(result.colors.astype(np.float64))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = o3d.io.write_point_cloud(str(output_path), pcd)
        if not success:
            raise PointCloudGenerationError(
                f"Open3D reported failure writing point cloud to '{output_path}'"
            )
    except PointCloudGenerationError:
        raise
    except Exception as exc:
        raise PointCloudGenerationError(
            f"Failed to save point cloud to '{output_path}': {exc}"
        ) from exc

    logger.info("Saved point cloud to '%s' (%d points)", output_path, result.point_count)
    return output_path