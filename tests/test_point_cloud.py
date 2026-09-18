"""Tests for src/point_cloud.py"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.point_cloud import (
    CameraIntrinsics,
    PointCloudGenerationError,
    build_point_cloud,
    save_point_cloud,
)


@pytest.fixture
def flat_distance_map() -> np.ndarray:
    # Simple 10x10 map, constant relative distance of 0.5.
    return np.full((10, 10), 0.5, dtype=np.float32)


def test_intrinsics_from_image_size():
    intrinsics = CameraIntrinsics.from_image_size(width=640, height=480)
    assert intrinsics.fx == 640
    assert intrinsics.fy == 640
    assert intrinsics.cx == 320
    assert intrinsics.cy == 240


def test_build_point_cloud_shapes(flat_distance_map):
    result = build_point_cloud(flat_distance_map, stride=1)
    expected_points = flat_distance_map.shape[0] * flat_distance_map.shape[1]
    assert result.points.shape == (expected_points, 3)
    assert result.point_count == expected_points
    assert result.colors is None


def test_build_point_cloud_with_color(flat_distance_map):
    rgb = Image.new("RGB", (10, 10), color=(255, 0, 0))
    result = build_point_cloud(flat_distance_map, rgb_image=rgb, stride=1)

    assert result.colors is not None
    assert result.colors.shape == result.points.shape
    # Red channel should be 1.0 (255/255) everywhere.
    assert np.allclose(result.colors[:, 0], 1.0)
    assert np.allclose(result.colors[:, 1], 0.0)


def test_stride_reduces_point_count(flat_distance_map):
    full = build_point_cloud(flat_distance_map, stride=1)
    strided = build_point_cloud(flat_distance_map, stride=2)
    assert strided.point_count < full.point_count


def test_mismatched_rgb_size_raises(flat_distance_map):
    wrong_size_rgb = Image.new("RGB", (5, 5))
    with pytest.raises(PointCloudGenerationError):
        build_point_cloud(flat_distance_map, rgb_image=wrong_size_rgb)


def test_invalid_stride_raises(flat_distance_map):
    with pytest.raises(PointCloudGenerationError):
        build_point_cloud(flat_distance_map, stride=0)


def test_non_2d_distance_map_raises():
    bad_shape = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(PointCloudGenerationError):
        build_point_cloud(bad_shape)


def test_save_point_cloud_writes_file(flat_distance_map, tmp_path: Path):
    result = build_point_cloud(flat_distance_map, stride=1)
    output_path = tmp_path / "scene.ply"

    saved_path = save_point_cloud(result, output_path)

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0