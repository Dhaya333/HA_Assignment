"""Tests for src/metadata.py"""

from pathlib import Path

import pytest

from src.depth_estimator import DepthEstimationResult
from src.image_loader import LoadedImage
from src.metadata import build_metadata, save_metadata
from src.point_cloud import CameraIntrinsics, PointCloudResult
import numpy as np
from PIL import Image


@pytest.fixture
def sample_inputs():
    loaded_image = LoadedImage(
        image=Image.new("RGB", (64, 48)),
        path=Path("data/input/test.jpg"),
        width=64,
        height=48,
    )
    depth_result = DepthEstimationResult(
        depth_map=np.random.rand(48, 64).astype(np.float32),
        inference_seconds=0.123,
        original_size=(64, 48),
    )
    point_cloud_result = PointCloudResult(
        points=np.zeros((100, 3), dtype=np.float32),
        colors=None,
        intrinsics=CameraIntrinsics(fx=64, fy=64, cx=32, cy=24),
        point_count=100,
    )
    return loaded_image, depth_result, point_cloud_result


def test_build_metadata_contains_expected_fields(sample_inputs, tmp_path: Path):
    loaded_image, depth_result, point_cloud_result = sample_inputs
    output_paths = {"original_image": tmp_path / "original.png"}

    metadata = build_metadata(loaded_image, depth_result, point_cloud_result, output_paths)

    assert metadata.input_image["width"] == 64
    assert metadata.model["inference_seconds"] == 0.123
    assert metadata.point_cloud["point_count"] == 100
    assert "convention" in metadata.depth["raw_convention"] or "closer" in metadata.depth["raw_convention"]


def test_save_metadata_writes_valid_json(sample_inputs, tmp_path: Path):
    loaded_image, depth_result, point_cloud_result = sample_inputs
    metadata = build_metadata(loaded_image, depth_result, point_cloud_result, {})
    output_path = tmp_path / "metadata.json"

    save_metadata(metadata, output_path)

    assert output_path.exists()
    import json
    with open(output_path) as f:
        data = json.load(f)
    assert data["input_image"]["width"] == 64