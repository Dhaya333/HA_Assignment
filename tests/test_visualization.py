"""Tests for src/visualization.py"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.visualization import (
    VisualizationError,
    save_depth_visualization,
    save_original_image,
)


def test_save_original_image(tmp_path: Path):
    image = Image.new("RGB", (20, 20), color=(10, 20, 30))
    output_path = tmp_path / "original.png"

    result_path = save_original_image(image, output_path)

    assert result_path.exists()
    saved = Image.open(result_path)
    assert saved.size == (20, 20)


def test_save_original_image_creates_parent_dirs(tmp_path: Path):
    image = Image.new("RGB", (10, 10))
    output_path = tmp_path / "nested" / "dir" / "original.png"

    save_original_image(image, output_path)
    assert output_path.exists()


def test_save_depth_visualization(tmp_path: Path):
    depth = np.random.rand(30, 40).astype(np.float32)
    output_path = tmp_path / "depth.png"

    result_path = save_depth_visualization(depth, output_path)

    assert result_path.exists()
    saved = Image.open(result_path)
    assert saved.size == (40, 30)  # PIL is (width, height)
    assert saved.mode == "RGB"


def test_save_depth_visualization_non_2d_raises(tmp_path: Path):
    bad_shape = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(VisualizationError):
        save_depth_visualization(bad_shape, tmp_path / "depth.png")


def test_save_depth_visualization_write_failure_raises(tmp_path: Path):
    depth = np.random.rand(10, 10).astype(np.float32)
    with patch("src.visualization.Image.fromarray") as mock_fromarray:
        mock_fromarray.return_value.save.side_effect = OSError("disk full")
        with pytest.raises(VisualizationError):
            save_depth_visualization(depth, tmp_path / "depth.png")