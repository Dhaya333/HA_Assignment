"""
Tests for src/depth_estimator.py

Model loading is mocked throughout — these tests verify our wrapper
logic (error handling, normalization, result shape), not the actual
pretrained model, so they run offline and fast in CI.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.depth_estimator import DepthEstimator, ModelLoadError


@pytest.fixture
def mock_transformers():
    """Patch AutoImageProcessor/AutoModelForDepthEstimation with mocks."""
    with patch("src.depth_estimator.AutoImageProcessor") as mock_processor_cls, \
         patch("src.depth_estimator.AutoModelForDepthEstimation") as mock_model_cls:

        mock_processor = MagicMock()
        mock_model = MagicMock()

        mock_processor_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        yield mock_processor, mock_model


def test_model_load_failure_raises_model_load_error():
    with patch(
        "src.depth_estimator.AutoImageProcessor.from_pretrained",
        side_effect=OSError("network unreachable"),
    ):
        with pytest.raises(ModelLoadError):
            DepthEstimator(device="cpu")


def test_successful_load_sets_eval_mode(mock_transformers):
    _, mock_model = mock_transformers
    DepthEstimator(device="cpu")
    mock_model.eval.assert_called_once()


def test_estimate_returns_normalized_depth(mock_transformers):
    mock_processor, mock_model = mock_transformers

    # Fake a plausible model output: a (H, W) tensor with varying values.
    fake_depth_tensor = torch.rand(48, 64) * 10.0  # arbitrary non-[0,1] range
    mock_processor.return_value.to.return_value = MagicMock()
    mock_processor.post_process_depth_estimation.return_value = [
        {"predicted_depth": fake_depth_tensor}
    ]
    mock_model.return_value = MagicMock()

    estimator = DepthEstimator(device="cpu")
    image = Image.new("RGB", (64, 48), color=(50, 100, 150))

    result = estimator.estimate(image)

    assert result.depth_map.shape == (48, 64)
    assert result.depth_map.min() >= 0.0
    assert result.depth_map.max() <= 1.0
    assert np.isclose(result.depth_map.max(), 1.0)
    assert np.isclose(result.depth_map.min(), 0.0)
    assert result.original_size == (64, 48)
    assert result.inference_seconds >= 0.0


def test_estimate_raises_on_flat_depth_map(mock_transformers):
    mock_processor, mock_model = mock_transformers

    # All-identical values -> should be rejected as invalid.
    flat_tensor = torch.ones(10, 10) * 5.0
    mock_processor.post_process_depth_estimation.return_value = [
        {"predicted_depth": flat_tensor}
    ]

    estimator = DepthEstimator(device="cpu")
    image = Image.new("RGB", (10, 10))

    with pytest.raises(RuntimeError):
        estimator.estimate(image)


def test_estimate_converts_non_rgb_image(mock_transformers):
    mock_processor, mock_model = mock_transformers
    fake_tensor = torch.rand(20, 20) * 5.0
    mock_processor.post_process_depth_estimation.return_value = [
        {"predicted_depth": fake_tensor}
    ]

    estimator = DepthEstimator(device="cpu")
    grayscale_image = Image.new("L", (20, 20))  # not RGB

    # Should not raise — estimate() converts to RGB internally.
    result = estimator.estimate(grayscale_image)
    assert result.depth_map.shape == (20, 20)


def test_estimate_raises_runtime_error_on_inference_failure(mock_transformers):
    mock_processor, mock_model = mock_transformers
    mock_processor.side_effect = RuntimeError("out of memory")

    estimator = DepthEstimator(device="cpu")
    image = Image.new("RGB", (32, 32))

    with pytest.raises(RuntimeError):
        estimator.estimate(image)


def test_default_device_uses_config_device(mock_transformers):
    with patch("src.depth_estimator.config.DEVICE", "cpu"):
        estimator = DepthEstimator()  # no device passed
        assert estimator.device == "cpu"


def test_model_moved_to_correct_device(mock_transformers):
    _, mock_model = mock_transformers
    DepthEstimator(device="cpu")
    mock_model.to.assert_called_once_with("cpu")