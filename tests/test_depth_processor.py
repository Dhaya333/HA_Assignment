"""Tests for src/depth_processor.py"""

import numpy as np
import pytest

from src.depth_processor import InvalidDepthMapError, process_depth
from config import config


def test_process_depth_inverts_and_rescales():
    # Raw model convention: 0 = far, 1 = close.
    raw = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    result = process_depth(raw)

    # After inversion + rescale: raw 1.0 (closest) -> MIN_RELATIVE_DEPTH
    #                            raw 0.0 (farthest) -> MAX_RELATIVE_DEPTH
    assert np.isclose(result.distance[1, 0], config.MIN_RELATIVE_DEPTH)
    assert np.isclose(result.distance[0, 0], config.MAX_RELATIVE_DEPTH)


def test_process_depth_output_range():
    raw = np.random.rand(20, 20).astype(np.float32)
    result = process_depth(raw)

    assert result.distance.min() >= config.MIN_RELATIVE_DEPTH - 1e-6
    assert result.distance.max() <= config.MAX_RELATIVE_DEPTH + 1e-6


def test_process_depth_preserves_ordering():
    # Closer objects (higher raw value) must end up with SMALLER distance
    # after conversion, since distance convention is "larger = farther".
    raw = np.array([[0.9, 0.1]], dtype=np.float32)
    result = process_depth(raw)
    assert result.distance[0, 0] < result.distance[0, 1]


def test_non_2d_input_raises():
    bad_shape = np.zeros((5, 5, 3), dtype=np.float32)
    with pytest.raises(InvalidDepthMapError):
        process_depth(bad_shape)


def test_out_of_range_input_raises():
    out_of_range = np.array([[0.0, 1.5]], dtype=np.float32)  # 1.5 > 1.0
    with pytest.raises(InvalidDepthMapError):
        process_depth(out_of_range)


def test_negative_input_raises():
    negative = np.array([[-0.1, 0.5]], dtype=np.float32)
    with pytest.raises(InvalidDepthMapError):
        process_depth(negative)