"""Tests for src/image_loader.py"""

from pathlib import Path

import pytest
from PIL import Image
from unittest.mock import MagicMock, patch

from src.image_loader import (
    ImageNotFoundError,
    InvalidImageError,
    UnsupportedImageFormatError,
    load_image,
)

def test_os_error_during_open_raises_invalid_image_error(tmp_path: Path):
    path = tmp_path / "locked.jpg"
    Image.new("RGB", (10, 10)).save(path)

    with patch("src.image_loader.Image.open", side_effect=OSError("permission denied")):
        with pytest.raises(InvalidImageError):
            load_image(path)


def test_zero_dimension_image_raises(tmp_path: Path):
    path = tmp_path / "zero.jpg"
    Image.new("RGB", (10, 10)).save(path)

    fake_image = MagicMock()
    fake_image.size = (0, 0)
    fake_image.mode = "RGB"

    with patch("src.image_loader.Image.open", return_value=fake_image):
        with pytest.raises(InvalidImageError):
            load_image(path)

@pytest.fixture
def valid_jpg(tmp_path: Path) -> Path:
    path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 48), color=(120, 80, 200)).save(path)
    return path


@pytest.fixture
def valid_png(tmp_path: Path) -> Path:
    path = tmp_path / "test.png"
    Image.new("RGBA", (32, 32), color=(10, 20, 30, 255)).save(path)
    return path


def test_load_valid_jpg(valid_jpg: Path):
    result = load_image(valid_jpg)
    assert result.width == 64
    assert result.height == 48
    assert result.image.mode == "RGB"
    assert result.path == valid_jpg


def test_load_valid_png_converts_to_rgb(valid_png: Path):
    # Input is RGBA; loader should convert it to RGB.
    result = load_image(valid_png)
    assert result.image.mode == "RGB"


def test_nonexistent_path_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(ImageNotFoundError):
        load_image(missing)


def test_unsupported_extension_raises(tmp_path: Path):
    bad_ext = tmp_path / "test.bmp"
    Image.new("RGB", (10, 10)).save(bad_ext)
    with pytest.raises(UnsupportedImageFormatError):
        load_image(bad_ext)


def test_corrupted_file_with_valid_extension_raises(tmp_path: Path):
    fake_jpg = tmp_path / "corrupted.jpg"
    fake_jpg.write_bytes(b"this is not actually image data")
    with pytest.raises(InvalidImageError):
        load_image(fake_jpg)


def test_directory_instead_of_file_raises(tmp_path: Path):
    directory = tmp_path / "a_directory.jpg"
    directory.mkdir()
    with pytest.raises(InvalidImageError):
        load_image(directory)


def test_extension_case_insensitive(tmp_path: Path):
    path = tmp_path / "test.JPG"
    Image.new("RGB", (10, 10)).save(path)
    result = load_image(path)
    assert result.width == 10