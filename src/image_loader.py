"""
Image loading and validation module — Phase 3.

Validates and loads a 2D RGB image from a local filesystem path before
it reaches depth estimation. No model or depth logic lives here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from config import config

logger = logging.getLogger(__name__)


class ImageNotFoundError(FileNotFoundError):
    """Raised when the given image path does not exist."""


class UnsupportedImageFormatError(ValueError):
    """Raised when the image extension isn't in the supported set."""


class InvalidImageError(ValueError):
    """Raised when the file exists but isn't a readable/valid image."""


@dataclass
class LoadedImage:
    """Container for a validated, loaded image and its basic metadata."""

    image: Image.Image       # RGB PIL image
    path: Path
    width: int
    height: int


def load_image(path: str | Path) -> LoadedImage:
    """
    Validate and load an image from a filesystem path.

    Checks, in order (Section 2):
        1. File existence
        2. Supported format (extension)
        3. Image readability
        4. Image dimensions (non-zero)
        5. RGB conversion

    Raises ImageNotFoundError, UnsupportedImageFormatError, or
    InvalidImageError with a clear message on failure.
    """
    image_path = Path(path)

    if not image_path.exists():
        raise ImageNotFoundError(
            f"Image not found: '{image_path}'. Check the path is correct "
            f"and relative to the current working directory."
        )

    if not image_path.is_file():
        raise InvalidImageError(f"Path exists but is not a file: '{image_path}'")

    extension = image_path.suffix.lower()
    if extension not in config.SUPPORTED_IMAGE_FORMATS:
        supported = ", ".join(sorted(config.SUPPORTED_IMAGE_FORMATS))
        raise UnsupportedImageFormatError(
            f"Unsupported image format '{extension}' for '{image_path.name}'. "
            f"Supported formats: {supported}"
        )

    try:
        image = Image.open(image_path)
        image.load()  # force full read now, not lazily later mid-pipeline
    except UnidentifiedImageError as exc:
        raise InvalidImageError(
            f"File '{image_path.name}' has a supported extension but "
            f"isn't a readable image (corrupted or misnamed file)."
        ) from exc
    except OSError as exc:
        raise InvalidImageError(
            f"Failed to read image '{image_path.name}': {exc}"
        ) from exc

    width, height = image.size
    if width == 0 or height == 0:
        raise InvalidImageError(
            f"Image '{image_path.name}' has invalid dimensions: {width}x{height}"
        )

    if image.mode != "RGB":
        logger.info("Converting image from mode '%s' to RGB", image.mode)
        image = image.convert("RGB")

    logger.info("Loaded image '%s' (%dx%d)", image_path.name, width, height)

    return LoadedImage(image=image, path=image_path, width=width, height=height)