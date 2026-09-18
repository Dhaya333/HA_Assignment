"""
Pipeline module — orchestrates all phases into a single run.

Image -> Depth -> Processed Depth -> Point Cloud -> Visualizations -> Metadata

Each phase's exceptions are allowed to propagate up as-is (they're
already specific and descriptive per-module) rather than being wrapped
or swallowed here, so the real failure point is always visible.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import config
from src.depth_estimator import DepthEstimator
from src.depth_processor import process_depth
from src.image_loader import load_image
from src.metadata import build_metadata, save_metadata
from src.point_cloud import build_point_cloud, save_point_cloud
from src.visualization import save_depth_visualization, save_original_image

logger = logging.getLogger(__name__)


def run_pipeline(image_path: str | Path, estimator: DepthEstimator) -> Path:
    """
    Run the full 2D -> 3D pipeline on a single image.

    `estimator` is passed in (not constructed here) so the model is
    loaded once by the caller and reused across multiple images,
    rather than reloaded per run (Section 15).

    Returns the output directory for this run.
    """
    logger.info("=" * 60)
    logger.info("Starting pipeline for: %s", image_path)

    # Phase 3: load + validate image
    loaded_image = load_image(image_path)

    # Set up a per-image output folder: data/output/<image_stem>/
    run_output_dir = config.OUTPUT_DIR / loaded_image.path.stem
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2/3: depth estimation
    depth_result = estimator.estimate(loaded_image.image)

    # Phase 4: depth processing + 3D reconstruction
    processed = process_depth(depth_result.depth_map)
    point_cloud_result = build_point_cloud(
        distance=processed.distance,
        rgb_image=loaded_image.image,
    )

    # Phase 5: visualizations
    original_path = save_original_image(
        loaded_image.image, run_output_dir / config.ORIGINAL_IMAGE_FILENAME
    )
    depth_viz_path = save_depth_visualization(
        depth_result.depth_map, run_output_dir / config.DEPTH_MAP_FILENAME
    )
    point_cloud_path = save_point_cloud(
        point_cloud_result, run_output_dir / config.POINT_CLOUD_FILENAME
    )

    # Phase 6: metadata
    output_paths = {
        "original_image": original_path,
        "depth_map": depth_viz_path,
        "point_cloud": point_cloud_path,
    }
    metadata = build_metadata(loaded_image, depth_result, point_cloud_result, output_paths)
    metadata_path = save_metadata(metadata, run_output_dir / config.METADATA_FILENAME)
    output_paths["metadata"] = metadata_path

    logger.info("Pipeline completed. Output: %s", run_output_dir)
    logger.info("=" * 60)

    return run_output_dir