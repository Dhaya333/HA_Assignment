"""
Metadata module — Phase 6.

Assembles and writes a per-run metadata.json describing what was run,
with what model/device, and the depth/coordinate conventions used —
so downstream consumers of scene.ply never have to guess.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config import config
from src.depth_estimator import DepthEstimationResult
from src.image_loader import LoadedImage
from src.point_cloud import PointCloudResult

logger = logging.getLogger(__name__)


class MetadataError(RuntimeError):
    """Raised when metadata assembly or writing fails."""


@dataclass
class RunMetadata:
    timestamp_utc: str
    input_image: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    depth: dict = field(default_factory=dict)
    point_cloud: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)


def build_metadata(
    loaded_image: LoadedImage,
    depth_result: DepthEstimationResult,
    point_cloud_result: PointCloudResult,
    output_paths: dict[str, Path],
) -> RunMetadata:
    """Assemble a RunMetadata record from each pipeline phase's output."""
    try:
        return RunMetadata(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            input_image={
                "filename": loaded_image.path.name,
                "width": loaded_image.width,
                "height": loaded_image.height,
            },
            model={
                "name": config.MODEL_NAME,
                "device": config.DEVICE,
                "inference_seconds": round(depth_result.inference_seconds, 4),
            },
            depth={
                "type": "relative",
                "raw_convention": "larger value = closer (model output)",
                "projection_convention": "larger value = farther (after inversion for 3D projection)",
                "note": "Not metric distance. No camera calibration used.",
            },
            point_cloud={
                "point_count": point_cloud_result.point_count,
                "stride": config.POINT_CLOUD_STRIDE,
                "intrinsics": {
                    "fx": point_cloud_result.intrinsics.fx,
                    "fy": point_cloud_result.intrinsics.fy,
                    "cx": point_cloud_result.intrinsics.cx,
                    "cy": point_cloud_result.intrinsics.cy,
                    "note": "Assumed/pseudo intrinsics, not real camera calibration.",
                },
                "distance_range": {
                    "min": config.MIN_RELATIVE_DEPTH,
                    "max": config.MAX_RELATIVE_DEPTH,
                },
            },
            outputs={key: str(path) for key, path in output_paths.items()},
        )
    except Exception as exc:
        raise MetadataError(f"Failed to assemble run metadata: {exc}") from exc


def save_metadata(metadata: RunMetadata, output_path: str | Path) -> Path:
    """Write a RunMetadata record to a JSON file."""
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2)
    except Exception as exc:
        raise MetadataError(f"Failed to write metadata to '{output_path}': {exc}") from exc

    logger.info("Saved run metadata to '%s'", output_path)
    return output_path