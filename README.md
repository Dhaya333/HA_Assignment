# Hawk_Aerospace

Version 1 prototype: converts a single 2D RGB image into a **relative 3D spatial representation** of the observed scene, intended as a future spatial-perception input for drone obstacle avoidance and path planning.

## What this is (and isn't)

This system estimates **relative** depth from a single monocular image using a pretrained AI model. It does **not** produce real-world metric measurements. A single 2D image has no inherent absolute scale — without camera calibration, stereo vision, or a known reference object, "how far in meters" cannot be answered. This version answers "what's relatively closer or farther, and roughly where" instead.

Flight control, GPS navigation, motor control, and physical drone integration are **not** part of this version.

## Pipeline


## Model

**Depth Anything V2 — Small** (`depth-anything/Depth-Anything-V2-Small-hf`), run via Hugging Face `transformers`.

Chosen for:
- CPU-friendly (24.8M params, by far the lightest of the Depth Anything V2 scales)
- Reliable install — no custom CUDA build or manual checkpoint download, just `pip install`
- Apache-2.0 licensed (the only Depth Anything V2 scale with this license — Base/Large/Giant are CC-BY-NC-4.0, non-commercial)
- Strong relative-depth quality for a lightweight model

The model automatically uses GPU (CUDA) if available, otherwise falls back to CPU.

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py --image data/input/outdoor_scene.jpg
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`

## Output

For each input image, a result folder is created under `data/output/` containing:

| File | Description |
|---|---|
| `original.png` | Copy of the input image |
| `depth.png` | Visualized depth map |
| `scene.ply` | 3D point cloud (X, Y, Z, and optionally R, G, B) |
| `metadata.json` | Run metadata — model used, device, image dimensions, depth/coordinate type |

## Limitations

- **Relative depth only.** Depth and 3D coordinates are relative, not metric. Do not interpret them as real-world distances.
- **No real camera intrinsics.** 3D projection uses assumed/pseudo camera parameters (documented in `src/point_cloud.py`), not a calibrated camera. These are placeholders designed so real intrinsics can be substituted later.
- **Single image only.** No temporal consistency, no motion, no multi-view geometry in this version.
- **Not validated for flight safety.** This is a perception-layer prototype, not a certified navigation input.

## Future Work

Camera calibration, stereo/multi-image input, visual odometry, SLAM, semantic segmentation for obstacle classification, occupancy/free-space mapping, and eventual path-planning + drone controller integration. See project spec for the full version roadmap.

## Project Structure

## Development Status

- [x] Phase 1 — Environment
- [x] Phase 2 — Model
- [x] Phase 3 — Image → Depth
- [x] Phase 4 — Depth → 3D
- [x] Phase 5 — Visualization
- [x] Phase 6 — Metadata
- [x] Phase 7 — Testing
- [x] Phase 8 — Documentation