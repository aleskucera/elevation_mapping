# Terrain Toolkit

GPU-accelerated point cloud to heightmap conversion using [NVIDIA Warp](https://github.com/NVIDIA/warp).

## Features

- **HeightMapBuilder** — rasterizes `(N, 3)` point clouds into `max`, `mean`, `min`, and `count` layers in a single GPU kernel pass
- **Multigrid inpainting** — fills empty cells via iterative diffusion with multigrid acceleration
- **Gaussian smoothing** — NaN-aware separable blur on GPU
- **Geometric traversability** — slope, step-height, and surface-roughness cost layers combined into a single traversability map
- **Support-ratio filter** — rejects cells without enough local measurements, inflates obstacles, and applies temporal hysteresis across frames
- **TerrainPipeline** — points in, multi-layer terrain map out, in a single call
- All kernels run on CUDA via Warp

## Requirements

- Python >= 3.12
- NVIDIA GPU with CUDA support
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
uv sync
```

For dev dependencies (matplotlib, plotly for visualization):

```bash
uv sync --group dev
```

## Usage

### Low-level: just the builder

```python
import numpy as np
from terrain_toolkit import HeightMapBuilder, multigrid_inpaint, gaussian_smooth

builder = HeightMapBuilder(
    resolution=0.1,             # meters per cell
    bounds=(-5, 5, -5, 5),      # (xmin, xmax, ymin, ymax)
)
layers = builder.build(points)  # points: (N, 3) float32
# layers.max, layers.mean, layers.min — (H, W) float32, NaN where empty
# layers.count                          — (H, W) int32

heightmap = multigrid_inpaint(layers.max)
heightmap = gaussian_smooth(heightmap, sigma=1.5)
```

### Full pipeline: points → traversability cost map

```python
from terrain_toolkit import (
    TerrainPipeline, TraversabilityConfig, FilterConfig,
)

pipe = TerrainPipeline(
    resolution=0.1,
    bounds=(-5, 5, -5, 5),
    primary="max",                       # which reduction feeds the cost stack
    inpaint=True,
    smooth_sigma=1.5,
    traversability=TraversabilityConfig(
        max_slope_deg=60.0,
        max_step_height_m=0.55,
        max_roughness_m=0.2,
        step_window_radius_m=0.15,
        roughness_window_radius_m=0.3,
        slope_weight=0.2,
        step_weight=0.2,
        roughness_weight=0.6,
    ),
    filter=FilterConfig(                 # optional; set to None to skip
        support_radius_m=0.1,
        support_ratio=0.75,
        inflation_radius_m=0.4,
        obstacle_threshold=0.8,
    ),
)

tm = pipe.process(points)
# tm.max, tm.mean, tm.min, tm.count    — raw reductions
# tm.elevation                          — primary after inpaint + smooth
# tm.slope_cost, tm.step_cost, tm.roughness_cost, tm.traversability
```

## Test scripts

**Synthetic data** — generates a tilted plane + Gaussian bump with configurable noise:

```bash
uv run python test_synthetic.py --preset noisy
uv run python test_synthetic.py --preset very_noisy
uv run python test_synthetic.py --preset sparse
```

Presets: `clean`, `noisy`, `very_noisy`, `sparse`. Individual overrides: `--noise-std`, `--outlier-frac`, `--dropout-frac`, `--smooth-sigma`.

**Real lidar data** (requires `.npy` point cloud files):

```bash
uv run python test_livox.py --path livox.npy
uv run python test_ouster.py --path ouster.npy
```

All test scripts output interactive Plotly HTML files (`heightmap.html`, `heightmap_ouster.html`, etc.).

## Benchmark

```bash
uv run python benchmark.py
```

Typical results on RTX A500 (68k points, 158x374 grid):

| Stage | Time |
|---|---|
| build (max) | ~2 ms |
| multigrid inpaint | ~5 ms |
| gaussian smooth | ~2.5 ms |
| **full pipeline** | **~7 ms** |
