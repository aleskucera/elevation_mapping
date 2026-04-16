# Elevation Mapping

GPU-accelerated point cloud to heightmap conversion using [NVIDIA Warp](https://github.com/NVIDIA/warp).

## Features

- **HeightMapBuilder** — rasterizes `(N, 3)` point clouds into 2D heightmaps with `max` or `mean` reduction
- **Multigrid inpainting** — fills empty cells via iterative diffusion with multigrid acceleration
- **Gaussian smoothing** — NaN-aware separable blur on GPU
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

```python
import numpy as np
from elevation_mapping import HeightMapBuilder, multigrid_inpaint, gaussian_smooth

# Build heightmap from point cloud
builder = HeightMapBuilder(
    resolution=0.1,             # meters per cell
    bounds=(-5, 5, -5, 5),      # (xmin, xmax, ymin, ymax)
    reduction="max",            # "max" or "mean"
)
heightmap = builder.build(points)  # points: (N, 3) float32

# Fill empty cells and smooth
heightmap = multigrid_inpaint(heightmap)
heightmap = gaussian_smooth(heightmap, sigma=1.5)
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
