from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

wp.init()


@wp.kernel
def _rasterize_all(
    points: wp.array(dtype=wp.vec3),
    xmin: float,
    ymin: float,
    inv_res: float,
    width: int,
    height: int,
    max_map: wp.array2d(dtype=wp.float32),
    min_map: wp.array2d(dtype=wp.float32),
    sum_map: wp.array2d(dtype=wp.float32),
    count_map: wp.array2d(dtype=wp.int32),
):
    tid = wp.tid()
    p = points[tid]
    j = int((p[0] - xmin) * inv_res)
    i = int((p[1] - ymin) * inv_res)
    if i < 0 or i >= height or j < 0 or j >= width:
        return
    wp.atomic_max(max_map, i, j, p[2])
    wp.atomic_min(min_map, i, j, p[2])
    wp.atomic_add(sum_map, i, j, p[2])
    wp.atomic_add(count_map, i, j, 1)


@wp.kernel
def _finalize(
    sum_map: wp.array2d(dtype=wp.float32),
    count_map: wp.array2d(dtype=wp.int32),
    max_map: wp.array2d(dtype=wp.float32),
    min_map: wp.array2d(dtype=wp.float32),
    mean_map: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    c = count_map[i, j]
    nan = wp.float32(wp.nan)
    if c > 0:
        mean_map[i, j] = sum_map[i, j] / float(c)
    else:
        mean_map[i, j] = nan
        max_map[i, j] = nan
        min_map[i, j] = nan


@dataclass
class HeightMapLayers:
    """Per-cell reductions from a single point-cloud rasterization pass.

    Empty cells are NaN in max/mean/min; count is 0.
    """

    max: np.ndarray
    mean: np.ndarray
    min: np.ndarray
    count: np.ndarray

    def __getitem__(self, name: str) -> np.ndarray:
        return getattr(self, name)


@dataclass
class HeightMapBuilder:
    resolution: float
    bounds: tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)

    def __post_init__(self) -> None:
        xmin, xmax, ymin, ymax = self.bounds
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("Invalid bounds.")
        self.width = int(math.ceil((xmax - xmin) / self.resolution))
        self.height = int(math.ceil((ymax - ymin) / self.resolution))

    def build(self, points: np.ndarray) -> HeightMapLayers:
        """Scatter (N, 3) points into a grid and return max/mean/min/count layers."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3).")

        xmin, _, ymin, _ = self.bounds
        pts = np.ascontiguousarray(points, dtype=np.float32)
        pts_wp = wp.array(pts, dtype=wp.vec3)
        shape = (self.height, self.width)

        max_map = wp.array(np.full(shape, -np.inf, dtype=np.float32), dtype=wp.float32)
        min_map = wp.array(np.full(shape, np.inf, dtype=np.float32), dtype=wp.float32)
        sum_map = wp.zeros(shape, dtype=wp.float32)
        count_map = wp.zeros(shape, dtype=wp.int32)
        mean_map = wp.zeros(shape, dtype=wp.float32)

        wp.launch(
            _rasterize_all,
            dim=pts.shape[0],
            inputs=[
                pts_wp,
                float(xmin),
                float(ymin),
                float(1.0 / self.resolution),
                int(self.width),
                int(self.height),
                max_map,
                min_map,
                sum_map,
                count_map,
            ],
        )
        wp.launch(
            _finalize,
            dim=shape,
            inputs=[sum_map, count_map, max_map, min_map, mean_map],
        )
        wp.synchronize()

        return HeightMapLayers(
            max=max_map.numpy().copy(),
            mean=mean_map.numpy().copy(),
            min=min_map.numpy().copy(),
            count=count_map.numpy().copy(),
        )
