from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .kernels import (
    compact_inliers_kernel,
    knn_mean_dist_kernel,
    reduce_mean_stats_kernel,
)


@dataclass
class OutlierFilterConfig:
    """Configuration for `StatisticalOutlierFilter`."""

    # Number of nearest neighbors used to compute each point's mean distance.
    k: int = 20
    # Maximum search radius (meters) for neighbor lookup. Points with fewer than
    # `k` neighbors inside this radius are rejected outright.
    search_radius_m: float = 0.5
    # Reject points whose range-normalized mean k-NN distance exceeds
    # μ + std_multiplier·σ, where statistics are taken over all valid points.
    std_multiplier: float = 1.0
    # Sensor origin in the same frame as the points. The mean k-NN distance is
    # divided by ||p - origin|| before thresholding, which compensates for the
    # ≈ linear growth of lidar point spacing with range.
    sensor_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Floor on the range divisor to avoid blow-up for points at the sensor.
    range_eps_m: float = 0.1


def _hashgrid_dims(points_np: np.ndarray, radius: float) -> tuple[int, int, int]:
    mins = points_np.min(axis=0)
    maxs = points_np.max(axis=0)
    extent = np.maximum(maxs - mins, radius)
    cells = np.ceil(extent / max(radius, 1.0e-6)).astype(int)
    cells = np.clip(cells, 8, 256)
    return int(cells[0]), int(cells[1]), int(cells[2])


class StatisticalOutlierFilter:
    """GPU-native k-NN distance-based statistical outlier removal.

    Per-point range-normalized mean distance to k nearest neighbors; reject
    points exceeding μ + std_multiplier·σ (statistics over all valid points).
    Points with fewer than k neighbors in `search_radius_m` are also rejected.

    All reductions and compaction happen on the GPU — only μ/σ scalars and the
    output count are read back (≤ 16 bytes per call). Accepts numpy or
    `wp.array` input; returns the matching type.
    """

    def __init__(
        self,
        config: OutlierFilterConfig | None = None,
        *,
        device: wp.context.Device | None = None,
    ):
        self.config = config or OutlierFilterConfig()
        self.device = device if device is not None else wp.get_device()
        self._grid: wp.HashGrid | None = None

        # Per-point scratch + outputs, grown on demand.
        self._scratch: wp.array | None = None
        self._mean_dist: wp.array | None = None
        self._valid: wp.array | None = None
        self._out_pts: wp.array | None = None
        self._capacity: int = 0

        # Scalar reduction buffers (fixed size).
        with wp.ScopedDevice(self.device):
            self._sum = wp.zeros(1, dtype=wp.float32)
            self._sum_sq = wp.zeros(1, dtype=wp.float32)
            self._count = wp.zeros(1, dtype=wp.int32)
            self._out_counter = wp.zeros(1, dtype=wp.int32)

    def _ensure_grid(self, radius: float, points_np: np.ndarray) -> wp.HashGrid:
        dims = _hashgrid_dims(points_np, radius)
        if self._grid is None or self._grid.device != self.device:
            self._grid = wp.HashGrid(*dims, device=self.device)
        return self._grid

    def _ensure_buffers(self, n: int, k: int) -> None:
        if (
            self._capacity >= n
            and self._scratch is not None
            and self._scratch.shape[1] == k
        ):
            return
        with wp.ScopedDevice(self.device):
            self._scratch = wp.empty((n, k), dtype=wp.float32)
            self._mean_dist = wp.empty(n, dtype=wp.float32)
            self._valid = wp.empty(n, dtype=wp.int32)
            self._out_pts = wp.empty(n, dtype=wp.vec3)
        self._capacity = n

    def apply(self, points: np.ndarray | wp.array) -> np.ndarray | wp.array:
        """Return `points` with outliers removed. Input and output types match."""
        cfg = self.config
        return_numpy = isinstance(points, np.ndarray)

        if return_numpy:
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"points must be (N, 3); got {points.shape}")
            n = len(points)
            pts_np_f32 = np.ascontiguousarray(points, dtype=np.float32)
            pts_wp = wp.array(pts_np_f32, dtype=wp.vec3, device=self.device)
            grid_points_np = pts_np_f32
        else:
            n = len(points)
            pts_wp = points
            # HashGrid dims need CPU min/max; one D2H readback for the bbox.
            grid_points_np = points.numpy()

        if n <= cfg.k:
            return points

        big_init = float(cfg.search_radius_m) * float(cfg.search_radius_m) + 1.0

        with wp.ScopedDevice(self.device):
            grid = self._ensure_grid(cfg.search_radius_m, grid_points_np)
            grid.build(points=pts_wp, radius=float(cfg.search_radius_m))

            self._ensure_buffers(n, cfg.k)
            origin = wp.vec3(
                float(cfg.sensor_origin[0]),
                float(cfg.sensor_origin[1]),
                float(cfg.sensor_origin[2]),
            )

            wp.launch(
                knn_mean_dist_kernel,
                dim=n,
                inputs=[
                    grid.id,
                    pts_wp,
                    float(cfg.search_radius_m),
                    int(cfg.k),
                    big_init,
                    origin,
                    float(cfg.range_eps_m),
                    self._scratch,
                ],
                outputs=[self._mean_dist, self._valid],
            )

            self._sum.zero_()
            self._sum_sq.zero_()
            self._count.zero_()
            wp.launch(
                reduce_mean_stats_kernel,
                dim=n,
                inputs=[self._mean_dist, self._valid],
                outputs=[self._sum, self._sum_sq, self._count],
            )
            wp.synchronize()
            s = float(self._sum.numpy()[0])
            ssq = float(self._sum_sq.numpy()[0])
            count_valid = int(self._count.numpy()[0])

            if count_valid == 0:
                if return_numpy:
                    return points[:0]
                return wp.empty(0, dtype=wp.vec3, device=self.device)

            mu = s / count_valid
            var = max(0.0, ssq / count_valid - mu * mu)
            sigma = math.sqrt(var)
            threshold = mu + cfg.std_multiplier * sigma

            self._out_counter.zero_()
            wp.launch(
                compact_inliers_kernel,
                dim=n,
                inputs=[
                    pts_wp,
                    self._mean_dist,
                    self._valid,
                    float(threshold),
                ],
                outputs=[self._out_counter, self._out_pts],
            )
            wp.synchronize()
            n_out = int(self._out_counter.numpy()[0])

            if return_numpy:
                return self._out_pts.numpy()[:n_out].astype(points.dtype, copy=True)

            # GPU path: allocate a right-sized output and copy from the compact buffer.
            out = wp.empty(n_out, dtype=wp.vec3, device=self.device)
            if n_out > 0:
                wp.copy(out, self._out_pts, 0, 0, n_out)
            return out
