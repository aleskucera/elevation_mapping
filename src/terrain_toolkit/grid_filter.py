from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .filter_kernels import count_obstacles_kernel
from .filter_kernels import filter_grid
from .filter_kernels import inflate_obstacles_kernel
from .grid_utils import meters_to_cells


@dataclass
class FilterConfig:
    """Configuration for `GridMapFilter`."""

    # Support ratio: cells whose neighborhood has fewer than this fraction of
    # measured cells are rejected (set to NaN).
    support_radius_m: float = 0.5
    support_ratio: float = 0.5

    # Gaussian-weighted obstacle inflation. Only cells above `obstacle_threshold`
    # act as sources; their influence decays with exp(-d²/2σ²). The kernel
    # window extends to 3σ.
    inflation_sigma_m: float = 0.3
    obstacle_threshold: float = 0.8

    # Temporal hysteresis: reject frames where obstacle count grows by more than
    # `obstacle_growth_threshold` relative to the last accepted frame, up to
    # `rejection_limit_frames` consecutive rejections before force-accepting.
    obstacle_growth_threshold: float = 2.0
    rejection_limit_frames: int = 5
    min_obstacle_baseline: int = 10


class GridMapFilter:
    """Support-ratio filter + obstacle inflation + temporal rejection hysteresis.

    Keeps state across calls (last-frame obstacle count, consecutive-rejection
    counter), so the same instance must be reused across frames for the
    hysteresis logic to work.
    """

    def __init__(
        self,
        resolution: float,
        height: int,
        width: int,
        config: FilterConfig | None = None,
        *,
        device: wp.context.Device | None = None,
        verbose: bool = False,
    ):
        self.resolution = resolution
        self.height = height
        self.width = width
        self.shape = (height, width)
        self.config = config or FilterConfig()
        self.device = device if device is not None else wp.get_device()
        self.verbose = verbose

        cfg = self.config
        self.support_radius_cells = meters_to_cells(cfg.support_radius_m, resolution)

        # Gaussian inflation: sigma in cells, kernel radius = ceil(3σ).
        import math
        sigma_cells = cfg.inflation_sigma_m / resolution
        self.inflation_radius_cells = int(math.ceil(3.0 * sigma_cells))
        self.inv_two_sigma_sq = 1.0 / (2.0 * sigma_cells * sigma_cells) if sigma_cells > 0 else 0.0

        # Temporal state.
        self._last_obstacle_count = 0
        self._consecutive_rejections = 0

        with wp.ScopedDevice(self.device):
            self._elev = wp.zeros(self.shape, dtype=wp.float32)
            self._cost = wp.zeros(self.shape, dtype=wp.float32)
            self._filtered = wp.zeros(self.shape, dtype=wp.float32)
            self._inflated = wp.zeros(self.shape, dtype=wp.float32)
            self._num_obstacles = wp.zeros(1, dtype=wp.int32)

    def _is_stable(self, current: int) -> bool:
        cfg = self.config
        if self._last_obstacle_count < cfg.min_obstacle_baseline:
            self._last_obstacle_count = current
            self._consecutive_rejections = 0
            return True

        growth = current / self._last_obstacle_count
        if growth > cfg.obstacle_growth_threshold:
            self._consecutive_rejections += 1
            if self._consecutive_rejections <= cfg.rejection_limit_frames:
                return False
            # Force-accept: establish a new baseline.

        self._consecutive_rejections = 0
        self._last_obstacle_count = current
        return True

    def apply(self, raw_elevation: np.ndarray, cost_map: np.ndarray) -> np.ndarray:
        """Run support-ratio filter + inflation. Returns all-NaN when the frame
        is rejected by the growth hysteresis.

        `raw_elevation` is the pre-inpaint heightmap (NaN in unmeasured cells).
        """
        if raw_elevation.shape != self.shape or cost_map.shape != self.shape:
            raise ValueError("raw_elevation and cost_map must match filter shape")

        cfg = self.config
        self._elev.assign(
            wp.from_numpy(np.ascontiguousarray(raw_elevation, dtype=np.float32), device=self.device)
        )
        self._cost.assign(
            wp.from_numpy(np.ascontiguousarray(cost_map, dtype=np.float32), device=self.device)
        )

        with wp.ScopedTimer("GridMapFilter", active=self.verbose):
            # 1. Inflate obstacles on the full (NaN-free) cost map first.
            wp.launch(
                inflate_obstacles_kernel,
                dim=self.shape,
                inputs=[
                    self._cost,
                    self.height,
                    self.width,
                    self.inflation_radius_cells,
                    float(self.inv_two_sigma_sq),
                    float(cfg.obstacle_threshold),
                ],
                outputs=[self._inflated],
                device=self.device,
            )

            # 2. Count obstacles on the inflated map for hysteresis.
            self._num_obstacles.zero_()
            wp.launch(
                count_obstacles_kernel,
                dim=self.shape,
                inputs=[self._inflated, float(cfg.obstacle_threshold)],
                outputs=[self._num_obstacles],
                device=self.device,
            )
            wp.synchronize()
            current = int(self._num_obstacles.numpy()[0])

            if not self._is_stable(current):
                return np.full(self.shape, np.nan, dtype=np.float32)

            # 3. Filter by support ratio — NaN-out cells without enough
            #    real measurements in the raw elevation neighborhood.
            wp.launch(
                filter_grid,
                dim=self.shape,
                inputs=[
                    self._elev,
                    self._inflated,
                    self.height,
                    self.width,
                    self.support_radius_cells,
                    float(cfg.support_ratio),
                ],
                outputs=[self._filtered],
                device=self.device,
            )
        wp.synchronize()
        return self._filtered.numpy().copy()
