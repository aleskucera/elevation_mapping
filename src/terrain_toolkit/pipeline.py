from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp

from .heightmap import HeightMapBuilder
from .heightmap import gaussian_smooth
from .heightmap import multigrid_inpaint
from .outlier import OutlierFilterConfig
from .outlier import StatisticalOutlierFilter
from .traversability import FilterConfig
from .traversability import GeometricTraversabilityAnalyzer
from .traversability import ObstacleInflator
from .traversability import SupportRatioMask
from .traversability import TemporalGate
from .traversability import TraversabilityConfig

PrimaryLayer = Literal["max", "mean", "min"]


@dataclass
class TerrainMap:
    """Full output of `TerrainPipeline.process`.

    Always populated: `max`, `mean`, `min`, `count`, `elevation` (the primary
    reduction after inpaint + smooth).

    Populated when the corresponding stage is configured: `slope_cost`,
    `step_cost`, `roughness_cost`, `traversability`.
    """

    resolution: float
    bounds: tuple[float, float, float, float]

    # raw reductions
    max: np.ndarray
    mean: np.ndarray
    min: np.ndarray
    count: np.ndarray

    # primary reduction after inpaint + smooth (always present)
    elevation: np.ndarray

    # geometric cost layers (None when traversability is disabled)
    slope_cost: np.ndarray | None = None
    step_cost: np.ndarray | None = None
    roughness_cost: np.ndarray | None = None
    traversability: np.ndarray | None = None

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return all non-None layers as a flat name → array dict."""
        d = {
            "max": self.max,
            "mean": self.mean,
            "min": self.min,
            "count": self.count,
            "elevation": self.elevation,
        }
        for name in ("slope_cost", "step_cost", "roughness_cost", "traversability"):
            arr = getattr(self, name)
            if arr is not None:
                d[name] = arr
        return d


class TerrainPipeline:
    """Points → (max, mean, min, count) → inpaint → smooth → cost → filter.

    Single entry point: `process(points)` returns a fully-populated
    `TerrainMap`. Stateful: reuses GPU buffers and filter hysteresis across
    calls, so the same instance should be reused frame-to-frame.

    Internally keeps data on the GPU: point cloud is uploaded once, every
    stage consumes and produces `wp.array`, and a single download happens
    at the end to build the numpy-backed `TerrainMap` for the caller.
    """

    def __init__(
        self,
        resolution: float,
        bounds: tuple[float, float, float, float],
        *,
        primary: PrimaryLayer = "max",
        inpaint: bool = True,
        smooth_sigma: float = 0.0,
        inpaint_iters_per_level: int = 50,
        inpaint_coarse_iters: int = 200,
        z_max: float | None = None,
        outlier: OutlierFilterConfig | None = None,
        traversability: TraversabilityConfig | None = None,
        filter: FilterConfig | None = None,
    ):
        if primary not in ("max", "mean", "min"):
            raise ValueError(f"primary must be 'max', 'mean', or 'min'; got {primary!r}")
        if traversability is not None and not inpaint:
            raise ValueError(
                "traversability requires inpaint=True — the cost kernels assume a filled grid"
            )
        if filter is not None and traversability is None:
            raise ValueError("filter is only meaningful when traversability is enabled")

        self.resolution = resolution
        self.bounds = bounds
        self.primary = primary
        self.z_max = z_max
        self.inpaint = inpaint
        self.smooth_sigma = smooth_sigma
        self.inpaint_iters_per_level = inpaint_iters_per_level
        self.inpaint_coarse_iters = inpaint_coarse_iters

        self.builder = HeightMapBuilder(resolution=resolution, bounds=bounds)
        self.height = self.builder.height
        self.width = self.builder.width

        self.outlier_filter: StatisticalOutlierFilter | None = None
        if outlier is not None:
            self.outlier_filter = StatisticalOutlierFilter(config=outlier)

        self.analyzer: GeometricTraversabilityAnalyzer | None = None
        if traversability is not None:
            self.analyzer = GeometricTraversabilityAnalyzer(
                resolution=resolution,
                height=self.height,
                width=self.width,
                config=traversability,
            )

        self.inflator: ObstacleInflator | None = None
        self.temporal_gate: TemporalGate | None = None
        self.support_mask: SupportRatioMask | None = None
        if filter is not None:
            self.inflator = ObstacleInflator(
                resolution=resolution,
                height=self.height,
                width=self.width,
                config=filter,
            )
            self.temporal_gate = TemporalGate(config=filter)
            self.support_mask = SupportRatioMask(
                resolution=resolution,
                height=self.height,
                width=self.width,
                config=filter,
            )

    def process(self, points: np.ndarray) -> TerrainMap:
        if self.z_max is not None:
            points = points[points[:, 2] <= self.z_max]

        # Single upload to GPU — every stage below consumes wp.array.
        pts_wp = wp.array(
            np.ascontiguousarray(points, dtype=np.float32), dtype=wp.vec3,
        )
        if self.outlier_filter is not None:
            pts_wp = self.outlier_filter.apply(pts_wp)

        layers = self.builder.build(pts_wp)
        primary_layer = layers[self.primary]  # wp.array

        elevation = primary_layer
        if self.inpaint:
            elevation = multigrid_inpaint(
                elevation,
                iters_per_level=self.inpaint_iters_per_level,
                coarse_iters=self.inpaint_coarse_iters,
            )
        if self.smooth_sigma > 0.0:
            elevation = gaussian_smooth(elevation, sigma=self.smooth_sigma)

        traversability_wp: wp.array | None = None
        costs_wp: dict[str, wp.array] | None = None
        if self.analyzer is not None:
            costs = self.analyzer.compute(elevation)
            total = costs.total
            if self.inflator is not None:
                # Inflate obstacles, then temporally gate, then mask by support ratio.
                # Inpainted cells with enough real neighbors keep their cost; the rest
                # become NaN, and frames that spike in obstacle count are rejected.
                inflated = self.inflator.apply(total)
                if self.temporal_gate.is_stable(inflated):
                    total = self.support_mask.apply(primary_layer, inflated)
                else:
                    total = self.support_mask.rejected_frame()
            traversability_wp = total
            costs_wp = {"slope": costs.slope, "step": costs.step, "roughness": costs.roughness}

        # Single download barrier: sync once, then batch-copy all layers to CPU.
        wp.synchronize()
        tm = TerrainMap(
            resolution=self.resolution,
            bounds=self.bounds,
            max=layers.max.numpy().copy(),
            mean=layers.mean.numpy().copy(),
            min=layers.min.numpy().copy(),
            count=layers.count.numpy().copy(),
            elevation=elevation.numpy().copy(),
        )
        if costs_wp is not None:
            tm.slope_cost = costs_wp["slope"].numpy().copy()
            tm.step_cost = costs_wp["step"].numpy().copy()
            tm.roughness_cost = costs_wp["roughness"].numpy().copy()
        if traversability_wp is not None:
            tm.traversability = traversability_wp.numpy().copy()
        return tm
