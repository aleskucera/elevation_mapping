from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .grid_filter import FilterConfig, GridMapFilter
from .height_map_builder import HeightMapBuilder
from .postprocess import gaussian_smooth, multigrid_inpaint
from .traversability import GeometricTraversabilityAnalyzer, TraversabilityConfig

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
        self.inpaint = inpaint
        self.smooth_sigma = smooth_sigma
        self.inpaint_iters_per_level = inpaint_iters_per_level
        self.inpaint_coarse_iters = inpaint_coarse_iters

        self.builder = HeightMapBuilder(resolution=resolution, bounds=bounds)
        self.height = self.builder.height
        self.width = self.builder.width

        self.analyzer: GeometricTraversabilityAnalyzer | None = None
        if traversability is not None:
            self.analyzer = GeometricTraversabilityAnalyzer(
                resolution=resolution,
                height=self.height,
                width=self.width,
                config=traversability,
            )

        self.filter: GridMapFilter | None = None
        if filter is not None:
            self.filter = GridMapFilter(
                resolution=resolution,
                height=self.height,
                width=self.width,
                config=filter,
            )

    def process(self, points: np.ndarray) -> TerrainMap:
        layers = self.builder.build(points)
        primary_layer = getattr(layers, self.primary)

        elevation = primary_layer
        if self.inpaint:
            elevation = multigrid_inpaint(
                elevation,
                iters_per_level=self.inpaint_iters_per_level,
                coarse_iters=self.inpaint_coarse_iters,
            )
        if self.smooth_sigma > 0.0:
            elevation = gaussian_smooth(elevation, sigma=self.smooth_sigma)

        tm = TerrainMap(
            resolution=self.resolution,
            bounds=self.bounds,
            max=layers.max,
            mean=layers.mean,
            min=layers.min,
            count=layers.count,
            elevation=elevation,
        )

        if self.analyzer is not None:
            costs = self.analyzer.compute(elevation)
            tm.slope_cost = costs.slope
            tm.step_cost = costs.step
            tm.roughness_cost = costs.roughness
            total = costs.total
            if self.filter is not None:
                # Support-ratio filter decides which cells are trustworthy —
                # inpainted cells with enough real neighbors keep their cost.
                total = self.filter.apply(primary_layer, total)
            tm.traversability = total

        return tm
