from __future__ import annotations

import warp as wp

wp.init()


@wp.kernel
def filter_grid(
    elevation_map: wp.array(dtype=wp.float32, ndim=2),
    cost_map: wp.array(dtype=wp.float32, ndim=2),
    map_height: wp.int32,
    map_width: wp.int32,
    support_radius: wp.int32,
    support_ratio: wp.float32,
    filtered_cost: wp.array(dtype=wp.float32, ndim=2),
):
    """Keep cost where the local neighborhood has enough measured cells; else NaN."""
    r, c = wp.tid()
    measured = int(0)
    total = int(0)
    for dr in range(-support_radius, support_radius + 1):
        for dc in range(-support_radius, support_radius + 1):
            nr = r + dr
            nc = c + dc
            if nr >= 0 and nr < map_height and nc >= 0 and nc < map_width:
                val = elevation_map[nr, nc]
                total += 1
                if not wp.isnan(val):
                    measured += 1
    ratio = float(0.0)
    if total > 0:
        ratio = float(measured) / float(total)
    if ratio >= support_ratio:
        filtered_cost[r, c] = cost_map[r, c]
    else:
        filtered_cost[r, c] = wp.float32(wp.nan)


@wp.kernel
def count_obstacles_kernel(
    cost_map: wp.array(dtype=wp.float32, ndim=2),
    obstacle_threshold: wp.float32,
    num_obstacles: wp.array(dtype=wp.int32),
):
    """Atomically count cells whose cost exceeds the obstacle threshold."""
    r, c = wp.tid()
    v = cost_map[r, c]
    if not wp.isnan(v) and v > obstacle_threshold:
        wp.atomic_add(num_obstacles, 0, 1)


@wp.kernel
def inflate_obstacles_kernel(
    cost_map: wp.array(dtype=wp.float32, ndim=2),
    map_height: wp.int32,
    map_width: wp.int32,
    inflation_radius: wp.int32,
    inv_two_sigma_sq: wp.float32,
    obstacle_threshold: wp.float32,
    inflated_cost_map: wp.array(dtype=wp.float32, ndim=2),
):
    """Gaussian-weighted obstacle dilation.

    Only cells above `obstacle_threshold` act as sources.  Each source's
    influence decays with squared distance via exp(-d²/(2σ²)).  The output
    is max(original_cost, max_over_sources(source_cost * weight)), so costs
    are only ever raised, never lowered.
    """
    r, c = wp.tid()
    own = cost_map[r, c]
    best = own
    for dr in range(-inflation_radius, inflation_radius + 1):
        for dc in range(-inflation_radius, inflation_radius + 1):
            nr = r + dr
            nc = c + dc
            if nr >= 0 and nr < map_height and nc >= 0 and nc < map_width:
                v = cost_map[nr, nc]
                if not wp.isnan(v) and v > obstacle_threshold:
                    dist_sq = float(dr * dr + dc * dc)
                    weight = wp.exp(-dist_sq * inv_two_sigma_sq)
                    candidate = v * weight
                    if candidate > best:
                        best = candidate
    inflated_cost_map[r, c] = best
