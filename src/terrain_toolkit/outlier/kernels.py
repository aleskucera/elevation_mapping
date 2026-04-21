from __future__ import annotations

import warp as wp

wp.init()


@wp.kernel
def knn_mean_dist_kernel(
    grid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    search_radius: wp.float32,
    k: wp.int32,
    big_init: wp.float32,
    sensor_origin: wp.vec3,
    range_eps: wp.float32,
    scratch: wp.array(dtype=wp.float32, ndim=2),
    mean_dist: wp.array(dtype=wp.float32),
    valid: wp.array(dtype=wp.int32),
):
    """Per-point mean k-NN distance, normalized by range to `sensor_origin`.

    Output is `mean_knn_dist / max(||p - origin||, range_eps)` — scale-invariant
    against lidar's linearly-increasing point spacing with range. `scratch` is a
    per-thread (N, k) workspace holding the k smallest squared distances seen so
    far. `valid[i] = 0` when fewer than k neighbors lie within `search_radius`.
    """
    i = wp.tid()
    p = points[i]

    # Initialize this thread's scratch row.
    for j in range(k):
        scratch[i, j] = big_init

    max_idx = int(0)
    max_val = big_init
    count = int(0)
    r2 = search_radius * search_radius

    neighbors = wp.hash_grid_query(grid, p, search_radius)
    for index in neighbors:
        if index == i:
            continue
        q = points[index]
        diff = q - p
        d2 = wp.dot(diff, diff)
        if d2 > r2:
            continue
        count += 1
        if d2 < max_val:
            scratch[i, max_idx] = d2
            # Find the new max in O(k).
            max_val = scratch[i, 0]
            max_idx = int(0)
            for j in range(1, k):
                v = scratch[i, j]
                if v > max_val:
                    max_val = v
                    max_idx = j

    if count < k:
        mean_dist[i] = float(0.0)
        valid[i] = 0
        return

    s = float(0.0)
    for j in range(k):
        s += wp.sqrt(scratch[i, j])
    mean = s / float(k)

    to_sensor = p - sensor_origin
    r = wp.sqrt(wp.dot(to_sensor, to_sensor))
    if r < range_eps:
        r = range_eps
    mean_dist[i] = mean / r
    valid[i] = 1


@wp.kernel
def reduce_mean_stats_kernel(
    mean_dist: wp.array(dtype=wp.float32),
    valid: wp.array(dtype=wp.int32),
    out_sum: wp.array(dtype=wp.float32),
    out_sum_sq: wp.array(dtype=wp.float32),
    out_count: wp.array(dtype=wp.int32),
):
    """Atomic reduction of mean_dist over the valid subset → (sum, sum_sq, count)."""
    i = wp.tid()
    if valid[i] == 1:
        v = mean_dist[i]
        wp.atomic_add(out_sum, 0, v)
        wp.atomic_add(out_sum_sq, 0, v * v)
        wp.atomic_add(out_count, 0, 1)


@wp.kernel
def compact_inliers_kernel(
    points: wp.array(dtype=wp.vec3),
    mean_dist: wp.array(dtype=wp.float32),
    valid: wp.array(dtype=wp.int32),
    threshold: wp.float32,
    out_counter: wp.array(dtype=wp.int32),
    out_points: wp.array(dtype=wp.vec3),
):
    """Write surviving points (valid and mean_dist ≤ threshold) to a compact buffer."""
    i = wp.tid()
    if valid[i] == 1 and mean_dist[i] <= threshold:
        slot = wp.atomic_add(out_counter, 0, 1)
        out_points[slot] = points[i]
