"""
voronoi.py
----------
Phase 3 — Zone seeding & spatial partitioning.

Algorithm:
  1. Scatter N candidate seed points, rejecting those in high-slope /
     high-water / border cells.  Uses a Poisson-disk-inspired minimum
     spacing to avoid clustering.
  2. BFS from all seeds simultaneously (multi-source BFS = Voronoi fill).
  3. Lloyd relaxation: move each seed to its cell centroid, re-run BFS.
     Repeat LLOYD_ITERS times.  This produces evenly-spaced, organic cells.
  4. Build adjacency graph: two districts are neighbours if any of their
     cells share an edge.  Record shared-edge count.

Design decisions:
  - BFS (flood-fill) rather than scipy.Voronoi because BFS respects the
    grid topology and is O(n) in the number of cells.  scipy.Voronoi in
    Euclidean space doesn't account for water/cliff blocking (you'd still
    need a flood-fill afterwards to correct).  We need grid BFS anyway for
    adjacency, so do it once.
  - Poisson-disk rejection replaces pure random seeding from Tome/grimoire.
    Pure random seeding with Lloyd converges but wastes iterations.
    Poisson-disk gives a better start and needs fewer Lloyd iterations.
  - We seed INTO valid (buildable-ish) terrain but do NOT exclude water
    cells from the BFS expansion, so districts that straddle water/land are
    classified correctly in Phase 4 based on their aggregate water_pct.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Optional

import numpy as np

from .terrain_types import District, TerrainMap

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLOYD_ITERS = 3  # number of Lloyd relaxation passes
MIN_SEED_DIST = 12  # Poisson-disk minimum distance between seeds (cells)
MIN_DIST_SCALE = (
    0.55  # actual min_seed_dist = MIN_SEED_DIST + rand * scale*MIN_SEED_DIST
)
# Seed rejection thresholds
MAX_SEED_SLOPE = 5.0  # reject seed if slope exceeds this (blocks/cell)
MAX_SEED_WATER = 0.5  # reject seed if local water fraction > 50 %
MIN_DISTRICT_SIZE = 30  # districts smaller than this are merged in Phase 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_districts(
    terrain_map: TerrainMap,
    n_districts: int | None = None,
    *,
    rng: random.Random | None = None,
    seed_bias_toward_center: bool = True,
) -> list[District]:
    """
    Run the full Phase-3 pipeline (seed → BFS → Lloyd × n → adjacency).

    Parameters
    ----------
    terrain_map   : must have .features populated by feature_extractor.
    n_districts   : target count; if None, auto-computed from build area.
    rng           : random.Random instance for reproducibility.
    seed_bias_toward_center : upweight central cells slightly during seed
                              selection (avoids degenerate edge-heavy patterns).

    Returns
    -------
    List of District objects; also writes terrain_map.district_map and
    terrain_map.districts in-place.
    """
    feat = terrain_map.features
    depth, width = feat.shape

    if rng is None:
        rng = random.Random(42)

    # Auto-compute district count: ~1 per 500 cells.
    # Finer granularity improves zone boundary resolution and rural sub-zoning.
    if n_districts is None:
        n_districts = max(12, int(round(depth * width / 500)))

    # -- Seed placement -------------------------------------------------------
    seeds = _poisson_disk_seeds(feat, n_districts, rng, seed_bias_toward_center)

    # -- BFS (Voronoi fill) + Lloyd relaxation --------------------------------
    district_map = _voronoi_bfs(seeds, depth, width)
    for _ in range(LLOYD_ITERS):
        seeds = _lloyd_step(seeds, district_map, depth, width)
        district_map = _voronoi_bfs(seeds, depth, width)

    # -- Build District objects -----------------------------------------------
    districts = _build_districts(seeds, district_map, feat, terrain_map.biome_name_map)

    # -- Adjacency graph -------------------------------------------------------
    _compute_adjacency(districts, district_map, depth, width)

    # -- Write results --------------------------------------------------------
    terrain_map.district_map = district_map
    terrain_map.districts = districts
    return districts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _poisson_disk_seeds(
    feat: np.ndarray,
    n: int,
    rng: random.Random,
    center_bias: bool,
) -> list[tuple[int, int]]:
    """
    Approximate Poisson-disk sampling of seed points.

    Strategy: repeatedly sample uniformly (with optional center-bias weight),
    reject if too close to any already-placed seed OR if the cell fails
    terrain validity.  After MAX_ATTEMPTS failures, relax the minimum distance
    slightly.  This guarantees we always reach n seeds.
    """
    depth, width = feat.shape
    seeds: list[tuple[int, int]] = []
    min_d = MIN_SEED_DIST
    MAX_ATTEMPTS = 400

    # Pre-compute a validity mask vectorised (was a 65k-iteration Python loop
    # for a 256x256 build area).
    valid_mask = (
        (~feat["is_border"])
        & (feat["slope"] <= MAX_SEED_SLOPE)
        & (feat["water_pct"] <= MAX_SEED_WATER)
    )
    valid_indices = np.argwhere(valid_mask)  # shape (K, 2)

    if len(valid_indices) == 0:
        # Degenerate map — fall back to grid seeds ignoring terrain.
        step = max(1, int(math.sqrt(depth * width / n)))
        for z in range(step // 2, depth, step):
            for x in range(step // 2, width, step):
                seeds.append((z, x))
                if len(seeds) >= n:
                    break
            if len(seeds) >= n:
                break
        return seeds[:n]

    # Optional center bias: weight each valid cell by exp(-dist_to_center²/2σ²)
    if center_bias and len(valid_indices) > 0:
        cz, cx = depth / 2.0, width / 2.0
        dz = valid_indices[:, 0] - cz
        dx = valid_indices[:, 1] - cx
        sigma = min(depth, width) * 0.4
        weights = np.exp(-(dz**2 + dx**2) / (2 * sigma**2))
        weights /= weights.sum()
    else:
        weights = None

    fails = 0
    while len(seeds) < n:
        # Sample a candidate
        idx = rng.choices(range(len(valid_indices)), weights=weights, k=1)[0]
        cz, cx = int(valid_indices[idx, 0]), int(valid_indices[idx, 1])

        # Check Poisson-disk distance to all existing seeds
        too_close = False
        for sz, sx in seeds:
            if (sz - cz) ** 2 + (sx - cx) ** 2 < min_d**2:
                too_close = True
                break

        if not too_close:
            seeds.append((cz, cx))
            fails = 0
        else:
            fails += 1
            if fails >= MAX_ATTEMPTS:
                # Relax minimum distance and retry
                min_d = max(4, int(min_d * 0.85))
                fails = 0

    return seeds


def _voronoi_bfs(
    seeds: list[tuple[int, int]],
    depth: int,
    width: int,
) -> np.ndarray:
    """
    Multi-source BFS from all seeds simultaneously.
    Each cell is labelled with the index (0-based) of the nearest seed.
    Returns int32 array of shape (depth, width); -1 = unreachable.
    """
    district_map = np.full((depth, width), -1, dtype=np.int32)
    queue: deque[tuple[int, int, int]] = deque()  # (z, x, district_id)

    for did, (sz, sx) in enumerate(seeds):
        if 0 <= sz < depth and 0 <= sx < width:
            if district_map[sz, sx] == -1:
                district_map[sz, sx] = did
                queue.append((sz, sx, did))

    neighbours_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while queue:
        z, x, did = queue.popleft()
        for dz, dx in neighbours_offsets:
            nz, nx = z + dz, x + dx
            if 0 <= nz < depth and 0 <= nx < width and district_map[nz, nx] == -1:
                district_map[nz, nx] = did
                queue.append((nz, nx, did))

    return district_map


def _lloyd_step(
    seeds: list[tuple[int, int]],
    district_map: np.ndarray,
    depth: int,
    width: int,
) -> list[tuple[int, int]]:
    """
    Move each seed to the integer centroid of its assigned cells.
    Seeds that have no cells (can happen at map boundaries) stay put.
    """
    n = len(seeds)
    sum_z = np.zeros(n, dtype=np.int64)
    sum_x = np.zeros(n, dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)

    # Vectorised accumulation
    zz, xx = np.mgrid[0:depth, 0:width]
    flat_d = district_map.ravel()
    flat_z = zz.ravel()
    flat_x = xx.ravel()

    valid = flat_d >= 0
    np.add.at(sum_z, flat_d[valid], flat_z[valid])
    np.add.at(sum_x, flat_d[valid], flat_x[valid])
    np.add.at(counts, flat_d[valid], 1)

    new_seeds = []
    for did in range(n):
        if counts[did] > 0:
            nz = int(round(sum_z[did] / counts[did]))
            nx = int(round(sum_x[did] / counts[did]))
            nz = max(0, min(depth - 1, nz))
            nx = max(0, min(width - 1, nx))
            new_seeds.append((nz, nx))
        else:
            new_seeds.append(seeds[did])  # keep original if orphaned

    return new_seeds


def _build_districts(
    seeds: list[tuple[int, int]],
    district_map: np.ndarray,
    feat: np.ndarray,
    biome_name_map: Optional[np.ndarray],
) -> list[District]:
    """
    Construct District objects and compute aggregate terrain statistics.
    """
    n = len(seeds)
    districts = [District(district_id=i, seed=seeds[i]) for i in range(n)]

    # Collect cell coordinates per district
    depth, width = district_map.shape
    zz, xx = np.mgrid[0:depth, 0:width]
    for did in range(n):
        mask = district_map == did
        cells_z = zz[mask].astype(np.int32)
        cells_x = xx[mask].astype(np.int32)
        districts[did].cells = np.stack([cells_z, cells_x], axis=1)

    # Compute aggregate stats
    for d in districts:
        if d.size == 0:
            continue
        cz, cx = d.cells[:, 0], d.cells[:, 1]
        cell_feat = feat[cz, cx]

        heights = cell_feat["height"].astype(np.float64)
        d.mean_height = float(heights.mean())
        d.height_range = float(heights.max() - heights.min())
        d.mean_slope = float(cell_feat["slope"].mean())
        d.mean_roughness = float(cell_feat["roughness"].mean())
        d.mean_water_dist = float(cell_feat["water_dist"].mean())
        d.water_pct = float(cell_feat["water_pct"].mean())
        d.tree_density = float(cell_feat["tree_density"].mean())
        d.is_border = bool(cell_feat["is_border"].any())

        # Dominant biome: mode of raw biome strings
        if biome_name_map is not None and len(d.cells) > 0:
            cz2, cx2 = d.cells[:, 0], d.cells[:, 1]
            cell_biomes = biome_name_map[cz2, cx2]
            unique, counts = np.unique(cell_biomes, return_counts=True)
            d.dominant_biome = str(unique[counts.argmax()])

    return districts


def _compute_adjacency(
    districts: list[District],
    district_map: np.ndarray,
    depth: int,
    width: int,
) -> None:
    """
    Fill district.neighbours: {neighbour_id: shared_edge_count}.
    We scan only rightward and downward to avoid double-counting.
    """
    # Rightward neighbours (same z, x vs x+1)
    left = district_map[:, :-1]
    right = district_map[:, 1:]
    diff = left != right
    pairs_right = np.stack([left[diff], right[diff]], axis=1)

    # Downward neighbours (z vs z+1, same x)
    top = district_map[:-1, :]
    bottom = district_map[1:, :]
    diff2 = top != bottom
    pairs_down = np.stack([top[diff2], bottom[diff2]], axis=1)

    all_pairs = np.vstack([pairs_right, pairs_down])

    for a, b in all_pairs:
        if a < 0 or b < 0:
            continue
        districts[a].neighbours[b] = districts[a].neighbours.get(b, 0) + 1
        districts[b].neighbours[a] = districts[b].neighbours.get(a, 0) + 1
