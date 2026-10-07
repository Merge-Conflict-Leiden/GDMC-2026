"""
Road network generation using A* with ACO-inspired pheromone reinforcement.

Three layers build the network from most important to least, sharing a single
cost_mod array that is reinforced after each road is placed. Later roads
therefore prefer corridors established by earlier ones.

  Layer 1 — Primary backbone:  castle/town_center → gates → harbors → bridges
  Layer 2 — Secondary connectors: residential district centroids → nearest road
  Layer 3 — Tertiary rural tracks: rural districts and windmill sites → nearest gate

All coordinates are (z, x) — z is the row index, x is the column index.
"""

from __future__ import annotations

import heapq
import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np

from .terrain_types import SubZone, TerrainMap, ZoneType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEIGHT_COEFF = 1.5  # cost multiplier on (Δheight)² per step — quadratic so
# steep climbs are strongly avoided: paths contour around hills the way a
# human would walk, instead of cutting straight up a slope.
MAX_STEP_HEIGHT = 8  # blocks — cliffs steeper than this are impassable

WATER_COST = 1e9  # truly impassable (off-limits zones, cliff steps)
BRIDGE_WATER_COST = 5.0  # water at a designated bridge site
WATER_CROSS_COST = 35.0  # open water — expensive but finite, so a road whose
# target lies across a river crosses it instead of dead-ending on the bank.
# A ~10-cell river costs like a ~300-cell land detour, so crossings only
# happen where they are genuinely needed.  Water cells along placed paths are
# rendered as spruce-plank bridges by RoadPlacer.

ZONE_COST: dict[ZoneType, float] = {
    ZoneType.URBAN: 1.00,
    ZoneType.WALL_PERIMETER: 1.05,
    ZoneType.RURAL: 1.20,
    ZoneType.HARBOR: 1.40,
    ZoneType.OFF_LIMITS: 1e9,
    ZoneType.UNCLASSIFIED: 1.20,
}

# ACO pheromone reinforcement
REINFORCE_DECAY = 0.70  # multiply cost_mod along placed path by this
REINFORCE_FLOOR = 0.25  # do not let cost_mod fall below this

MAX_SECONDARY_DIST = 200  # skip residential district if nearest road is farther
MAX_TERTIARY_DIST = 120  # skip rural target if nearest road is farther
WALL_CORRIDOR_RADIUS = (
    13  # cells from a gate site that remain passable in the wall-perimeter zone
)

# A* visit limits per layer
MAX_VISITS_PRIMARY = 150_000
MAX_VISITS_SECONDARY = 50_000
MAX_VISITS_TERTIARY = 30_000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class RoadClass(IntEnum):
    TERTIARY = 1  # rural dirt track,  1 cell wide
    SECONDARY = 2  # district connector, 2 cells wide
    PRIMARY = 3  # main spine,         3 cells wide


@dataclass
class Road:
    """A single road segment stored as a sequence of (z, x) cell coordinates."""

    path: list[tuple[int, int]]
    road_class: RoadClass

    @property
    def width(self) -> int:
        return int(self.road_class)


@dataclass
class RoadNetwork:
    """
    Complete road network for the build area.

    road_map is a (depth, width) uint8 array where each cell holds the
    highest RoadClass value of any road passing through it (0 = no road).
    """

    primary_roads: list[Road] = field(default_factory=list)
    secondary_roads: list[Road] = field(default_factory=list)
    tertiary_roads: list[Road] = field(default_factory=list)
    road_map: Optional[np.ndarray] = None  # uint8, 0/1/2/3

    @property
    def all_roads(self) -> list[Road]:
        return self.primary_roads + self.secondary_roads + self.tertiary_roads


# ---------------------------------------------------------------------------
# Wall-barrier helper
# ---------------------------------------------------------------------------


def _build_wall_barrier(
    districts: list,
    tm: TerrainMap,
    depth: int,
    width: int,
) -> set[tuple[int, int]]:
    """Return WALL_PERIMETER cells that are outside every gate corridor.

    Passing this set as *exclude* to A* forces secondary and tertiary roads to
    cross the wall only through gate openings rather than at arbitrary points.
    """
    if not tm.gate_sites:
        return set()

    gate_arr = np.array(tm.gate_sites, dtype=np.float32)  # (G, 2)
    r2 = float(WALL_CORRIDOR_RADIUS**2)
    barrier: set[tuple[int, int]] = set()

    for d in districts:
        if d.zone_type != ZoneType.WALL_PERIMETER or d.size == 0:
            continue
        for cell in d.cells:
            lz, lx = int(cell[0]), int(cell[1])
            diffs = gate_arr - np.array([[lz, lx]], dtype=np.float32)
            if float(np.min(np.sum(diffs**2, axis=1))) > r2:
                barrier.add((lz, lx))

    return barrier


# ---------------------------------------------------------------------------
# Gate-crossing helpers
# ---------------------------------------------------------------------------


def _find_external_exit(
    gate_site: tuple[int, int],
    hub: tuple[int, int] | None,
    feat: np.ndarray,
    dmap: np.ndarray,
    districts: list,
    depth: int,
    width: int,
) -> tuple[int, int] | None:
    """
    Walk outward from gate_site (away from hub) and return the first passable
    cell that is outside the WALL_PERIMETER and URBAN zones — i.e. the natural
    exit point on the rural/outer side of the wall.
    """
    gz, gx = gate_site
    if hub is not None:
        dz, dx = gz - hub[0], gx - hub[1]
    else:
        dz, dx = 1, 0
    norm = math.sqrt(dz * dz + dx * dx) or 1.0
    ndz, ndx = dz / norm, dx / norm

    inner = {ZoneType.URBAN, ZoneType.WALL_PERIMETER, ZoneType.OFF_LIMITS}
    for step in range(1, 70):
        lz = int(round(gz + ndz * step))
        lx = int(round(gx + ndx * step))
        if not (0 <= lz < depth and 0 <= lx < width):
            break
        if float(feat[lz, lx]["water_pct"]) > 0.5:
            continue
        did = int(dmap[lz, lx])
        zone = (
            districts[did].zone_type
            if 0 <= did < len(districts)
            else ZoneType.UNCLASSIFIED
        )
        if zone not in inner:
            return (lz, lx)
    return None


def _extract_gate_sites_from_roads(
    primary_roads: list[Road],
    districts: list,
    tm: TerrainMap,
) -> None:
    """
    Detect where primary roads actually cross the WALL_PERIMETER zone and
    replace tm.gate_sites with those crossing centroids.

    Each cluster of adjacent crossing cells (within a neighbourhood of ~3 cells)
    becomes one gate site.  If no crossings are found the existing gate_sites
    are kept as a fallback.
    """
    wall_cells: set[tuple[int, int]] = set()
    for d in districts:
        if d.zone_type == ZoneType.WALL_PERIMETER and d.size > 0:
            for cell in d.cells:
                wall_cells.add((int(cell[0]), int(cell[1])))

    crossing_cells: set[tuple[int, int]] = set()
    for road in primary_roads:
        for cell in road.path:
            if cell in wall_cells:
                crossing_cells.add(cell)

    if not crossing_cells:
        logger.debug(
            "road_network: no wall crossings in primary roads; keeping original gate_sites."
        )
        return

    # Cluster: cells within a 3-cell neighbourhood belong to the same gate.
    remaining = set(crossing_cells)
    gate_positions: list[tuple[int, int]] = []

    while remaining:
        seed = min(remaining)
        cluster: list[tuple[int, int]] = []
        queue: deque[tuple[int, int]] = deque([seed])
        seen = {seed}
        while queue:
            cell = queue.popleft()
            if cell not in remaining:
                continue
            remaining.discard(cell)
            cluster.append(cell)
            lz, lx = cell
            for dz in range(-3, 4):
                for dx in range(-3, 4):
                    nb = (lz + dz, lx + dx)
                    if nb in remaining and nb not in seen:
                        seen.add(nb)
                        queue.append(nb)

        cz = int(round(sum(c[0] for c in cluster) / len(cluster)))
        cx = int(round(sum(c[1] for c in cluster) / len(cluster)))
        gate_positions.append((cz, cx))

    logger.info(
        "road_network: %d gate(s) detected from primary road crossings: %s",
        len(gate_positions),
        gate_positions,
    )
    tm.gate_sites[:] = gate_positions


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_roads(
    terrain_map: TerrainMap,
    *,
    rng: random.Random | None = None,
) -> RoadNetwork:
    """
    Run all three road-generation layers and return a populated RoadNetwork.

    The RoadNetwork.road_map array (and terrain_map.road_map) are written as
    a side-effect so downstream phases can inspect the road layout.
    """
    if rng is None:
        rng = random.Random(0)

    feat = terrain_map.features
    dmap = terrain_map.district_map
    districts = terrain_map.districts

    if feat is None or dmap is None:
        raise ValueError(
            "terrain_map must have features and district_map populated (run Phase 2+)."
        )

    depth, width = feat.shape
    network = RoadNetwork()

    road_map = np.zeros((depth, width), dtype=np.uint8)
    road_cells: set[tuple[int, int]] = set()
    bridge_set: set[tuple[int, int]] = set(terrain_map.bridge_sites)

    # Shared ACO cost_mod — reinforced after every road across all three layers.
    # Seeded with per-cell wiggle noise (±15%) so A* breaks symmetry and produces
    # naturally curved paths instead of ruler-straight diagonals.
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    cost_mod = np_rng.uniform(0.85, 1.15, (depth, width)).astype(np.float32)

    # Primary roads are routed first — they cross the wall freely and their
    # wall crossings become the authoritative gate positions.
    _build_primary(
        network,
        road_map,
        road_cells,
        cost_mod,
        terrain_map,
        feat,
        dmap,
        districts,
        bridge_set,
    )
    # Detect exactly where primary roads crossed the wall and update gate_sites.
    _extract_gate_sites_from_roads(network.primary_roads, districts, terrain_map)

    # Barrier: wall-perimeter cells outside the detected gate corridors.
    # Built after primary roads so it uses the roads-derived gate positions.
    wall_barrier = _build_wall_barrier(districts, terrain_map, depth, width)

    _build_secondary(
        network,
        road_map,
        road_cells,
        cost_mod,
        terrain_map,
        feat,
        dmap,
        districts,
        bridge_set,
        wall_barrier,
    )
    _build_tertiary(
        network,
        road_map,
        road_cells,
        cost_mod,
        terrain_map,
        feat,
        dmap,
        districts,
        bridge_set,
        wall_barrier,
    )

    network.road_map = road_map
    terrain_map.road_map = road_map

    return network


# ---------------------------------------------------------------------------
# Layer 1 — Primary backbone
# ---------------------------------------------------------------------------


def _build_primary(
    network: RoadNetwork,
    road_map: np.ndarray,
    road_cells: set[tuple[int, int]],
    cost_mod: np.ndarray,
    tm: TerrainMap,
    feat: np.ndarray,
    dmap: np.ndarray,
    districts: list,
    bridge_set: set[tuple[int, int]],
) -> None:
    """Connect castle→town_center, then gates, harbors, and bridge sites."""
    depth, width = feat.shape

    def _snap(pos):
        if pos is None:
            return None
        return _nearest_passable(feat, dmap, districts, pos, radius=10)

    castle = _snap(tm.castle_site)
    town = _snap(tm.town_center)
    harbors = [h for h in (_snap(h) for h in tm.harbor_sites) if h is not None]
    bridges = [b for b in (_snap(b) for b in tm.bridge_sites) if b is not None]

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if castle is not None and town is not None:
        pairs.append((castle, town))

    hub = town if town is not None else castle
    if hub is not None:
        # For each gate_site, route past the wall to an external exit point.
        # The road crosses the wall naturally — wall crossings are detected
        # afterward and become the authoritative gate positions.
        for gs in tm.gate_sites:
            ext = _find_external_exit(gs, hub, feat, dmap, districts, depth, width)
            if ext is not None:
                snapped = _snap(ext)
                if snapped is not None:
                    pairs.append((hub, snapped))

        for h in harbors:
            pairs.append((hub, h))

        for b in bridges:
            pairs.append((hub, b))

    for start, end in pairs:
        path = _astar(
            feat,
            dmap,
            districts,
            start,
            end,
            bridge_set=bridge_set,
            cost_mod=cost_mod,
            max_visits=MAX_VISITS_PRIMARY,
            depth=depth,
            width=width,
        )
        if path is None:
            continue
        road = Road(path=path, road_class=RoadClass.PRIMARY)
        network.primary_roads.append(road)
        _paint_road(road_map, road_cells, path, RoadClass.PRIMARY)
        _reinforce(cost_mod, path)


# ---------------------------------------------------------------------------
# Layer 2 — Secondary connectors
# ---------------------------------------------------------------------------


def _build_secondary(
    network: RoadNetwork,
    road_map: np.ndarray,
    road_cells: set[tuple[int, int]],
    cost_mod: np.ndarray,
    tm: TerrainMap,
    feat: np.ndarray,
    dmap: np.ndarray,
    districts: list,
    bridge_set: set[tuple[int, int]],
    wall_barrier: set[tuple[int, int]] = frozenset(),
) -> None:
    """Connect each residential district centroid to the nearest existing road cell."""
    depth, width = feat.shape

    residential = [
        d
        for d in districts
        if d.zone_type == ZoneType.URBAN
        and d.sub_zone == SubZone.RESIDENTIAL
        and d.size > 0
    ]

    if not road_cells:
        return

    road_arr = np.array(list(road_cells), dtype=np.int32)

    for d in residential:
        cz, cx = d.centroid
        centroid = _nearest_passable(
            feat, dmap, districts, (int(round(cz)), int(round(cx))), radius=10
        )
        if centroid is None:
            continue

        diff = road_arr - np.array([[centroid[0], centroid[1]]], dtype=np.float32)
        dists = np.hypot(diff[:, 0], diff[:, 1])
        idx = int(np.argmin(dists))

        if float(dists[idx]) > MAX_SECONDARY_DIST:
            continue

        nearest_road = (int(road_arr[idx, 0]), int(road_arr[idx, 1]))
        path = _astar(
            feat,
            dmap,
            districts,
            centroid,
            nearest_road,
            bridge_set=bridge_set,
            cost_mod=cost_mod,
            max_visits=MAX_VISITS_SECONDARY,
            depth=depth,
            width=width,
            exclude=wall_barrier or None,
        )
        if path is None:
            continue

        road = Road(path=path, road_class=RoadClass.SECONDARY)
        network.secondary_roads.append(road)
        _paint_road(road_map, road_cells, path, RoadClass.SECONDARY)
        _reinforce(cost_mod, path)

        # Rebuild so subsequent districts can connect to newly placed roads
        road_arr = np.array(list(road_cells), dtype=np.int32)


# ---------------------------------------------------------------------------
# Layer 3 — Tertiary rural tracks
# ---------------------------------------------------------------------------


def _build_tertiary(
    network: RoadNetwork,
    road_map: np.ndarray,
    road_cells: set[tuple[int, int]],
    cost_mod: np.ndarray,
    tm: TerrainMap,
    feat: np.ndarray,
    dmap: np.ndarray,
    districts: list,
    bridge_set: set[tuple[int, int]],
    wall_barrier: set[tuple[int, int]] = frozenset(),
) -> None:
    """Connect each rural target to the nearest existing road cell (one path each)."""
    depth, width = feat.shape

    rural_targets: list[tuple[int, int]] = []
    rural_sub_targets = {SubZone.FARMLAND, SubZone.ANIMAL_PEN, SubZone.ORCHARD}

    for d in districts:
        if (
            d.zone_type == ZoneType.RURAL
            and d.sub_zone in rural_sub_targets
            and d.size > 0
        ):
            cz, cx = d.centroid
            snapped = _nearest_passable(
                feat, dmap, districts, (int(round(cz)), int(round(cx))), radius=10
            )
            if snapped is not None:
                rural_targets.append(snapped)

    for wz, wx in tm.windmill_sites:
        snapped = _nearest_passable(feat, dmap, districts, (wz, wx), radius=10)
        if snapped is not None:
            rural_targets.append(snapped)

    if not rural_targets or not road_cells:
        return

    for target in rural_targets:
        road_arr = np.array(list(road_cells), dtype=np.int32)
        dists = np.hypot(road_arr[:, 0] - target[0], road_arr[:, 1] - target[1])
        idx = int(np.argmin(dists))
        if float(dists[idx]) > MAX_TERTIARY_DIST:
            continue
        nearest = (int(road_arr[idx, 0]), int(road_arr[idx, 1]))

        path = _astar(
            feat,
            dmap,
            districts,
            target,
            nearest,
            bridge_set=bridge_set,
            cost_mod=cost_mod,
            max_visits=MAX_VISITS_TERTIARY,
            depth=depth,
            width=width,
            exclude=wall_barrier or None,
        )
        if path is None:
            continue

        road = Road(path=path, road_class=RoadClass.TERTIARY)
        network.tertiary_roads.append(road)
        _paint_road(road_map, road_cells, path, RoadClass.TERTIARY)
        _reinforce(cost_mod, path)


# ---------------------------------------------------------------------------
# ACO reinforcement
# ---------------------------------------------------------------------------


def _reinforce(cost_mod: np.ndarray, path: list[tuple[int, int]]) -> None:
    """Reduce cost_mod along path so future roads prefer established corridors."""
    for z, x in path:
        cost_mod[z, x] = max(REINFORCE_FLOOR, cost_mod[z, x] * REINFORCE_DECAY)


# ---------------------------------------------------------------------------
# A* (terrain-aware, 8-connected)
# ---------------------------------------------------------------------------

_NEIGHBORS_8 = [
    (-1, -1, 1.4142),
    (-1, 0, 1.0),
    (-1, 1, 1.4142),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, 1.4142),
    (1, 0, 1.0),
    (1, 1, 1.4142),
]


def _astar(
    feat: np.ndarray,
    dmap: np.ndarray,
    districts: list,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    bridge_set: set[tuple[int, int]],
    cost_mod: np.ndarray | None,
    max_visits: int,
    depth: int,
    width: int,
    exclude: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    """8-connected A* from start to goal. Returns path as (z, x) list or None.

    *exclude* is an optional set of (z, x) cells that are treated as
    impassable — used to prevent paths from cutting through building footprints.
    """
    sz, sx = start
    gz, gx = goal

    if not (0 <= sz < depth and 0 <= sx < width):
        return None
    if not (0 <= gz < depth and 0 <= gx < width):
        return None

    total = depth * width
    INF = float("inf")

    g_score = np.full(total, INF, dtype=np.float64)
    came_from = np.full(total, -1, dtype=np.int64)

    start_idx = sz * width + sx
    goal_idx = gz * width + gx

    g_score[start_idx] = 0.0
    open_heap: list[tuple[float, int]] = []
    heapq.heappush(open_heap, (_octile(sz, sx, gz, gx), start_idx))

    visits = 0
    while open_heap:
        f_cur, cur_idx = heapq.heappop(open_heap)
        visits += 1
        if visits > max_visits:
            break

        if cur_idx == goal_idx:
            return _reconstruct(came_from, cur_idx, width)

        cz, cx = divmod(cur_idx, width)
        g_cur = g_score[cur_idx]

        if f_cur > g_cur + _octile(cz, cx, gz, gx) + 1e-9:
            continue

        for dz, dx, base_cost in _NEIGHBORS_8:
            nz, nx = cz + dz, cx + dx
            if not (0 <= nz < depth and 0 <= nx < width):
                continue
            if exclude is not None and (nz, nx) in exclude:
                continue

            nidx = nz * width + nx
            cell_cost = _terrain_cost(
                feat, dmap, districts, (cz, cx), (nz, nx), bridge_set
            )
            if cell_cost >= WATER_COST:
                continue

            move_cost = base_cost * cell_cost
            if cost_mod is not None:
                move_cost *= float(cost_mod[nz, nx])

            tentative_g = g_cur + move_cost
            if tentative_g < g_score[nidx]:
                g_score[nidx] = tentative_g
                came_from[nidx] = cur_idx
                h = _octile(nz, nx, gz, gx)
                heapq.heappush(open_heap, (tentative_g + h, nidx))

    return None


def _terrain_cost(
    feat: np.ndarray,
    dmap: np.ndarray,
    districts: list,
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
    bridge_set: set[tuple[int, int]],
) -> float:
    """
    Cost of moving from pos_a to adjacent pos_b.

    Water is expensive (WATER_CROSS_COST, cheaper at designated bridge sites)
    but never impassable, so roads can bridge a river to reach the far bank.
    OFF_LIMITS zones are impassable. Height change adds a quadratic penalty
    scaled by HEIGHT_COEFF so paths prefer contouring over climbing.
    """
    bz, bx = pos_b
    cell_b = feat[bz, bx]

    if float(cell_b["water_pct"]) > 0.5:
        # No zone or height penalty on water: feat["height"] of a water cell
        # is the RIVER BOTTOM (OCEAN_FLOOR heightmap), which is irrelevant
        # for a bridge deck spanning at bank level.
        return BRIDGE_WATER_COST if pos_b in bridge_set else WATER_CROSS_COST

    did_b = int(dmap[bz, bx])
    if 0 <= did_b < len(districts):
        zone_cost = ZONE_COST.get(districts[did_b].zone_type, 1.20)
        if zone_cost >= WATER_COST:
            return WATER_COST
    else:
        zone_cost = 1.20

    az, ax = pos_a
    if float(feat[az, ax]["water_pct"]) > 0.5:
        # Stepping off a bridge onto the bank: skip the height check — the
        # water cell's bottom height vs the bank is not a real climb.
        return zone_cost
    dh = abs(float(cell_b["height"]) - float(feat[az, ax]["height"]))
    if dh > MAX_STEP_HEIGHT:
        return WATER_COST
    return zone_cost * (1.0 + HEIGHT_COEFF * dh * dh)


def _octile(az: int, ax: int, bz: int, bx: int) -> float:
    dz = abs(bz - az)
    dx = abs(bx - ax)
    return max(dz, dx) + (math.sqrt(2) - 1.0) * min(dz, dx)


def _reconstruct(
    came_from: np.ndarray, goal_idx: int, width: int
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    idx = goal_idx
    while idx >= 0:
        z, x = divmod(int(idx), width)
        path.append((z, x))
        prev = int(came_from[idx])
        if prev == idx or prev < 0:
            break
        idx = prev
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nearest_passable(
    feat: np.ndarray,
    dmap: np.ndarray,
    districts: list,
    pos: tuple[int, int],
    radius: int = 10,
) -> tuple[int, int] | None:
    """Spiral search for the nearest non-water, non-OFF_LIMITS cell within radius."""
    depth, width = feat.shape
    z0, x0 = pos

    for r in range(radius + 1):
        if r == 0:
            candidates = [(z0, x0)]
        else:
            candidates = []
            for i in range(-r, r + 1):
                candidates.append((z0 - r, x0 + i))
                candidates.append((z0 + r, x0 + i))
            for i in range(-r + 1, r):
                candidates.append((z0 + i, x0 - r))
                candidates.append((z0 + i, x0 + r))

        for cz, cx in candidates:
            if not (0 <= cz < depth and 0 <= cx < width):
                continue
            if float(feat[cz, cx]["water_pct"]) > 0.5:
                continue
            did = int(dmap[cz, cx])
            if (
                0 <= did < len(districts)
                and districts[did].zone_type == ZoneType.OFF_LIMITS
            ):
                continue
            return (cz, cx)

    return None


def _paint_road(
    road_map: np.ndarray,
    road_cells: set[tuple[int, int]],
    path: list[tuple[int, int]],
    road_class: RoadClass,
) -> None:
    """Mark path cells in road_map at the given class level (higher wins)."""
    depth, width = road_map.shape
    val = int(road_class)
    for z, x in path:
        if 0 <= z < depth and 0 <= x < width:
            if road_map[z, x] < val:
                road_map[z, x] = val
            road_cells.add((z, x))
