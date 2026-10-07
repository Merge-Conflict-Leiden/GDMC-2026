"""
Geometry & placement scoring helpers (pure, module-level).

Everything here is stateless: footprint rotation into world space, the
candidate search that scores anchor/building/rotation triples against
terrain and road context, and small grid utilities shared by all placement
modes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from terrain.terrain_types import TerrainMap
from utils import rotate_offset

from .model import Building

MAX_SLOPE = 4.0
MAX_HEIGHT_DIFF = 8  # max height range across a building footprint
MAX_WATER_PCT = 0.05  # reject footprints with any cell this wet
ROAD_BONUS = 8
BOUNDARY_BONUS = 2  # weight for jigsaw-to-open-boundary alignment score
OPENNESS_BONUS = 2  # weight for facing open ground (not into a hillside/wall)

# Gate→road approach scoring (opt-in via score_gate_road, used for the
# castle).  Penalty weights per unit of each connection-quality measure;
# the blocked weight dominates so a gate whose road line crosses the
# building's own footprint (i.e. faces away from the road) is ruled out.
GATE_ROAD_DIST_W = 0.10  # per block of straight-line gate→road distance
GATE_ROAD_ELEV_W = 0.80  # per block of height gap between gate and road end
GATE_ROAD_BUMP_W = 2.00  # per block of mean ridge/dip along the connection
GATE_ROAD_BLOCKED_W = 5.0  # per line cell crossing the building's footprint


def bresenham_heights(start: int, end: int, n: int) -> list[int]:
    """
    Return *n* integer heights stepping evenly from *start* to *end*.

    Uses Bresenham's line algorithm so consecutive values differ by at most
    ±1 block and the full delta is spread as uniformly as possible.  When the
    path is shorter than |end - start| steps, the sequence reaches as close to
    *end* as a ±1-per-step constraint allows.
    """
    if n <= 0:
        return []
    if n == 1:
        return [start]
    delta = end - start
    sign = 1 if delta >= 0 else -1
    dy = abs(delta)
    dx = n - 1
    D = 2 * dy - dx
    cur = start
    result: list[int] = []
    for i in range(n):
        result.append(cur)
        if i < n - 1 and cur != end:
            if D > 0:
                cur += sign
                D -= 2 * dx
            D += 2 * dy
    return result


def world_footprint(
    anchor: tuple[int, int],
    footprint: frozenset[tuple[int, int]],
    W: int,
    D: int,
    direction: int,
) -> list[tuple[int, int]]:
    alz, alx = anchor
    return [
        (alz + oz, alx + ox)
        for sx, sz in footprint
        for ox, oz in [rotate_offset(sx, sz, W, D, direction)]
    ]


def _boundary_alignment_score(
    anchor: tuple[int, int],
    b: Building,
    direction: int,
    fp: list[tuple[int, int]],
    road_map: Optional[np.ndarray],
    occupied: set[tuple[int, int]],
    depth: int,
    width: int,
) -> float:
    """
    Score how well jigsaw entrance(s) align with the footprint's open boundary.

    Builds the set of footprint perimeter cells (those with at least one free
    cardinal neighbour outside the footprint and not occupied), biases toward
    perimeter cells that adjoin a road, then returns the negative sum of the
    minimum Chebyshev distances from each jigsaw connector to the nearest
    target perimeter cell.

    Score of 0 means every entrance sits exactly on the road-facing open
    boundary (ideal).  Increasingly negative values indicate the entrance is
    buried inside the footprint or faces a blocked / back side.

    Falls back to door_x/door_z when the building has no jigsaw offsets.
    """
    fp_set = set(fp)

    # Perimeter: footprint cells that have at least one free cardinal neighbour.
    perimeter: list[tuple[int, int]] = []
    for lz, lx in fp_set:
        for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nlz, nlx = lz + dz, lx + dx
            if (
                (nlz, nlx) not in fp_set
                and (nlz, nlx) not in occupied
                and 0 <= nlz < depth
                and 0 <= nlx < width
            ):
                perimeter.append((lz, lx))
                break

    if not perimeter:
        return 0.0

    # Prefer perimeter cells that border a road; fall back to all perimeter cells
    # when no road is nearby (rural buildings, windmills, etc.).
    if road_map is not None:
        road_facing = [
            (lz, lx)
            for lz, lx in perimeter
            if any(
                0 <= lz + dz < road_map.shape[0]
                and 0 <= lx + dx < road_map.shape[1]
                and road_map[lz + dz, lx + dx] > 0
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
        ]
        targets = road_facing if road_facing else perimeter
    else:
        targets = perimeter

    # Use jigsaw connectors as entrance points; fall back to door position.
    entrance_pts = (
        [(sx, sz) for sx, sy, sz in b.jigsaw_offsets]
        if b.jigsaw_offsets
        else [(b.door_x, b.door_z)]
    )

    total = 0.0
    for sx, sz in entrance_pts:
        ox, oz = rotate_offset(sx, sz, b.W, b.D, direction)
        elz, elx = anchor[0] + oz, anchor[1] + ox
        min_dist = min(max(abs(elz - plz), abs(elx - plx)) for plz, plx in targets)
        total -= float(min_dist)

    return total


def _entrance_openness(
    anchor: tuple[int, int],
    b: Building,
    direction: int,
    fp: list[tuple[int, int]],
    feat: np.ndarray,
    depth: int,
    width: int,
    look: int = 5,
) -> float:
    """
    Score how open the terrain is directly in front of the entrance.

    Samples a few cells straight out from the door and returns a negative
    penalty when the ground rises into a wall/hillside (or runs off the build
    edge) — ~0 when the front is open.  Nearer obstructions weigh more, and
    gentle slopes (rise ≤ 2) are ignored so only genuine walls are avoided.
    This keeps entrances — the castle gate especially — from facing a cliff.
    """
    if not fp:
        return 0.0
    ox, oz = rotate_offset(b.door_x, b.door_z, b.W, b.D, direction)
    elz, elx = anchor[0] + oz, anchor[1] + ox
    fcz = sum(z for z, _ in fp) / len(fp)
    fcx = sum(x for _, x in fp) / len(fp)
    # Outward cardinal = from the footprint centre toward the entrance.
    if abs(elz - fcz) >= abs(elx - fcx):
        ddz, ddx = (1 if elz >= fcz else -1), 0
    else:
        ddz, ddx = 0, (1 if elx >= fcx else -1)

    fp_ref = sum(int(feat[z, x]["height"]) for z, x in fp) / len(fp)
    penalty = 0.0
    for k in range(1, look + 1):
        cz, cx = elz + ddz * k, elx + ddx * k
        weight = (look - k + 1) / look  # nearer blockage matters more
        if not (0 <= cz < depth and 0 <= cx < width):
            penalty += weight  # facing off the build edge is not "open"
            continue
        rise = int(feat[cz, cx]["height"]) - fp_ref
        if rise > 2:  # a genuine wall / hillside, not a gentle slope
            penalty += (rise - 2) * weight
    return -penalty


def _bresenham_line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Grid cells on the straight line from *a* to *b* (both inclusive)."""
    (z0, x0), (z1, x1) = a, b
    dz, dx = abs(z1 - z0), abs(x1 - x0)
    sz, sx = (1 if z1 >= z0 else -1), (1 if x1 >= x0 else -1)
    err = dz - dx
    cells: list[tuple[int, int]] = []
    z, x = z0, x0
    while True:
        cells.append((z, x))
        if (z, x) == (z1, x1):
            return cells
        e2 = 2 * err
        if e2 > -dx:
            err -= dx
            z += sz
        if e2 < dz:
            err += dz
            x += sx


def _gate_road_approach_score(
    anchor: tuple[int, int],
    b: Building,
    direction: int,
    fp_set: set[tuple[int, int]],
    feat: np.ndarray,
    road_cells: np.ndarray,
    road_heights: np.ndarray,
    depth: int,
    width: int,
) -> float:
    """
    Score the future approach track from the entrance to the road network.

    For each jigsaw entrance (falling back to the door), pick the road cell
    the approach builder itself would target (nearest by distance AND grade,
    mirroring _connect_jigsaw), then rate the straight-line connection:

      - distance         — shorter approach tracks are better;
      - elevation gap    — gate and road at similar height connect flush;
      - bumpiness        — mean deviation of the terrain along the line from
                           an even Bresenham ramp: the part terracing or the
                           track's clamping would otherwise have to hide;
      - footprint cross  — line cells inside the building's own footprint
                           mean the gate faces AWAY from the road and the
                           real path must detour all the way around.

    Returns 0 for a perfect flush connection, increasingly negative
    otherwise.  This is what actually orients the castle: the gate ends up
    on the side with a road AND the least elevation to bridge.
    """
    entrance_pts = (
        [(sx, sz) for sx, _sy, sz in b.jigsaw_offsets]
        if b.jigsaw_offsets
        else [(b.door_x, b.door_z)]
    )

    total = 0.0
    scored = 0
    for sx, sz in entrance_pts:
        ox, oz = rotate_offset(sx, sz, b.W, b.D, direction)
        elz, elx = anchor[0] + oz, anchor[1] + ox
        if not (0 <= elz < depth and 0 <= elx < width):
            continue
        gate_h = float(feat[elz, elx]["height"])

        # Same target choice as _connect_jigsaw: a slightly farther road
        # cell near the gate's grade beats the nearest one up a bank.
        dists = (road_cells[:, 0] - elz) ** 2 + (road_cells[:, 1] - elx) ** 2
        elev_gap = road_heights - gate_h
        ti = int(np.argmin(dists + 6.0 * elev_gap * elev_gap))
        tlz, tlx = int(road_cells[ti, 0]), int(road_cells[ti, 1])
        road_h = float(road_heights[ti])

        line = _bresenham_line((elz, elx), (tlz, tlx))
        ramp = bresenham_heights(int(round(gate_h)), int(round(road_h)), len(line))
        bump = 0.0
        blocked = 0
        for (clz, clx), ramp_h in zip(line[1:], ramp[1:]):
            if (clz, clx) in fp_set:
                blocked += 1
                continue
            dev = abs(int(feat[clz, clx]["height"]) - ramp_h)
            if dev > 1:  # ±1 rides with the ramp; more needs smoothing
                bump += dev - 1
        bump_avg = bump / max(len(line) - 1, 1)

        total -= (
            GATE_ROAD_DIST_W * float(np.sqrt(dists[ti]))
            + GATE_ROAD_ELEV_W * abs(road_h - gate_h)
            + GATE_ROAD_BUMP_W * bump_avg
            + GATE_ROAD_BLOCKED_W * blocked
        )
        scored += 1

    return total / scored if scored else 0.0


def dilated_cells(
    cells: np.ndarray,
    radius: int,
    depth: int,
    width: int,
) -> np.ndarray:
    """Return *cells* grown outward by *radius* (Manhattan), clipped to the map."""
    from scipy.ndimage import binary_dilation

    mask = np.zeros((depth, width), dtype=bool)
    mask[cells[:, 0], cells[:, 1]] = True
    mask = binary_dilation(mask, iterations=radius)
    return np.argwhere(mask)


def find_best_placement(
    district,
    buildings: list[Building],
    feat: np.ndarray,
    road_map: Optional[np.ndarray],
    occupied: set[tuple[int, int]],
    depth: int,
    width: int,
    *,
    prefer_town: bool = False,
    tm: Optional[TerrainMap] = None,
    max_height_diff: int = MAX_HEIGHT_DIFF,
    max_candidates: int = 250,
    score_gate_road: bool = False,
) -> Optional[tuple[tuple[int, int], Building, int]]:
    district_cells: set[tuple[int, int]] = {
        (int(r[0]), int(r[1])) for r in district.cells
    }
    candidates = _sort_by_road_proximity(
        [c for c in district_cells if c not in occupied],
        road_map,
    )
    sorted_b = sorted(buildings, key=lambda b: len(b.footprint), reverse=True)

    best_score: float = float("-inf")
    best = None

    town_cz = town_cx = None
    if prefer_town and tm is not None and tm.town_center is not None:
        town_cz, town_cx = tm.town_center

    # Gate→road scoring precomputation (castle): the road raster and its
    # terrain heights, shared across every candidate evaluation.
    gr_cells: Optional[np.ndarray] = None
    gr_heights: Optional[np.ndarray] = None
    if score_gate_road and road_map is not None:
        rc = np.argwhere(road_map > 0)
        if len(rc) > 0:
            gr_cells = rc
            gr_heights = feat[rc[:, 0], rc[:, 1]]["height"].astype(np.float64)

    for anchor in candidates[:max_candidates]:
        if anchor in occupied:
            continue
        for b in sorted_b:
            for direction in range(4):
                fp = world_footprint(anchor, b.footprint, b.W, b.D, direction)
                if not fp:
                    continue
                # Bounds/occupancy use the all-y shadow so roof overhangs
                # can't reach over the footprint into other structures.
                sh = (
                    fp
                    if b.shadow == b.footprint
                    else world_footprint(anchor, b.shadow, b.W, b.D, direction)
                )
                if any(not (0 <= lz < depth and 0 <= lx < width) for lz, lx in sh):
                    continue
                if any(c in occupied for c in sh):
                    continue
                if any(
                    float(feat[lz, lx]["water_pct"]) > MAX_WATER_PCT for lz, lx in fp
                ):
                    continue

                heights = [int(feat[lz, lx]["height"]) for lz, lx in fp]
                height_range = max(heights) - min(heights)
                if height_range > max_height_diff:
                    continue  # terrain too uneven; skip rather than level aggressively

                ox, oz = rotate_offset(b.door_x, b.door_z, b.W, b.D, direction)
                ent_lz = max(0, min(depth - 1, anchor[0] + oz))
                ent_lx = max(0, min(width - 1, anchor[1] + ox))
                road_score = _road_adj(ent_lz, ent_lx, road_map, depth, width)

                # Town-facing bonus: prefer direction whose entrance points toward town.
                town_bonus = 0.0
                if prefer_town and town_cz is not None:
                    dz = town_cz - anchor[0]
                    dx = town_cx - anchor[1]
                    dominant = 0 if abs(dz) >= abs(dx) else 1
                    expected_dir = (
                        (0 if dz >= 0 else 2)
                        if dominant == 0
                        else (3 if dx >= 0 else 1)
                    )
                    town_bonus = 5.0 if direction == expected_dir else 0.0

                slope_penalty = sum(
                    max(0.0, float(feat[lz, lx]["slope"]) - MAX_SLOPE) for lz, lx in fp
                )

                boundary_score = _boundary_alignment_score(
                    anchor, b, direction, fp, road_map, occupied, depth, width
                )

                openness = _entrance_openness(
                    anchor, b, direction, fp, feat, depth, width
                )

                # Gate→road approach quality: dominates orientation for the
                # castle (already weighted internally, so added at 1.0).
                gate_road = 0.0
                if gr_cells is not None and gr_heights is not None:
                    gate_road = _gate_road_approach_score(
                        anchor,
                        b,
                        direction,
                        set(fp),
                        feat,
                        gr_cells,
                        gr_heights,
                        depth,
                        width,
                    )

                score = (
                    road_score * ROAD_BONUS
                    - height_range
                    + town_bonus
                    - slope_penalty
                    + boundary_score * BOUNDARY_BONUS
                    + openness * OPENNESS_BONUS
                    + gate_road
                )
                if score > best_score:
                    best_score = score
                    best = (anchor, b, direction)

    return best


def find_fallback_placement(
    district,
    buildings: list[Building],
    feat: np.ndarray,
    depth: int,
    width: int,
) -> Optional[tuple[tuple[int, int], Building, int]]:
    """
    Last-resort placement for a must-place building (the castle) when the
    scored search fails everywhere.

    Occupancy, slope, and height caps are all ignored — terracing/levelling
    will bulldoze what it must — and only the map bounds stay hard.  Among
    all anchors it minimizes water under the footprint first and terrain
    unevenness second, so the forced placement still lands on the driest,
    flattest spot available.
    """
    cells = [(int(r[0]), int(r[1])) for r in district.cells]
    if not cells or not buildings:
        return None

    # Evenly subsample anchors and footprint cells so huge dilated anchor
    # sets stay tractable while still covering the whole area.
    stride = max(1, len(cells) // 250)

    best = None
    best_key: Optional[tuple[int, int]] = None
    for anchor in cells[::stride]:
        for b in buildings:
            for direction in range(4):
                # Bounds check via the rotated box corners (exact for a
                # rectangular schematic box, O(1) per candidate).
                corner_offsets = [
                    rotate_offset(sx, sz, b.W, b.D, direction)
                    for sx, sz in (
                        (0, 0),
                        (b.W - 1, 0),
                        (0, b.D - 1),
                        (b.W - 1, b.D - 1),
                    )
                ]
                if any(
                    not (0 <= anchor[0] + oz < depth and 0 <= anchor[1] + ox < width)
                    for ox, oz in corner_offsets
                ):
                    continue

                fp = world_footprint(anchor, b.footprint, b.W, b.D, direction)[::3]
                if not fp:
                    continue
                wet = sum(
                    1
                    for lz, lx in fp
                    if float(feat[lz, lx]["water_pct"]) > MAX_WATER_PCT
                )
                heights = [int(feat[lz, lx]["height"]) for lz, lx in fp]
                key = (wet, max(heights) - min(heights))
                if best_key is None or key < best_key:
                    best_key = key
                    best = (anchor, b, direction)

    return best


def mark_footprint(
    anchor: tuple[int, int],
    b: Building,
    direction: int,
    occupied: set[tuple[int, int]],
    pad: int = 1,
) -> None:
    # Mark the all-y shadow (not just the y=0 footprint) so overhanging
    # roofs/canopies of this building are protected from later placements.
    for lz, lx in world_footprint(anchor, b.shadow, b.W, b.D, direction):
        for dlz in range(-pad, pad + 1):
            for dlx in range(-pad, pad + 1):
                occupied.add((lz + dlz, lx + dlx))


def _road_adj(
    lz: int, lx: int, road_map: Optional[np.ndarray], depth: int, width: int
) -> int:
    if road_map is None:
        return 0
    return sum(
        1
        for dz in range(-1, 2)
        for dx in range(-1, 2)
        if 0 <= lz + dz < depth
        and 0 <= lx + dx < width
        and road_map[lz + dz, lx + dx] > 0
    )


def _sort_by_road_proximity(
    candidates: list[tuple[int, int]],
    road_map: Optional[np.ndarray],
) -> list[tuple[int, int]]:
    if not candidates or road_map is None:
        return candidates
    road_cells = np.argwhere(road_map > 0).astype(np.float32)
    if len(road_cells) == 0:
        return candidates
    arr = np.array(candidates, dtype=np.float32)  # (N, 2)
    dists = np.min(
        np.sum((road_cells[None] - arr[:, None]) ** 2, axis=2), axis=1
    )  # (N,)
    order = np.argsort(dists)
    return [candidates[i] for i in order]


def road_occupied(road_map: Optional[np.ndarray]) -> set[tuple[int, int]]:
    occ: set[tuple[int, int]] = set()
    if road_map is not None:
        zs, xs = np.where(road_map > 0)
        for lz, lx in zip(zs.tolist(), xs.tolist()):
            occ.add((lz, lx))
    return occ
