"""
wall_layout.py
--------------
Phase 7 — Wall layout generation.

Extracts the outer boundary of the fortified inner zone (CASTLE + CIVIC +
TOWN_CENTER sub-zones, plus the enclosing WALL_PERIMETER ring) and simplifies
it to straight segments using Douglas-Peucker.

  GATE_WIDTH = 25   gate opening (no wall placed in this span)

Gate positions are assigned by projecting each gate_site onto the nearest
polygon edge.  One gate per edge; edges shorter than GATE_WIDTH are skipped.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from skimage.measure import find_contours

from .terrain_types import SubZone, TerrainMap, ZoneType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structural constants
# ---------------------------------------------------------------------------

GATE_WIDTH = 25  # gate opening width in blocks

# D-P epsilon for contour simplification (cells).  Larger → fewer, longer edges.
_DP_EPSILON = 7.0

# Maximum perpendicular distance (cells) from a gate_site to the nearest
# polygon edge for the gate to be assigned to that edge.
_GATE_SNAP_MAX_DIST = 60


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class WallLayout:
    """
    Closed polygon of wall corner positions plus optional gate information.

    corners[i] → corners[(i+1) % n] is one straight wall run.

    For each run i:
      - towers are placed at corners[i] and corners[(i+1) % n]
      - gate_t[i] is the normalised parameter (0 = corners[i], 1 = next corner)
        at which the gate centre sits, or None if there is no gate on this run
    """

    corners: List[Tuple[int, int]] = field(default_factory=list)
    gate_t: List[Optional[float]] = field(default_factory=list)  # len == len(corners)

    @property
    def n(self) -> int:
        return len(self.corners)

    @property
    def is_valid(self) -> bool:
        return self.n >= 3 and len(self.gate_t) == self.n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_wall_layout(terrain_map: TerrainMap) -> Optional[WallLayout]:
    """
    Build a WallLayout from the terrain segmentation data.

    Returns None if the map has no fortified zone or if the boundary cannot
    be resolved into a usable polygon.
    """
    # 1. Binary mask of the area the wall should surround.
    mask = _build_fortified_mask(terrain_map)
    if mask is None or not mask.any():
        logger.warning("WallLayout: no fortified zone found.")
        return None

    # 2. Outer boundary as an ordered list of integer cell coordinates.
    raw_poly = _extract_boundary_polygon(mask)
    if len(raw_poly) < 6:
        logger.warning(
            "WallLayout: boundary polygon has too few points (%d).", len(raw_poly)
        )
        return None

    # 3. Simplify to straight-ish segments (Douglas-Peucker).
    corners = _dp_simplify_closed(raw_poly, epsilon=_DP_EPSILON)
    if len(corners) < 3:
        logger.warning(
            "WallLayout: simplified polygon has too few corners (%d).", len(corners)
        )
        return None

    logger.info("WallLayout: %d corners after simplification.", len(corners))

    # 4. Assign gate positions.
    gate_t = _assign_gates(corners, terrain_map.gate_sites)

    layout = WallLayout(corners=corners, gate_t=gate_t)
    n_gates = sum(1 for g in gate_t if g is not None)
    logger.info("WallLayout: %d corners, %d gate(s).", layout.n, n_gates)
    return layout


# ---------------------------------------------------------------------------
# Step 1 — Binary mask of the fortified zone
# ---------------------------------------------------------------------------


def _build_fortified_mask(terrain_map: TerrainMap) -> Optional[np.ndarray]:
    """
    True for every cell that belongs to the fortified inner zone or the
    surrounding wall-perimeter ring.
    """
    dm = terrain_map.district_map
    if dm is None:
        return None

    depth, width = dm.shape
    mask = np.zeros((depth, width), dtype=np.bool_)

    inner_subs = {SubZone.CASTLE, SubZone.CIVIC, SubZone.TOWN_CENTER}

    for d in terrain_map.districts:
        if d.size == 0:
            continue
        if d.zone_type == ZoneType.WALL_PERIMETER or d.sub_zone in inner_subs:
            mask[d.cells[:, 0], d.cells[:, 1]] = True

    return mask


# ---------------------------------------------------------------------------
# Step 2 — Extract outer boundary as an ordered polygon
# ---------------------------------------------------------------------------


def _extract_boundary_polygon(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return the outer boundary of `mask` as an ordered list of (z, x) integer
    cell coordinates, using skimage's contour tracing (Marching Squares).

    The contour runs at the midpoint between True and False pixels (level 0.5).
    """
    contours = find_contours(mask.astype(np.float32), level=0.5)
    if not contours:
        return []

    # Longest contour = outer boundary.
    contour = max(contours, key=len)

    # Round to nearest integer cell, remove consecutive duplicates.
    points: List[Tuple[int, int]] = []
    for z, x in contour:
        p = (int(round(z)), int(round(x)))
        if not points or p != points[-1]:
            points.append(p)

    # Remove the closing duplicate (find_contours closes the contour).
    if len(points) > 1 and points[-1] == points[0]:
        points = points[:-1]

    return points


# ---------------------------------------------------------------------------
# Step 3 — Douglas-Peucker simplification (closed polygon)
# ---------------------------------------------------------------------------


def _dp_simplify_closed(
    points: List[Tuple[int, int]],
    epsilon: float,
) -> List[Tuple[int, int]]:
    """
    Simplify a closed polygon using Douglas-Peucker.

    Treats the polygon as a closed loop by finding the two points farthest
    from each other as the initial split points.
    """
    if len(points) <= 3:
        return list(points)

    # Find the pair of points farthest apart → use as the split for a
    # "maximum inscribed chord" approach to handle the closed nature.
    n = len(points)
    arr = np.array(points, dtype=np.float64)

    # Cheap approximation: find index of point farthest from points[0].
    d0 = np.hypot(arr[:, 0] - arr[0, 0], arr[:, 1] - arr[0, 1])
    far_idx = int(np.argmax(d0))

    # Split into two open chains and simplify each.
    chain1 = points[: far_idx + 1]  # 0 … far_idx
    chain2 = points[far_idx:] + [points[0]]  # far_idx … 0 (wrapped)

    half1 = _dp_simplify_open(chain1, epsilon)
    half2 = _dp_simplify_open(chain2, epsilon)

    # Merge: half1 ends at far_idx, half2 starts at far_idx → remove duplicate.
    merged = half1[:-1] + half2[:-1]  # both share the junction points
    return merged if merged else list(points)


def _dp_simplify_open(
    points: List[Tuple[int, int]],
    epsilon: float,
) -> List[Tuple[int, int]]:
    """Standard Douglas-Peucker on an open polyline."""
    if len(points) <= 2:
        return list(points)

    sz, sx = points[0]
    ez, ex = points[-1]
    dz, dx = ez - sz, ex - sx
    line_len_sq = dz * dz + dx * dx

    max_dist = 0.0
    max_idx = 0

    for i in range(1, len(points) - 1):
        pz, px = points[i]
        if line_len_sq == 0.0:
            d = math.sqrt((pz - sz) ** 2 + (px - sx) ** 2)
        else:
            t = ((pz - sz) * dz + (px - sx) * dx) / line_len_sq
            t = max(0.0, min(1.0, t))
            d = math.sqrt((pz - sz - t * dz) ** 2 + (px - sx - t * dx) ** 2)
        if d > max_dist:
            max_dist = d
            max_idx = i

    if max_dist > epsilon:
        left = _dp_simplify_open(points[: max_idx + 1], epsilon)
        right = _dp_simplify_open(points[max_idx:], epsilon)
        return left[:-1] + right
    return [points[0], points[-1]]


# ---------------------------------------------------------------------------
# Step 4 — Assign gate positions to polygon edges
# ---------------------------------------------------------------------------


def _assign_gates(
    corners: List[Tuple[int, int]],
    gate_sites: List[Tuple[int, int]],
) -> List[Optional[float]]:
    """
    For each gate_site, find the nearest polygon edge and store a normalised
    parameter t (0 = start corner, 1 = end corner) for the gate centre.

    An edge receives a gate only if:
      - Perpendicular distance from the gate_site to the edge ≤ _GATE_SNAP_MAX_DIST.
      - Edge length ≥ GATE_WIDTH, leaving room for the opening.
      - No gate has already been assigned to that edge.
    """
    n = len(corners)
    gate_t: List[Optional[float]] = [None] * n
    assigned_edges: set[int] = set()

    for gz, gx in gate_sites:
        best_edge = -1
        best_dist = float("inf")
        best_t = 0.5

        for i in range(n):
            p1 = corners[i]
            p2 = corners[(i + 1) % n]

            dz = p2[0] - p1[0]
            dx = p2[1] - p1[1]
            edge_len_sq = dz * dz + dx * dx
            if edge_len_sq == 0:
                continue

            edge_len = math.sqrt(edge_len_sq)

            t = ((gz - p1[0]) * dz + (gx - p1[1]) * dx) / edge_len_sq
            t = max(0.0, min(1.0, t))
            proj_z = p1[0] + t * dz
            proj_x = p1[1] + t * dx
            dist = math.sqrt((gz - proj_z) ** 2 + (gx - proj_x) ** 2)

            if (
                dist < best_dist
                and dist <= _GATE_SNAP_MAX_DIST
                and edge_len >= GATE_WIDTH
                and i not in assigned_edges
            ):
                best_dist = dist
                best_edge = i
                best_t = t

        if best_edge >= 0:
            p1 = corners[best_edge]
            p2 = corners[(best_edge + 1) % n]
            edge_len = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            # Clamp so the full gate opening fits within the edge.
            half_gate = GATE_WIDTH / 2
            gate_center_dist = max(
                half_gate,
                min(edge_len - half_gate, best_t * edge_len),
            )
            gate_t[best_edge] = gate_center_dist / edge_len
            assigned_edges.add(best_edge)
            logger.debug(
                "WallLayout: gate on edge %d at t=%.3f (dist=%.1f).",
                best_edge,
                gate_t[best_edge],
                best_dist,
            )

    return gate_t
