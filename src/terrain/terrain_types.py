"""
terrain_types.py
----------------
Core data structures for GDMC terrain segmentation.

Design notes:
  - All 2D arrays are indexed [z, x] to match GDPC conventions (z=row, x=col).
  - Coordinates are always integers in Minecraft block space.
  - ZoneType uses IntEnum so it can be stored directly in numpy uint8 arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


def _expand_road_map(road_map: np.ndarray) -> np.ndarray:
    """
    Return a copy of road_map dilated to road widths:
      tertiary  (1) — 3 cells wide (radius 1, symmetric square)
      secondary (2) — 3 cells wide (radius 1, symmetric square)
      primary   (3) — 5 cells wide (radius 2, symmetric square)

    Processing order: lowest class first so higher-class cells overwrite.
    """
    depth, width = road_map.shape
    expanded = np.zeros_like(road_map)

    for cls in (1, 2, 3):
        positions = np.argwhere(road_map == cls)
        if len(positions) == 0:
            continue
        radius = 2 if cls == 3 else 1
        for z, x in positions:
            for dz in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nz, nx = z + dz, x + dx
                    if 0 <= nz < depth and 0 <= nx < width:
                        expanded[nz, nx] = max(expanded[nz, nx], cls)

    return expanded


# ---------------------------------------------------------------------------
# Zone types
# ---------------------------------------------------------------------------


class ZoneType(IntEnum):
    """
    Coarse zone classification for every Voronoi cell in the build area.

    Priority order (lower = evaluated first during hard-override checks):
      OFF_LIMITS < WALL_PERIMETER < RURAL < HARBOR < URBAN
    """

    UNCLASSIFIED = 0
    OFF_LIMITS = 1  # water, steep cliffs, build-area edge buffer
    WALL_PERIMETER = 2  # ring of rural just outside urban convex hull
    RURAL = 3  # farmland, outer buildings, orchards, pens
    HARBOR = 4  # water-adjacent urban boundary → dock / pier
    URBAN = 5  # housing, roads, market, castle, civic buildings


class SubZone(IntEnum):
    """
    Fine-grained label within a coarse ZoneType.
    Set after Phase 5 (special zone detection).
    """

    NONE = 0
    # Rural sub-zones
    FARMLAND = 10
    ORCHARD = 11
    ANIMAL_PEN = 12
    FOREST_BUFFER = 13
    GRAVEYARD = 14  # fenced burial ground just outside the civic core
    MAZE = 15  # ornamental hedge labyrinth (pleasure garden by the keep)
    # Urban sub-zones
    TOWN_CENTER = 20  # market square, fountain
    CIVIC = 21  # church, library, tavern
    RESIDENTIAL = 22  # houses (various professions)
    CASTLE = 23  # castle / keep
    # Infrastructure
    GATE_SITE = 33  # city gate in the wall
    TOWER_SITE = 34  # wall tower
    # Waterfront
    HARBOR_DOCK = 40  # actual pier / boat landing
    BRIDGE_SITE = 42  # river crossing


# ---------------------------------------------------------------------------
# Per-cell terrain features  (stored as a structured array for efficiency)
# ---------------------------------------------------------------------------

FEATURE_DTYPE = np.dtype(
    [
        ("height", np.int16),  # surface Y level
        ("slope", np.float32),  # Sobel gradient magnitude (blocks/block)
        ("roughness", np.float32),  # std-dev of height in 7×7 window
        ("water_dist", np.float32),  # distance (blocks) to nearest water
        ("water_pct", np.float32),  # fraction of 5×5 neighbourhood that is water
        ("tree_density", np.float32),  # log blocks per cell in 5×5 window [0,1]
        ("is_border", np.bool_),  # touches build-area edge
        ("biome_id", np.uint16),  # dominant Minecraft biome ID in cell
    ]
)


# ---------------------------------------------------------------------------
# Voronoi district  (one per seed point after BFS + relaxation)
# ---------------------------------------------------------------------------


@dataclass
class District:
    """
    A single Voronoi cell and all derived metadata.

    Coordinates are stored as numpy arrays for fast bulk operations.
    """

    district_id: int
    seed: tuple[int, int]  # (z, x) initial seed; updated by Lloyd

    # Spatial membership
    cells: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int32))
    # cells[:, 0] = z  cells[:, 1] = x

    # Adjacency: maps neighbour district_id → number of shared-edge cells
    neighbours: dict[int, int] = field(default_factory=dict)

    # Aggregate terrain stats (computed from cells after BFS)
    mean_height: float = 0.0
    height_range: float = 0.0
    mean_slope: float = 0.0
    mean_roughness: float = 0.0
    mean_water_dist: float = 1e9
    water_pct: float = 0.0
    tree_density: float = 0.0
    dominant_biome: str = ""
    is_border: bool = False  # any cell touches build-area edge

    # Suitability scores (continuous, set in Phase 4)
    urban_score: float = 0.0
    rural_score: float = 0.0
    harbor_score: float = 0.0

    # Final classification (set in Phase 4 / 5)
    zone_type: ZoneType = ZoneType.UNCLASSIFIED
    sub_zone: SubZone = SubZone.NONE

    # Betweenness centrality in the district adjacency graph (set in Phase 5)
    centrality: float = 0.0

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def centroid(self) -> tuple[float, float]:
        if self.size == 0:
            return float(self.seed[0]), float(self.seed[1])
        return float(self.cells[:, 0].mean()), float(self.cells[:, 1].mean())


# ---------------------------------------------------------------------------
# SuperDistrict  (contiguous group of same-typed districts)
# ---------------------------------------------------------------------------


@dataclass
class SuperDistrict:
    """
    A contiguous cluster of districts sharing the same ZoneType.
    Used for wall placement, large-scale path routing, and statistics.
    """

    super_id: int
    zone_type: ZoneType
    district_ids: list[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.district_ids)


# ---------------------------------------------------------------------------
# Full terrain segmentation result  (returned by TerrainSegmenter.run())
# ---------------------------------------------------------------------------


@dataclass
class TerrainMap:
    """
    Complete annotated zone map for the build area.

    Shapes are (depth, width) i.e. (z_size, x_size) — matching GDPC.
    """

    # Build-area geometry
    x0: int
    z0: int  # world-space origin
    x1: int
    z1: int  # world-space end (inclusive)

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def depth(self) -> int:
        return self.z1 - self.z0 + 1

    # 2D arrays  [z, x]
    features: Optional[np.ndarray] = None  # structured, dtype=FEATURE_DTYPE
    biome_name_map: Optional[np.ndarray] = (
        None  # object dtype, raw biome strings e.g. "minecraft:cherry_grove"
    )
    district_map: Optional[np.ndarray] = (
        None  # int32, district_id per cell (-1=unassigned)
    )
    zone_map: Optional[np.ndarray] = None  # uint8, ZoneType per cell
    sub_zone_map: Optional[np.ndarray] = None  # uint8, SubZone per cell
    road_map: Optional[np.ndarray] = None  # uint8, RoadClass value per cell (0=none)
    road_network: Optional[object] = None  # RoadNetwork | None  (from road_network.py)
    # Cached expansion of road_map to full rendered width (3–5 cells).
    # Set to None when road_map is assigned; computed lazily on first access.
    _road_map_expanded: Optional[object] = field(
        default=None, init=False, repr=False, compare=False
    )

    water_mask: Optional[np.ndarray] = None
    # MOTION_BLOCKING_NO_PLANTS heightmap [z, x]: one above the top solid or
    # FLUID block.  For water cells this is the true water SURFACE, whereas
    # features["height"] is the ocean floor — use this for anything that
    # must float (boats, docks) rather than sit on the river bed.
    surface_heightmap: Optional[np.ndarray] = None

    # District objects
    districts: list[District] = field(default_factory=list)
    super_districts: list[SuperDistrict] = field(default_factory=list)

    # Wall layout (populated by Phase 7, if run)
    wall_layout: object = None  # WallLayout | None  (from wall_layout.py)
    # Actual placed wall pike cells (set by main.py after WallPlacer runs).
    # Approach-path A* uses this to route around the physical wall.
    wall_cells: set[tuple[int, int]] = field(default_factory=set)

    # Special locations (local [z, x] coordinates)
    town_center: Optional[tuple[int, int]] = None
    castle_site: Optional[tuple[int, int]] = None
    castle_entrance: Optional[tuple[int, int]] = None
    # gate position; set by SchematicBuildingPlacer.place_at_subzone(CASTLE)
    gate_sites: list[tuple[int, int]] = field(default_factory=list)
    harbor_sites: list[tuple[int, int]] = field(default_factory=list)
    bridge_sites: list[tuple[int, int]] = field(default_factory=list)
    windmill_sites: list[tuple[int, int]] = field(default_factory=list)
    # Burial-ground anchors (local [z, x]); set in detect_special_zones.
    # graveyard_site stays the primary one (the crypt digs beneath it).
    graveyard_site: Optional[tuple[int, int]] = None
    graveyard_sites: list[tuple[int, int]] = field(default_factory=list)
    # Hedge-maze anchors (local [z, x]); set in detect_special_zones.
    maze_site: Optional[tuple[int, int]] = None
    maze_sites: list[tuple[int, int]] = field(default_factory=list)

    @property
    def road_map_expanded(self) -> Optional[np.ndarray]:
        """road_map dilated to full rendered road width (3–5 cells)."""
        if self.road_map is None:
            return None
        if self._road_map_expanded is None:
            self._road_map_expanded = _expand_road_map(self.road_map)
        return self._road_map_expanded  # type: ignore[return-value]

    def world_to_local(self, wx: int, wz: int) -> tuple[int, int]:
        return wz - self.z0, wx - self.x0

    def local_to_world(self, lz: int, lx: int) -> tuple[int, int]:
        return self.x0 + lx, self.z0 + lz
