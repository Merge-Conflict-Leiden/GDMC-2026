"""
garden.py
---------
Circular garden/farmland generator for the GDMC medieval city pipeline.

Adapted from the original GardenGenerator to use the TerrainMap API so no
live heightmap fetch is required.  Heights come from terrain_map.features,
which is already populated by Phase 1 of the segmenter.

Two modes (both via the same generate() method)
------------------------------------------------
Near-building  — pass building_lw / building_ld so the search starts far
                 enough from the building to leave a visible gap.
Standalone     — leave building_lw / building_ld at 0; the garden is centred
                 right on / near the given local (lz, lx).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np
from gdpc.block import Block
from gdpc.editor import Editor

from utils import log

if TYPE_CHECKING:
    from terrain.terrain_types import TerrainMap


@dataclass
class GardenConfig:
    min_radius: int = 3
    max_radius: int = 7
    padding: int = 3
    attempts: int = 30  # raised: more tries now each one is cheaper
    min_crop_density: float = 0.2
    max_crop_density: float = 0.8
    max_height_variance: int = 3
    min_farm_fraction: float = 0.75  # min share of inner circle that must be free
    min_ring_fraction: float = 0.80  # min share of ring cells that must be free


class GardenGenerator:
    """Generate circular farmland/garden plots using TerrainMap height data."""

    WATER_RADIUS = 4

    _EDGE_COBBLE: List[Block] = [
        Block("minecraft:cobblestone_slab", {"type": "bottom"}),
        Block("minecraft:cobblestone_stairs", {"facing": "north"}),
        Block("minecraft:cobblestone_stairs", {"facing": "south"}),
        Block("minecraft:cobblestone_stairs", {"facing": "east"}),
        Block("minecraft:cobblestone_stairs", {"facing": "west"}),
        Block("minecraft:mossy_cobblestone_slab", {"type": "bottom"}),
        Block("minecraft:mossy_cobblestone_stairs", {"facing": "north"}),
        Block("minecraft:mossy_cobblestone_stairs", {"facing": "south"}),
        Block("minecraft:mossy_cobblestone_stairs", {"facing": "east"}),
        Block("minecraft:mossy_cobblestone_stairs", {"facing": "west"}),
    ]
    _EDGE_LEAVES: List[Block] = [
        Block("minecraft:oak_leaves", {"persistent": "true"}),
    ]
    _EDGE_SPRUCE: List[Block] = [
        Block("minecraft:spruce_slab", {"type": "bottom"}),
        Block("minecraft:spruce_stairs", {"facing": "north"}),
        Block("minecraft:spruce_stairs", {"facing": "south"}),
        Block("minecraft:spruce_stairs", {"facing": "east"}),
        Block("minecraft:spruce_stairs", {"facing": "west"}),
        Block(
            "minecraft:spruce_trapdoor",
            {"half": "bottom", "open": "false", "facing": "north"},
        ),
    ]
    _EDGE_SANDSTONE: List[Block] = [
        Block("minecraft:sandstone_slab", {"type": "bottom"}),
        Block("minecraft:sandstone_stairs", {"facing": "north"}),
        Block("minecraft:sandstone_stairs", {"facing": "south"}),
        Block("minecraft:sandstone_stairs", {"facing": "east"}),
        Block("minecraft:sandstone_stairs", {"facing": "west"}),
        Block("minecraft:smooth_sandstone_slab", {"type": "bottom"}),
    ]

    _CROP_SETS: List[List[Tuple[str, Optional[int]]]] = [
        [("minecraft:beetroots", 3), ("minecraft:sweet_berry_bush", 3)],
        [("minecraft:carrots", 7), ("minecraft:potatoes", 7)],
        [("minecraft:wheat", 7), ("minecraft:short_grass", None)],
        [
            ("minecraft:large_fern", None),
            ("minecraft:fern", None),
            ("minecraft:orange_tulip", None),
            ("minecraft:pink_tulip", None),
            ("minecraft:white_tulip", None),
            ("minecraft:red_tulip", None),
        ],
        [
            ("minecraft:azalea", None),
            ("minecraft:rose_bush", None),
            ("minecraft:peony", None),
            ("minecraft:flowering_azalea", None),
            ("minecraft:lilac", None),
        ],
    ]

    def __init__(self, config: GardenConfig | None = None) -> None:
        self.config = config or GardenConfig()

    # ------------------------------------------------------------------
    # Terrain helpers
    # ------------------------------------------------------------------

    def _h(self, feat: np.ndarray, lz: int, lx: int) -> int:
        """Return surface Y (one ABOVE the surface block) for local cell."""
        lz = max(0, min(lz, feat.shape[0] - 1))
        lx = max(0, min(lx, feat.shape[1] - 1))
        return int(feat["height"][lz, lx])

    def _suitable(self, feat: np.ndarray, lz: int, lx: int, radius: int) -> bool:
        depth, width = feat.shape
        heights = [
            int(feat["height"][lz + dz, lx + dx])
            for dz in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dz * dz + dx * dx <= radius * radius
            and 0 <= lz + dz < depth
            and 0 <= lx + dx < width
        ]
        return bool(heights) and (
            max(heights) - min(heights) <= self.config.max_height_variance
        )

    # ------------------------------------------------------------------
    # Spot finder
    # ------------------------------------------------------------------

    def _find_spot(
        self,
        feat: np.ndarray,
        center_lz: int,
        center_lx: int,
        building_lw: int,
        building_ld: int,
        radius: int,
        occupied: Set[Tuple[int, int]],
        allowed_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Find a suitable garden center near (center_lz, center_lx).

        A candidate is accepted only when:
          • terrain height variance within the circle ≤ max_height_variance
          • ≥ min_farm_fraction of inner-circle cells pass the occupancy/zone check
          • ≥ min_ring_fraction of ring cells pass the occupancy/zone check

        For standalone mode (building_lw=building_ld=0) the search starts AT
        the trigger cell and expands outward; no forced offset is applied.
        For near-building mode a minimum distance from the building is kept.
        """
        depth, width = feat.shape
        r2 = radius * radius
        outer_r2 = (radius + 1) ** 2

        # Precompute total cell counts so fraction thresholds are relative.
        farm_total = sum(
            1
            for dlz in range(-radius, radius + 1)
            for dlx in range(-radius, radius + 1)
            if dlz * dlz + dlx * dlx <= r2
        )
        ring_total = sum(
            1
            for dlz in range(-radius - 1, radius + 2)
            for dlx in range(-radius - 1, radius + 2)
            if r2 < dlz * dlz + dlx * dlx <= outer_r2
        )
        min_farm = max(5, int(farm_total * self.config.min_farm_fraction))
        min_ring = max(radius * 2, int(ring_total * self.config.min_ring_fraction))

        def _ok(lz: int, lx: int) -> bool:
            if not (0 <= lz < depth and 0 <= lx < width):
                return False
            if (lz, lx) in occupied:
                return False
            if allowed_cells is not None and (lz, lx) not in allowed_cells:
                return False
            return True

        def _candidate_ok(clz: int, clx: int) -> bool:
            """Full acceptance test for a candidate center."""
            # Must have room for ring (radius+1 from every edge)
            if not (
                radius + 1 <= clz < depth - radius - 1
                and radius + 1 <= clx < width - radius - 1
            ):
                return False
            if not self._suitable(feat, clz, clx, radius):
                return False
            farm_ok = sum(
                1
                for dlz in range(-radius, radius + 1)
                for dlx in range(-radius, radius + 1)
                if dlz * dlz + dlx * dlx <= r2 and _ok(clz + dlz, clx + dlx)
            )
            if farm_ok < min_farm:
                return False
            ring_ok = sum(
                1
                for dlz in range(-radius - 1, radius + 2)
                for dlx in range(-radius - 1, radius + 2)
                if r2 < dlz * dlz + dlx * dlx <= outer_r2 and _ok(clz + dlz, clx + dlx)
            )
            return ring_ok >= min_ring

        standalone = building_lw == 0 and building_ld == 0

        if standalone:
            # For standalone: search close to the trigger cell (no forced offset).
            # Expand outward from 0 distance up to radius+padding.
            search_r = radius + self.config.padding
            candidates: List[Tuple[int, int]] = []
            for dz in range(-search_r, search_r + 1):
                for dx in range(-search_r, search_r + 1):
                    candidates.append((center_lz + dz, center_lx + dx))
            candidates.sort(
                key=lambda p: (
                    (p[0] - center_lz) ** 2
                    + (p[1] - center_lx) ** 2
                    + random.random() * 4
                )
            )
            for lz, lx in candidates[: self.config.attempts]:
                if _candidate_ok(lz, lx):
                    return lz, lx
            return None

        # Near-building mode: keep a distance from the building centre.
        building_size = max(building_lw, building_ld, 1)
        min_dist = radius + building_size // 2 + self.config.padding
        max_dist = max(min_dist + 2, building_size * 2 + self.config.padding)

        candidates = []
        for dz in range(-max_dist, max_dist + 1):
            for dx in range(-max_dist, max_dist + 1):
                if max(abs(dz), abs(dx)) < min_dist:
                    continue
                candidates.append((center_lz + dz, center_lx + dx))

        candidates.sort(
            key=lambda p: (
                (p[0] - center_lz) ** 2 + (p[1] - center_lx) ** 2 + random.random()
            )
        )

        for lz, lx in candidates[: self.config.attempts]:
            if _candidate_ok(lz, lx):
                return lz, lx
        return None

    # ------------------------------------------------------------------
    # Water irrigation
    # ------------------------------------------------------------------

    def _water_safe(
        self,
        feat: np.ndarray,
        lz: int,
        lx: int,
        farm_y: int,
        farmland_lookup: Dict[Tuple[int, int], int],
    ) -> bool:
        depth, width = feat.shape
        for ddz, ddx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nlz, nlx = lz + ddz, lx + ddx
            nbr_y = farmland_lookup.get((nlz, nlx))
            if nbr_y is None:
                if 0 <= nlz < depth and 0 <= nlx < width:
                    if self._h(feat, nlz, nlx) - 1 < farm_y:
                        return False
            elif nbr_y < farm_y:
                return False
        return True

    def _irrigate(
        self,
        feat: np.ndarray,
        terrain_map: "TerrainMap",
        editor: Editor,
        farmland_positions: List[
            Tuple[int, int, int]
        ],  # (wx, wy, wz) wy = surface block y
    ) -> None:
        by_y: Dict[int, Set[Tuple[int, int]]] = {}
        for wx, wy, wz in farmland_positions:
            by_y.setdefault(wy, set()).add((wx, wz))

        farm_lookup_wx: Dict[Tuple[int, int], int] = {
            (wx, wz): wy for wx, wy, wz in farmland_positions
        }

        water_block = Block("minecraft:water")
        base_block = Block("minecraft:dirt")

        for farm_y, xz_set in by_y.items():
            uncovered: Set[Tuple[int, int]] = set(xz_set)

            while uncovered:
                best_pos = None
                best_cover: Set[Tuple[int, int]] = set()

                for cx, cz in uncovered:
                    cover = {
                        (fx, fz)
                        for fx, fz in uncovered
                        if (fx - cx) ** 2 + (fz - cz) ** 2 <= self.WATER_RADIUS**2
                    }
                    if len(cover) > len(best_cover):
                        best_cover = cover
                        best_pos = (cx, cz)

                if best_pos is None:
                    break

                wx_w, wz_w = best_pos
                lz_w, lx_w = terrain_map.world_to_local(wx_w, wz_w)

                local_lookup: Dict[Tuple[int, int], int] = {}
                for (fx, fz), fy in farm_lookup_wx.items():
                    flz, flx = terrain_map.world_to_local(fx, fz)
                    local_lookup[(flz, flx)] = fy

                if not self._water_safe(feat, lz_w, lx_w, farm_y, local_lookup):
                    placed = False
                    for fx, fz in best_cover:
                        flz, flx = terrain_map.world_to_local(fx, fz)
                        if self._water_safe(feat, flz, flx, farm_y, local_lookup):
                            wx_w, wz_w = fx, fz
                            placed = True
                            break
                    if not placed:
                        uncovered -= best_cover
                        continue

                editor.placeBlock((wx_w, farm_y - 1, wz_w), base_block)
                editor.placeBlock((wx_w, farm_y, wz_w), water_block)
                uncovered -= best_cover

    # ------------------------------------------------------------------
    # Crops & edges
    # ------------------------------------------------------------------

    def _edge_palette(self, biome: str) -> List[Block]:
        if "desert" in biome or "badlands" in biome:
            return self._EDGE_SANDSTONE
        if "snowy" in biome or "taiga" in biome:
            return self._EDGE_SPRUCE
        return random.choice([self._EDGE_COBBLE, self._EDGE_LEAVES, self._EDGE_SPRUCE])

    def _place_crop(
        self, editor: Editor, farmland: List[Tuple[int, int, int]], biome: str
    ) -> None:
        if "jungle" in biome or "swamp" in biome:
            eligible = [self._CROP_SETS[3], self._CROP_SETS[4]]
        elif "plains" in biome or "meadow" in biome:
            eligible = [self._CROP_SETS[2], self._CROP_SETS[3]]
        else:
            eligible = self._CROP_SETS

        raw_set = random.choice(eligible)

        def make_crop(name: str, max_age: Optional[int]) -> Block:
            if max_age is None:
                return Block(name)
            return Block(name, {"age": str(random.randint(1, max_age))})

        crops = [make_crop(n, a) for n, a in raw_set]
        primary = random.choice(crops)

        for wx, wy, wz in farmland:
            if random.random() > self.config.max_crop_density:
                continue
            crop = primary if random.random() < 0.7 else random.choice(crops)
            editor.placeBlock((wx, wy + 1, wz), crop)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        editor: Editor,
        terrain_map: "TerrainMap",
        center_lz: int,
        center_lx: int,
        occupied: Set[Tuple[int, int]],
        *,
        building_lw: int = 0,
        building_ld: int = 0,
        allowed_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> Tuple[bool, int, int, int]:
        """
        Try to place a circular garden near (center_lz, center_lx).

        When building_lw / building_ld > 0 the garden is kept at a comfortable
        distance from the building so it reads as a backyard rather than
        overlapping.  When both are 0 the garden is planted as close to the
        centre as fits.

        *allowed_cells* — if provided, farmland and ring blocks are only placed
        on cells in this set.  Use it to enforce zone boundaries (walls, roads)
        so a garden whose circle extends outside its zone gets cleanly clipped
        rather than partially placed across the boundary.

        The center returned from _find_spot has at least min_farm_fraction of
        farmland cells and min_ring_fraction of ring cells free.  The placed
        garden is then *shaped to the free space*: farmland only goes on cells
        whose full 8-neighbourhood is free, and the outline ring is drawn on
        the boundary of that region — so the border is always complete even
        when part of the ideal circle is blocked.

        Returns (success, center_lz, center_lx, radius).
        """
        feat = terrain_map.features
        if feat is None:
            return False, 0, 0, 0

        radius = random.randint(self.config.min_radius, self.config.max_radius)
        depth, width = feat.shape

        pos = self._find_spot(
            feat,
            center_lz,
            center_lx,
            building_lw,
            building_ld,
            radius,
            occupied,
            allowed_cells,
        )
        if pos is None:
            return False, 0, 0, 0

        glz, glx = pos
        outer_r2 = (radius + 1) ** 2

        def _cell_ok(lz: int, lx: int) -> bool:
            if not (0 <= lz < depth and 0 <= lx < width):
                return False
            if (lz, lx) in occupied:
                return False
            if allowed_cells is not None and (lz, lx) not in allowed_cells:
                return False
            return True

        # ── Form the garden to the available space ─────────────────────
        # Collect every free cell within the outer circle, keep only the
        # connected region around the centre, then split it into interior
        # (farmland) and outline (edge decoration).  The outline is thereby
        # always complete: where part of the circle is blocked, the garden
        # shrinks to fit instead of being cut off mid-shape.
        free: Set[Tuple[int, int]] = set()
        for dlz in range(-radius - 1, radius + 2):
            for dlx in range(-radius - 1, radius + 2):
                if dlz * dlz + dlx * dlx > outer_r2:
                    continue
                lz, lx = glz + dlz, glx + dlx
                if _cell_ok(lz, lx):
                    free.add((lz, lx))
        if not free:
            return False, 0, 0, 0

        seed = min(free, key=lambda c: (c[0] - glz) ** 2 + (c[1] - glx) ** 2)
        region: Set[Tuple[int, int]] = {seed}
        stack = [seed]
        while stack:
            sz_, sx_ = stack.pop()
            for ddz, ddx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (sz_ + ddz, sx_ + ddx)
                if nb in free and nb not in region:
                    region.add(nb)
                    stack.append(nb)

        # Interior = cells with all 8 neighbours inside the region (so every
        # farmland cell is surrounded); outline = region cells touching it.
        farm_cells = {
            (lz, lx)
            for lz, lx in region
            if all(
                (lz + ddz, lx + ddx) in region
                for ddz in (-1, 0, 1)
                for ddx in (-1, 0, 1)
            )
        }
        if len(farm_cells) < 5:
            return False, 0, 0, 0
        ring_cells = {
            c
            for c in region - farm_cells
            if any(
                (c[0] + ddz, c[1] + ddx) in farm_cells
                for ddz in (-1, 0, 1)
                for ddx in (-1, 0, 1)
            )
        }

        wx_g, wz_g = terrain_map.local_to_world(glz, glx)
        surface_y = self._h(feat, glz, glx) - 1
        biome = editor.getBiome((wx_g, surface_y, wz_g)).lower()

        farmland_block = Block("minecraft:farmland", {"moisture": "7"})
        edge_palette = self._edge_palette(biome)
        air = Block("minecraft:air")

        placed: List[
            Tuple[int, int, int]
        ] = []  # (wx, surface_y, wz) for farmland cells

        for lz, lx in sorted(ring_cells | farm_cells):
            wx, wz = terrain_map.local_to_world(lz, lx)
            h = self._h(feat, lz, lx)  # one above surface block

            if (lz, lx) in ring_cells:
                editor.placeBlock((wx, h, wz), random.choice(edge_palette))
            else:
                for dy in range(1, 9):
                    editor.placeBlock((wx, h + dy, wz), air)
                editor.placeBlock((wx, h, wz), farmland_block)
                placed.append((wx, h, wz))

        if not placed:
            return False, 0, 0, 0

        self._irrigate(feat, terrain_map, editor, placed)

        if "desert" not in biome and "badlands" not in biome:
            self._place_crop(editor, placed, biome)

        log(f"[Garden] radius={radius} biome={biome[:12]} local({glz},{glx})")
        return True, glz, glx, radius
