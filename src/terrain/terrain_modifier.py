"""
terrain_modifier.py
-------------------
Terrain preparation for the GDMC pipeline. Runs after planning (Phase 5-7)
and before block placement, so all placers work on already-prepared ground.

Techniques:
  Road smoothing      — Multi-pass ±1 carve + convergence loop so no two
                        adjacent road cells differ by more than 1 block.
  Column fill         — Samples the surface block as fill material; raises
                        or lowers a terrain column to an exact target Y.
  Foundation sealing  — Interior levelled to a common floor; perimeter gets
                        a vertical retaining wall down to the surrounding
                        terrain, and every column is sealed against caves.
  Vegetation clearing — standard across all reviewed entries; clears tree
                        trunks, leaves, and tall plants above road/wall paths.
"""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING

import numpy as np
from gdpc.block import Block
from gdpc.editor import Editor

from .terrain_types import TerrainMap

if TYPE_CHECKING:
    from .road_network import RoadNetwork
    from .wall_layout import WallLayout

logger = logging.getLogger(__name__)

_AIR = Block("minecraft:air")
_DIRT = Block("minecraft:dirt")
_GRAVEL = Block("minecraft:gravel")
_STONE = Block("minecraft:stone")
_COBBLESTONE = Block("minecraft:cobblestone")

# Natural surface blocks that are safe to mirror when filling holes.
_NATURAL_SURFACE = frozenset(
    {
        "grass_block",
        "sand",
        "red_sand",
        "podzol",
        "mycelium",
        "gravel",
        "coarse_dirt",
        "dirt",
        "snow_block",
        "stone",
        "sandstone",
        "red_sandstone",
        "mud",
        "moss_block",
    }
)

# Biome keyword → surface block for single-step fill blending.
# Checked in order; first matching substring wins; falls back to grass_block.
_BIOME_SURFACE: tuple[tuple[str, str], ...] = (
    # cherry_grove must precede the bare "grove" (snowy) entry — substring
    # matching would otherwise blanket cherry blossom terrain in snow.
    ("cherry_grove", "grass_block"),
    ("desert", "sand"),
    ("badlands", "red_sand"),
    ("beach", "sand"),
    ("mushroom", "mycelium"),
    ("frozen", "snow_block"),
    ("snowy", "snow_block"),
    ("ice", "snow_block"),
    ("grove", "snow_block"),
    ("swamp", "mud"),
    ("taiga", "podzol"),
)


def _biome_surface_block(biome: str | None) -> Block:
    if biome:
        b = biome.lower()
        for key, block_id in _BIOME_SURFACE:
            if key in b:
                return Block(f"minecraft:{block_id}")
    return Block("minecraft:grass_block")


_AIR_IDS = frozenset({"air", "cave_air", "void_air"})

# Thin ground cover a downward foundation scan must punch through rather
# than rest on: stopping at a snow layer would entomb it between the fill
# and the ground and hide any void beneath it.
_SNOW_COVER_IDS = frozenset({"snow", "powder_snow"})

# Natural soil converted to fill when a foundation pillar lands on it, so
# the pillar meets the ground stone-on-stone instead of stone-on-grass.
_DIRT_FAMILY_IDS = frozenset(
    {
        "grass_block",
        "dirt",
        "coarse_dirt",
        "rooted_dirt",
        "podzol",
        "mycelium",
        "dirt_path",
        "farmland",
        "mud",
    }
)

# Visible foundation palette used when stone_bricks is passed as fill_block.
_STONE_BRICKS_PALETTE = (
    Block("minecraft:stone_bricks"),
    Block("minecraft:stone_bricks"),
    Block("minecraft:stone_bricks"),
    Block("minecraft:mossy_stone_bricks"),
    Block("minecraft:cracked_stone_bricks"),
    Block("minecraft:cobblestone"),
)


class TerrainModifier:
    """
    Prepares Minecraft terrain before block placement.

    Typical call order in main():
        modifier = TerrainModifier(editor, terrain_map)
        modifier.prepare_roads(terrain_map.road_network)
        modifier.prepare_wall(terrain_map.wall_layout)
        # … then run placers, passing modifier so they can fill columns …
    """

    def __init__(self, editor: Editor, terrain_map: TerrainMap) -> None:
        self._editor = editor
        self._tm = terrain_map

    # ------------------------------------------------------------------
    # High-level preparation  (call before the matching placer)
    # ------------------------------------------------------------------

    def fill_terrain_holes(
        self,
        *,
        drop_threshold: int = 4,
        min_higher_neighbors: int = 3,
    ) -> int:
        """
        Fill isolated sharp dips (holes) in the terrain.

        A cell qualifies as a hole when at least *min_higher_neighbors* of its
        four cardinal neighbours have a surface height >= *drop_threshold* blocks
        above it.  Each qualifying cell is filled from its current surface up to
        the minimum height of those neighbours, using dirt as body fill and a
        surface block sampled from the adjacent terrain as the top layer (grass,
        sand, podzol, etc.).  features["height"] is updated to stay in sync.

        Water cells and border cells are skipped so rivers, ponds, and build-
        area edges are left untouched.

        Returns the number of holes filled.
        """
        feat = self._tm.features
        if feat is None:
            return 0
        depth, width = feat.shape
        H = feat["height"].astype(np.int32)
        n_filled = 0

        # Run multiple passes so chain-holes (a hole adjacent to another hole)
        # are caught even if the upstream cell was visited first.
        for _pass in range(5):
            pass_filled = 0
            for lz in range(depth):
                for lx in range(width):
                    if feat[lz, lx]["is_border"]:
                        continue
                    if float(feat[lz, lx]["water_pct"]) > 0.0:
                        continue

                    cell_h = int(H[lz, lx])
                    nbrs = [
                        (lz + dz, lx + dx, int(H[lz + dz, lx + dx]))
                        for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                        if 0 <= lz + dz < depth and 0 <= lx + dx < width
                    ]
                    higher = [
                        (nz, nx, nh)
                        for nz, nx, nh in nbrs
                        if nh - cell_h >= drop_threshold
                    ]
                    if len(higher) < min_higher_neighbors:
                        continue

                    # Fill to the minimum height of the qualifying neighbours.
                    target_y = min(nh for _, _, nh in higher)

                    # Sample the surface block from the lowest qualifying neighbour
                    # to match the top material of the surrounding terrain.
                    ref_nz, ref_nx, _ = min(higher, key=lambda t: t[2])
                    rwx, rwz = self._tm.local_to_world(ref_nz, ref_nx)
                    ref_block = self._editor.getBlock((rwx, target_y - 1, rwz))
                    top_block = _surface_block(ref_block.id or "")

                    wx, wz = self._tm.local_to_world(lz, lx)
                    # Use depth-appropriate fill: stone deep down, gravel mid,
                    # dirt near surface — mirrors natural terrain layering.
                    for y in range(cell_h, target_y - 1):
                        depth_from_top = (target_y - 1) - y
                        self._editor.placeBlock(
                            (wx, y, wz), _fill_for_depth(depth_from_top)
                        )
                    self._editor.placeBlock((wx, target_y - 1, wz), top_block)

                    H[lz, lx] = target_y
                    pass_filled += 1

            n_filled += pass_filled
            if pass_filled == 0:
                break  # converged — no new holes found

        feat["height"] = H.astype(np.int16)
        logger.info("TerrainModifier: filled %d terrain holes.", n_filled)
        return n_filled

    def prepare_roads(self, road_network: "RoadNetwork") -> None:
        """
        Smooth road heights, clear vegetation, and grade terrain under all roads.

        Updates features["height"] so RoadPlacer reads the smoothed heights.
        """
        feat = self._tm.features
        if feat is None or self._tm.road_map is None or road_network is None:
            logger.warning("TerrainModifier: skipping road prep — terrain not ready.")
            return

        # Clear the road corridor of whole trees FIRST, while feat["height"]
        # still matches the natural terrain — so trunks are fully removed (rather
        # than sliced by the later grading, which would leave floating canopy).
        # Uses the same shared clearer as walls/buildings/farmland.
        self._clear_road_vegetation()

        orig = feat["height"].copy()  # snapshot before smoothing

        logger.info("TerrainModifier: smoothing road height profiles...")
        self._smooth_road_heights(road_network)

        # Water cells along roads are bridge spans (rendered as plank decks
        # by RoadPlacer) — exclude them from smoothing stats, shoulder
        # grading, vegetation clearing, and terrain grading alike.  Use the
        # exact water mask: water_pct is a 5x5 blur and misses edge cells.
        water = (
            self._tm.water_mask
            if self._tm.water_mask is not None
            else feat["water_pct"] > 0.5
        )
        road_cells: set[tuple[int, int]] = {
            (z, x)
            for road in road_network.all_roads
            for z, x in road.path
            if not water[z, x]
        }

        # Log smoothing delta summary
        smoothed = feat["height"]
        deltas = smoothed.astype(np.int32) - orig.astype(np.int32)
        road_deltas = [int(deltas[z, x]) for z, x in road_cells]
        n_changed = sum(1 for d in road_deltas if d != 0)
        if road_deltas:
            logger.info(
                "TerrainModifier: smoothing changed %d/%d road cells  "
                "(delta min=%d max=%d mean=%.2f)",
                n_changed,
                len(road_deltas),
                min(road_deltas),
                max(road_deltas),
                sum(road_deltas) / len(road_deltas),
            )
        else:
            logger.warning(
                "TerrainModifier: no road cells found — smoothing had nothing to do."
            )

        # Spread the smoothed centreline height across the full rendered road
        # width (RoadPlacer dilates roads to 3-5 cells), so the surface is
        # flat across its cross-section instead of stepping sideways with the
        # raw terrain.
        shoulder_targets = self._shoulder_targets()
        for (slz, slx), target in shoulder_targets.items():
            if water[slz, slx]:
                continue  # never grade a water shoulder up to bank height
            # Cliffside shoulders: grading more than ±4 blocks would raise a
            # tall stone sliver (or dig a trench) beside the road — worse
            # than leaving the natural drop at the road's edge.
            if abs(int(feat["height"][slz, slx]) - target) > 4:
                continue
            feat["height"][slz, slx] = np.int16(target)

        # Vegetation clearing along roads is done inline by RoadPlacer
        # (per-cell 19-block air column above each placed road block) so we
        # do not duplicate it here.
        all_cells = road_cells | set(shoulder_targets)
        logger.info("TerrainModifier: grading terrain along roads...")
        self._grade_cells(all_cells, orig)

    def prepare_wall(self, wall_layout: "WallLayout") -> None:
        """Clear vegetation along the wall centreline."""
        feat = self._tm.features
        if feat is None or wall_layout is None or not wall_layout.is_valid:
            return

        wall_cells = self._wall_cells(wall_layout)
        orig = feat["height"].copy()
        logger.info(
            "TerrainModifier: clearing vegetation along %d wall cells...",
            len(wall_cells),
        )
        self._clear_cells(wall_cells, orig, clear_above=14)

    # ------------------------------------------------------------------
    # Column-level utilities  (called by placers during block placement)
    # ------------------------------------------------------------------

    def set_surface_height(
        self,
        lz: int,
        lx: int,
        target_y: int,
        fill_block: list[Block] | None = None,
    ) -> None:
        """
        Raise or lower terrain column (lz, lx) to target_y.

          curr_y > target_y  →  cut  (place air from target_y to curr_y − 1)
          curr_y < target_y  →  fill (place fill_block from curr_y to target_y − 1)

        Keeps features["height"] in sync so subsequent calls stay consistent.
        fill_block defaults to a depth-appropriate material (dirt / gravel / stone).
        """
        feat = self._tm.features
        if feat is None:
            return
        curr_y = int(feat[lz, lx]["height"])
        if curr_y == target_y:
            return

        wx, wz = self._tm.local_to_world(lz, lx)

        if curr_y > target_y:
            for y in range(target_y, curr_y):
                self._editor.placeBlock((wx, y, wz), _AIR)
        else:
            gap = target_y - curr_y
            blocks = [_fill_for_depth(gap)] if fill_block is None else fill_block
            for y in range(curr_y, target_y):
                self._editor.placeBlock((wx, y, wz), blocks)

        feat["height"][lz, lx] = np.int16(target_y)

    def fill_column_down(
        self,
        wx: int,
        wy: int,
        wz: int,
        fill_block: list[Block] | None = None,
    ) -> None:
        """
        Fill from wy − 1 down to the terrain surface below the block at (wx, wy, wz).

        Call AFTER placing any block that might be floating — roads on an embankment,
        wall segments on a slope, building floors above a hollow, etc.

        Uses features["height"] as the authoritative ground level; this reflects any
        prior calls to set_surface_height / level_footprint, so the fill stops at the
        correct modified terrain, not the original raw terrain.
        """
        lz, lx = self._tm.world_to_local(wx, wz)
        feat = self._tm.features
        if feat is None:
            return
        depth, width = feat.shape
        if not (0 <= lz < depth and 0 <= lx < width):
            return

        terrain_y = int(feat[lz, lx]["height"])
        if wy <= terrain_y:
            return  # already at or below ground

        gap = wy - terrain_y
        block = [_fill_for_depth(gap)] if fill_block is None else fill_block
        for y in range(terrain_y, wy):
            self._editor.placeBlock((wx, y, wz), block)

    def level_footprint(
        self,
        cells: list[tuple[int, int]],
        fill_block: list[Block] | None = None,
        strategy: str = "max",
        entrance_cell: tuple[int, int] | None = None,
    ) -> int:
        """
        Level all cells in a footprint to a common floor height and return it.

        strategy="entrance" — use the entrance/jigsaw cell's terrain height as
                         the floor so the front door sits flush with the terrain.
                         Falls back to min(heights) when entrance_cell is None.
        strategy="max"   (default) — use the highest cell as the floor.
                         All lower cells are filled up; nothing is cut.
        strategy="blend" — 70 % average + 30 % minimum blend.
                         Biases toward the low side so cuts dominate.
        """
        feat = self._tm.features
        if feat is None or not cells:
            return 0

        heights = [int(feat[lz, lx]["height"]) for lz, lx in cells]
        if strategy == "entrance":
            min_y = min(heights)
            avg_y = sum(heights) / len(heights)

            if entrance_cell is not None:
                ez, ex = entrance_cell
                entrance_y = int(feat[ez, ex]["height"])
                floor_y = round(0.5 * min_y + 0.3 * avg_y + 0.2 * entrance_y)
            else:
                floor_y = round(0.6 * min_y + 0.4 * avg_y)
        elif strategy == "max":
            floor_y = max(heights)
        else:
            avg_y = sum(heights) / len(heights)
            min_y = min(heights)
            floor_y = int(round(0.70 * avg_y + 0.30 * min_y))

        block = [_COBBLESTONE] if fill_block is None else fill_block
        for lz, lx in cells:
            self.set_surface_height(lz, lx, floor_y, fill_block=block)

        return floor_y

    def fill_footprint_walls(
        self,
        cells: list[tuple[int, int]],
        floor_y: int,
        fill_block: list[Block] | None = None,
    ) -> None:
        """
        For every *perimeter* cell of the footprint (a cell that has at least
        one 4-connected neighbour outside the footprint), place fill_block
        straight down from floor_y to the lowest adjacent outer-terrain height.

        *floor_y* is the leveled surface height (one-above convention, i.e. the
        value returned by level_footprint() before applying y_offset).

        This replaces the expanding pyramid skirt with a clean vertical wall
        that follows the exact outline of the placed schematic.
        """
        feat = self._tm.features
        if feat is None or not cells:
            return
        depth, width = feat.shape

        fp_set = set(cells)
        base_block = [_COBBLESTONE] if fill_block is None else fill_block
        use_palette = any(
            b.id in ("minecraft:stone_bricks", "stone_bricks") for b in base_block
        )

        for lz, lx in cells:
            outer = [
                (lz + dz, lx + dx)
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if (lz + dz, lx + dx) not in fp_set
                and 0 <= lz + dz < depth
                and 0 <= lx + dx < width
            ]
            if not outer:
                continue  # interior cell — skip

            min_outer_y = min(int(feat[nlz, nlx]["height"]) for nlz, nlx in outer)
            if min_outer_y >= floor_y:
                continue  # terrain already at or above floor

            wx, wz = self._tm.local_to_world(lz, lx)
            for y in range(min_outer_y, floor_y):
                blk = (
                    random.choice(_STONE_BRICKS_PALETTE)
                    if use_palette
                    else random.choice(base_block)
                )
                self._editor.placeBlock((wx, y, wz), blk)

    def make_building_apron(
        self,
        cells: list[tuple[int, int]],
        floor_y: int,
        fill_block: list[Block] | None = None,
        radius: int = 2,
        protected: set[tuple[int, int]] | None = None,
        biome: str | None = None,
    ) -> None:
        """
        Clear and reinforce a *radius*-cell apron around the building footprint.

        *floor_y* is the leveled surface height (height-map convention, one-above).

        Rings 1 … radius-1 are breathing room: they are levelled to the floor
        height (cutting into hillsides, filling dips) so high terrain never
        presses a retaining wall directly against the building.  The
        outermost ring gets the classic treatment:
          uphill  (terrain > floor) — cobblestone retaining wall from floor level
                   up to the terrain surface so the hillside cut looks reinforced.
          downhill (terrain < floor) — cobblestone fill from terrain surface up to
                   floor level, extending the foundation outward one cell.
          level   (terrain == floor) — no fill; just clear 3 blocks of vegetation.

        Cells in *protected* (roads, walls, other buildings) are left untouched.
        """
        feat = self._tm.features
        if feat is None or not cells:
            return
        depth, width = feat.shape
        base_block = [_COBBLESTONE] if fill_block is None else fill_block
        if protected is None:
            protected = set()

        water = (
            self._tm.water_mask
            if self._tm.water_mask is not None
            else feat["water_pct"] > 0.5
        )

        # Build rings by Chebyshev distance from the footprint.
        visited: set[tuple[int, int]] = set(cells)
        frontier: set[tuple[int, int]] = set(cells)
        rings: list[set[tuple[int, int]]] = []
        for _ in range(max(1, radius)):
            ring: set[tuple[int, int]] = set()
            for lz, lx in frontier:
                for dz in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        nlz, nlx = lz + dz, lx + dx
                        if (nlz, nlx) in visited:
                            continue
                        if not (0 <= nlz < depth and 0 <= nlx < width):
                            continue
                        visited.add((nlz, nlx))
                        ring.add((nlz, nlx))
            rings.append(ring)
            frontier = ring

        for i, ring in enumerate(rings):
            outermost = i == len(rings) - 1
            for nlz, nlx in ring:
                if (nlz, nlx) in protected:
                    continue
                if water[nlz, nlx]:
                    continue  # never touch coastline or water cells
                terrain_y = int(feat[nlz, nlx]["height"])
                wx, wz = self._tm.local_to_world(nlz, nlx)

                if not outermost:
                    # Breathing room: level to the floor height.
                    if terrain_y != floor_y:
                        self.set_surface_height(
                            nlz, nlx, floor_y, fill_block=base_block
                        )
                    clear_top = floor_y
                else:
                    if terrain_y > floor_y:
                        # Uphill: retaining wall from floor to terrain surface.
                        # For a single-step cut the topmost block uses the biome
                        # surface material so it blends with the surrounding ground.
                        diff = terrain_y - floor_y
                        surface_blk = _biome_surface_block(biome) if diff == 1 else None
                        for y in range(floor_y - 1, terrain_y):
                            blk = (
                                surface_blk
                                if (surface_blk is not None and y == floor_y)
                                else base_block
                            )
                            self._editor.placeBlock((wx, y, wz), blk)
                    elif terrain_y < floor_y:
                        # Downhill: fill from terrain surface up to floor level.
                        # Start at terrain_y (one above surface block) so the
                        # original ground block is preserved under the fill.
                        # For a single-step fill the one visible block uses the
                        # biome surface material instead of stone.
                        diff = floor_y - terrain_y
                        for y in range(terrain_y, floor_y):
                            blk = (
                                _biome_surface_block(biome) if diff == 1 else base_block
                            )
                            self._editor.placeBlock((wx, y, wz), blk)
                        # Keep feat["height"] in sync so path A* and subsequent
                        # placers see the raised surface, not the old stale value.
                        feat["height"][nlz, nlx] = np.int16(floor_y)
                    clear_top = max(terrain_y, floor_y)

                # Clear vegetation above the final top level
                for dy in range(3):
                    self._editor.placeBlock((wx, clear_top + dy, wz), _AIR)

    def fill_footprint_base(
        self,
        cells: list[tuple[int, int]],
        floor_y: int,
        fill_block: list[Block] | None = None,
        scan_depth: int = 15,
    ) -> None:
        """
        For every footprint cell scan downward from floor_y and fill any air
        with fill_block, stopping at the first solid block per column.

        Snow layers and leftover vegetation do not stop the scan — they are
        replaced by fill, never entombed under it. Once a column has punched
        through such a gap, the soil it lands on (grass_block, dirt, …) is
        converted to fill as well, so the pillar base does not sit directly
        on a grass surface.

        This seals cave roofs and voids that terrain levelling exposes under
        the structure floor — necessary when a cut removes terrain that was
        sitting above a cave, leaving an air gap directly under the building.
        """
        from builds._vegetation import is_clearable_veg

        base_block = [_COBBLESTONE] if fill_block is None else fill_block
        for lz, lx in cells:
            wx, wz = self._tm.local_to_world(lz, lx)
            punched = False  # column was filled through air or ground cover
            for dy in range(1, scan_depth + 1):
                y = floor_y - dy
                if y < 0:
                    break
                blk = self._editor.getBlock((wx, y, wz))
                bid = (blk.id or "") if blk is not None else ""
                # An unknown/unloaded block (id None → "") is treated as "not
                # air": stop rather than risk filling past real terrain.
                if not bid:
                    break
                name = bid.removeprefix("minecraft:")
                if name in _AIR_IDS or name in _SNOW_COVER_IDS or is_clearable_veg(bid):
                    self._editor.placeBlock((wx, y, wz), base_block)
                    punched = True
                elif punched and name in _DIRT_FAMILY_IDS:
                    self._editor.placeBlock((wx, y, wz), base_block)
                else:
                    break

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _smooth_road_heights(self, road_network: "RoadNetwork") -> None:
        """
        Three-pass height smoothing along road paths (yawgmoth/GDMC25).

        Pass 1 — forward  ±1 carve: each cell is clamped to prev ± 1.
        Pass 2 — backward ±1 carve: each cell is clamped to next ± 1.
        Pass 3 — convergence: iteratively clamp any adjacent road-cell pair
                 that differs by more than 1 until the whole network settles.

        Writes the smoothed heights back to features["height"].
        """
        feat = self._tm.features
        H = feat["height"].astype(np.int32)
        water = (
            self._tm.water_mask
            if self._tm.water_mask is not None
            else feat["water_pct"] > 0.5
        )

        for road in road_network.all_roads:
            path = road.path
            if len(path) < 2:
                continue
            # Forward.  Water cells are bridge spans: leave their heights
            # untouched (they hold the river bottom) and never let them
            # constrain the bank heights — otherwise the bottom drags both
            # banks down, carving ramps into the river.
            for i in range(1, len(path)):
                pz, px = path[i]
                qz, qx = path[i - 1]
                if water[pz, px] or water[qz, qx]:
                    continue
                ph = int(H[qz, qx])
                H[pz, px] = max(ph - 1, min(ph + 1, int(H[pz, px])))
            # Backward
            for i in range(len(path) - 2, -1, -1):
                pz, px = path[i]
                qz, qx = path[i + 1]
                if water[pz, px] or water[qz, qx]:
                    continue
                nh = int(H[qz, qx])
                H[pz, px] = max(nh - 1, min(nh + 1, int(H[pz, px])))

        road_cells: set[tuple[int, int]] = {
            (z, x)
            for road in road_network.all_roads
            for z, x in road.path
            if not water[z, x]
        }
        changed = True
        iters = 0
        while changed and iters < 20:
            changed = False
            iters += 1
            for z, x in road_cells:
                for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nz, nx = z + dz, x + dx
                    if (nz, nx) in road_cells:
                        diff = int(H[z, x]) - int(H[nz, nx])
                        if abs(diff) > 1:
                            H[z, x] = int(H[nz, nx]) + (1 if diff > 0 else -1)
                            changed = True

        feat["height"] = H.astype(np.int16)
        logger.debug("TerrainModifier: road convergence in %d iterations.", iters)

    def _shoulder_targets(self) -> dict[tuple[int, int], int]:
        """
        Map every off-centreline road-surface cell to the smoothed height of
        its nearest centreline cell.

        Uses the same dilation rule as _expand_road_map (radius 2 for primary,
        1 otherwise) so the graded area matches exactly what RoadPlacer will
        cover with road blocks.  Must run AFTER _smooth_road_heights so the
        centreline heights are final.
        """
        road_map = self._tm.road_map
        feat = self._tm.features
        if road_map is None or feat is None:
            return {}

        H = feat["height"]
        depth, width = road_map.shape
        # cell → (dist² to its centreline cell, target height); nearest wins.
        best: dict[tuple[int, int], tuple[int, int]] = {}

        for z, x in np.argwhere(road_map > 0):
            z, x = int(z), int(x)
            radius = 2 if road_map[z, x] == 3 else 1
            h = int(H[z, x])
            for dz in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nz, nx = z + dz, x + dx
                    if not (0 <= nz < depth and 0 <= nx < width):
                        continue
                    if road_map[nz, nx] > 0:
                        continue  # centreline cells keep their own height
                    d2 = dz * dz + dx * dx
                    cur = best.get((nz, nx))
                    if cur is None or d2 < cur[0]:
                        best[(nz, nx)] = (d2, h)

        return {cell: h for cell, (_d2, h) in best.items()}

    def _clear_cells(
        self,
        cells: set[tuple[int, int]],
        orig_heights: np.ndarray,
        clear_above: int = 5,
    ) -> None:
        """Place air above each cell to remove trees and tall vegetation."""
        for lz, lx in cells:
            terrain_y = int(orig_heights[lz, lx])
            wx, wz = self._tm.local_to_world(lz, lx)
            for dy in range(0, clear_above):
                self._editor.placeBlock((wx, terrain_y + dy, wz), _AIR)

    def _clear_road_vegetation(self) -> None:
        """
        Whole-tree clear of the full (dilated) road corridor with the shared
        clearer, so roadside trees are removed as units — trunk plus their own
        canopy — with no floating leaves or half-cut trees along the road.

        Runs before smoothing/grading so feat["height"] still matches the
        natural terrain and every trunk base is found intact.  Water/bridge
        cells are skipped by clear_vegetation itself.
        """
        from builds._vegetation import clear_vegetation

        from .terrain_types import _expand_road_map

        feat = self._tm.features
        road_map = self._tm.road_map
        if feat is None or road_map is None:
            return
        depth, width = feat.shape
        expanded = _expand_road_map(road_map)
        wm = self._tm.water_mask

        def _is_water(z: int, x: int) -> bool:
            if wm is not None:
                return bool(wm[z, x])
            return float(feat[z, x]["water_pct"]) > 0.5

        cells = [
            (int(z), int(x))
            for z, x in np.argwhere(expanded > 0)
            if not _is_water(int(z), int(x))
        ]
        if cells:
            clear_vegetation(cells, feat, depth, width, self._editor, self._tm)

    def _grade_cells(
        self,
        cells: set[tuple[int, int]],
        orig_heights: np.ndarray,
    ) -> None:
        """
        Cut or fill each cell so MC terrain matches the smoothed features["height"].

        orig_heights is the snapshot taken before smoothing; it represents the
        current MC terrain height.  features["height"] is the smoothed target.
        """
        feat = self._tm.features
        n_cut = n_fill = 0
        max_cut = max_fill = 0

        for lz, lx in cells:
            orig_y = int(orig_heights[lz, lx])
            target_y = int(feat[lz, lx]["height"])
            if orig_y == target_y:
                continue
            wx, wz = self._tm.local_to_world(lz, lx)
            if orig_y > target_y:
                delta = orig_y - target_y
                n_cut += 1
                max_cut = max(max_cut, delta)
                for y in range(target_y, orig_y):
                    self._editor.placeBlock((wx, y, wz), _AIR)
            else:
                delta = target_y - orig_y
                n_fill += 1
                max_fill = max(max_fill, delta)
                block = _fill_for_depth(delta)
                for y in range(orig_y, target_y):
                    self._editor.placeBlock((wx, y, wz), block)

        logger.info(
            "TerrainModifier: graded %d cells  "
            "(cut=%d max_cut=%d  fill=%d max_fill=%d)",
            n_cut + n_fill,
            n_cut,
            max_cut,
            n_fill,
            max_fill,
        )

    def _wall_cells(self, wall_layout: "WallLayout") -> set[tuple[int, int]]:
        """Return every local cell on the wall centrelines."""
        feat = self._tm.features
        depth, width = feat.shape
        cells: set[tuple[int, int]] = set()

        n = wall_layout.n
        for i in range(n):
            lz1, lx1 = wall_layout.corners[i]
            lz2, lx2 = wall_layout.corners[(i + 1) % n]
            dz = lz2 - lz1
            dx = lx2 - lx1
            L = math.sqrt(dz * dz + dx * dx)
            if L < 1.0:
                continue
            for step in range(int(round(L)) + 1):
                t = step / L
                lz = int(round(lz1 + dz * t))
                lx = int(round(lx1 + dx * t))
                if 0 <= lz < depth and 0 <= lx < width:
                    cells.add((lz, lx))

        return cells


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _fill_for_depth(gap: int) -> Block:
    """Choose a contextually appropriate fill block based on gap size."""
    if gap <= 2:
        return _DIRT
    if gap <= 5:
        return _GRAVEL
    return _STONE


def _surface_block(block_id: str) -> Block:
    """Return the block as-is if it is a natural surface material, else dirt."""
    bid = block_id.replace("minecraft:", "")
    if bid in _NATURAL_SURFACE:
        return Block(f"minecraft:{bid}")
    return _DIRT
