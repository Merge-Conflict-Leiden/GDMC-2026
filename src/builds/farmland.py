"""
This module defines a generator for filling farmland districts with crops,
augmented with random schematic decorations.
"""

import random
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
from gdpc.block import Block
from gdpc.editor import Editor

from blocks.block_processor import BlockProcessor
from builds._vegetation import clear_vegetation
from consts import BUILD_DIR, BUILD_OUTPUT_DIR
from structures.structure_cache import StructureCache
from structures.structure_placer import StructurePlacer
from terrain.terrain_types import SubZone, TerrainMap
from utils import log


class FarmlandGenerator:
    """
    Fill 'FARMLAND'-zoned districts with wheat fields, irrigation channels, and
    field paths.

    To guarantee every crop cell is within 4 blocks of a water source, while
    leaving clear visual rows separated by paths:
      PATH_SPACING = 10 -> one dirt-path column every 10 blocks
      ROW_SPACING  =  9 -> one water channel every 9 blocks
    """

    SCHEMATIC_DIR = "farmland"
    SCHEMATIC_CHANCE = 0.02  # 2% chance for a special schematic to spawn on top
    WINDMILL_EXCLUSION_RADIUS = 8  # local cells to leave empty around windmills
    ROW_SPACING = 9  # water channel every ROW_SPACING blocks (Z)
    PATH_SPACING = 10  # field divider every PATH_SPACING blocks (X)
    CLEAR_HEIGHT = 6  # blocks of vegetation cleared above surface
    MAX_CELL_SLOPE = 2.5  # skip cells steeper than this (irrigation won't work)
    BARE_CHANCE = 0.00  # probability of leaving a crop cell as bare farmland
    MAX_SCHEMATIC_HEIGHT_DIFF = (
        3  # reject schematic footprints with more height variation
    )

    _AIR = Block("air")
    _FARMLAND = Block("farmland", states={"moisture": "7"})
    _WATER = Block("water")
    _DIRT = Block("dirt")
    _PATH = Block("dirt_path")
    _HAY = Block("hay_block", states={"axis": "y"})
    _COBBLESTONE = Block("cobblestone")

    _WHEAT_POOL: Tuple[Optional[Block], ...] = tuple(
        [Block("wheat", states={"age": "7"})] * 12
        + [Block("wheat", states={"age": "6"})] * 4
        + [Block("wheat", states={"age": "5"})] * 2
        + [Block("wheat", states={"age": "4"})]
        + [Block("wheat", states={"age": "0"})]
    )

    def __init__(self) -> None:
        self._cache = StructureCache(BUILD_DIR, BUILD_OUTPUT_DIR)
        self._placer = StructurePlacer(
            structure_cache=self._cache,
            block_processor=BlockProcessor(),
            block_palette=None,
            furniture_replacer=None,
        )
        self._schematics: List[Tuple[str, int, int]] = []  # (csv_path, width, depth)
        self._load_schematics()

    def _load_schematics(self) -> None:
        """Discover all CSV schematics in builds/output/farmland/."""
        schem_dir = Path(BUILD_DIR) / BUILD_OUTPUT_DIR / self.SCHEMATIC_DIR
        if not schem_dir.is_dir():
            return
        for csv_file in sorted(schem_dir.glob("*.csv")):
            rel_path = f"{self.SCHEMATIC_DIR}/{csv_file.name}"
            try:
                _, width, _, depth = self._cache.load_structure(rel_path)
                self._schematics.append((rel_path, width, depth))
                log(f"[Farmland] Loaded schematic {rel_path} ({width}w × {depth}d)")
            except Exception as e:
                log(f"[Farmland] Could not load schematic {rel_path}: {e}")

    def _windmill_exclusion(
        self, windmill_sites: List[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """Circular exclusion set around every windmill site (local coords)."""
        r = self.WINDMILL_EXCLUSION_RADIUS
        excluded: Set[Tuple[int, int]] = set()
        for wz, wx in windmill_sites:
            for dz in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dz * dz + dx * dx <= r * r:
                        excluded.add((wz + dz, wx + dx))
        return excluded

    def _fits_in_farmland(
        self,
        lz: int,
        lx: int,
        s_width: int,
        s_depth: int,
        farmland_set: Set[Tuple[int, int]],
        feat: np.ndarray,
    ) -> Tuple[bool, Set[Tuple[int, int]]]:
        """
        Return (True, footprint) when the full schematic footprint lies within
        the farmland district, is free of roads, and has acceptable terrain flatness.
        """
        footprint: Set[Tuple[int, int]] = set()
        for dz in range(s_depth):
            for dx in range(s_width):
                cell = (lz + dz, lx + dx)
                if cell not in farmland_set:
                    return False, set()
                footprint.add(cell)
        heights = [int(feat[fz, fx]["height"]) for fz, fx in footprint]
        if max(heights) - min(heights) > self.MAX_SCHEMATIC_HEIGHT_DIFF:
            return False, set()
        return True, footprint

    def _is_on_road(self, lz: int, lx: int, road_map: Optional[np.ndarray]) -> bool:
        if road_map is None:
            return False
        return (
            0 <= lz < road_map.shape[0]
            and 0 <= lx < road_map.shape[1]
            and road_map[lz, lx] != 0
        )

    def _is_water(
        self, lz: int, lx: int, water_mask: Optional[np.ndarray], feat: np.ndarray
    ) -> bool:
        if water_mask is not None:
            return bool(water_mask[lz, lx])
        return float(feat[lz, lx]["water_pct"]) > 0.5

    def generate(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        *,
        pre_occupied: Optional[Set[Tuple[int, int]]] = None,
    ) -> None:
        """
        Two-pass farmland generation:

          Pass 1 — place all terrain: paths, irrigation channels, farmland
                   blocks and wheat crops.
          Pass 2 — overlay schematics (skip_air=True) so wheat planted in
                   Pass 1 shows through wherever the schematic has no block.
        """
        sub_zone_map = terrain_map.sub_zone_map
        road_map = terrain_map.road_map_expanded

        if sub_zone_map is None:
            log("[Farmland] sub_zone_map not available, skipping.")
            return

        farmland_coords = np.argwhere(sub_zone_map == int(SubZone.FARMLAND))
        if len(farmland_coords) == 0:
            log("[Farmland] No FARMLAND zones found.")
            return

        farmland_set: Set[Tuple[int, int]] = {
            (int(r[0]), int(r[1])) for r in farmland_coords
        }
        excluded = self._windmill_exclusion(terrain_map.windmill_sites)
        if pre_occupied:
            excluded = excluded | pre_occupied
        # Remove excluded cells up front so every later pass (crops, dams,
        # schematic footprints) treats them as outside the field.
        farmland_set -= excluded

        min_lz = int(farmland_coords[:, 0].min())
        min_lx = int(farmland_coords[:, 1].min())

        feat = terrain_map.features
        map_depth, map_width = feat.shape
        water_mask = terrain_map.water_mask

        # Cells that actually receive farmland/path/water this pass.  Roads,
        # water cells, and steep cells keep their original terrain.
        active: Set[Tuple[int, int]] = {
            (lz, lx)
            for lz, lx in farmland_set
            if not self._is_on_road(lz, lx, road_map)
            and not self._is_water(lz, lx, water_mask, feat)
            and float(feat[lz, lx]["slope"]) <= self.MAX_CELL_SLOPE
        }

        clear_vegetation(
            list(farmland_set),
            feat,
            map_depth,
            map_width,
            editor,
            terrain_map,
            protected=excluded,
        )

        n_wheat = n_water = n_path = 0
        # Cobblestone dams are collected here and placed after all farmland
        # blocks so they never accidentally overwrite farmland/path cells.
        cobblestone_dams: Set[Tuple[int, int, int]] = set()  # (lz, lx, wy)
        # Road-edge dams: a single path block placed at water level on a road
        # cell that sits lower than the adjacent water channel.  The road
        # surface is unaffected; this just caps the exposed face so water
        # cannot flow onto the road.
        road_dams: Set[Tuple[int, int, int]] = set()  # (lz, lx, wy)

        for lz, lx in sorted(active):
            wx, wz = terrain_map.local_to_world(lz, lx)
            # features["height"] is ONE ABOVE the surface block
            surface_y = int(feat[lz, lx]["height"]) - 1

            # Clear vegetation above the surface so crops are unobstructed
            for dy in range(1, self.CLEAR_HEIGHT + 2):
                editor.placeBlock((wx, surface_y + dy, wz), self._AIR)

            row_off = (lz - min_lz) % self.ROW_SPACING
            col_off = (lx - min_lx) % self.PATH_SPACING
            is_water_row = row_off == self.ROW_SPACING // 2
            is_path_col = col_off == 0

            # ---- path × water intersection: decorative hay bale --------
            if is_path_col and is_water_row:
                editor.placeBlock((wx, surface_y, wz), self._DIRT)
                editor.placeBlock((wx, surface_y + 1, wz), self._HAY)
                n_path += 1
                continue

            # ---- field-divider path column ------------------------------
            if is_path_col:
                editor.placeBlock((wx, surface_y, wz), self._PATH)
                n_path += 1
                continue

            # ---- irrigation channel (middle of each 9-row strip) --------
            if is_water_row:
                # Pre-check: every neighbour must be within the build area so
                # a cobblestone dam can be placed if needed.  If any neighbour
                # is outside the map we cannot contain the water, so fall back
                # to a dirt-path strip instead.
                all_in_bounds = all(
                    0 <= lz + dz_n < map_depth and 0 <= lx + dx_n < map_width
                    for dz_n, dx_n in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )
                if not all_in_bounds:
                    editor.placeBlock((wx, surface_y, wz), self._PATH)
                    n_path += 1
                    continue

                # Water at surface_y is sealed on a side only when that
                # neighbour column is solid at the same level.  A lower
                # neighbour cannot be dammed when it is active (the dam would
                # replace crops), a road (no cobble lumps on roads), or
                # pre-occupied (the dam would clip an existing structure) —
                # in those cases the cell degrades to a path block.  Only
                # inactive, free, open columns get a scheduled dam.
                sealed = True
                dams_here: List[Tuple[int, int, int]] = []
                road_dams_here: List[Tuple[int, int, int]] = []
                for dz_n, dx_n in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nlz, nlx = lz + dz_n, lx + dx_n
                    n_sy = int(feat[nlz, nlx]["height"]) - 1
                    if n_sy >= surface_y:
                        continue  # neighbour terrain/placement seals this side
                    if (nlz, nlx) in active or (nlz, nlx) in excluded:
                        sealed = False
                        break
                    if self._is_on_road(nlz, nlx, road_map):
                        # Road cell is lower than the water channel.  Rather
                        # than abandoning the water, cap the exposed face with
                        # a path block at water level so it cannot flow onto
                        # the road surface.
                        road_dams_here.append((nlz, nlx, surface_y))
                        continue
                    dams_here.append((nlz, nlx, surface_y))
                if not sealed:
                    editor.placeBlock((wx, surface_y, wz), self._PATH)
                    n_path += 1
                    continue

                editor.placeBlock((wx, surface_y - 1, wz), self._DIRT)
                editor.placeBlock((wx, surface_y, wz), self._WATER)
                n_water += 1
                cobblestone_dams.update(dams_here)
                road_dams.update(road_dams_here)
                continue

            # ---- crop cell ----------------------------------------------
            editor.placeBlock((wx, surface_y, wz), self._FARMLAND)
            if random.random() >= self.BARE_CHANCE:
                editor.placeBlock(
                    (wx, surface_y + 1, wz), random.choice(self._WHEAT_POOL)
                )
            n_wheat += 1

        # Place cobblestone dams.  Every scheduled position is an inactive,
        # free, non-road column that is open at water level, so placement is
        # unconditional — skipping any of these would let water spill.  The
        # cells are recorded so they join the shared occupied set below:
        # later phases (vegetation sweeps, gardens, misc) must never remove
        # or overbuild a dam, or the channel leaks.
        n_dams = 0
        dam_cells: Set[Tuple[int, int]] = set()
        for dlz, dlx, wy in cobblestone_dams:
            wx, wz = terrain_map.local_to_world(dlz, dlx)
            editor.placeBlock((wx, wy, wz), self._COBBLESTONE)
            dam_cells.add((dlz, dlx))
            n_dams += 1

        # Road-edge dams: a single path block at water level caps the gap
        # between the water channel and the lower road surface.
        for dlz, dlx, wy in road_dams:
            wx, wz = terrain_map.local_to_world(dlz, dlx)
            editor.placeBlock((wx, wy, wz), self._PATH)
            dam_cells.add((dlz, dlx))
            n_dams += 1

        n_schem = 0
        placed_footprints: Set[Tuple[int, int]] = set()
        if self._schematics:
            for lz, lx in sorted(farmland_set):
                if self._is_on_road(lz, lx, road_map):
                    continue
                if self._is_water(lz, lx, water_mask, feat):
                    continue
                if (lz, lx) in placed_footprints:
                    continue

                row_off = (lz - min_lz) % self.ROW_SPACING
                col_off = (lx - min_lx) % self.PATH_SPACING
                if row_off == self.ROW_SPACING // 2 or col_off == 0:
                    continue  # don't anchor on water/path cells

                if random.random() >= self.SCHEMATIC_CHANCE:
                    continue

                schem_path, sw, sd = random.choice(self._schematics)
                ok, footprint = self._fits_in_farmland(
                    lz, lx, sw, sd, farmland_set, feat
                )
                if not ok:
                    continue

                wx, wz = terrain_map.local_to_world(lz, lx)
                surface_y = int(feat[lz, lx]["height"]) - 1
                with editor.pushTransform((wx, surface_y, wz)):
                    self._placer.place_structure(editor, schem_path, skip_air=True)
                placed_footprints.update(footprint)
                n_schem += 1

        # Expose every cell we worked on so the caller's shared occupied set
        # reflects the farmland.  Later phases (gardens, misc) then skip it.
        if pre_occupied is not None:
            pre_occupied.update(active)
            pre_occupied.update(placed_footprints)
            pre_occupied.update(dam_cells)

        log(
            f"[Farmland] {n_wheat} crop cells, {n_water} water sources, "
            f"{n_dams} cobblestone dams, {n_path} path/hay blocks, "
            f"{n_schem} decorations placed."
        )
