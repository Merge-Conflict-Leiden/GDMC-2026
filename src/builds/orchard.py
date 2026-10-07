"""
This module defines a generator for ORCHARD-zoned districts.

Layout strategy
---------------
  • Ground is replaced with biome-appropriate grass/podzol/rooted/coarse dirt.
  • Fruit trees are planted on a TREE_SPACING grid.  Each tree is a trunk of
    2–4 log blocks topped by a spherical crown of biome-appropriate leaves.
    Leaf density tapers toward the outer shell so crowns look natural.
  • One well schematic is placed near the district centroid (if it fits and
    the CSV is available).
  • Scatter decorations (piles, carts) are placed with low probability on
    non-tree, non-road cells using skip_air=True.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

_AIR = Block("minecraft:air")

# ---------------------------------------------------------------------------
# Biome lookup tables
# ---------------------------------------------------------------------------

_BIOME_TREE: Dict[str, Tuple[str, str]] = {
    # biome-key: (log_block, leaf_block)
    "cherry": ("cherry_log", "cherry_leaves"),
    "jungle": ("jungle_log", "jungle_leaves"),
    "savanna": ("acacia_log", "acacia_leaves"),
    "badlands": ("acacia_log", "acacia_leaves"),
    "taiga": ("spruce_log", "spruce_leaves"),
    "snowy": ("spruce_log", "spruce_leaves"),
    "frozen": ("spruce_log", "spruce_leaves"),
    "birch": ("birch_log", "birch_leaves"),
    "flower": ("birch_log", "birch_leaves"),
    "meadow": ("oak_log", "oak_leaves"),
    "plains": ("oak_log", "oak_leaves"),
    "default": ("oak_log", "oak_leaves"),
}

_BIOME_GROUND: Dict[str, Tuple[Block, ...]] = {
    "taiga": (
        Block("podzol"),
        Block("podzol"),
        Block("podzol"),
        Block("podzol"),
        Block("rooted_dirt"),
        Block("coarse_dirt"),
    ),
    "snowy": (
        Block("podzol"),
        Block("podzol"),
        Block("rooted_dirt"),
        Block("coarse_dirt"),
    ),
    "frozen": (
        Block("podzol"),
        Block("podzol"),
        Block("rooted_dirt"),
        Block("coarse_dirt"),
    ),
    "jungle": (
        Block("podzol"),
        Block("podzol"),
        Block("rooted_dirt"),
        Block("coarse_dirt"),
        Block("grass_block"),
        Block("grass_block"),
    ),
    "savanna": (Block("rooted_dirt"), Block("coarse_dirt")),
    "badlands": (Block("coarse_dirt"), Block("red_sand")),
    "default": (
        Block("grass_block"),
        Block("grass_block"),
        Block("grass_block"),
        Block("grass_block"),
        Block("rooted_dirt"),
        Block("coarse_dirt"),
    ),
}


def _biome_key(biome: str) -> str:
    for key in _BIOME_TREE:
        if key != "default" and key in biome:
            return key
    return "default"


class OrchardGenerator:
    """Generate ORCHARD districts as grid-planted fruit orchards with decorations."""

    SCHEMATIC_DIR = "orchard"
    WINDMILL_EXCLUSION_RADIUS = 8  # keep trees clear of windmill blades
    TREE_SPACING = 7  # one tree every N cells in both X and Z
    TRUNK_MIN = 2
    TRUNK_MAX = 4
    CROWN_RADIUS = 3  # voxel sphere radius for leaf crown
    CLEAR_HEIGHT = 8  # vegetation cleared above surface before placing
    DECOR_CHANCE = 0.008  # ~0.8 % per non-tree, non-road cell

    # Schematic names that are classified as "well" (placed once, near centroid)
    WELL_NAMES = {"well"}
    # Schematic names that are scatter decorations
    DECOR_NAMES = {"small_pile", "medium_pile", "big_cart", "small_cart"}

    def __init__(self) -> None:
        self._cache = StructureCache(BUILD_DIR, BUILD_OUTPUT_DIR)
        self._placer = StructurePlacer(
            structure_cache=self._cache,
            block_processor=BlockProcessor(),
            block_palette=None,
            furniture_replacer=None,
        )
        self._well_schematics: List[Tuple[str, int, int]] = []
        self._decor_schematics: List[Tuple[str, int, int]] = []
        self._load_schematics()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _load_schematics(self) -> None:
        schem_dir = Path(BUILD_DIR) / BUILD_OUTPUT_DIR / self.SCHEMATIC_DIR
        if not schem_dir.is_dir():
            return
        for csv_file in sorted(schem_dir.glob("*.csv")):
            rel = f"{self.SCHEMATIC_DIR}/{csv_file.name}"
            stem = csv_file.stem.lower()
            try:
                _, w, _, d = self._cache.load_structure(rel)
                entry = (rel, w, d)
                if any(name in stem for name in self.WELL_NAMES):
                    self._well_schematics.append(entry)
                    log(f"[Orchard] Loaded well schematic {rel} ({w}w × {d}d)")
                else:
                    self._decor_schematics.append(entry)
                    log(f"[Orchard] Loaded decor schematic {rel} ({w}w × {d}d)")
            except Exception as e:
                log(f"[Orchard] Could not load {rel}: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _windmill_exclusion(self, sites: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Circular exclusion set around every windmill site (local coords)."""
        r = self.WINDMILL_EXCLUSION_RADIUS
        excluded: Set[Tuple[int, int]] = set()
        for wz, wx in sites:
            for dz in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dz * dz + dx * dx <= r * r:
                        excluded.add((wz + dz, wx + dx))
        return excluded

    def _on_road(self, lz: int, lx: int, road_map: Optional[np.ndarray]) -> bool:
        if road_map is None:
            return False
        return (
            0 <= lz < road_map.shape[0]
            and 0 <= lx < road_map.shape[1]
            and road_map[lz, lx] != 0
        )

    def _fits(
        self,
        lz: int,
        lx: int,
        sw: int,
        sd: int,
        orchard_set: Set[Tuple[int, int]],
        road_map: Optional[np.ndarray],
        occupied: Set[Tuple[int, int]],
    ) -> Tuple[bool, Set[Tuple[int, int]]]:
        fp: Set[Tuple[int, int]] = set()
        for dz in range(sd):
            for dx in range(sw):
                cell = (lz + dz, lx + dx)
                if cell not in orchard_set:
                    return False, set()
                if self._on_road(cell[0], cell[1], road_map):
                    return False, set()
                if cell in occupied:
                    return False, set()
                fp.add(cell)
        return True, fp

    def _place_tree(
        self,
        editor: Editor,
        wx: int,
        surface_y: int,
        wz: int,
        log_block: Block,
        leaf_block: Block,
    ) -> None:
        trunk_height = random.randint(self.TRUNK_MIN, self.TRUNK_MAX)
        for dy in range(trunk_height):
            editor.placeBlock((wx, surface_y + dy, wz), log_block)

        # Spherical crown centred one block above trunk top
        crown_base = surface_y + trunk_height
        r = self.CROWN_RADIUS
        for dz in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    dist_sq = dz * dz + dy * dy + dx * dx
                    if dist_sq > r * r:
                        continue
                    # Skip innermost core column (trunk takes it)
                    if dz == 0 and dx == 0 and dy <= 0:
                        continue
                    # Thin the outermost shell randomly for a natural look
                    if dist_sq > (r - 1) ** 2 and random.random() < 0.35:
                        continue
                    editor.placeBlock(
                        (wx + dx, crown_base + dy, wz + dz),
                        Block(
                            leaf_block.id,
                            states={"persistent": "true", "distance": "1"},
                        ),
                    )

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def generate(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        *,
        pre_occupied: Optional[Set[Tuple[int, int]]] = None,
    ) -> None:
        """Plant fruit trees, place well and scatter decorations in ORCHARD zones."""
        sub_map = terrain_map.sub_zone_map
        road_map = terrain_map.road_map_expanded
        feat = terrain_map.features

        if sub_map is None or feat is None:
            log("[Orchard] Missing map data, skipping.")
            return

        coords = np.argwhere(sub_map == int(SubZone.ORCHARD))
        if len(coords) == 0:
            log("[Orchard] No ORCHARD zones found.")
            return

        orchard_set: Set[Tuple[int, int]] = {(int(r[0]), int(r[1])) for r in coords}
        excl: Set[Tuple[int, int]] = set(pre_occupied) if pre_occupied else set()
        # Windmill blades sweep well past the building footprint — keep the
        # whole radius clear of trees, matching farmland and animal pens.
        excl |= self._windmill_exclusion(terrain_map.windmill_sites)
        # Remove excluded cells up front so ground replacement, the well, and
        # decorations (whose _fits only checks orchard_set) all respect them.
        orchard_set -= excl
        if not orchard_set:
            log("[Orchard] Zone fully excluded, skipping.")
            return

        min_lz = int(coords[:, 0].min())
        min_lx = int(coords[:, 1].min())
        map_depth, map_width = feat.shape

        clear_vegetation(
            list(orchard_set),
            feat,
            map_depth,
            map_width,
            editor,
            terrain_map,
            protected=excl,
        )

        # Biome at district centroid
        cz = int(coords[:, 0].mean())
        cx = int(coords[:, 1].mean())
        wx_c, wz_c = terrain_map.local_to_world(cz, cx)
        biome = editor.getBiome((wx_c, int(feat[cz, cx]["height"]) - 1, wz_c)).lower()

        bkey = _biome_key(biome)
        log_id, leaf_id = _BIOME_TREE[bkey]
        log_block = Block(log_id, states={"axis": "y"})
        leaf_block = Block(leaf_id)
        ground_pool = _BIOME_GROUND.get(bkey, _BIOME_GROUND["default"])

        # Cells occupied by trees or schematics (to prevent overlap)
        occupied: Set[Tuple[int, int]] = set()
        tree_cells: Set[Tuple[int, int]] = set()

        # Identify tree grid positions
        for lz, lx in orchard_set:
            if self._on_road(lz, lx, road_map):
                continue
            if (lz - min_lz) % self.TREE_SPACING == 0 and (
                lx - min_lx
            ) % self.TREE_SPACING == 0:
                # Keep the crown (radius 3) clear of neighbouring structures:
                # skip trees whose immediate surroundings are pre-occupied.
                if any(
                    (lz + dz, lx + dx) in excl
                    for dz in range(-2, 3)
                    for dx in range(-2, 3)
                ):
                    continue
                tree_cells.add((lz, lx))

        # ------------------------------------------------------------------
        # Pass 1 – ground replacement and clearing
        # ------------------------------------------------------------------
        for lz, lx in sorted(orchard_set):
            if (lz, lx) in excl or self._on_road(lz, lx, road_map):
                continue

            wx, wz = terrain_map.local_to_world(lz, lx)
            surface_y = int(feat[lz, lx]["height"]) - 1

            for dy in range(1, self.CLEAR_HEIGHT + 1):
                editor.placeBlock((wx, surface_y + dy, wz), _AIR)

            editor.placeBlock((wx, surface_y, wz), random.choice(ground_pool))

        # ------------------------------------------------------------------
        # Pass 2 – plant trees on grid
        # ------------------------------------------------------------------
        n_trees = 0
        for lz, lx in sorted(tree_cells):
            wx, wz = terrain_map.local_to_world(lz, lx)
            surface_y = int(feat[lz, lx]["height"]) - 1
            self._place_tree(editor, wx, surface_y + 1, wz, log_block, leaf_block)
            # Mark crown footprint as occupied (rough 3×3 around tree base)
            for dz in range(-1, 2):
                for dx in range(-1, 2):
                    occupied.add((lz + dz, lx + dx))
            n_trees += 1

        # ------------------------------------------------------------------
        # Pass 3 – well near centroid (once)
        # ------------------------------------------------------------------
        n_well = 0
        if self._well_schematics:
            well_schem, ww, wd = random.choice(self._well_schematics)
            # Search outward from centroid for a fitting position
            search_order = sorted(
                [
                    (lz, lx)
                    for lz, lx in orchard_set
                    if (lz, lx) not in excl and not self._on_road(lz, lx, road_map)
                ],
                key=lambda c: (c[0] - cz) ** 2 + (c[1] - cx) ** 2,
            )
            for lz, lx in search_order:
                ok, fp = self._fits(lz, lx, ww, wd, orchard_set, road_map, occupied)
                if not ok:
                    continue
                fp_hs = [int(feat[fz, fx]["height"]) for fz, fx in fp]
                if max(fp_hs) - min(fp_hs) > 0:
                    continue
                wx, wz = terrain_map.local_to_world(lz, lx)
                surface_y = fp_hs[0]
                with editor.pushTransform((wx, surface_y, wz)):
                    self._placer.place_structure(editor, well_schem, skip_air=True)
                # Replace command-block shaft placeholders with air so the
                # well interior is hollow rather than filled with command blocks.
                for world_pos, _ in self._placer.command_block_positions:
                    editor.placeBlock(world_pos, _AIR)
                self._placer.command_block_positions.clear()
                occupied.update(fp)
                n_well = 1
                log(f"[Orchard] Placed well '{well_schem}'.")
                break

        # ------------------------------------------------------------------
        # Pass 4 – scatter decorations
        # ------------------------------------------------------------------
        n_decor = 0
        if self._decor_schematics:
            for lz, lx in sorted(orchard_set):
                if (lz, lx) in excl or self._on_road(lz, lx, road_map):
                    continue
                if (lz, lx) in occupied:
                    continue
                if (lz, lx) in tree_cells:
                    continue
                if random.random() >= self.DECOR_CHANCE:
                    continue

                schem_path, sw, sd = random.choice(self._decor_schematics)
                ok, fp = self._fits(lz, lx, sw, sd, orchard_set, road_map, occupied)
                if not ok:
                    continue
                fp_hs = [int(feat[fz, fx]["height"]) for fz, fx in fp]
                if max(fp_hs) - min(fp_hs) > 0:
                    continue

                wx, wz = terrain_map.local_to_world(lz, lx)
                surface_y = fp_hs[0]
                with editor.pushTransform((wx, surface_y, wz)):
                    self._placer.place_structure(editor, schem_path, skip_air=True)
                occupied.update(fp)
                n_decor += 1

        # Expose every cell we worked on so the caller's shared occupied set
        # reflects the orchard.  Later phases (gardens, misc) then skip it.
        if pre_occupied is not None:
            for lz, lx in orchard_set:
                if not self._on_road(lz, lx, road_map) and (lz, lx) not in excl:
                    pre_occupied.add((lz, lx))

        log(f"[Orchard] {n_trees} trees, {n_well} well, {n_decor} decorations placed.")
