import ast
import random
from collections import deque
from pathlib import Path
from typing import FrozenSet, List, Optional, Set, Tuple

import numpy as np
from gdpc.block import Block
from gdpc.editor import Editor

from blocks.block_palette import BlockPalette
from blocks.block_processor import BlockProcessor
from builds._vegetation import clear_vegetation
from consts import BUILD_DIR, BUILD_OUTPUT_DIR
from furniture.furniture_placer import FurniturePlacer
from furniture.loot import container_loot_data
from structures.structure_cache import StructureCache
from structures.structure_placer import StructurePlacer
from terrain import _ROAD_BLOCKS
from terrain.terrain_types import SubZone, TerrainMap
from utils import log, rotate_offset

# The 16 uniform wool colours a boat sail can be dyed.
_WOOL_COLOURS = [
    "white",
    "orange",
    "magenta",
    "light_blue",
    "yellow",
    "lime",
    "pink",
    "gray",
    "light_gray",
    "cyan",
    "purple",
    "blue",
    "brown",
    "green",
    "red",
    "black",
]

# Blocks that count as "water in the way" when drying out a moored hull.
_WATER_LIKE = {
    "minecraft:water",
    "minecraft:kelp",
    "minecraft:kelp_plant",
    "minecraft:seagrass",
    "minecraft:tall_seagrass",
    "minecraft:bubble_column",
    "minecraft:ice",
    "minecraft:frosted_ice",
}

# ---------------------------------------------------------------------------
# Wood-type palettes
# ---------------------------------------------------------------------------
_WOOD_PALETTES = [
    (
        "oak_log",
        "oak_planks",
        "oak_slab",
        "oak_stairs",
        "oak_fence",
        "oak_trapdoor",
        "dirt_path",
        "oak",
    ),
    (
        "spruce_log",
        "spruce_planks",
        "spruce_slab",
        "spruce_stairs",
        "spruce_fence",
        "spruce_trapdoor",
        "packed_mud",
        "spruce",
    ),
    (
        "dark_oak_log",
        "dark_oak_planks",
        "dark_oak_slab",
        "dark_oak_stairs",
        "dark_oak_fence",
        "dark_oak_trapdoor",
        "coarse_dirt",
        "dark_oak",
    ),
]

_AIR = Block("minecraft:air")

_WATER_PCT_THRESHOLD = 0.40
_MIN_OCEAN_CELLS = 32
_LANDLOCKED_WATER_THRESHOLD = 200


_TIER_XS = "xs"
_TIER_SM = "sm"
_TIER_MD = "md"
_TIER_LG = "lg"

_MIN_PIER_WIDTH = 24
_PIER_DEPTH: dict = {_TIER_XS: 5, _TIER_SM: 8, _TIER_MD: 10, _TIER_LG: 14}
BOARDWALK_DEPTH: dict = {_TIER_XS: 3, _TIER_SM: 5, _TIER_MD: 7, _TIER_LG: 10}
_MAX_CRANES_BY_TIER: dict = {_TIER_XS: 0, _TIER_SM: 0, _TIER_MD: 1, _TIER_LG: 2}
_DECOR_CHANCE_BY_TIER: dict = {
    _TIER_XS: 0.03,
    _TIER_SM: 0.05,
    _TIER_MD: 0.06,
    _TIER_LG: 0.08,
}

_SCHEM_TIER_SMALL = {"small_", "wheelbarrow", "crate", "barrel_pile"}
_SCHEM_TIER_MEDIUM = {"medium_", "cart", "er1", "er3", "shack", "cargo_pile"}
_SCHEM_TIER_LARGE = {"big_", "crane", "warehouse", "boat_docked", "dockmaster"}


def _schem_tier(stem: str) -> str:
    s = stem.lower()
    for prefix in _SCHEM_TIER_LARGE:
        if s.startswith(prefix):
            return "large"
    for prefix in _SCHEM_TIER_MEDIUM:
        if s.startswith(prefix):
            return "medium"
    return "small"


def _water_cells_to_tier(n: int) -> str:
    if n < 10:
        return _TIER_XS
    if n < 30:
        return _TIER_SM
    if n < 70:
        return _TIER_MD
    return _TIER_LG


def _water_span(water_cells: Set[Tuple[int, int]]) -> int:
    if not water_cells:
        return 0
    zs = [z for z, _ in water_cells]
    xs = [x for _, x in water_cells]
    return max(max(zs) - min(zs), max(xs) - min(xs))


def _make_harbor_palette(wood: str) -> BlockPalette:
    unidirectional = {
        "log": [f"{wood}_log"],
        "planks": [f"{wood}_planks"],
        "stairs": [f"{wood}_stairs"],
        "slab": [f"{wood}_slab"],
        "fence": [f"{wood}_fence"],
        "gate": [f"{wood}_fence_gate"],
        "sign": [f"{wood}_sign"],
        "trapdoor": [f"{wood}_trapdoor"],
        "door": [f"{wood}_door"],
    }
    wood_types = [
        "oak",
        "spruce",
        "birch",
        "jungle",
        "acacia",
        "dark_oak",
        "mangrove",
        "cherry",
        "pale_oak",
        "bamboo",
    ]
    block_to_palette: dict = {}
    for wt in wood_types:
        for k, v in unidirectional.items():
            block_to_palette[f"{wt}_{k if k != 'gate' else 'fence_gate'}"] = k

    return BlockPalette(
        unidirectional_palettes=unidirectional,
        bidirectional_palettes={},
        block_to_palette=block_to_palette,
    )


def _split_components(cells: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
    """Split *cells* into 4-connected components."""
    remaining = set(cells)
    out: List[Set[Tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        comp = {seed}
        queue: deque = deque([seed])
        while queue:
            lz, lx = queue.popleft()
            for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (lz + dz, lx + dx)
                if nb in remaining:
                    remaining.discard(nb)
                    comp.add(nb)
                    queue.append(nb)
        out.append(comp)
    return out


def _trim_water_cells_bfs(
    water_cells: Set[Tuple[int, int]], shore_cells: Set[Tuple[int, int]], max_depth: int
) -> Set[Tuple[int, int]]:
    if max_depth <= 0 or not shore_cells:
        return set()
    visited: Set[Tuple[int, int]] = set(shore_cells)
    queue: deque = deque([(cell, 0) for cell in shore_cells])
    result: Set[Tuple[int, int]] = set()

    while queue:
        (lz, lx), depth = queue.popleft()
        for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (lz + dz, lx + dx)
            if nb in water_cells and nb not in visited:
                visited.add(nb)
                result.add(nb)
                if depth + 1 < max_depth:
                    queue.append((nb, depth + 1))
    return result


class HarborGenerator:
    SCHEMATIC_DIR = "harbor"
    DECK_ABOVE_WATER = 1
    PILING_DEPTH = 8
    PILING_SPACING = 3  # a support piling every N cells; the deck spans the gaps
    CLEAR_HEIGHT = 14
    MIN_PIER_FOR_CRANE = 6
    WATER_SCAN_RADIUS = 16

    BOAT_PATH = "boat/boat.csv"
    BOAT_SUBMERGE = 2  # keel sits this many blocks below the water surface
    BOAT_MAX_ANCHORS = 900  # cap the moor-spot search for performance
    BOAT_BERTH_REACH = 24  # how far beyond the district the berth may extend
    BOAT_BERTH_MAX_CELLS = 2000  # cap on extra berth water collected by BFS

    def __init__(self) -> None:
        (
            piling_id,
            deck_id,
            slab_id,
            stair_id,
            fence_id,
            trapdoor_id,
            path_id,
            wood_label,
        ) = random.choice(_WOOD_PALETTES)
        self._piling = Block(piling_id, states={"axis": "y"})
        self._deck_planks = Block(deck_id)
        self._stripped_log = Block(f"stripped_{wood_label}_log", states={"axis": "y"})
        self._slab = Block(slab_id)
        self._stairs = Block(stair_id)
        self._fence = Block(fence_id)
        self._trapdoor = Block(trapdoor_id)
        self._path = Block(path_id)
        self._leaves = Block(
            f"{wood_label}_leaves", states={"distance": "7", "persistent": "true"}
        )
        self._wood = wood_label

        log(f"[Harbor] Initialized with wood type: {wood_label}")
        self._cache = StructureCache(BUILD_DIR, BUILD_OUTPUT_DIR)
        self._placer = StructurePlacer(
            structure_cache=self._cache,
            block_processor=BlockProcessor(),
            block_palette=_make_harbor_palette(wood_label),
            furniture_replacer=None,
        )

        self._crane_schematics: List[
            Tuple[str, int, int, FrozenSet[Tuple[int, int]]]
        ] = []
        self._decor_schematics: List[Tuple[str, int, int, str]] = []
        self._load_schematics()

    def _load_schematics(self) -> None:
        schem_dir = Path(BUILD_DIR) / BUILD_OUTPUT_DIR / self.SCHEMATIC_DIR
        if not schem_dir.is_dir():
            return
        for csv_file in sorted(schem_dir.glob("*.csv")):
            rel = f"{self.SCHEMATIC_DIR}/{csv_file.name}"
            stem = csv_file.stem.lower()
            try:
                df, w, _, d = self._cache.load_structure(rel)
                if "crane" in stem:
                    base_fp = self._base_xz_cells(df)
                    self._crane_schematics.append((rel, w, d, base_fp))
                else:
                    tier = _schem_tier(stem)
                    self._decor_schematics.append((rel, w, d, tier))
            except Exception as e:
                log(f"[Harbor] Error loading schematic {rel}: {e}")

    def _place_boat(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        feat: np.ndarray,
        water_cells: Set[Tuple[int, int]],
        pier_cells: Set[Tuple[int, int]],
        occupied: Set[Tuple[int, int]],
        sea_level: int,
        pre_occupied: Optional[Set[Tuple[int, int]]],
    ) -> bool:
        """
        Moor the boat schematic in open water beside the dock.

        The hull is seated ``BOAT_SUBMERGE`` blocks below the water surface so it
        rides in the water.  Its all-white sail is dyed a single random wool
        colour, the furniture markers inside are dressed like any other
        structure (with harbour-themed loot), and the submerged hull columns are
        drained so the vessel is not full of water — the open sea around it is
        left untouched.

        The berth search is NOT limited to water inside the harbor district:
        it extends up to ``BOAT_BERTH_REACH`` cells into connected open water
        beyond the district border, so a thin shoreline sliver of a district
        still gets its boat moored in the adjacent sea.  Before placement the
        berth airspace is cleared of non-water obstructions (lily pads, ice,
        overhanging swamp/mangrove canopy) that would clip through the vessel.

        Returns True if a boat was placed.
        """
        try:
            df, W, H, D = self._cache.load_structure(self.BOAT_PATH)
        except Exception as e:
            log(f"[Harbor] Boat schematic unavailable ({e}); skipping boat.")
            return False

        # Schematic columns: every (x, z) the boat occupies, and the subset that
        # reaches the waterline (a solid block at or below the surface) — only
        # those get drained, so an overhanging sail never punches a hole in the
        # sea beside the hull.
        foot_cols: Set[Tuple[int, int]] = set()
        hull_cols: Set[Tuple[int, int]] = set()
        for pos in df["Position"].map(ast.literal_eval):
            sx, sy, sz = int(pos[0]), int(pos[1]), int(pos[2])
            foot_cols.add((sx, sz))
            if sy <= self.BOAT_SUBMERGE:
                hull_cols.add((sx, sz))
        if not foot_cols:
            return False

        boat_base_y = sea_level - self.BOAT_SUBMERGE
        if boat_base_y < 1:
            return False

        depth_g, width_g = feat.shape
        blocked = occupied | pier_cells

        # Berth water: the district's own water plus connected open water just
        # beyond its border, so the boat can moor in the adjacent sea when the
        # district itself is a thin shoreline sliver.
        berth_water = self._expanded_berth_water(water_cells, feat)

        # Prefer a berth close to the dock so the vessel reads as moored.
        # Blocked cells (the dock itself, earlier boats) are filtered out up
        # front so they don't eat the anchor budget before open water is
        # even considered.
        ref_cells = pier_cells or water_cells
        rz = sum(c[0] for c in ref_cells) / len(ref_cells)
        rx = sum(c[1] for c in ref_cells) / len(ref_cells)
        candidates = sorted(
            (c for c in berth_water if c not in blocked),
            key=lambda c: (c[0] - rz) ** 2 + (c[1] - rx) ** 2,
        )

        chosen: Optional[Tuple[Tuple[int, int], int, List[Tuple[int, int]]]] = None
        for anchor in candidates[: self.BOAT_MAX_ANCHORS]:
            alz, alx = anchor
            for direction in range(4):
                fp: List[Tuple[int, int]] = []
                ok = True
                for sx, sz in foot_cols:
                    ox, oz = rotate_offset(sx, sz, W, D, direction)
                    cell = (alz + oz, alx + ox)
                    if not (0 <= cell[0] < depth_g and 0 <= cell[1] < width_g):
                        ok = False
                        break
                    if cell not in berth_water or cell in blocked:
                        ok = False
                        break
                    fp.append(cell)
                if ok:
                    chosen = (anchor, direction, fp)
                    break
            if chosen is not None:
                break

        if chosen is None:
            log("[Harbor] No open water large enough for the boat; skipping.")
            return False

        (alz, alx), direction, fp = chosen

        # Dye the sail: recolour every white_wool to one uniform wool colour,
        # cached under a colour-specific key so we never mutate the base df.
        colour = random.choice(_WOOL_COLOURS)
        key = f"boat/__boat_{colour}.csv"
        if key not in self._cache._cache:
            df2 = df.copy()
            df2.loc[df2["BlockID"] == "minecraft:white_wool", "BlockID"] = (
                f"minecraft:{colour}_wool"
            )
            self._cache._cache[key] = (df2, W, H, D)

        # Dedicated placer: no harbour wood palette (keep the boat's own mix of
        # woods) but with furniture replacement + harbour-themed container loot.
        boat_placer = StructurePlacer(
            structure_cache=self._cache,
            block_processor=BlockProcessor(),
            block_palette=None,
            furniture_replacer=FurniturePlacer(),
        )

        # Clear the berth airspace: lily pads, snow layers and overhanging
        # swamp/mangrove canopy in the boat's columns would clip through the
        # hull and sail (placement skips air, so it never removes them itself).
        editor.flushBuffer()
        top_y = min(boat_base_y + H, 255)
        for sx, sz in foot_cols:
            ox, oz = rotate_offset(sx, sz, W, D, direction)
            cwx, cwz = terrain_map.local_to_world(alz + oz, alx + ox)
            for wy in range(sea_level + 1, top_y + 1):
                if editor.getBlock((cwx, wy, cwz)).id != "minecraft:air":
                    editor.placeBlock((cwx, wy, cwz), _AIR)

        wx0, wz0 = terrain_map.local_to_world(alz, alx)
        with editor.pushTransform((wx0, boat_base_y, wz0)):
            boat_placer.place_structure(
                editor, key, biome=None, direction=direction, skip_air=True
            )
        boat_placer.place_furniture(editor, loot_category="harbor")

        # Drain the hull: within the submerged columns, turn any trapped water
        # into air (only water — never carve seabed or the boat's own blocks).
        editor.flushBuffer()
        air = Block("air")
        for sx, sz in hull_cols:
            ox, oz = rotate_offset(sx, sz, W, D, direction)
            cwx, cwz = terrain_map.local_to_world(alz + oz, alx + ox)
            for wy in range(boat_base_y, sea_level + 1):
                if editor.getBlock((cwx, wy, cwz)).id in _WATER_LIKE:
                    editor.placeBlock((cwx, wy, cwz), air)

        occupied.update(fp)
        if pre_occupied is not None:
            pre_occupied.update(fp)
        log(
            f"[Harbor] Moored a {colour}-sailed boat at local ({alz},{alx}) "
            f"dir={direction}, base y={boat_base_y}."
        )
        return True

    def _expanded_berth_water(
        self,
        water_cells: Set[Tuple[int, int]],
        feat: np.ndarray,
    ) -> Set[Tuple[int, int]]:
        """
        Grow the berth area from the district's water cells into connected
        open water on the feature map, up to ``BOAT_BERTH_REACH`` steps and
        ``BOAT_BERTH_MAX_CELLS`` extra cells.  Keeps the moor-spot search
        local while freeing the boat from the district's arbitrary border.
        """
        depth, width = feat.shape
        visited: Set[Tuple[int, int]] = set(water_cells)
        queue: deque = deque((c, 0) for c in water_cells)
        extra = 0
        while queue and extra < self.BOAT_BERTH_MAX_CELLS:
            (lz, lx), d = queue.popleft()
            if d >= self.BOAT_BERTH_REACH:
                continue
            for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (lz + dz, lx + dx)
                if nb in visited:
                    continue
                if not (0 <= nb[0] < depth and 0 <= nb[1] < width):
                    continue
                if float(feat[nb[0], nb[1]]["water_pct"]) > _WATER_PCT_THRESHOLD:
                    visited.add(nb)
                    queue.append((nb, d + 1))
                    extra += 1
        return visited

    def _is_edge(self, cell: Tuple[int, int], pier_cells: Set[Tuple[int, int]]) -> bool:
        lz, lx = cell
        return any(
            (lz + dz, lx + dx) not in pier_cells
            for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )

    def _on_road(self, lz: int, lx: int, road_map: Optional[np.ndarray]) -> bool:
        if road_map is None:
            return False
        return (
            0 <= lz < road_map.shape[0]
            and 0 <= lx < road_map.shape[1]
            and road_map[lz, lx] != 0
        )

    def _connected_water_size(
        self,
        seed_cells: Set[Tuple[int, int]],
        feat: np.ndarray,
        cap: int = _LANDLOCKED_WATER_THRESHOLD + 1,
    ) -> int:
        """BFS outward from seed_cells through water cells in feat; returns size (capped)."""
        depth, width = feat.shape
        visited: Set[Tuple[int, int]] = set(seed_cells)
        queue: deque = deque(seed_cells)
        count = len(seed_cells)
        while queue and count < cap:
            lz, lx = queue.popleft()
            for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (lz + dz, lx + dx)
                if nb in visited:
                    continue
                nlz, nlx = nb
                if not (0 <= nlz < depth and 0 <= nlx < width):
                    continue
                if float(feat[nlz, nlx]["water_pct"]) > _WATER_PCT_THRESHOLD:
                    visited.add(nb)
                    queue.append(nb)
                    count += 1
        return count

    def _is_water(self, lz: int, lx: int, feat: np.ndarray) -> bool:
        if not (0 <= lz < feat.shape[0] and 0 <= lx < feat.shape[1]):
            return False
        return float(feat[lz, lx]["water_pct"]) > _WATER_PCT_THRESHOLD

    def _random_stair(self):
        return Block(
            self._stairs.id,
            states={
                "facing": random.choice(["north", "south", "east", "west"]),
                "half": random.choice(["bottom", "top"]),
                "shape": "straight",
            },
        )

    def _fits(
        self,
        lz: int,
        lx: int,
        sw: int,
        sd: int,
        valid_set: Set[Tuple[int, int]],
        road_map: Optional[np.ndarray],
        occupied: Set[Tuple[int, int]],
    ) -> Tuple[bool, Set[Tuple[int, int]]]:
        fp: Set[Tuple[int, int]] = set()
        for dz in range(sd):
            for dx in range(sw):
                cell = (lz + dz, lx + dx)
                if (
                    cell not in valid_set
                    or self._on_road(cell[0], cell[1], road_map)
                    or cell in occupied
                ):
                    return False, set()
                fp.add(cell)
        return True, fp

    def _expand_cells(
        self, cells: Set[Tuple[int, int]], radius: int = 2
    ) -> Set[Tuple[int, int]]:
        result = set(cells)
        for lz, lx in cells:
            for dz in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    result.add((lz + dz, lx + dx))
        return result

    def _base_xz_cells(self, df) -> FrozenSet[Tuple[int, int]]:
        """Return the (x, z) schematic coords occupied at the minimum Y layer."""
        rows = [ast.literal_eval(p) for p in df["Position"]]
        if not rows:
            return frozenset()
        min_y = min(y for _, y, _ in rows)
        return frozenset((x, z) for x, y, z in rows if y == min_y)

    def _rotated_base_offsets(
        self,
        base_xz: FrozenSet[Tuple[int, int]],
        sw: int,
        sd: int,
        direction: int,
    ) -> Set[Tuple[int, int]]:
        """
        Convert schematic base-layer (x, z) coords to (dz, dx) offsets from the
        anchor after rotation.  Rotation convention (looking down, +x=East, +z=South):
          dir 0: identity
          dir 1: 90° CW  → new (dz=x,   dx=sd-1-z)  extents: (sd, sw)
          dir 2: 180°    → new (dz=sd-1-z, dx=sw-1-x) extents: (sw, sd)
          dir 3: 270° CW → new (dz=sw-1-x, dx=z)      extents: (sd, sw)
        """
        offsets: Set[Tuple[int, int]] = set()
        for x, z in base_xz:
            if direction == 0:
                offsets.add((z, x))
            elif direction == 1:
                offsets.add((x, sd - 1 - z))
            elif direction == 2:
                offsets.add((sd - 1 - z, sw - 1 - x))
            else:  # direction == 3
                offsets.add((sw - 1 - x, z))
        return offsets

    def _water_facing_direction(self, lz: int, lx: int, feat: np.ndarray) -> int:
        r = self.WATER_SCAN_RADIUS
        counts = [0, 0, 0, 0]  # N, E, S, W
        for dz in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if not (0 <= lz + dz < feat.shape[0] and 0 <= lx + dx < feat.shape[1]):
                    continue
                if float(feat[lz + dz, lx + dx]["water_pct"]) > _WATER_PCT_THRESHOLD:
                    if abs(dz) >= abs(dx):
                        counts[0 if dz < 0 else 2] += 1
                    else:
                        counts[1 if dx > 0 else 3] += 1
        return int(np.argmax(counts))

    def _generate_rectangular_piers(
        self,
        water_cells: Set[Tuple[int, int]],
        base_boardwalk_cells: Set[Tuple[int, int]],
        max_length: int,
        tier: str,
    ) -> Set[Tuple[int, int]]:
        pier_fingers: Set[Tuple[int, int]] = set()
        if not base_boardwalk_cells:
            return pier_fingers

        pier_width = (
            3 if tier in (_TIER_XS, _TIER_SM) else (5 if tier == _TIER_LG else 4)
        )
        spacing = 7 if tier in (_TIER_XS, _TIER_SM) else 10

        # Only extend piers from the water-facing outer edge of the dock body
        outer_boardwalk_edge = {
            (lz, lx)
            for lz, lx in base_boardwalk_cells
            if any(
                (lz + dz, lx + dx) in water_cells
                and (lz + dz, lx + dx) not in base_boardwalk_cells
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
        }

        edge_list = sorted(list(outer_boardwalk_edge))

        for i in range(0, len(edge_list), spacing):
            slz, slx = edge_list[i]
            best_dir = (1, 0)
            max_water_count = 0

            for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                w_count = 0
                for step in range(1, max_length + 1):
                    target_cell = (slz + dz * step, slx + dx * step)
                    if (
                        target_cell in water_cells
                        and target_cell not in base_boardwalk_cells
                    ):
                        w_count += 1
                if w_count > max_water_count:
                    max_water_count = w_count
                    best_dir = (dz, dx)

            if max_water_count < 4:
                continue

            dz, dx = best_dir
            perp_z, perp_x = -dx, dz

            for step in range(1, max_length + 1):
                cz = slz + dz * step
                cx = slx + dx * step
                # Only add cells that are genuinely in open water (not dock body)
                all_in_water = all(
                    (
                        cz + perp_z * (w - pier_width // 2),
                        cx + perp_x * (w - pier_width // 2),
                    )
                    in water_cells
                    for w in range(pier_width)
                )
                if not all_in_water:
                    break  # stop pier if it would leave water
                for w in range(pier_width):
                    offset_z = cz + perp_z * (w - pier_width // 2)
                    offset_x = cx + perp_x * (w - pier_width // 2)
                    pier_fingers.add((offset_z, offset_x))

        # Guarantee piers never cut into the dock body
        return pier_fingers - base_boardwalk_cells

    def generate(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        *,
        pre_occupied: Optional[Set[Tuple[int, int]]] = None,
    ) -> bool:
        """Build every harbor.  Returns True only if at least one was placed.

        Each connected HARBOR_DOCK patch is built as its OWN harbor — a map
        with several dock districts gets several independent docks, each with
        its own sea level, road connection and moored boats.  (Merging them,
        as before, averaged the sea level across bays, aimed the boat search
        at the mean centroid of all piers — often dry land between two bays —
        and placed at most ONE boat for the entire map.)

        The planner may reserve a HARBOR_DOCK zone that later proves
        unbuildable (no open water, landlocked pond, no land foothold, …); in
        those cases nothing is rendered for that site so callers — the
        chronicle in particular — never claim a harbor that does not exist.
        """
        sub_map = terrain_map.sub_zone_map
        feat = terrain_map.features
        if sub_map is None or feat is None:
            return False
        coords = np.argwhere(sub_map == int(SubZone.HARBOR_DOCK))
        if len(coords) == 0:
            return False

        harbor_all: Set[Tuple[int, int]] = {(int(r[0]), int(r[1])) for r in coords}
        built_any = False
        for comp in sorted(_split_components(harbor_all), key=len, reverse=True):
            if len(comp) < 12:
                continue  # sliver — not worth a dock
            built_any |= self._generate_site(editor, terrain_map, comp, pre_occupied)
        return built_any

    def _generate_site(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        harbor_set: Set[Tuple[int, int]],
        pre_occupied: Optional[Set[Tuple[int, int]]],
    ) -> bool:
        """Build one dock (plus boats) on a single connected harbor patch."""
        road_map = terrain_map.road_map_expanded
        feat = terrain_map.features
        excl: Set[Tuple[int, int]] = set(pre_occupied) if pre_occupied else set()

        water_cells: Set[Tuple[int, int]] = set()
        land_cells: Set[Tuple[int, int]] = set()

        for lz, lx in harbor_set:
            if (lz, lx) in excl or self._on_road(lz, lx, road_map):
                continue
            if self._is_water(lz, lx, feat):
                water_cells.add((lz, lx))
            else:
                land_cells.add((lz, lx))

        # Need BOTH a water body and a land foothold to build a harbor.  If
        # either side is missing/tiny the harbor would either land on a
        # puddle or have nowhere to stand — skip rather than crash on
        # np.median([]) below.
        if not water_cells or len(land_cells) < 3:
            log(
                "[Harbor] Skipping degenerate harbor zone (insufficient water or land)."
            )
            return False
        if len(water_cells) < _MIN_OCEAN_CELLS and len(land_cells) < 3:
            log("[Harbor] Skipping isolated miniature water body/puddle.")
            return False

        # ── Skip landlocked lakes/ponds ──────────────────────────────
        if water_cells:
            connected = self._connected_water_size(water_cells, feat)
            if connected < _LANDLOCKED_WATER_THRESHOLD:
                log(
                    f"[Harbor] Skipping landlocked water body "
                    f"(connected={connected} < {_LANDLOCKED_WATER_THRESHOLD})."
                )
                return False

        if land_cells:
            clear_vegetation(
                list(land_cells),
                feat,
                feat.shape[0],
                feat.shape[1],
                editor,
                terrain_map,
                protected=excl,
            )

        tier = _water_cells_to_tier(len(water_cells))
        max_depth = _PIER_DEPTH[tier]
        crane_cap = _MAX_CRANES_BY_TIER[tier]
        decor_prob = _DECOR_CHANCE_BY_TIER[tier]

        shore_cells: Set[Tuple[int, int]] = {
            (lz, lx)
            for lz, lx in land_cells
            if any(
                (lz + dz, lx + dx) in water_cells
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
        }

        # True water surface: feat["height"] of a water cell is the OCEAN
        # FLOOR, so deriving sea level from it seats boats (keel = sea_level
        # - BOAT_SUBMERGE) on the river bed wherever the water is deep.  Use
        # the surface heightmap for the real water level and only fall back
        # to the floor-based estimate when it is unavailable.
        surf = terrain_map.surface_heightmap
        if surf is not None:
            sea_level = int(
                np.median([int(surf[lz, lx]) - 1 for lz, lx in water_cells])
            )
        else:
            sea_level = int(
                np.median([int(feat[lz, lx]["height"]) - 1 for lz, lx in water_cells])
            )
        land_height = (
            int(np.median([int(feat[lz, lx]["height"]) - 1 for lz, lx in shore_cells]))
            if shore_cells
            else sea_level
        )
        deck_y = max(sea_level + self.DECK_ABOVE_WATER, land_height)

        # ------------------------------------------------------------------
        # Dock body: a shallow coast-hugging fringe wide enough to walk on.
        # Depth is kept small so the dock silhouette still follows the shore
        # contour rather than flood-filling deep into the water body.
        # ------------------------------------------------------------------
        _DOCK_FRINGE_DEPTH = {_TIER_XS: 2, _TIER_SM: 3, _TIER_MD: 4, _TIER_LG: 5}
        fringe_depth = _DOCK_FRINGE_DEPTH[tier]
        anchor_cells = shore_cells or land_cells
        dock_water_fringe: Set[Tuple[int, int]] = _trim_water_cells_bfs(
            water_cells, anchor_cells, fringe_depth
        )
        base_boardwalk_cells = anchor_cells | dock_water_fringe

        # Piers extend outward from the dock's water-facing edge; guaranteed
        # disjoint from the dock body by the subtraction in _generate_rectangular_piers.
        if _water_span(water_cells) >= _MIN_PIER_WIDTH:
            pier_fingers = self._generate_rectangular_piers(
                water_cells, base_boardwalk_cells, max_depth, tier
            )
        else:
            pier_fingers = set()
        pier_cells: Set[Tuple[int, int]] = base_boardwalk_cells | pier_fingers

        if len(pier_cells) < 2:
            pier_cells = base_boardwalk_cells | land_cells

        occupied: Set[Tuple[int, int]] = set(excl)

        # ------------------------------------------------------------------
        # Pass 1 – Core Structural Framing
        # ------------------------------------------------------------------
        for lz, lx in sorted(pier_cells):
            wx, wz = terrain_map.local_to_world(lz, lx)
            surface_y = int(feat[lz, lx]["height"]) - 1

            for dy in range(1, self.CLEAR_HEIGHT + 1):
                editor.placeBlock((wx, deck_y + dy, wz), _AIR)

            piling_bottom = max(0, surface_y - self.PILING_DEPTH)
            is_piling_station = (lz + lx) % self.PILING_SPACING == 0

            if (lz, lx) in dock_water_fringe:
                # Spaced stone pilings under the coast-hugging dock; the deck
                # spans the open water between them instead of a solid quay fill.
                if is_piling_station:
                    for y in range(piling_bottom, deck_y):
                        editor.placeBlock((wx, y, wz), Block("stone_bricks"))

            elif (lz, lx) in pier_fingers:
                # Spaced timber pilings under the pier extensions.
                if is_piling_station:
                    for y in range(piling_bottom, deck_y):
                        editor.placeBlock((wx, y, wz), self._piling)
                    # Cross-brace to a neighbouring piling just below the deck.
                    for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if (lz + dz, lx + dx) in pier_cells:
                            editor.placeBlock(
                                (wx + dx, deck_y - 1, wz + dz), self._stripped_log
                            )
            else:
                # Land/shore dock cell — level the ground up to the deck (solid).
                for y in range(min(surface_y, deck_y), max(surface_y, deck_y) + 1):
                    editor.placeBlock(
                        (wx, y, wz),
                        self._piling
                        if random.random() < 0.2
                        else Block("stone_bricks"),
                    )

            floor_block = (
                [self._stripped_log] * 7
                + [self._deck_planks] * 40
                + [self._random_stair() for _ in range(3)]
            )
            editor.placeBlock((wx, deck_y, wz), floor_block)

            if self._is_edge((lz, lx), pier_cells):
                editor.placeBlock((wx, deck_y, wz), self._stripped_log)
                edge_rand = random.random()
                if edge_rand < 0.15:
                    editor.placeBlock((wx, deck_y + 1, wz), self._fence)
                elif edge_rand < 0.25:
                    editor.placeBlock((wx, deck_y + 1, wz), self._piling)
                    if random.random() < 0.40:
                        editor.placeBlock(
                            (wx, deck_y + 2, wz),
                            Block("lantern", states={"hanging": "false"}),
                        )

        # ------------------------------------------------------------------
        # Pass 2 – Advanced Micro-Cluttering & Clustering
        # ------------------------------------------------------------------
        for lz, lx in pier_cells:
            if (
                (lz, lx) in occupied
                or random.random() > 0.05
                or self._is_edge((lz, lx), pier_cells)
            ):
                continue

            wx, wz = terrain_map.local_to_world(lz, lx)
            choice = random.randint(0, 8)

            blocks_to_place = [(wx, wz)]
            if random.random() < 0.4:
                for dz, dx in ((1, 0), (0, 1), (-1, 0)):
                    if (
                        (lz + dz, lx + dx) in pier_cells
                        and not self._is_edge((lz + dz, lx + dx), pier_cells)
                        and random.random() < 0.4
                    ):
                        blocks_to_place.append(
                            terrain_map.local_to_world(lz + dz, lx + dx)
                        )

            for cx, cz in blocks_to_place:
                if choice <= 1:
                    editor.placeBlock(
                        (cx, deck_y + 1, cz),
                        Block(
                            "barrel",
                            states={"facing": "up"},
                            data=container_loot_data("harbor", is_barrel=True),
                        ),
                    )
                elif choice == 2:
                    editor.placeBlock(
                        (cx, deck_y + 1, cz),
                        Block(
                            "chest",
                            states={
                                "facing": random.choice(
                                    ["north", "south", "east", "west"]
                                )
                            },
                            data=container_loot_data("harbor"),
                        ),
                    )
                elif choice == 3:
                    editor.placeBlock(
                        (cx, deck_y + 1, cz),
                        Block("composter", states={"level": str(random.randint(0, 8))}),
                    )
                elif choice == 4:
                    editor.placeBlock((cx, deck_y + 1, cz), self._fence)
                    editor.placeBlock(
                        (cx, deck_y + 2, cz),
                        Block("lantern", states={"hanging": "false"}),
                    )
                elif choice == 5:
                    editor.placeBlock((cx, deck_y + 1, cz), Block("raw_iron_block"))
                elif choice == 6:
                    editor.placeBlock(
                        (cx, deck_y + 1, cz),
                        Block(
                            "potted_mangrove_propagule"
                            if random.random() < 0.5
                            else "potted_fern"
                        ),
                    )
                else:
                    editor.placeBlock((cx, deck_y + 1, cz), self._trapdoor)

                occupied.add((lz, lx))

        # ------------------------------------------------------------------
        # Pass 3 – Structural Cranes Placement
        # ------------------------------------------------------------------
        n_cranes = 0
        if (
            self._crane_schematics
            and crane_cap > 0
            and len(base_boardwalk_cells) >= self.MIN_PIER_FOR_CRANE
        ):
            available = base_boardwalk_cells - occupied
            crane_limit = min(crane_cap, max(1, len(pier_cells) // 10))

            for lz, lx in sorted(available, key=lambda c: (c[0], c[1])):
                if n_cranes >= crane_limit:
                    break
                if (lz, lx) in occupied:
                    continue

                schem_path, sw, sd, base_xz = self._crane_schematics[-1]

                # The y=0 base layer must sit on the dock body; the rest of
                # the bounding box (the log arm) must stay above DECK cells —
                # otherwise the arm ends up as floating logs over open water.
                direction = self._water_facing_direction(lz, lx, feat)
                base_offsets = self._rotated_base_offsets(base_xz, sw, sd, direction)
                if not all(
                    (lz + dz, lx + dx) in base_boardwalk_cells
                    and not self._on_road(lz + dz, lx + dx, road_map)
                    and (lz + dz, lx + dx) not in occupied
                    for dz, dx in base_offsets
                ):
                    continue
                ew, ed = (sd, sw) if direction % 2 == 1 else (sw, sd)
                fp = {(lz + dz, lx + dx) for dz in range(ed) for dx in range(ew)}
                if not all(c in pier_cells for c in fp):
                    continue
                wx, wz = terrain_map.local_to_world(lz, lx)

                with editor.pushTransform((wx, deck_y + 1, wz)):
                    self._placer.place_structure(
                        editor, schem_path, direction=direction, skip_air=False
                    )

                occupied.update(self._expand_cells({(lz, lx)}, radius=7))
                n_cranes += 1

        # ------------------------------------------------------------------
        # Pass 4 – Grid Decoration Blueprint Scaling
        # ------------------------------------------------------------------
        allowed_tiers = {"small"}
        if tier == _TIER_MD:
            allowed_tiers.add("medium")
        elif tier == _TIER_LG:
            allowed_tiers.update(["medium", "large"])

        eligible_decor = [p for p in self._decor_schematics if p[3] in allowed_tiers]
        n_decor = 0

        if eligible_decor:
            for lz, lx in sorted(pier_cells):
                if (lz, lx) in occupied or random.random() >= decor_prob:
                    continue
                schem_path, sw, sd, _t = random.choice(eligible_decor)

                if _t in ["medium", "large"] and (lz, lx) not in base_boardwalk_cells:
                    continue

                # Shuffle rotations so placement isn't biased toward direction 0.
                # For each candidate direction, check the correctly-sized footprint.
                # direction % 2 == 1  →  width and depth swap (90° / 270° rotation).
                directions = list(range(4))
                random.shuffle(directions)
                for direction in directions:
                    ew, ed = (sd, sw) if direction % 2 == 1 else (sw, sd)
                    ok, fp = self._fits(lz, lx, ew, ed, pier_cells, road_map, occupied)
                    if ok:
                        wx, wz = terrain_map.local_to_world(lz, lx)
                        with editor.pushTransform((wx, deck_y + 1, wz)):
                            self._placer.place_structure(
                                editor, schem_path, direction=direction, skip_air=True
                            )
                        occupied.update(self._expand_cells(fp, radius=3))
                        n_decor += 1
                        break

        # ------------------------------------------------------------------
        # Pass 5 – Infrastructure Connectivity (Smooth Gradient Handshake)
        # ------------------------------------------------------------------
        raw_road_map = terrain_map.road_map
        if raw_road_map is not None:
            road_cells_arr = np.argwhere(raw_road_map > 0)
            if len(road_cells_arr) > 0 and (shore_cells or land_cells):
                candidates = list(shore_cells) or list(land_cells)
                entrance = min(
                    candidates,
                    key=lambda c: float(
                        np.min(
                            np.sum(
                                (road_cells_arr - np.array([[c[0], c[1]]])) ** 2, axis=1
                            )
                        )
                    ),
                )

                from terrain.road_network import _astar

                depth, width = feat.shape
                ez, ex = entrance
                best = int(
                    np.argmin(
                        np.sum((road_cells_arr - np.array([[ez, ex]])) ** 2, axis=1)
                    )
                )
                target = (int(road_cells_arr[best, 0]), int(road_cells_arr[best, 1]))

                path = _astar(
                    feat,
                    terrain_map.district_map,
                    terrain_map.districts,
                    entrance,
                    target,
                    bridge_set=set(),
                    cost_mod=None,
                    max_visits=8000,
                    depth=depth,
                    width=width,
                    exclude=(occupied | pier_cells) - {entrance},
                )

                if path:
                    target_road_y = int(feat[target[0], target[1]]["height"]) - 1
                    path_len = len(path)

                    for index, (plz, plx) in enumerate(path):
                        if raw_road_map[plz, plx] > 0 and index > 0:
                            break
                        t = index / max(1, path_len - 1)
                        lerp_y = int(round(deck_y + t * (target_road_y - deck_y)))
                        pwx, pwz = terrain_map.local_to_world(plz, plx)

                        next_y = lerp_y
                        nlz, nlx = plz, plx
                        if index < path_len - 1:
                            nlz, nlx = path[index + 1]
                            next_y = int(
                                round(
                                    deck_y
                                    + ((index + 1) / max(1, path_len - 1))
                                    * (target_road_y - deck_y)
                                )
                            )

                        delta_y = next_y - lerp_y

                        # Clear air above path
                        for dy in range(1, 4):
                            editor.placeBlock((pwx, lerp_y + dy, pwz), Block("air"))

                        # Support below the path: spaced timber pilings over
                        # water (the walkway spans the gaps), solid stone on land.
                        natural_surface = int(feat[plz, plx]["height"]) - 1
                        if (plz, plx) in water_cells:
                            if (plz + plx) % self.PILING_SPACING == 0:
                                bottom = max(0, natural_surface - self.PILING_DEPTH)
                                for y in range(bottom, lerp_y):
                                    editor.placeBlock((pwx, y, pwz), self._piling)
                        else:
                            for y in range(
                                max(0, min(natural_surface, lerp_y)), lerp_y
                            ):
                                editor.placeBlock((pwx, y, pwz), Block("stone_bricks"))

                        # Surface block: stair facing direction of travel, flat path otherwise
                        if delta_y != 0:
                            dz_dir = nlz - plz
                            dx_dir = nlx - plx
                            if dz_dir != 0 or dx_dir != 0:
                                facing = (
                                    "south"
                                    if dz_dir > 0
                                    else "north"
                                    if dz_dir < 0
                                    else "east"
                                    if dx_dir > 0
                                    else "west"
                                )
                            else:
                                facing = "south"
                            half = "bottom" if delta_y > 0 else "top"
                            editor.placeBlock(
                                (pwx, lerp_y, pwz),
                                Block(
                                    self._stairs.id,
                                    states={
                                        "facing": facing,
                                        "half": half,
                                        "shape": "straight",
                                    },
                                ),
                            )
                        else:
                            editor.placeBlock(
                                (pwx, lerp_y, pwz),
                                random.choice(_ROAD_BLOCKS),
                            )

        if pre_occupied is not None:
            pre_occupied.update(pier_cells)

        # Moor boats in the open water beside the dock (stops as soon as no
        # further berth fits).  Bigger harbors host a small fleet.
        n_boats = {_TIER_XS: 1, _TIER_SM: 1, _TIER_MD: 2, _TIER_LG: 3}[tier]
        for _ in range(n_boats):
            if not self._place_boat(
                editor,
                terrain_map,
                feat,
                water_cells,
                pier_cells,
                occupied,
                sea_level,
                pre_occupied,
            ):
                break

        log(
            f"[Harbor] Rendered structured tier={tier} harbor body with clustered cargo, vegetation, and zoned large structures."
        )
        return True
