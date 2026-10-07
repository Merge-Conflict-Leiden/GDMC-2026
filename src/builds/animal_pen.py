"""
This module defines a generator for ANIMAL_PEN-zoned districts.

Layout strategy
---------------
  • All cells receive a biome-appropriate earthy ground replacement (podzol,
    coarse_dirt, grass_block depending on biome).
  • Edge cells (adjacent to non-pen territory) get a fence post so the pen is
    enclosed.  On sloped terrain, fence posts are stacked up to the higher
    neighbour's ground level so no gaps appear (step pattern).
  • A gate is placed at a road-adjacent border cell and oriented to face the
    road.  Internal divider rows each get a centred gate oriented along the
    divider run.
  • Internal fence rows are added every PEN_DIVISION blocks in Z, dividing the
    district into multiple sub-pens.
  • The pen itself is kept realistic: it sits on one terrain terrace (total
    relief capped), 1-cell-wide tendrils are pruned so the fence line reads
    as a deliberate paddock, and the area is capped so a huge district yields
    a paddock rather than a fenced county.
  • One barn schematic (the largest that fits) is placed in the interior and
    the terrain under its footprint is flattened to the median height.  All
    four rotations of several candidate anchors are scored: the entrance
    (jigsaw markers) must open onto level, in-pen ground — never straight
    into a wall of terrain — and sites that hug rising ground or need heavy
    terraforming are penalised.
  • Biome-appropriate livestock (a mix of 1–2 fence-safe species per pen) are
    summoned in the interior after the buffer is flushed so they spawn inside the
    completed fence.
"""

import ast
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from gdpc.block import Block
from gdpc.editor import Editor

from blocks.block_processor import BlockProcessor
from builds._vegetation import clear_vegetation
from consts import BUILD_DIR, BUILD_OUTPUT_DIR
from structures.structure_cache import StructureCache
from structures.structure_placer import StructurePlacer
from terrain.terrain_types import SubZone, TerrainMap
from utils import log, rotate_offset, seated_path_height

_AIR = Block("minecraft:air")
_DIRT = Block("minecraft:dirt")
_MUD = Block("minecraft:packed_mud")
_FENCE_BLOCKS = (Block("oak_fence"), Block("spruce_fence"))


def _load_jigsaw_offsets(rel_path: str) -> List[Tuple[int, int]]:
    """Return (sx, sz) for every minecraft:jigsaw block in the schematic CSV."""
    full = Path(BUILD_DIR) / BUILD_OUTPUT_DIR / rel_path
    try:
        df = pd.read_csv(full).dropna(subset=["BlockID"])
        mask = df["BlockID"].str.contains("jigsaw", case=False, na=False)
        if not mask.any():
            return []
        pos = df.loc[mask, "Position"].map(ast.literal_eval)
        coords = pd.DataFrame(pos.tolist(), columns=["x", "y", "z"])
        return [(int(r.x), int(r.z)) for r in coords.itertuples(index=False)]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Biome lookup tables
# ---------------------------------------------------------------------------

# Only large, fence-safe livestock — no goats (long-jump over fences), rabbits
# or frogs (small/twitchy escapees).  Camels, donkeys, mules, pandas and
# sniffers are all big and slow enough to stay penned, so they join the biome-
# appropriate pools below.  A pen mixes 1–2 of its biome's species.
_BIOME_ANIMALS: Dict[str, List[str]] = {
    "snowy": ["sheep", "cow"],
    "frozen": ["sheep", "cow"],
    "taiga": ["pig", "sheep", "cow", "donkey"],
    "desert": ["llama", "horse", "camel", "donkey", "mule"],
    "jungle": ["pig", "cow", "panda", "sniffer"],
    "savanna": ["horse", "cow", "llama", "donkey", "mule"],
    "swamp": ["pig", "cow", "sniffer"],
    "mushroom": ["mooshroom"],
    "meadow": ["cow", "sheep", "horse", "donkey", "sniffer"],
    "default": ["cow", "sheep", "pig", "horse", "donkey", "mule"],
}

_BIOME_GROUND: Dict[str, Tuple[Block, ...]] = {
    "snowy": (Block("dirt"), Block("coarse_dirt")),
    "desert": (Block("coarse_dirt"), Block("coarse_dirt"), Block("sand")),
    "mushroom": (Block("mycelium"),),
    "swamp": (Block("podzol"), Block("podzol"), Block("coarse_dirt")),
    "default": (
        Block("coarse_dirt"),
        Block("coarse_dirt"),
        Block("podzol"),
        Block("grass_block"),
    ),
}


def _biome_key(biome: str) -> str:
    for key in _BIOME_ANIMALS:
        if key != "default" and key in biome:
            return key
    return "default"


def _fence_material(_biome: str) -> str:
    """Randomly pick oak or spruce for the pen's fence material."""
    return random.choice(["oak", "spruce"])


def _gate_facing_to_road(lz: int, lx: int, road_map: Optional[np.ndarray]) -> str:
    """Return the fence-gate facing that points toward the adjacent road."""
    if road_map is not None:
        for dz, dx, face in [
            (-1, 0, "north"),
            (1, 0, "south"),
            (0, 1, "east"),
            (0, -1, "west"),
        ]:
            nlz, nlx = lz + dz, lx + dx
            if (
                0 <= nlz < road_map.shape[0]
                and 0 <= nlx < road_map.shape[1]
                and road_map[nlz, nlx] != 0
            ):
                return face
    return "north"


def _largest_component(cells: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Return the largest 4-connected component of *cells*."""
    remaining = set(cells)
    best: Set[Tuple[int, int]] = set()
    while remaining:
        seed = remaining.pop()
        comp = {seed}
        queue = [seed]
        while queue:
            lz, lx = queue.pop()
            for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (lz + dz, lx + dx)
                if nb in remaining:
                    remaining.discard(nb)
                    comp.add(nb)
                    queue.append(nb)
        if len(comp) > len(best):
            best = comp
    return best


def _gate_facing_by_run(lz: int, lx: int, all_fence: Set[Tuple[int, int]]) -> str:
    """Return gate facing based on which axis the fence runs along."""
    runs_ew = (lz, lx - 1) in all_fence or (lz, lx + 1) in all_fence
    # facing="north" → gate bar runs E-W, posts on E+W, connects to E-W fence
    # facing="east"  → gate bar runs N-S, posts on N+S, connects to N-S fence
    return "north" if runs_ew else "east"


def _gate_facing_along_road(lz: int, lx: int, road_map: Optional[np.ndarray]) -> str:
    """
    Return gate facing parallel to the road direction through this cell.

    A gate lets traffic through along the direction it faces, so a road
    running N-S needs facing='north' (gate posts on E/W sides).
    """
    if road_map is None:
        return "north"
    ns = any(
        0 <= lz + dz < road_map.shape[0]
        and 0 <= lx < road_map.shape[1]
        and road_map[lz + dz, lx] != 0
        for dz in (-1, 1)
    )
    return "north" if ns else "east"


class AnimalPenGenerator:
    """Generate ANIMAL_PEN districts as fenced livestock areas with barns."""

    SCHEMATIC_DIR = "animal_pen"
    WINDMILL_EXCLUSION_RADIUS = 8
    BARN_CHANCE = 0.80  # probability of attempting barn placement per district
    MAX_ZONE_SLOPE = 2.5  # skip entire zone if mean slope exceeds this
    MAX_CELL_SLOPE = 2.0  # individual cells steeper than this are unusable
    MIN_PEN_CELLS = 40  # skip the zone if fewer usable cells remain
    MAX_PEN_RELIEF = 3  # pen stays on one terrace: max height from the median
    MAX_PEN_CELLS = 1200  # cap: paddock, not a fenced county
    BARN_CANDIDATE_FITS = 30  # how many fitting (anchor, rotation) sites to score
    MAX_FENCE_EXTRA = (
        3  # fence post can be at most this many blocks above its own surface_y
    )
    CHICKEN_PEN_NAME = "chicken_pen"
    CHICKEN_PEN_CHANCE = 0.30  # probability of placing the chicken pen decoration
    OVERLAY_PREFIX = "over_"
    OVERLAY_CHANCE = 0.65  # probability of placing an over_* decoration
    PEN_DIVISION = 14  # internal fence divider every N blocks in Z
    ANIMAL_DENSITY = 8  # ~one animal per N interior cells
    MAX_ANIMALS = 20
    CLEAR_HEIGHT = 8
    BARN_CLEAR_HEIGHT = 22  # generous clear for tall barn structures

    def __init__(self) -> None:
        self._cache = StructureCache(BUILD_DIR, BUILD_OUTPUT_DIR)
        self._placer = StructurePlacer(
            structure_cache=self._cache,
            block_processor=BlockProcessor(),
            block_palette=None,
            furniture_replacer=None,
        )
        self._schematics: List[Tuple[str, int, int, List[Tuple[int, int]]]] = []
        self._chicken_pen_schem: Optional[Tuple[str, int, int]] = None
        self._overlay_schematics: List[Tuple[str, int, int]] = []
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
            try:
                _, w, _, d = self._cache.load_structure(rel)
                stem = csv_file.stem.lower()
                if self.CHICKEN_PEN_NAME in stem:
                    self._chicken_pen_schem = (rel, w, d)
                    log(f"[AnimalPen] Loaded chicken pen schematic {rel} ({w}w × {d}d)")
                elif stem.startswith(self.OVERLAY_PREFIX):
                    self._overlay_schematics.append((rel, w, d))
                    log(f"[AnimalPen] Loaded overlay schematic {rel} ({w}w × {d}d)")
                else:
                    jig = _load_jigsaw_offsets(rel)
                    self._schematics.append((rel, w, d, jig))
                    log(
                        f"[AnimalPen] Loaded schematic {rel} ({w}w × {d}d jig={len(jig)})"
                    )
            except Exception as e:
                log(f"[AnimalPen] Could not load {rel}: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _windmill_exclusion(self, sites: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        r = self.WINDMILL_EXCLUSION_RADIUS
        ex: Set[Tuple[int, int]] = set()
        for wz, wx in sites:
            for dz in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dz * dz + dx * dx <= r * r:
                        ex.add((wz + dz, wx + dx))
        return ex

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
        pen_set: Set[Tuple[int, int]],
        road_map: Optional[np.ndarray],
    ) -> Tuple[bool, Set[Tuple[int, int]]]:
        fp: Set[Tuple[int, int]] = set()
        for dz in range(sd):
            for dx in range(sw):
                cell = (lz + dz, lx + dx)
                if cell not in pen_set or self._on_road(cell[0], cell[1], road_map):
                    return False, set()
                fp.add(cell)
        return True, fp

    @staticmethod
    def _grounded_y(fp: Set[Tuple[int, int]], feat: np.ndarray) -> Optional[int]:
        """Surface y if every footprint cell sits at the same ground height.

        Decorations are placed as-is (no terraforming like the barn), so the
        footprint must touch the ground everywhere — one cell lower means a
        floating corner, one cell higher means clipping into terrain.
        Returns None when the footprint is not uniformly grounded.
        """
        heights = {int(feat[lz, lx]["height"]) - 1 for lz, lx in fp}
        return heights.pop() if len(heights) == 1 else None

    def _barn_site_cost(
        self,
        lz: int,
        lx: int,
        sw: int,
        sd: int,
        direction: int,
        fp: Set[Tuple[int, int]],
        barn_y: int,
        heights: List[int],
        jig_offsets: List[Tuple[int, int]],
        feat: np.ndarray,
        pen_set: Set[Tuple[int, int]],
        fence_cells: Set[Tuple[int, int]],
    ) -> float:
        """
        Cost of placing the barn at (lz, lx) with the given rotation.
        Lower is better.  Three concerns, entrance weighted heaviest:

          * doorway (jigsaw) openness — the cells in front of the entrance
            must be inside the pen, off the fence line, and roughly level
            with the barn floor (no wall of terrain against the door);
          * wall-hugging — surrounding ground rising above the barn floor;
          * terraforming — height spread across the footprint itself.
        """
        depth, width = feat.shape
        cost = float(max(heights) - min(heights))

        # Ring around the footprint: rising ground means the barn is shoved
        # against a slope face.
        for flz, flx in fp:
            for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (flz + dz, flx + dx)
                if nb in fp or not (0 <= nb[0] < depth and 0 <= nb[1] < width):
                    continue
                rise = int(feat[nb[0], nb[1]]["height"]) - 1 - barn_y
                if rise > 1:
                    cost += min(rise, 6) * 0.25

        # Doorway openness: probe a few cells straight out from every jigsaw
        # marker along the outward normal of its footprint edge.
        ew, ed = (sd, sw) if direction % 2 == 1 else (sw, sd)
        for sx, sz in jig_offsets:
            ox, oz = rotate_offset(sx, sz, sw, sd, direction)
            edge_dist = {
                (0, -1): ox,
                (0, 1): ew - 1 - ox,
                (-1, 0): oz,
                (1, 0): ed - 1 - oz,
            }
            ndz, ndx = min(edge_dist, key=edge_dist.get)
            for step in (1, 2, 3):
                cell = (lz + oz + ndz * step, lx + ox + ndx * step)
                if (
                    not (0 <= cell[0] < depth and 0 <= cell[1] < width)
                    or cell not in pen_set
                    or cell in fence_cells
                ):
                    cost += 2.0  # doorway leads out of the pen / into the fence
                    continue
                rise = int(feat[cell[0], cell[1]]["height"]) - 1 - barn_y
                if rise > 0:
                    cost += rise * 3.0  # terrain wall in front of the door
                elif rise < -2:
                    cost += 1.0  # sheer drop off the doorstep

        return cost

    def _connect_barn_jigsaws(
        self,
        lz: int,
        lx: int,
        jig_offsets: List[Tuple[int, int]],
        fp: Set[Tuple[int, int]],
        road_map: np.ndarray,
        terrain_map: TerrainMap,
        feat: np.ndarray,
        depth: int,
        width: int,
        editor: Editor,
        fence_cells: Optional[Set[Tuple[int, int]]] = None,
        gate_cells: Optional[Set[Tuple[int, int]]] = None,
        excl: Optional[Set[Tuple[int, int]]] = None,
    ) -> None:
        from terrain.road_network import _astar

        road_cells = np.argwhere(road_map > 0)
        if len(road_cells) == 0:
            return

        # Fences are impassable except at gates, so the doorstep track leaves
        # the pen through a gate instead of dead-ending against the fence.
        # The physical city wall is impassable too.  Pre-occupied cells
        # (lantern posts, approach paths) are also blocked so the path doesn't
        # overwrite them.
        blocked: Set[Tuple[int, int]] = set(fp) | terrain_map.wall_cells
        if fence_cells:
            blocked |= fence_cells - (gate_cells or set())
        if excl:
            blocked |= excl

        for sx, sz in jig_offsets:
            jig_lz = lz + sz
            jig_lx = lx + sx
            if not (0 <= jig_lz < depth and 0 <= jig_lx < width):
                continue

            dists = np.sum((road_cells - np.array([[jig_lz, jig_lx]])) ** 2, axis=1)
            best = int(np.argmin(dists))
            target = (int(road_cells[best, 0]), int(road_cells[best, 1]))

            path = _astar(
                feat,
                terrain_map.district_map,
                terrain_map.districts,
                (jig_lz, jig_lx),
                target,
                bridge_set=set(),
                cost_mod=None,
                max_visits=6_000,
                depth=depth,
                width=width,
                exclude=blocked - {(jig_lz, jig_lx)},
            )
            if path is None:
                continue

            for i, (plz, plx) in enumerate(path):
                if road_map[plz, plx] > 0:
                    break
                # Seat toward the lower flanking terrain so the doorstep
                # track lies in the ground, not on a slope shoulder.
                h = seated_path_height(feat, path, i, depth, width)
                pwx, pwz = terrain_map.local_to_world(plz, plx)
                editor.placeBlock((pwx, h, pwz), _MUD)

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def generate(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        *,
        pre_occupied: Optional[Set[Tuple[int, int]]] = None,
        path_cells: Optional[Set[Tuple[int, int]]] = None,
    ) -> None:
        """Place ground, fences, a barn, and summon animals in ANIMAL_PEN zones.

        *path_cells* — cells of jigsaw approach paths (building doorstep
        tracks).  These are walkable infrastructure like roads: where one
        crosses the pen border it gets a gate, and inside the pen it is left
        untouched.
        """
        sub_map = terrain_map.sub_zone_map
        road_map = terrain_map.road_map_expanded
        feat = terrain_map.features

        if sub_map is None or feat is None:
            log("[AnimalPen] Missing map data, skipping.")
            return

        coords = np.argwhere(sub_map == int(SubZone.ANIMAL_PEN))
        if len(coords) == 0:
            log("[AnimalPen] No ANIMAL_PEN zones found.")
            return

        pen_set: Set[Tuple[int, int]] = {(int(r[0]), int(r[1])) for r in coords}
        excl = self._windmill_exclusion(terrain_map.windmill_sites)
        if pre_occupied:
            excl = excl | pre_occupied

        # Approach paths are merged into the local road map: border crossings
        # then get gates and interior path cells stay untouched, exactly like
        # roads.  They must NOT stay in excl — excluding them would carve an
        # unfenced corridor straight through the border.
        paths: Set[Tuple[int, int]] = set(path_cells) if path_cells else set()
        if paths:
            excl -= paths
            road_map = (
                road_map.copy()
                if road_map is not None
                else np.zeros(feat.shape, dtype=np.uint8)
            )
            for plz, plx in paths:
                if 0 <= plz < road_map.shape[0] and 0 <= plx < road_map.shape[1]:
                    road_map[plz, plx] = max(road_map[plz, plx], 1)

        # Remove excluded cells from the pen up front.  Border detection then
        # treats them as "outside", so the fence wraps around such holes and
        # the enclosure stays closed (otherwise animals escape through cells
        # that were silently skipped).  This also keeps the barn and
        # decorations off pre-occupied cells, since _fits checks pen_set.
        pen_set -= excl

        # Per-cell steepness filter: a pen only works on reasonably flat
        # ground (the zone-mean check alone lets half-flat/half-cliff zones
        # through, putting fences and animals on the cliff).  Steep cells are
        # treated as "outside" so the fence wraps around them, and only the
        # largest connected flat region is kept so the pen isn't fragmented
        # into slivers between steep patches.
        pen_set = {
            (lz, lx)
            for lz, lx in pen_set
            if float(feat[lz, lx]["slope"]) <= self.MAX_CELL_SLOPE
            and float(feat[lz, lx]["water_pct"]) == 0.0  # no fence/barn in ponds
        }
        pen_set = _largest_component(pen_set)

        # Terrace constraint: a real paddock sits on one level of the land.
        # Cells far above/below the pen's median height are dropped so the
        # fence never marches up a hillside even when every step is gentle.
        if pen_set:
            med_h = float(
                np.median([int(feat[lz, lx]["height"]) for lz, lx in pen_set])
            )
            pen_set = {
                c
                for c in pen_set
                if abs(int(feat[c[0], c[1]]["height"]) - med_h) <= self.MAX_PEN_RELIEF
            }
            pen_set = _largest_component(pen_set)

        # Prune 1-cell-wide tendrils (cells with fewer than two cardinal
        # neighbours) so the fence outline reads as a deliberate paddock
        # instead of spaghetti hugging every terrain wrinkle.
        changed = True
        while changed:
            changed = False
            for c in sorted(pen_set):
                n_nb = sum(
                    1
                    for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if (c[0] + dz, c[1] + dx) in pen_set
                )
                if n_nb < 2:
                    pen_set.discard(c)
                    changed = True
        pen_set = _largest_component(pen_set)

        # Area cap: grow outward from the pen centroid so an oversized zone
        # yields one compact paddock instead of fencing the whole district.
        if len(pen_set) > self.MAX_PEN_CELLS:
            pcz = sum(c[0] for c in pen_set) / len(pen_set)
            pcx = sum(c[1] for c in pen_set) / len(pen_set)
            seed = min(pen_set, key=lambda c: (c[0] - pcz) ** 2 + (c[1] - pcx) ** 2)
            kept: Set[Tuple[int, int]] = {seed}
            queue = [seed]
            while queue and len(kept) < self.MAX_PEN_CELLS:
                lz, lx = queue.pop(0)
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (lz + dz, lx + dx)
                    if nb in pen_set and nb not in kept:
                        kept.add(nb)
                        queue.append(nb)
                        if len(kept) >= self.MAX_PEN_CELLS:
                            break
            pen_set = kept

        if len(pen_set) < self.MIN_PEN_CELLS:
            log(
                f"[AnimalPen] Skipping zone — only {len(pen_set)} usable flat "
                f"cells (need {self.MIN_PEN_CELLS})."
            )
            return
        min_lz = int(coords[:, 0].min())
        map_depth, map_width = feat.shape

        clear_vegetation(
            list(pen_set),
            feat,
            map_depth,
            map_width,
            editor,
            terrain_map,
            protected=excl,
        )

        # Skip zones where the terrain is too steep for fences to enclose animals.
        active_cells = [
            (lz, lx)
            for lz, lx in pen_set
            if (lz, lx) not in excl and not self._on_road(lz, lx, road_map)
        ]
        if active_cells:
            mean_slope = sum(
                float(feat[lz, lx]["slope"]) for lz, lx in active_cells
            ) / len(active_cells)
            if mean_slope > self.MAX_ZONE_SLOPE:
                log(
                    f"[AnimalPen] Skipping zone — terrain too steep (mean slope {mean_slope:.1f})."
                )
                return

        # Biome at district centroid
        cz = int(coords[:, 0].mean())
        cx = int(coords[:, 1].mean())
        wx_c, wz_c = terrain_map.local_to_world(cz, cx)
        biome = editor.getBiome((wx_c, int(feat[cz, cx]["height"]) - 1, wz_c)).lower()

        bkey = _biome_key(biome)
        animals = _BIOME_ANIMALS[bkey]
        ground_pool = _BIOME_GROUND.get(bkey, _BIOME_GROUND["default"])
        mat = _fence_material(biome)

        # ------------------------------------------------------------------
        # Classify cells; record surface heights for fence step pattern
        # ------------------------------------------------------------------
        border_cells: Set[Tuple[int, int]] = set()
        interior: List[Tuple[int, int]] = []
        road_border: Set[Tuple[int, int]] = set()
        # Border cells that sit ON a road cell — get a gate instead of a fence.
        road_gates: Set[Tuple[int, int]] = set()
        # Maps every fence/div cell to its surface y for step-gap calculation
        fence_heights: Dict[Tuple[int, int], int] = {}

        for lz, lx in pen_set:
            if (lz, lx) in excl:
                continue
            on_road = self._on_road(lz, lx, road_map)
            sy = int(feat[lz, lx]["height"]) - 1
            is_bdr = any(
                (lz + dz, lx + dx) not in pen_set
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            adj_road = any(
                self._on_road(lz + dz, lx + dx, road_map)
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            if on_road:
                # Only border cells on a road get a crossing gate; interior
                # road cells are left untouched.
                if is_bdr:
                    road_gates.add((lz, lx))
                    fence_heights[(lz, lx)] = sy
                continue
            if is_bdr:
                border_cells.add((lz, lx))
                fence_heights[(lz, lx)] = sy
                if adj_road:
                    road_border.add((lz, lx))
            else:
                interior.append((lz, lx))

        # ------------------------------------------------------------------
        # Fix diagonal gaps: ensure Manhattan (no diagonal-only) connectivity
        # ------------------------------------------------------------------
        extra_border: Set[Tuple[int, int]] = set()
        for lz, lx in list(border_cells):
            for ddz, ddx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                diag = (lz + ddz, lx + ddx)
                if diag not in border_cells:
                    continue
                card_h = (lz, lx + ddx)  # horizontal-first candidate
                card_v = (lz + ddz, lx)  # vertical-first candidate
                if card_h in border_cells or card_v in border_cells:
                    continue  # already bridged cardinally
                for candidate in (card_h, card_v):
                    clz, clx = candidate
                    if (
                        candidate in pen_set
                        and candidate not in excl
                        and not self._on_road(clz, clx, road_map)
                    ):
                        extra_border.add(candidate)
                        break
        border_cells |= extra_border
        for lz, lx in extra_border:
            fence_heights[(lz, lx)] = int(feat[lz, lx]["height"]) - 1
        interior = [(lz, lx) for (lz, lx) in interior if (lz, lx) not in extra_border]

        # ------------------------------------------------------------------
        # Gate positions
        # ------------------------------------------------------------------
        gates: Set[Tuple[int, int]] = set()

        # Road-crossing gates: every border cell that sits on a road gets a gate.
        gates |= road_gates

        # Entrance gate: road-adjacent border cell (if no road crossing covers it)
        if road_border:
            gates.add(random.choice(list(road_border)))
        elif border_cells:
            gates.add(random.choice(list(border_cells)))

        # Internal fence dividers every PEN_DIVISION blocks in Z
        div_cells: Set[Tuple[int, int]] = set()
        div_z_offsets: Set[int] = set()
        for lz, lx in pen_set:
            if (lz, lx) in excl or self._on_road(lz, lx, road_map):
                continue
            if (lz, lx) in border_cells:
                continue
            if (lz - min_lz) % self.PEN_DIVISION == 0:
                div_cells.add((lz, lx))
                fence_heights[(lz, lx)] = int(feat[lz, lx]["height"]) - 1
                div_z_offsets.add(lz - min_lz)

        # One gate per divider row, at the middle column
        for z_off in div_z_offsets:
            row = sorted(
                [(lz, lx) for lz, lx in div_cells if (lz - min_lz) == z_off],
                key=lambda c: c[1],
            )
            if row:
                gates.add(row[len(row) // 2])

        all_fence: Set[Tuple[int, int]] = border_cells | div_cells | road_gates

        # ------------------------------------------------------------------
        # Fix gate flanking: every gate must have a fence post on both sides
        # along its run axis; add extra border cells where missing.
        # ------------------------------------------------------------------
        extra_flank: Set[Tuple[int, int]] = set()
        for gz, gx in list(gates):
            runs_ew = (gz, gx - 1) in all_fence or (gz, gx + 1) in all_fence
            runs_ns = (gz - 1, gx) in all_fence or (gz + 1, gx) in all_fence
            if runs_ew:
                for ddx in (-1, 1):
                    flank = (gz, gx + ddx)
                    if flank not in all_fence:
                        flz, flx = flank
                        if (
                            flank in pen_set
                            and flank not in excl
                            and not self._on_road(flz, flx, road_map)
                        ):
                            extra_flank.add(flank)
            elif runs_ns:
                for ddz in (-1, 1):
                    flank = (gz + ddz, gx)
                    if flank not in all_fence:
                        flz, flx = flank
                        if (
                            flank in pen_set
                            and flank not in excl
                            and not self._on_road(flz, flx, road_map)
                        ):
                            extra_flank.add(flank)
        border_cells |= extra_flank
        all_fence |= extra_flank
        for lz, lx in extra_flank:
            fence_heights[(lz, lx)] = int(feat[lz, lx]["height"]) - 1
        interior = [(lz, lx) for (lz, lx) in interior if (lz, lx) not in extra_flank]

        # ------------------------------------------------------------------
        # Pass 1 – ground, fences, gates
        # ------------------------------------------------------------------
        n_ground = n_fence = n_gate = 0

        for lz, lx in sorted(pen_set):
            if (lz, lx) in excl:
                continue
            on_road = self._on_road(lz, lx, road_map)
            # Interior road cells are left untouched; road_gate cells get a gate.
            if on_road and (lz, lx) not in road_gates:
                continue

            wx, wz = terrain_map.local_to_world(lz, lx)
            sy = int(feat[lz, lx]["height"]) - 1

            for dy in range(1, self.CLEAR_HEIGHT + 1):
                editor.placeBlock((wx, sy + dy, wz), _AIR)

            if not on_road:
                editor.placeBlock((wx, sy, wz), random.choice(ground_pool))
                n_ground += 1

            is_fence_cell = (
                (lz, lx) in border_cells
                or (lz, lx) in div_cells
                or (lz, lx) in road_gates
            )
            if not is_fence_cell:
                continue

            if (lz, lx) in gates:
                if (lz, lx) in road_gates:
                    facing = _gate_facing_along_road(lz, lx, road_map)
                elif (lz, lx) in road_border:
                    facing = _gate_facing_to_road(lz, lx, road_map)
                else:
                    facing = _gate_facing_by_run(lz, lx, all_fence)
                editor.placeBlock(
                    (wx, sy + 1, wz),
                    Block(
                        f"{mat}_fence_gate", states={"facing": facing, "open": "false"}
                    ),
                )
                n_gate += 1
            else:
                # Step pattern: an entity standing on a neighbour at height
                # nsy has its feet at nsy+1, so our fence must reach nsy+1
                # to block it — hence max_adj_sy + 1 as the floor.
                max_adj_sy = sy
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    adj = (lz + dz, lx + dx)
                    if adj in fence_heights:
                        max_adj_sy = max(max_adj_sy, fence_heights[adj])
                fence_top = min(
                    sy + self.MAX_FENCE_EXTRA,
                    max(sy + 1, max_adj_sy + 1),
                )
                for fy in range(sy + 1, fence_top + 1):
                    editor.placeBlock((wx, fy, wz), random.choice(_FENCE_BLOCKS))
                n_fence += 1

        # ------------------------------------------------------------------
        # Pass 2 – barn schematic (largest that fits, terrain flattened).
        # Several fitting (anchor, rotation) sites are scored and the best
        # taken: the entrance must open onto level in-pen ground (never into
        # a wall of terrain), and sites hugging rising ground or needing
        # heavy terraforming are penalised.
        # ------------------------------------------------------------------
        barn_fp: Set[Tuple[int, int]] = set()
        if self._schematics and random.random() < self.BARN_CHANCE and interior:
            by_size = sorted(self._schematics, key=lambda s: s[1] * s[2], reverse=True)
            random.shuffle(interior)
            for schem_path, sw, sd, jig_offsets in by_size:
                best: Optional[
                    tuple[float, int, int, int, Set[Tuple[int, int]], int]
                ] = None
                n_fits = 0
                for lz, lx in interior:
                    if n_fits >= self.BARN_CANDIDATE_FITS:
                        break
                    for direction in range(4):
                        ew, ed = (sd, sw) if direction % 2 == 1 else (sw, sd)
                        ok, fp = self._fits(lz, lx, ew, ed, pen_set, road_map)
                        if not ok:
                            continue
                        n_fits += 1
                        heights = [int(feat[flz, flx]["height"]) - 1 for flz, flx in fp]
                        barn_y = int(np.median(heights)) + 1
                        cost = self._barn_site_cost(
                            lz,
                            lx,
                            sw,
                            sd,
                            direction,
                            fp,
                            barn_y,
                            heights,
                            jig_offsets,
                            feat,
                            pen_set,
                            all_fence,
                        )
                        if best is None or cost < best[0]:
                            best = (cost, lz, lx, direction, fp, barn_y)
                if best is None:
                    continue
                _, lz, lx, direction, fp, barn_y = best
                wx, wz = terrain_map.local_to_world(lz, lx)
                for flz, flx in fp:
                    fwx, fwz = terrain_map.local_to_world(flz, flx)
                    fh = int(feat[flz, flx]["height"]) - 1
                    if fh < barn_y:
                        for fy in range(fh, barn_y):
                            editor.placeBlock((fwx, fy, fwz), _DIRT)
                    elif fh > barn_y:
                        for fy in range(barn_y + 1, fh + 1):
                            editor.placeBlock((fwx, fy, fwz), _AIR)
                    for dy in range(1, self.BARN_CLEAR_HEIGHT + 1):
                        editor.placeBlock((fwx, barn_y + dy, fwz), _AIR)
                with editor.pushTransform((wx, barn_y, wz)):
                    self._placer.place_structure(
                        editor, schem_path, direction=direction, skip_air=True
                    )
                barn_fp = fp
                log(f"[AnimalPen] Placed barn '{schem_path}' (rotation {direction}).")
                if jig_offsets and road_map is not None:
                    rotated_jigs = [
                        rotate_offset(sx, sz, sw, sd, direction)
                        for sx, sz in jig_offsets
                    ]
                    self._connect_barn_jigsaws(
                        lz,
                        lx,
                        rotated_jigs,
                        fp,
                        road_map,
                        terrain_map,
                        feat,
                        feat.shape[0],
                        feat.shape[1],
                        editor,
                        fence_cells=all_fence,
                        gate_cells=gates,
                        excl=excl,
                    )
                break

        # ------------------------------------------------------------------
        # Pass 2b – chicken pen decoration (small chance, central interior)
        # ------------------------------------------------------------------
        if self._chicken_pen_schem and random.random() < self.CHICKEN_PEN_CHANCE:
            cp_path, cpw, cpd = self._chicken_pen_schem
            # Restrict candidates to cells that are not barn footprint,
            # not a fence/div cell, and not immediately inside the border
            # (no direct neighbour is a border or div cell).
            deep_interior = [
                (lz, lx)
                for lz, lx in interior
                if (lz, lx) not in barn_fp
                and not any(
                    (lz + dz, lx + dx) in all_fence
                    for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )
            ]
            # Prefer cells closest to the district centroid.
            deep_interior.sort(key=lambda c: (c[0] - cz) ** 2 + (c[1] - cx) ** 2)
            placed_cp = False
            for lz, lx in deep_interior:
                ok, fp = self._fits(lz, lx, cpw, cpd, pen_set, road_map)
                if not ok:
                    continue
                # Footprint must stay clear of all fence/div and barn cells.
                if any(c in all_fence or c in barn_fp for c in fp):
                    continue
                sy = self._grounded_y(fp, feat)
                if sy is None:
                    continue  # footprint doesn't touch the ground everywhere
                wx, wz = terrain_map.local_to_world(lz, lx)
                with editor.pushTransform((wx, sy, wz)):
                    self._placer.place_structure(editor, cp_path, skip_air=True)
                log(f"[AnimalPen] Placed chicken pen '{cp_path}'.")
                placed_cp = True
                break
            if not placed_cp:
                log("[AnimalPen] No fitting position for chicken pen.")

        # ------------------------------------------------------------------
        # Pass 2c – overlay decoration (over_small/medium/large, random pick)
        # ------------------------------------------------------------------
        if self._overlay_schematics and random.random() < self.OVERLAY_CHANCE:
            # Build the candidate set once (reuse deep_interior logic).
            overlay_interior = [
                (lz, lx)
                for lz, lx in interior
                if (lz, lx) not in barn_fp
                and not any(
                    (lz + dz, lx + dx) in all_fence
                    for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )
            ]
            overlay_interior.sort(key=lambda c: (c[0] - cz) ** 2 + (c[1] - cx) ** 2)
            # Try schematics from largest to smallest so a big one is preferred
            # but smaller ones can fall back if space is tight.
            by_size = sorted(
                self._overlay_schematics, key=lambda s: s[1] * s[2], reverse=True
            )
            placed_ov = False
            for ov_path, ovw, ovd in by_size:
                if placed_ov:
                    break
                for lz, lx in overlay_interior:
                    ok, fp = self._fits(lz, lx, ovw, ovd, pen_set, road_map)
                    if not ok:
                        continue
                    if any(c in all_fence or c in barn_fp for c in fp):
                        continue
                    sy = self._grounded_y(fp, feat)
                    if sy is None:
                        continue  # footprint doesn't touch the ground everywhere
                    wx, wz = terrain_map.local_to_world(lz, lx)
                    with editor.pushTransform((wx, sy, wz)):
                        self._placer.place_structure(editor, ov_path, skip_air=True)
                    log(f"[AnimalPen] Placed overlay '{ov_path}'.")
                    placed_ov = True
                    break
            if not placed_ov:
                log("[AnimalPen] No fitting position for overlay schematic.")

        # ------------------------------------------------------------------
        # Pass 3 – summon animals (flush buffer first so the fence is built)
        # ------------------------------------------------------------------
        available = [c for c in interior if c not in barn_fp]
        n_to_spawn = max(
            2, min(self.MAX_ANIMALS, len(available) // max(1, self.ANIMAL_DENSITY))
        )
        # Mixed herds: draw 1–2 species from the biome list, then spawn a mix of
        # them so a pen can hold e.g. cows and sheep together.
        pen_species = random.sample(animals, min(len(animals), random.randint(1, 2)))
        n_spawned = 0

        if available and n_to_spawn > 0:
            editor.flushBuffer()
            spawn_cells = random.sample(available, min(n_to_spawn, len(available)))
            for lz, lx in spawn_cells:
                wx, wz = terrain_map.local_to_world(lz, lx)
                sy = int(feat[lz, lx]["height"]) - 1
                animal = random.choice(pen_species)
                editor.runCommand(f"summon minecraft:{animal} {wx} {sy + 1} {wz}")
                n_spawned += 1

        # Expose every cell we worked on so the caller's shared occupied set
        # reflects the pen.  Later phases (gardens, misc) then skip it.
        if pre_occupied is not None:
            for lz, lx in pen_set:
                if not self._on_road(lz, lx, road_map) and (lz, lx) not in excl:
                    pre_occupied.add((lz, lx))

        log(
            f"[AnimalPen] {n_ground} ground, {n_fence} fence, {n_gate} gates, "
            f"{'barn placed, ' if barn_fp else ''}"
            f"{n_spawned} animal(s) summoned ({'/'.join(pen_species)})."
        )
