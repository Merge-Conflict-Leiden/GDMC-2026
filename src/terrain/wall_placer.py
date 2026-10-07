"""
wall_placer.py
--------------
Perimeter wall placement — wooden palisade wall with corner/gate towers.

Each wall column alternates between stripped_spruce_wood and oak_log pikes,
6–7 blocks tall, with a 50 % chance of a spruce_fence or oak_fence spike on
top.  Cobblestone rubble piles are scattered on all four cardinal sides of
each pike, extending up to 2 blocks out, with heights tapering away from the
wall.

Towers (wall/wall_tower.csv, 3×3 base) are placed at:
  • The pike immediately flanking each gate opening on both sides.
  • Every TOWER_INTERVAL pike steps between those forced positions.

Four global phases ensure correctness:
  Phase 1 — classify every cell (pike vs gate) across ALL segments.
  Phase 2 — place pikes and gate air clearings.
  Phase 2.5 — place towers; tower footprint added to off_limits.
  Phase 3 — scatter rubble (off_limits = pikes | gates | tower cells).
"""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING, Optional

import numpy as np
from gdpc.block import Block
from gdpc.editor import Editor

from blocks.block_processor import BlockProcessor
from builds._vegetation import clear_vegetation
from consts import BUILD_DIR, BUILD_OUTPUT_DIR
from structures.structure_cache import StructureCache
from structures.structure_placer import StructurePlacer

from .terrain_types import TerrainMap
from .wall_layout import GATE_WIDTH, WallLayout

if TYPE_CHECKING:
    from .terrain_modifier import TerrainModifier

logger = logging.getLogger(__name__)

WALL_HEIGHT_MIN = 6
WALL_HEIGHT_MAX = 7
SPIKE_CHANCE = 0.5

# Pikes may wade into water at most this deep (blocks of water column);
# anything deeper stops the wall until the far bank.
MAX_WALL_WATER_DEPTH = 4

# A step-to-step ground jump larger than this is treated as a cliff face.
MAX_STEP_DY = 4
# How many steps ahead to scan for the terrain returning to the previous
# level; if it does, the cliff span is skipped and the wall resumes after.
CLIFF_LOOKAHEAD = 12

_AIR = Block("minecraft:air")
_PIKE_EVEN = Block("minecraft:stripped_spruce_wood", states={"axis": "y"})
_PIKE_ODD = Block("minecraft:oak_log", states={"axis": "y"})
_SPIKE_TYPES = (Block("minecraft:spruce_fence"), Block("minecraft:oak_fence"))
_FOUND_BLOCK = [
    Block("minecraft:cobblestone"),
    Block("minecraft:stone"),
    Block("minecraft:stone_bricks"),
]

_PILE_OPTIONS = [
    (1, False, 1.0),
    (1, True, 1.5),
    (2, False, 2.0),
    (2, True, 2.5),
]
_COBBLE = Block("minecraft:cobblestone")
_COBBLE_SLAB = Block("minecraft:cobblestone_slab", states={"type": "bottom"})

RUBBLE_D1_CHANCE = 0.70
RUBBLE_D2_CHANCE = 0.55

_CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))

TOWER_INTERVAL = 13  # pike steps between interval towers
_TOWER_SCHEM = "wall/wall_tower.csv"
_TOWER_HALF = 2  # footprint half-extent; covers 5×5 platform overhang

# Type alias for a classified wall step.
_Step = tuple[int, int, int, bool]  # (lz, lx, step_index, is_gate)


class WallPlacer:
    """Place perimeter walls as wooden palisade columns with cobblestone rubble."""

    def __init__(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        terrain_modifier: Optional["TerrainModifier"] = None,
    ) -> None:
        self._editor = editor
        self._tm = terrain_map
        self._modifier = terrain_modifier
        self._tower_placer: Optional[StructurePlacer] = None
        try:
            _cache = StructureCache(BUILD_DIR, BUILD_OUTPUT_DIR)
            _cache.load_structure(_TOWER_SCHEM)
            self._tower_placer = StructurePlacer(
                structure_cache=_cache,
                block_processor=BlockProcessor(),
                block_palette=None,
                furniture_replacer=None,
            )
        except Exception as _exc:
            logger.warning("WallPlacer: could not load tower schematic: %s", _exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def place_wall(
        self,
        layout: WallLayout,
        *,
        pre_occupied: set[tuple[int, int]] | None = None,
    ) -> tuple[int, set[tuple[int, int]]]:
        """Place all wall columns for the given WallLayout.

        Three global phases ensure correctness across segment boundaries:
          Phase 1 — classify every cell (pike vs gate) across ALL segments.
          Phase 2 — place pikes and gate air clearings.
          Phase 3 — scatter rubble with a global off_limits so no pile ever
                    lands on a pike from a neighbouring segment.

        *pre_occupied* — cells already claimed by castle or other structures;
        any wall step landing on one of these is silently skipped so the wall
        never cuts through previously placed buildings.

        Returns (block_count, wall_cells).
        """
        if not layout.is_valid:
            logger.warning("WallPlacer: layout is invalid (< 3 corners).")
            return 0, set()

        features = self._tm.features
        if features is None:
            return 0, set()
        depth, width = features.shape

        # Every wall step that lands on a rendered road surface becomes a
        # gate cell, so the wall can never block a road — even where the
        # assigned gate opening does not line up exactly with the crossing.
        road_cells: set[tuple[int, int]] = set()
        road_expanded = self._tm.road_map_expanded
        if road_expanded is not None:
            road_cells = {(int(z), int(x)) for z, x in np.argwhere(road_expanded > 0)}

        # Water is a natural defence, but shallow water (≤ MAX_WALL_WATER_DEPTH)
        # is still wall-able: pikes rise from the river bed and are extended by
        # the water depth so they stand full height above the surface.  Only
        # DEEP water stops the palisade at the bank until the far side.
        # (feat["height"] of a water cell is the RIVER BOTTOM; the
        # surface_heightmap is the water surface, so their difference is the
        # water-column depth.)
        wm = self._tm.water_mask
        surf_hm = self._tm.surface_heightmap
        if wm is not None:
            water_cells = {(int(z), int(x)) for z, x in np.argwhere(wm)}
        else:
            water_cells = {
                (int(z), int(x)) for z, x in np.argwhere(features["water_pct"] > 0.5)
            }
        if wm is not None and surf_hm is not None:
            water_depth = np.where(wm, surf_hm.astype(np.int32) - features["height"], 0)
            deep_water_cells = {
                (int(z), int(x))
                for z, x in np.argwhere(water_depth > MAX_WALL_WATER_DEPTH)
            }
        else:
            # No depth information — treat all water as deep (old behaviour).
            deep_water_cells = water_cells

        # ── Phase 1: classify ─────────────────────────────────────────
        all_steps: list[list[_Step]] = []
        all_pike_cells: set[tuple[int, int]] = set()
        all_gate_cells: set[tuple[int, int]] = set()

        n = layout.n
        for i in range(n):
            steps, pike_cells, gate_cells = self._classify_segment(
                layout.corners[i],
                layout.corners[(i + 1) % n],
                layout.gate_t[i],
                depth,
                width,
                features=features,
                pre_occupied=pre_occupied,
                road_cells=road_cells,
                water_cells=deep_water_cells,
            )
            all_steps.append(steps)
            all_pike_cells |= pike_cells
            all_gate_cells |= gate_cells

        # ── Phase 1.5: clear vegetation along the wall path ──────────
        # BFS from every wall cell with the standard search radius so trees
        # straddling the wall line on either side are removed before pikes are
        # placed.  Pre-occupied cells (castle, earlier buildings) are protected
        # so their blocks are never touched.
        wall_fp = list(all_pike_cells | all_gate_cells)
        if wall_fp:
            clear_vegetation(
                wall_fp,
                features,
                depth,
                width,
                self._editor,
                self._tm,
                protected=pre_occupied,
            )

        # ── Phase 2: place pikes and gate clearings ───────────────────
        count = 0
        for steps in all_steps:
            count += self._place_pikes(steps, features)

        # ── Phase 2.5: place towers ───────────────────────────────────
        tower_cells: set[tuple[int, int]] = set()
        if self._tower_placer is not None:
            for tlz, tlx in self._compute_tower_positions(all_steps):
                if float(features[tlz, tlx]["water_pct"]) > 0.0:  # type: ignore[index]
                    continue
                self._place_tower(tlz, tlx, features, depth, width)
                for ddz in range(-_TOWER_HALF, _TOWER_HALF + 1):
                    for ddx in range(-_TOWER_HALF, _TOWER_HALF + 1):
                        tower_cells.add((tlz + ddz, tlx + ddx))

        # ── Phase 3: scatter rubble globally ─────────────────────────
        # Road cells are off limits too: rubble piles must never land on a
        # road surface where it passes through (or runs beside) the wall.
        # Water cells likewise — a pile would sit on the river bottom.
        off_limits: set[tuple[int, int]] = (
            all_pike_cells | all_gate_cells | tower_cells | road_cells | water_cells
        )
        rubble_cells: set[tuple[int, int]] = set()
        for steps in all_steps:
            for lz, lx, _step, is_gate in steps:
                if is_gate:
                    continue
                self._place_rubble_piles(
                    lz,
                    lx,
                    features,
                    depth,
                    width,
                    off_limits,
                    rubble_cells,
                )

        logger.info(
            "WallPlacer: placed %d blocks across %d pike cells.",
            count,
            len(all_pike_cells),
        )
        return count, all_pike_cells

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_segment(
        p1: tuple[int, int],
        p2: tuple[int, int],
        gate_t: float | None,
        depth: int,
        width: int,
        *,
        features: np.ndarray,
        pre_occupied: set[tuple[int, int]] | None = None,
        road_cells: set[tuple[int, int]] | None = None,
        water_cells: set[tuple[int, int]] | None = None,
    ) -> tuple[list[_Step], set[tuple[int, int]], set[tuple[int, int]]]:
        """Return (steps, pike_cells, gate_cells) for one wall segment.

        Cells in *pre_occupied* are silently skipped — no pike, no gate
        clearing, no rubble — so the wall never overwrites an existing
        structure (e.g. castle).

        Cells in *road_cells* are classified as gate cells so a road
        crossing the wall line always gets an opening.

        Cells in *water_cells* (deep water only) are skipped entirely: the
        wall stops at the water's edge and resumes on the far bank.

        Cliff handling: when the ground jumps more than MAX_STEP_DY between
        consecutive placed steps, the next CLIFF_LOOKAHEAD steps are scanned.
        If the terrain returns to the previous level it is a temporary cliff
        — the span is skipped and the wall resumes after it.  Otherwise it is
        a genuine level change and the wall follows the terrain.
        """
        lz1, lx1 = p1
        lz2, lx2 = p2
        dz = lz2 - lz1
        dx = lx2 - lx1
        seg_len = math.sqrt(dz * dz + dx * dx)
        if seg_len < 1.0:
            return [], set(), set()

        gate_center: float | None = gate_t * seg_len if gate_t is not None else None
        half_gate = GATE_WIDTH / 2
        seg_steps = int(round(seg_len))

        def cell_at(s: int) -> tuple[int, int]:
            t = s / seg_len
            return int(round(lz1 + dz * t)), int(round(lx1 + dx * t))

        steps: list[_Step] = []
        pike_cells: set[tuple[int, int]] = set()
        gate_cells: set[tuple[int, int]] = set()
        prev_h: int | None = None  # ground height of the last placed step
        skip_until = -1  # last step index of a cliff span being skipped

        for step in range(seg_steps + 1):
            if step <= skip_until:
                continue
            lz, lx = cell_at(step)
            if not (0 <= lz < depth and 0 <= lx < width):
                prev_h = None
                continue
            if pre_occupied is not None and (lz, lx) in pre_occupied:
                prev_h = None
                continue
            if water_cells is not None and (lz, lx) in water_cells:
                prev_h = None
                continue

            h = int(features[lz, lx]["height"])
            if prev_h is not None and abs(h - prev_h) > MAX_STEP_DY:
                resume: int | None = None
                for ahead in range(
                    step + 1, min(step + CLIFF_LOOKAHEAD, seg_steps) + 1
                ):
                    alz, alx = cell_at(ahead)
                    if not (0 <= alz < depth and 0 <= alx < width):
                        continue
                    ah = int(features[alz, alx]["height"])
                    if abs(ah - prev_h) <= MAX_STEP_DY:
                        resume = ahead
                        break
                if resume is not None:
                    skip_until = resume - 1
                    continue
                # Terrain never comes back — genuine level change; fall
                # through and let the wall follow it.

            is_gate = (
                gate_center is not None and abs(step - gate_center) <= half_gate
            ) or (road_cells is not None and (lz, lx) in road_cells)
            if is_gate:
                gate_cells.add((lz, lx))
            else:
                pike_cells.add((lz, lx))
            steps.append((lz, lx, step, is_gate))
            prev_h = h

        return steps, pike_cells, gate_cells

    def _place_pikes(self, steps: list[_Step], features: object) -> int:
        """Place pikes and gate air clearings; return block count.

        Steps in shallow water rise from the river bed: the pike is extended
        by the water-column depth so it still stands full height above the
        water surface.  Gate clearings there start at the water surface, not
        the river bed, so no water blocks are carved out.
        """
        surf_hm = self._tm.surface_heightmap
        count = 0
        for lz, lx, step, is_gate in steps:
            wx, wz = self._tm.local_to_world(lz, lx)
            surface_y = int(features[lz, lx]["height"]) - 1  # type: ignore[index]
            water_depth = 0
            if surf_hm is not None:
                water_depth = min(
                    max(0, int(surf_hm[lz, lx]) - 1 - surface_y),
                    MAX_WALL_WATER_DEPTH,
                )

            if is_gate:
                for dy in range(1, WALL_HEIGHT_MAX + 1):
                    self._editor.placeBlock(
                        (wx, surface_y + water_depth + dy, wz),
                        Block("minecraft:air"),
                    )
                continue

            wall_height = random.randint(WALL_HEIGHT_MIN, WALL_HEIGHT_MAX) + water_depth
            pike = _PIKE_EVEN if step % 2 == 0 else _PIKE_ODD
            for dy in range(wall_height):
                self._editor.placeBlock((wx, surface_y + dy, wz), pike)
            count += wall_height

            if random.random() < SPIKE_CHANCE:
                self._editor.placeBlock(
                    (wx, surface_y + wall_height, wz),
                    random.choice(_SPIKE_TYPES),
                )
                count += 1

            if self._modifier is not None:
                self._modifier.fill_column_down(
                    wx, surface_y, wz, fill_block=_FOUND_BLOCK
                )

        return count

    @staticmethod
    def _compute_tower_positions(
        all_steps: list[list[_Step]],
    ) -> list[tuple[int, int]]:
        """Return (lz, lx) centres for towers.

        Forced positions: last pike before each gate section and first pike
        after each gate section.  Between all placed towers an interval tower
        is added every TOWER_INTERVAL pike steps.
        """
        forced: set[tuple[int, int]] = set()
        for steps in all_steps:
            prev_pike: tuple[int, int] | None = None
            was_gate = False
            for lz, lx, _, is_gate in steps:
                if not is_gate:
                    if was_gate:
                        forced.add((lz, lx))  # first pike after gate
                    prev_pike = (lz, lx)
                    was_gate = False
                else:
                    if not was_gate and prev_pike is not None:
                        forced.add(prev_pike)  # last pike before gate
                    was_gate = True

        positions: list[tuple[int, int]] = []
        placed: set[tuple[int, int]] = set()
        steps_since = 0

        for steps in all_steps:
            for lz, lx, _, is_gate in steps:
                if is_gate:
                    continue
                cell = (lz, lx)
                if cell in placed:
                    continue
                if (lz, lx) in forced or steps_since >= TOWER_INTERVAL:
                    positions.append(cell)
                    placed.add(cell)
                    steps_since = 0
                else:
                    steps_since += 1

        return positions

    def _place_tower(
        self,
        lz: int,
        lx: int,
        features: object,
        depth: int,
        width: int,
    ) -> None:
        """Place the wall_tower schematic centred at (lz, lx)."""
        wx, wz = self._tm.local_to_world(lz, lx)
        surface_y = int(features[lz, lx]["height"]) - 1  # type: ignore[index]

        # Schematic centre is at (x=2, z=2); shift origin so it lands on the pike.
        with self._editor.pushTransform((wx - 2, surface_y + 1, wz - 2)):
            self._tower_placer.place_structure(  # type: ignore[union-attr]
                self._editor,
                _TOWER_SCHEM,
                check_occupied=frozenset({"minecraft:spruce_trapdoor"}),
            )

        if self._modifier is not None:
            for ddz in range(-1, 2):
                for ddx in range(-1, 2):
                    nlz, nlx = lz + ddz, lx + ddx
                    if not (0 <= nlz < depth and 0 <= nlx < width):
                        continue
                    nwx, nwz = self._tm.local_to_world(nlz, nlx)
                    nsy = int(features[nlz, nlx]["height"]) - 1  # type: ignore[index]
                    self._modifier.fill_column_down(
                        nwx, nsy, nwz, fill_block=_FOUND_BLOCK
                    )
                    # Scan down past any log/wood blocks (wall pikes) to find solid base.
                    scan_y = nsy
                    while scan_y >= 0:
                        bid = self._editor.getBlock((nwx, scan_y, nwz)).id or ""
                        if "_log" not in bid and "_wood" not in bid:
                            break
                        scan_y -= 1
                    fill_start = scan_y + 1
                    for fill_y in range(fill_start, surface_y + 1):
                        self._editor.placeBlock((nwx, fill_y, nwz), _FOUND_BLOCK)

    def _place_rubble_piles(
        self,
        lz: int,
        lx: int,
        features: object,
        depth: int,
        width: int,
        off_limits: set[tuple[int, int]],
        rubble_cells: set[tuple[int, int]],
    ) -> None:
        """Scatter cobblestone rubble in all 4 cardinal directions.

        *off_limits*   — pike and gate cells; rubble is never placed here.
        *rubble_cells* — cells already used by a previous pile; each cell
                         receives rubble from at most one pile so slab heights
                         never stack from overlapping placements.
        """
        for ddz, ddx in _CARDINALS:
            if random.random() > RUBBLE_D1_CHANCE:
                continue

            # ── distance-1 pile ──────────────────────────────────────
            rlz1, rlx1 = lz + ddz, lx + ddx
            if not (0 <= rlz1 < depth and 0 <= rlx1 < width):
                continue
            if (rlz1, rlx1) in off_limits or (rlz1, rlx1) in rubble_cells:
                continue

            full1, slab1, h1 = random.choice(_PILE_OPTIONS)
            rwx1, rwz1 = self._tm.local_to_world(rlz1, rlx1)
            rsy1 = int(features[rlz1, rlx1]["height"]) - 1  # type: ignore[index]
            for i in range(full1):
                self._editor.placeBlock((rwx1, rsy1 + i, rwz1), _COBBLE)
            if slab1:
                self._editor.placeBlock((rwx1, rsy1 + full1, rwz1), _COBBLE_SLAB)
            rubble_cells.add((rlz1, rlx1))

            # ── distance-2 pile (must be strictly lower) ─────────────
            if random.random() > RUBBLE_D2_CHANCE:
                continue
            valid_d2 = [(f, s, h) for f, s, h in _PILE_OPTIONS if h < h1]
            if not valid_d2:
                continue

            rlz2, rlx2 = lz + 2 * ddz, lx + 2 * ddx
            if not (0 <= rlz2 < depth and 0 <= rlx2 < width):
                continue
            if (rlz2, rlx2) in off_limits or (rlz2, rlx2) in rubble_cells:
                continue

            full2, slab2, _ = random.choice(valid_d2)
            rwx2, rwz2 = self._tm.local_to_world(rlz2, rlx2)
            rsy2 = int(features[rlz2, rlx2]["height"]) - 1  # type: ignore[index]
            for i in range(full2):
                self._editor.placeBlock((rwx2, rsy2 + i, rwz2), _COBBLE)
            if slab2:
                self._editor.placeBlock((rwx2, rsy2 + full2, rwz2), _COBBLE_SLAB)
            rubble_cells.add((rlz2, rlx2))
