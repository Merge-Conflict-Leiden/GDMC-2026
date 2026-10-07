"""
Market square construction for the TOWN_CENTER district.

:class:`MarketSquareMixin` provides ``place_market_square`` on
:class:`~builds.schematic_placement.placer.SchematicBuildingPlacer`; it
relies on the host class for the editor, terrain map/modifier, and the
shared ``_place_one`` placement core.
"""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING

from gdpc.block import Block

from builds._vegetation import clear_vegetation
from terrain.terrain_types import SubZone

from .model import Building
from .scoring import mark_footprint, world_footprint

if TYPE_CHECKING:
    from gdpc.editor import Editor

    from terrain.terrain_modifier import TerrainModifier
    from terrain.terrain_types import TerrainMap

logger = logging.getLogger(__name__)

_AIR = Block("minecraft:air")
_COBBLE = Block("minecraft:cobblestone")


class MarketSquareMixin:
    """Market-square pass; mixed into SchematicBuildingPlacer."""

    # Provided by the host class (see SchematicBuildingPlacer.__init__).
    _editor: "Editor"
    _tm: "TerrainMap"
    _modifier: "TerrainModifier"
    fountain_anchor: tuple[int, int] | None
    fountain_direction: int
    fountain_floor_y: int
    fountain_building: "Building | None"

    def place_market_square(
        self,
        fountains: list[Building],
        stands: list[Building],
        occupied: set[tuple[int, int]],
        *,
        chronicle: "Building | None" = None,
        clear_height: int = 20,
        expand: int = 10,
        max_expand_slope: float = 2.0,
        max_expand_height_diff: int = 4,  # terrain spread absorbed into the flat terrace
    ) -> set[tuple[int, int]]:
        """
        Prepare and fill the TOWN_CENTER district as a paved market square.

        1. Grow the plaza by BFS flood-fill seeded from the *whole* free
           TOWN_CENTER district (interior road cells are absorbed and repaved),
           flowing outward into every flat, paveable, non-water cell within the
           allowed grade band around the seed.  Seeding from the full district —
           rather than only the cells that already pass the flatness test — keeps
           the square from collapsing or developing holes on gently sloped
           centres.  Roads encountered during growth are absorbed too when they
           sit below the seed grade — left alone they'd become a lower trench
           once their terraced neighbours rise around them — while roads at or
           above grade stay intact as real approaches (see the stepped-entrance
           pass below).  Every absorbed cell is terraced flat below (cut above
           the floor, cobble underfill below it).
        2. Terrace the whole plaza flat to a single floor, anchored to the
           approach-road grade so streets meet the square flush.  Cells whose
           terrain differs from that floor by more than *max_expand_height_diff*
           are dropped so no cut/fill is taller than the cap (contiguity kept).
        3. Smooth a 1-cell apron over natural terrain (roads excluded), then lay
           a short stone-brick STAIRCASE (up to *max_expand_height_diff* steps)
           wherever a road meets the plaza at a height gap, so every street
           entrance is walkable onto the square rather than a wall.
        4. Clear vegetation and pave with a distance-based stone pattern.
        5. Perimeter lantern posts (skipping road entries).
        6. Centrepiece: a fountain at the centroid (largest first), else the
           *chronicle* structure so the lectern always has a home.
        7. Market stalls on a spaced grid with a clear centre ring, plus corner
           tree planters framing a large square.
        8. Mark all plaza cells as occupied.

        Returns the updated *occupied* set.
        """
        feat = self._tm.features
        if feat is None:
            return occupied

        depth, width = feat.shape
        # Use the expanded road map so the market never paves over the
        # shoulders of roads already rendered at full width.
        road_map = self._tm.road_map_expanded

        def _free_cells(d):
            return [
                (int(c[0]), int(c[1]))
                for c in d.cells
                if 0 <= int(c[0]) < depth
                and 0 <= int(c[1]) < width
                and (int(c[0]), int(c[1])) not in occupied
            ]

        town_d = next(
            (
                d
                for d in self._tm.districts
                if d.sub_zone == SubZone.TOWN_CENTER and d.size > 0
            ),
            None,
        )
        cells = _free_cells(town_d) if town_d is not None else []

        if not cells:
            if town_d is None:
                logger.warning(
                    "place_market_square: no TOWN_CENTER district; "
                    "trying RESIDENTIAL fallback."
                )
            else:
                logger.warning(
                    "place_market_square: TOWN_CENTER fully occupied "
                    "(castle?); trying RESIDENTIAL fallback."
                )
            res_candidates = sorted(
                (
                    d
                    for d in self._tm.districts
                    if d.sub_zone == SubZone.RESIDENTIAL and d.size > 0
                ),
                key=lambda d: len(_free_cells(d)),
                reverse=True,
            )
            for d in res_candidates:
                cells = _free_cells(d)
                if cells:
                    town_d = d
                    break

        if not cells:
            logger.warning("place_market_square: no suitable district found, skipping.")
            return occupied

        icz, icx = town_d.centroid
        cz, cx = int(round(icz)), int(round(icx))

        # Growth reference: median height of the centroid region.  The final
        # floor is re-derived from the approach-road grade below (so streets meet
        # the plaza flush); growth just needs a stable anchor to measure spread.
        central = sorted(cells, key=lambda c: (c[0] - cz) ** 2 + (c[1] - cx) ** 2)
        central_hs = sorted(int(feat[lz, lx]["height"]) for lz, lx in central[:9])
        seed_y = central_hs[len(central_hs) // 2]

        wm = self._tm.water_mask

        def _is_water(lz: int, lx: int) -> bool:
            if wm is not None:
                return bool(wm[lz, lx])
            return float(feat[lz, lx]["water_pct"]) > 0.5

        def _paveable(lz: int, lx: int) -> bool:
            # Never pave over existing structures or water.
            if (lz, lx) in occupied or _is_water(lz, lx):
                return False
            is_road = (
                road_map is not None
                and 0 <= lz < road_map.shape[0]
                and 0 <= lx < road_map.shape[1]
                and road_map[lz, lx] > 0
            )
            if not is_road:
                return True
            # A road sitting below the plaza's reference grade would otherwise
            # be left as a lower trench cutting through the square once its
            # neighbours are terraced flat.  Absorb it into the plaza (it gets
            # repaved flush in step 1) instead.  Roads at or above grade are
            # left alone as real boundaries/approaches, handled by the
            # stepped-entrance pass below.
            return int(feat[lz, lx]["height"]) < seed_y

        def _plaza_ok(lz: int, lx: int) -> bool:
            if not (0 <= lz < depth and 0 <= lx < width):
                return False
            if not _paveable(lz, lx):
                return False
            if float(feat[lz, lx]["slope"]) > max_expand_slope:
                return False
            # Keep the plaza roughly flat: only absorb terrain within the
            # allowed spread of the seed grade (terraced flat, with cobble
            # underfill below the floor).
            return abs(int(feat[lz, lx]["height"]) - seed_y) <= max_expand_height_diff

        # ── 0. Grow the plaza by BFS flood-fill from the district ─────────
        # Seed from the *entire* free district so the plaza is always at least
        # the town-centre footprint (interior road cells are absorbed and
        # repaved into the square), then flood-fill outward into flat, paveable
        # ground within the allowed grade band.  Growing from the full district
        # rather than only its already-flat cells keeps the square from
        # collapsing or developing holes on real, gently sloped terrain, where a
        # rigid pre-filter would drop the high/low ends before growth begins.
        # Absorbed cells are terraced flat below: cut above the floor, cobble
        # underfill below it.  Water is never paved, in the seed or during growth.
        cell_set = {(lz, lx) for lz, lx in cells if not _is_water(lz, lx)}
        if not cell_set:
            cell_set = {(cz, cx)}
        frontier = set(cell_set)
        for _ in range(max(0, expand)):
            nxt: set[tuple[int, int]] = set()
            for lz, lx in frontier:
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (lz + dz, lx + dx)
                    if nb in cell_set:
                        continue
                    if _plaza_ok(nb[0], nb[1]):
                        cell_set.add(nb)
                        nxt.add(nb)
            if not nxt:
                break
            frontier = nxt
        cells = list(cell_set)

        # ── 1. Terrace the plaza flat to the approach-road grade ──────────
        # Anchor the single floor to the median grade of the plaza cells that
        # touch a road, so streets meet the square level rather than dropping or
        # floating at the edge.  The rest of the plaza is cut/filled (terraced)
        # to that height.
        road_ys: list[int] = []
        for lz, lx in cells:
            for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nlz, nlx = lz + dz, lx + dx
                if (
                    road_map is not None
                    and 0 <= nlz < road_map.shape[0]
                    and 0 <= nlx < road_map.shape[1]
                    and road_map[nlz, nlx] > 0
                ):
                    road_ys.append(int(feat[lz, lx]["height"]))
                    break
        if road_ys:
            road_ys.sort()
            floor_y = road_ys[len(road_ys) // 2]
        else:
            floor_y = seed_y

        # ── 1b. Cap the terrain-to-plaza-top difference ───────────────────
        # Drop any absorbed cell whose original ground sits more than
        # max_expand_height_diff blocks from the plaza floor, so terracing never
        # carves a cut/fill taller than the cap (no deep pits or tall retaining
        # walls at the edges).  Keep the plaza contiguous around the centre so
        # trimming an extreme edge can never leave interior holes or islands.
        band = max(0, max_expand_height_diff)
        capped = {
            c
            for c in cell_set
            if abs(int(feat[c[0], c[1]]["height"]) - floor_y) <= band
        }
        if capped:
            start = (
                (cz, cx)
                if (cz, cx) in capped
                else min(capped, key=lambda c: (c[0] - cz) ** 2 + (c[1] - cx) ** 2)
            )
            comp = {start}
            stack = [start]
            while stack:
                clz, clx = stack.pop()
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (clz + dz, clx + dx)
                    if nb in capped and nb not in comp:
                        comp.add(nb)
                        stack.append(nb)
            cell_set = comp
            cells = list(cell_set)
        else:
            logger.warning(
                "place_market_square: terrain too broken for the %d-block cap; "
                "terracing the full plaza.",
                band,
            )

        for lz, lx in cells:
            self._modifier.set_surface_height(lz, lx, floor_y, fill_block=[_COBBLE])

        # ── 2. Smooth 1-cell transition apron at market boundary ──────────
        # Step each immediately neighbouring non-market cell one block toward
        # floor_y so approach paths ramp gently rather than drop or float.
        # Roads are left untouched here — their transition onto the plaza is
        # handled by the stepped staircases in 3b (using true road grades).
        seen_apron: set[tuple[int, int]] = set()
        for lz, lx in cells:
            for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nlz, nlx = lz + dz, lx + dx
                if (
                    (nlz, nlx) in cell_set
                    or (nlz, nlx) in seen_apron
                    or (nlz, nlx) in occupied
                    or not (0 <= nlz < depth and 0 <= nlx < width)
                    or (
                        road_map is not None
                        and 0 <= nlz < road_map.shape[0]
                        and 0 <= nlx < road_map.shape[1]
                        and road_map[nlz, nlx] > 0
                    )
                ):
                    continue
                seen_apron.add((nlz, nlx))
                terrain_y = int(feat[nlz, nlx]["height"])
                diff = terrain_y - floor_y
                if abs(diff) <= 1:
                    continue
                new_y = terrain_y + (-1 if diff > 0 else 1)
                self._modifier.set_surface_height(nlz, nlx, new_y, fill_block=[_COBBLE])

        # ── 3. Clear vegetation & pave with distance-based pattern ───────
        clear_vegetation(
            cells, feat, depth, width, self._editor, self._tm, protected=occupied
        )
        max_dist_sq = max((lz - cz) ** 2 + (lx - cx) ** 2 for lz, lx in cells) or 1
        _CHISELED = Block("minecraft:chiseled_stone_bricks")
        _POLISHED = Block("minecraft:polished_andesite")
        _STONE_BR = Block("minecraft:stone_bricks")
        _TUFF_BR = Block("minecraft:tuff_bricks")
        for lz, lx in cells:
            wx, wz = self._tm.local_to_world(lz, lx)
            for dy in range(clear_height):
                self._editor.placeBlock((wx, floor_y + dy, wz), _AIR)
            is_edge = any(
                (lz + dz, lx + dx) not in cell_set
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            dist_sq = (lz - cz) ** 2 + (lx - cx) ** 2
            if is_edge:
                block = _CHISELED
            elif dist_sq < max_dist_sq * 0.15:
                block = _POLISHED
            elif (lz + lx) % 2 == 0:
                block = _STONE_BR
            else:
                block = _TUFF_BR
            self._editor.placeBlock((wx, floor_y - 1, wz), block)

        # ── 3b. Stepped road entrances ────────────────────────────────────
        # Wherever a street meets the plaza at a height gap (the plaza absorbed a
        # higher/lower road, or terracing raised the square above the approach),
        # cut a stone-brick STAIRCASE into the plaza's own edge cells — moving
        # inward from the border cell — so you can always walk onto the square.
        # The staircase is always carved out of plaza territory (never extends
        # onto the road): the border cell meets the road at its own grade, and
        # the innermost step lands flush on the plaza's existing flat floor.
        # The terrain cap (1b) keeps the gap within max_expand_height_diff, so a
        # run of that many steps is enough.
        _SB_STAIRS = "minecraft:stone_brick_stairs"
        _FACE = {(1, 0): "south", (-1, 0): "north", (0, 1): "east", (0, -1): "west"}
        plaza_top = floor_y - 1
        max_steps = max(1, max_expand_height_diff)

        def _is_road(rz: int, rx: int) -> bool:
            return (
                road_map is not None
                and 0 <= rz < road_map.shape[0]
                and 0 <= rx < road_map.shape[1]
                and road_map[rz, rx] > 0
            )

        def _place_step(clz: int, clx: int, top_y: int, facing: str) -> None:
            wx, wz = self._tm.local_to_world(clz, clx)
            for dy in range(1, 4):  # clear headroom above the step
                self._editor.placeBlock((wx, top_y + dy, wz), _AIR)
            self._editor.placeBlock(
                (wx, top_y, wz),
                Block(_SB_STAIRS, states={"facing": facing, "half": "bottom"}),
            )
            # Seal the column from just under the step down to the ground so the
            # staircase never floats over a dip in the road/terrain below.
            self._modifier.fill_column_down(wx, top_y, wz, fill_block=[_COBBLE])

        step_cells: set[tuple[int, int]] = set()
        handled_roads: set[tuple[int, int]] = set()
        for lz, lx in cells:
            if not any(
                (lz + dz, lx + dx) not in cell_set
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                continue  # interior cell, no entrance here
            for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nlz, nlx = lz + dz, lx + dx
                if (nlz, nlx) in cell_set or not _is_road(nlz, nlx):
                    continue
                if (nlz, nlx) in handled_roads:
                    continue
                handled_roads.add((nlz, nlx))
                road_top = int(feat[nlz, nlx]["height"]) - 1
                gap = plaza_top - road_top
                if gap == 0:
                    continue
                # Both cases (road below/above the plaza) cut inward from the
                # border cell, using only cells already claimed by the plaza
                # (cell_set) — the road side is never touched.
                sign = 1 if gap > 0 else -1
                facing = _FACE[(-sign * dz, -sign * dx)]
                for k in range(1, min(abs(gap), max_steps) + 1):
                    sclz, sclx = lz - (k - 1) * dz, lx - (k - 1) * dx
                    if (sclz, sclx) not in cell_set or (sclz, sclx) in step_cells:
                        break
                    _place_step(sclz, sclx, road_top + sign * k, facing)
                    step_cells.add((sclz, sclx))

        # ── 4. Lantern posts along the market border ──────────────────────
        # Sort border cells by angle around the centroid for even distribution;
        # skip road entry points so they remain open.
        # Seed sq_occ with the staircase cells so the fountain, market stalls,
        # and corner planters below never land on top of a stepped entrance.
        sq_occ: set[tuple[int, int]] = set(step_cells)
        border_cells = [
            (lz, lx)
            for lz, lx in cells
            if any(
                (lz + dz, lx + dx) not in cell_set
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
        ]
        border_cells.sort(key=lambda c: math.atan2(c[0] - cz, c[1] - cx))
        _PILLAR = Block("minecraft:stone_brick_wall")
        _LANTERN = Block("minecraft:lantern")
        last_lamp: tuple[int, int] | None = None
        for lz, lx in border_cells:
            if last_lamp is not None:
                d2 = (lz - last_lamp[0]) ** 2 + (lx - last_lamp[1]) ** 2
                if d2 < 25:  # enforce ~5-cell minimum spacing
                    continue
            if road_map is not None and any(
                0 <= lz + dz < road_map.shape[0]
                and 0 <= lx + dx < road_map.shape[1]
                and road_map[lz + dz, lx + dx] > 0
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                continue
            wx, wz = self._tm.local_to_world(lz, lx)
            self._editor.placeBlock((wx, floor_y, wz), _PILLAR)
            self._editor.placeBlock((wx, floor_y + 1, wz), _PILLAR)
            self._editor.placeBlock((wx, floor_y + 2, wz), _LANTERN)
            sq_occ.add((lz, lx))
            last_lamp = (lz, lx)

        # ── 5. Place fountain(s) at/near centroid ────────────────────────
        fountains_placed = 0
        center_first = sorted(cells, key=lambda c: (c[0] - cz) ** 2 + (c[1] - cx) ** 2)

        # Try each fountain in decreasing footprint size; place exactly one.
        for fountain in sorted(fountains, key=lambda b: len(b.footprint), reverse=True):
            placed_this = False
            for direction in range(4):
                if placed_this:
                    break
                for anchor in center_first:
                    if anchor in sq_occ or anchor in occupied:
                        continue
                    fp = world_footprint(
                        anchor, fountain.shadow, fountain.W, fountain.D, direction
                    )
                    if any(
                        not (0 <= lz2 < depth and 0 <= lx2 < width) for lz2, lx2 in fp
                    ):
                        continue
                    if any((lz2, lx2) not in cell_set for lz2, lx2 in fp):
                        continue
                    if any(
                        (lz2, lx2) in sq_occ or (lz2, lx2) in occupied
                        for lz2, lx2 in fp
                    ):
                        continue
                    self._place_one(
                        anchor,
                        fountain,
                        direction,
                        feat,
                        depth,
                        width,
                        level_terrain=False,
                        approach="none",
                        furniture=True,
                        clear_height=clear_height,
                        protected=occupied,
                    )
                    mark_footprint(anchor, fountain, direction, sq_occ, pad=1)
                    mark_footprint(anchor, fountain, direction, occupied, pad=1)
                    fountains_placed += 1
                    placed_this = True
                    self.fountain_anchor = anchor
                    self.fountain_direction = direction
                    self.fountain_floor_y = floor_y
                    self.fountain_building = fountain
                    break
            if placed_this:
                break

        # ── 5b. Chronicle fallback ────────────────────────────────────────
        # No fountain fit: guarantee a lectern home by placing the chronicle
        # structure.  Try a natural fit first (all rotations, centroid-first);
        # if nothing slots in cleanly, force-place at the centroid cell —
        # the chronicle is compact enough that this always produces a tidy result.
        if fountains_placed == 0 and chronicle is not None:
            placed_chronicle = False
            for direction in range(4):
                if placed_chronicle:
                    break
                for anchor in center_first:
                    if anchor in sq_occ or anchor in occupied:
                        continue
                    fp = world_footprint(
                        anchor, chronicle.shadow, chronicle.W, chronicle.D, direction
                    )
                    if any(
                        not (0 <= lz2 < depth and 0 <= lx2 < width) for lz2, lx2 in fp
                    ):
                        continue
                    if any((lz2, lx2) not in cell_set for lz2, lx2 in fp):
                        continue
                    if any(
                        (lz2, lx2) in sq_occ or (lz2, lx2) in occupied
                        for lz2, lx2 in fp
                    ):
                        continue
                    self._place_one(
                        anchor,
                        chronicle,
                        direction,
                        feat,
                        depth,
                        width,
                        level_terrain=False,
                        approach="none",
                        furniture=True,
                        clear_height=clear_height,
                        protected=occupied,
                    )
                    mark_footprint(anchor, chronicle, direction, sq_occ, pad=1)
                    mark_footprint(anchor, chronicle, direction, occupied, pad=1)
                    self.fountain_anchor = anchor
                    self.fountain_direction = direction
                    self.fountain_floor_y = floor_y
                    self.fountain_building = chronicle
                    placed_chronicle = True
                    break
            if not placed_chronicle:
                # Try all rotations across all cells before force-placing.
                force_anchor = center_first[0]
                force_direction = 0
                for direction in range(4):
                    for anchor in center_first:
                        if anchor in sq_occ or anchor in occupied:
                            continue
                        fp = world_footprint(
                            anchor,
                            chronicle.shadow,
                            chronicle.W,
                            chronicle.D,
                            direction,
                        )
                        if any(
                            not (0 <= lz2 < depth and 0 <= lx2 < width)
                            for lz2, lx2 in fp
                        ):
                            continue
                        if any((lz2, lx2) not in cell_set for lz2, lx2 in fp):
                            continue
                        force_anchor = anchor
                        force_direction = direction
                        break
                    else:
                        continue
                    break
                self._place_one(
                    force_anchor,
                    chronicle,
                    force_direction,
                    feat,
                    depth,
                    width,
                    level_terrain=False,
                    approach="none",
                    furniture=True,
                    clear_height=clear_height,
                    protected=occupied,
                )
                mark_footprint(force_anchor, chronicle, force_direction, sq_occ, pad=1)
                mark_footprint(
                    force_anchor, chronicle, force_direction, occupied, pad=1
                )
                self.fountain_anchor = force_anchor
                self.fountain_direction = force_direction
                self.fountain_floor_y = floor_y
                self.fountain_building = chronicle

        # ── 6. Market stalls in an orderly grid, open centre ─────────────
        # Stalls sit on a spaced grid inside a border margin, leaving aisles and
        # a clear ring around the centrepiece — an organised market rather than a
        # random scatter.  Each stall takes the first rotation that fits.
        zs = [z for z, _ in cells]
        xs = [x for _, x in cells]
        pz0, pz1, px0, px1 = min(zs), max(zs), min(xs), max(xs)
        stands_placed = 0
        if stands and len(cells) >= 25:
            margin = 2
            _OPEN_R2 = 3 * 3  # keep clear around the centrepiece
            _GRID = 3  # a stall every 3 cells → 1-cell aisles between rows
            for rz in range(pz0 + margin, pz1 - margin + 1, _GRID):
                for rx in range(px0 + margin, px1 - margin + 1, _GRID):
                    anchor = (rz, rx)
                    if (rz - cz) ** 2 + (rx - cx) ** 2 < _OPEN_R2:
                        continue
                    if anchor in sq_occ or anchor in occupied:
                        continue
                    b = random.choice(stands)
                    directions = list(range(4))
                    random.shuffle(directions)
                    for direction in directions:
                        fp = world_footprint(anchor, b.shadow, b.W, b.D, direction)
                        if any(
                            not (0 <= lz2 < depth and 0 <= lx2 < width)
                            for lz2, lx2 in fp
                        ):
                            continue
                        if any((lz2, lx2) not in cell_set for lz2, lx2 in fp):
                            continue
                        if any(
                            (lz2, lx2) in sq_occ or (lz2, lx2) in occupied
                            for lz2, lx2 in fp
                        ):
                            continue
                        self._place_one(
                            anchor,
                            b,
                            direction,
                            feat,
                            depth,
                            width,
                            level_terrain=False,
                            approach="none",
                            y_offset=1,
                            furniture=True,
                            clear_height=clear_height,
                            protected=occupied,
                        )
                        mark_footprint(anchor, b, direction, sq_occ, pad=1)
                        stands_placed += 1
                        break

        # ── 6b. Corner planters frame a large square ─────────────────────
        # A small tree near each corner defines the plaza edge; only on plazas
        # big enough that a tree won't crowd the stalls or the fountain.  Anchor
        # to the in-plaza cell nearest each bounding-box corner so the planters
        # still appear when the flood-filled plaza is irregular, and skip a
        # corner whose nearest cell is too far in (no real corner there).
        if (pz1 - pz0) >= 8 and (px1 - px0) >= 8:
            _leaf = Block("oak_leaves", states={"persistent": "true", "distance": "1"})
            _log = Block("oak_log", states={"axis": "y"})
            _dirt = Block("minecraft:dirt")
            for tz, tx in (
                (pz0 + 1, px0 + 1),
                (pz0 + 1, px1 - 1),
                (pz1 - 1, px0 + 1),
                (pz1 - 1, px1 - 1),
            ):
                clz, clx = min(
                    cell_set, key=lambda c: (c[0] - tz) ** 2 + (c[1] - tx) ** 2
                )
                if (
                    (clz - tz) ** 2 + (clx - tx) ** 2 > 9
                    or (clz, clx) in sq_occ
                    or (clz, clx) in occupied
                ):
                    continue
                wx, wz = self._tm.local_to_world(clz, clx)
                self._editor.placeBlock((wx, floor_y - 1, wz), _dirt)
                for h in range(3):
                    self._editor.placeBlock((wx, floor_y + h, wz), _log)
                for dz in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        self._editor.placeBlock((wx + dx, floor_y + 3, wz + dz), _leaf)
                self._editor.placeBlock((wx, floor_y + 4, wz), _leaf)
                sq_occ.add((clz, clx))

        # ── 7. Mark all market cells as occupied ─────────────────────────
        occupied.update(cells)

        logger.info(
            "place_market_square: y=%d  cells=%d  fountains=%d  stands=%d",
            floor_y,
            len(cells),
            fountains_placed,
            stands_placed,
        )
        return occupied
