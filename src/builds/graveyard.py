"""
This module defines a generator for the GRAVEYARD sub-zone.

A single fenced burial ground is placed just outside the civic core (site chosen
in ``classifier._detect_graveyard``).  It contains:

  * a low wall of mossy cobblestone and tuff around the plot with a single
    road-facing opening,
  * rows of headstones, each an inscribed sign naming one of the settlement's
    departed (drawn from the shared :class:`lore.Lore` roster),
  * a handful of older, illegible graves for atmosphere,
  * a central memorial naming the town, and lanterns along the wall.

The named dead here are the same people the chronicle mourns, so the book and
the world agree on who lies buried.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

import numpy as np
from gdpc.block import Block
from gdpc.editor import Editor

from builds._vegetation import clear_vegetation
from builds.animal_pen import _largest_component
from terrain.terrain_types import SubZone, TerrainMap
from utils import log, sign_nbt

if TYPE_CHECKING:
    from lore.names import Lore, Person

_AIR = Block("minecraft:air")
_WALL_POOL = (Block("mossy_cobblestone_wall"), Block("tuff_wall"))
_LANTERN = Block("lantern")
_GROUND_POOL = (
    Block("dead_fire_coral_block"),
    Block("dead_horn_coral_block"),
    Block("dead_brain_coral_block"),
    Block("dead_tube_coral_block"),
)
_GRASS = Block("grass_block")
GRASS_GROUND_CHANCE = 0.5
_MOUND_POOL = (Block("coarse_dirt"), Block("podzol"), Block("rooted_dirt"))
# Sign rotation → (dz, dx) the sign faces; the grave body lies that way.
_ROT_TO_DIR = {0: (1, 0), 4: (0, -1), 8: (-1, 0), 12: (0, 1)}
_GRAVE_FLOWERS = ("poppy", "wither_rose", "white_tulip", "dead_bush", "torchflower")
_HEADSTONES = (
    "stone_brick_wall",
    "mossy_stone_brick_wall",
    "cobblestone_wall",
    "andesite_wall",
)

_CLEAR_HEIGHT = 6


def _components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Split *cells* into 4-connected components."""
    remaining = set(cells)
    out: list[set[tuple[int, int]]] = []
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
        out.append(comp)
    return out


class GraveyardGenerator:
    """Build the fenced burial ground reserved by the classifier."""

    MAX_GRAVES = 14
    MAX_CELL_SLOPE = 1.5  # churchyard cells must be genuinely level ground
    MAX_PLOT_RELIEF = 2  # whole plot stays on one terrace (± of median)

    def generate(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        lore: Optional["Lore"] = None,
        *,
        pre_occupied: Optional[set[tuple[int, int]]] = None,
    ) -> int:
        """Render the graveyard.  Returns the number of headstones placed."""
        feat = terrain_map.features
        sub_map = terrain_map.sub_zone_map
        road_map = terrain_map.road_map_expanded

        if feat is None or sub_map is None or terrain_map.graveyard_site is None:
            log("[Graveyard] No site reserved, skipping.")
            return 0

        coords = np.argwhere(sub_map == int(SubZone.GRAVEYARD))
        if len(coords) == 0:
            log("[Graveyard] Site sub-zone empty, skipping.")
            return 0

        depth, width = feat.shape
        excl: set[tuple[int, int]] = set(pre_occupied) if pre_occupied else set()

        def _on_road(lz: int, lx: int) -> bool:
            return (
                road_map is not None
                and 0 <= lz < road_map.shape[0]
                and 0 <= lx < road_map.shape[1]
                and road_map[lz, lx] != 0
            )

        plot_all: set[tuple[int, int]] = {
            (int(r[0]), int(r[1]))
            for r in coords
            if (int(r[0]), int(r[1])) not in excl and not _on_road(int(r[0]), int(r[1]))
        }

        # The classifier may reserve several churchyard districts on a large
        # map — build each connected patch as its own graveyard.
        total_graves = 0
        for component in sorted(_components(plot_all), key=len, reverse=True):
            total_graves += self._build_one(
                editor, terrain_map, feat, road_map, component, lore, pre_occupied, excl
            )

        if total_graves == 0:
            log("[Graveyard] No flat plot in any reserved district, skipping.")
        return total_graves

    def _build_one(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        feat: np.ndarray,
        road_map,
        plot: set[tuple[int, int]],
        lore: Optional["Lore"],
        pre_occupied: Optional[set[tuple[int, int]]],
        excl: set[tuple[int, int]],
    ) -> int:
        """Render one churchyard on *plot*.  Returns headstones placed."""
        depth, width = feat.shape

        # A churchyard is level ground: keep only genuinely flat cells on one
        # terrace (largest connected patch), so the wall and headstone rows
        # never march down a hillside even when the reserved district slopes.
        plot = {
            (lz, lx)
            for lz, lx in plot
            if float(feat[lz, lx]["slope"]) <= self.MAX_CELL_SLOPE
        }
        plot = _largest_component(plot)
        if plot:
            med_h = float(np.median([int(feat[lz, lx]["height"]) for lz, lx in plot]))
            plot = {
                c
                for c in plot
                if abs(int(feat[c[0], c[1]]["height"]) - med_h) <= self.MAX_PLOT_RELIEF
            }
            plot = _largest_component(plot)

        if len(plot) < 9:
            return 0

        clear_vegetation(
            list(plot), feat, depth, width, editor, terrain_map, protected=excl
        )

        # ── Ground ────────────────────────────────────────────────────────
        for lz, lx in plot:
            wx, wz = terrain_map.local_to_world(lz, lx)
            gy = int(feat[lz, lx]["height"]) - 1
            for dy in range(1, _CLEAR_HEIGHT + 1):
                editor.placeBlock((wx, gy + dy, wz), _AIR)
            ground = (
                _GRASS
                if random.random() < GRASS_GROUND_CHANCE
                else random.choice(_GROUND_POOL)
            )
            editor.placeBlock((wx, gy, wz), ground)

        # ── Perimeter wall + road-facing gate opening ─────────────────────
        boundary = [
            (lz, lx)
            for (lz, lx) in plot
            if any(
                (lz + dz, lx + dx) not in plot
                for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
        ]
        gate_cell = self._nearest_road_boundary(boundary, road_map)
        for i, (lz, lx) in enumerate(sorted(boundary)):
            if (lz, lx) == gate_cell:
                continue  # leave the entrance open
            wx, wz = terrain_map.local_to_world(lz, lx)
            gy = int(feat[lz, lx]["height"]) - 1
            editor.placeBlock((wx, gy + 1, wz), random.choice(_WALL_POOL))
            if i % 6 == 0:  # lantern atop every sixth wall post
                editor.placeBlock((wx, gy + 2, wz), _LANTERN)

        # ── Graves ────────────────────────────────────────────────────────
        interior = sorted(plot - set(boundary))
        if not interior:
            interior = sorted(plot)
        min_lz = min(lz for lz, _ in interior)
        min_lx = min(lx for _, lx in interior)

        # Only name the dead not already carved into an earlier churchyard.
        the_dead = (
            [p for p in lore.deceased if not p.buried] if lore is not None else []
        )
        graves_placed = 0
        for lz, lx in interior:
            if graves_placed >= self.MAX_GRAVES:
                break
            if (lz - min_lz) % 2 != 0 or (lx - min_lx) % 3 != 0:
                continue
            person = the_dead[graves_placed] if graves_placed < len(the_dead) else None
            if self._place_grave(
                editor, terrain_map, feat, lz, lx, person, set(interior)
            ):
                if person is not None:
                    person.buried = True
                graves_placed += 1

        # ── Central memorial ──────────────────────────────────────────────
        self._place_memorial(editor, terrain_map, feat, plot, lore)

        # Hand the whole plot back so later phases skip it.
        if pre_occupied is not None:
            pre_occupied |= plot

        log(f"[Graveyard] Placed {graves_placed} graves.")
        return graves_placed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _nearest_road_boundary(
        self,
        boundary: list[tuple[int, int]],
        road_map,
    ) -> Optional[tuple[int, int]]:
        """Boundary cell closest to any road cell — becomes the gate opening."""
        if road_map is None or not boundary:
            return None
        road_cells = np.argwhere(road_map != 0)
        if len(road_cells) == 0:
            return boundary[0]
        best = None
        best_d = None
        for lz, lx in boundary:
            d = int(((road_cells[:, 0] - lz) ** 2 + (road_cells[:, 1] - lx) ** 2).min())
            if best_d is None or d < best_d:
                best_d, best = d, (lz, lx)
        return best

    def _place_grave(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        feat: np.ndarray,
        lz: int,
        lx: int,
        person: Optional["Person"],
        allowed: set[tuple[int, int]],
    ) -> bool:
        wx, wz = terrain_map.local_to_world(lz, lx)
        gy = int(feat[lz, lx]["height"]) - 1

        # The mound is two blocks long — a body laid out in front of the
        # stone, extending the way the sign faces.  Pick a facing whose foot
        # cell stays inside the plot interior; fall back to a lone mound.
        rotations = [0, 4, 8, 12]  # face a cardinal aisle
        random.shuffle(rotations)
        rotation = rotations[0]
        foot: Optional[tuple[int, int]] = None
        for rot in rotations:
            dz, dx = _ROT_TO_DIR[rot]
            cand = (lz + dz, lx + dx)
            if cand in allowed:
                rotation, foot = rot, cand
                break

        mound = random.choice(_MOUND_POOL)
        editor.placeBlock((wx, gy, wz), mound)
        if foot is not None:
            fwx, fwz = terrain_map.local_to_world(*foot)
            fgy = int(feat[foot[0], foot[1]]["height"]) - 1
            editor.placeBlock((fwx, fgy, fwz), mound)
            # Occasional flower on the grave itself, at the body's feet.
            if random.random() < 0.5:
                editor.placeBlock(
                    (fwx, fgy + 1, fwz), Block(random.choice(_GRAVE_FLOWERS))
                )

        # Headstone: a wall block with an inscribed sign atop it.
        editor.placeBlock((wx, gy + 1, wz), Block(random.choice(_HEADSTONES)))

        if person is not None and getattr(person, "name", None):
            given, _, surname = str(person.name).partition(" ")
            epitaph = getattr(person, "epitaph", None) or "Rest well"
            data = sign_nbt(["Here lies", given, surname, epitaph])
        else:
            # Older, weathered grave — no legible name.
            data = sign_nbt(["Here lies", "one long", "forgotten", ""])
        sign = Block("oak_sign", states={"rotation": str(rotation)}, data=data)
        editor.placeBlock((wx, gy + 2, wz), sign)
        return True

    def _place_memorial(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        feat: np.ndarray,
        plot: set[tuple[int, int]],
        lore: Optional["Lore"],
    ) -> None:
        cz = int(round(sum(lz for lz, _ in plot) / len(plot)))
        cx = int(round(sum(lx for _, lx in plot) / len(plot)))
        if (cz, cx) not in plot:
            cz, cx = next(iter(plot))
        wx, wz = terrain_map.local_to_world(cz, cx)
        gy = int(feat[cz, cx]["height"]) - 1

        editor.placeBlock((wx, gy + 1, wz), Block("chiseled_stone_bricks"))
        editor.placeBlock((wx, gy + 2, wz), Block("stone_bricks"))
        town = lore.settlement_name if lore is not None else "this town"
        data = sign_nbt(["In memory of", "the dead of", town, "gone before"])
        editor.placeBlock(
            (wx, gy + 3, wz),
            Block("oak_sign", states={"rotation": "0"}, data=data),
        )
