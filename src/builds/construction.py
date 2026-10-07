"""
This module stages a small number of "under construction" building sites.

A settlement that is visibly *still being built* reads as alive rather than
freshly stamped.  Sites come in several flavours — a half-raised stone shell,
a timber frame going up, a freshly dug foundation, and a builders' stockpile
yard — each surrounded by material piles with a couple of workers on hand.
It is a cheap, self-contained gesture toward a living town: no schematics,
just well-composed primitives.

Sites are placed on flat, empty, road-adjacent lots inside the urban zone
after all real buildings exist, so they never displace anything.  The number
of sites scales with the size of the city: a boomtown has more scaffolding.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

import numpy as np
from gdpc.block import Block
from gdpc.editor import Editor

from builds._vegetation import clear_vegetation
from entities.population import _summon_villager
from terrain.terrain_types import SubZone, TerrainMap
from utils import log

if TYPE_CHECKING:
    from lore.names import Lore, Person

_AIR = Block("minecraft:air")
_SCAFFOLD = Block("scaffolding")

# Weathered stone-brick palette for footings and half-raised walls (mirrors the
# weathered palette used elsewhere): mostly crisp brick with mossy/cracked/
# cobble accents, so a half-built shell looks aged and work-in-progress.
_STONE_BRICKS = (
    "stone_bricks",
    "stone_bricks",
    "stone_bricks",
    "mossy_stone_bricks",
    "cracked_stone_bricks",
    "cobblestone",
    "mossy_cobblestone",
)
# Dug-earth floor of a work site.
_FLOOR_DIRT = ("coarse_dirt", "coarse_dirt", "rooted_dirt")
# Loose materials heaped around the site.
_MATERIALS = (
    Block("oak_log", states={"axis": "y"}),
    Block("cobblestone"),
    Block("stone_bricks"),
    Block("oak_planks"),
)
# Neat stacks for the stockpile yard.
_STACKABLE = (
    Block("oak_log", states={"axis": "z"}),
    Block("spruce_log", states={"axis": "x"}),
    Block("stone_bricks"),
    Block("oak_planks"),
    Block("hay_block"),
    Block("smooth_stone"),
)
_HEAPS = ("gravel", "sand", "clay", "packed_mud")

_URBAN_SUBZONES = (SubZone.RESIDENTIAL, SubZone.CIVIC, SubZone.TOWN_CENTER)

# Site flavours: (name, pad size, worker profession), weighted by repetition.
_SITE_TYPES = (
    ("stone_shell", 5, "mason"),
    ("stone_shell", 5, "mason"),
    ("timber_frame", 6, "toolsmith"),
    ("timber_frame", 6, "toolsmith"),
    ("foundation", 6, "mason"),
    ("stockpile", 4, "mason"),
)
_MAX_PAD = max(p for _, p, _ in _SITE_TYPES)


class ConstructionSiteGenerator:
    """Stage a few half-built sites of varied flavours on empty urban lots."""

    BASE_SITES = 2
    MAX_SITES = 6
    MAX_RANGE = 1  # only genuinely flat lots — no terracing for a work site

    def generate(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        lore: Optional["Lore"] = None,
        *,
        pre_occupied: Optional[set[tuple[int, int]]] = None,
    ) -> int:
        feat = terrain_map.features
        sub_map = terrain_map.sub_zone_map
        road_map = terrain_map.road_map_expanded
        if feat is None or sub_map is None:
            return 0

        depth, width = feat.shape
        occupied: set[tuple[int, int]] = set(pre_occupied) if pre_occupied else set()

        def _free_flat(tz: int, tx: int, pad: int) -> Optional[list[tuple[int, int]]]:
            cells = []
            hs = []
            touches_road = False
            for dz in range(pad):
                for dx in range(pad):
                    lz, lx = tz + dz, tx + dx
                    if not (0 <= lz < depth and 0 <= lx < width):
                        return None
                    if (lz, lx) in occupied:
                        return None
                    if sub_map[lz, lx] not in (int(s) for s in _URBAN_SUBZONES):
                        return None
                    if road_map is not None and road_map[lz, lx] != 0:
                        return None
                    if float(feat[lz, lx]["water_pct"]) > 0.0:
                        return None
                    cells.append((lz, lx))
                    hs.append(int(feat[lz, lx]["height"]))
            if max(hs) - min(hs) > self.MAX_RANGE:
                return None
            # Require a road cell adjacent to the pad so the site is accessible.
            for lz, lx in cells:
                for ddz, ddx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nz, nx = lz + ddz, lx + ddx
                    if (
                        0 <= nz < depth
                        and 0 <= nx < width
                        and road_map is not None
                        and road_map[nz, nx] != 0
                    ):
                        touches_road = True
            return cells if touches_road else None

        # Collect candidate anchors (largest pad, so every flavour fits), then
        # greedily pick well-spread ones.
        anchors: list[tuple[int, int]] = []
        urban_cells = np.argwhere(np.isin(sub_map, [int(s) for s in _URBAN_SUBZONES]))
        for r in urban_cells:
            tz, tx = int(r[0]), int(r[1])
            if _free_flat(tz, tx, _MAX_PAD) is not None:
                anchors.append((tz, tx))

        if not anchors:
            log("[Construction] No free urban lot found, skipping.")
            return 0

        # A bigger city visibly builds more: scale the site count with the
        # urban area (~1 extra site per 8000 urban cells, capped).
        n_urban_cells = len(urban_cells)
        n_sites = min(self.MAX_SITES, self.BASE_SITES + n_urban_cells // 8000)

        rng = random.Random(len(anchors) * 2654435761 & 0xFFFFFFFF)
        rng.shuffle(anchors)
        chosen: list[tuple[int, int]] = []
        for a in anchors:
            if len(chosen) >= n_sites:
                break
            if all((a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2 >= 20**2 for c in chosen):
                chosen.append(a)

        placed = 0
        for tz, tx in chosen:
            site_type, pad, profession = rng.choice(_SITE_TYPES)
            cells = _free_flat(tz, tx, pad)
            if cells is None:
                continue
            self._build_site(
                editor,
                terrain_map,
                feat,
                tz,
                tx,
                pad,
                cells,
                site_type,
                lore,
                rng,
                profession,
            )
            occupied |= set(cells)
            if pre_occupied is not None:
                pre_occupied |= set(cells)
            placed += 1

        log(f"[Construction] Staged {placed} under-construction sites.")
        return placed

    # ------------------------------------------------------------------
    def _build_site(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        feat: np.ndarray,
        tz: int,
        tx: int,
        pad: int,
        cells: list[tuple[int, int]],
        site_type: str,
        lore: Optional["Lore"],
        rng: random.Random,
        profession: str,
    ) -> None:
        floor_y = int(np.median([int(feat[lz, lx]["height"]) for lz, lx in cells])) - 1

        # Clear vegetation, then lay the base pad for the flavour.
        clear_vegetation(
            cells,
            feat,
            feat.shape[0],
            feat.shape[1],
            editor,
            terrain_map,
            protected=set(),
        )
        dig = 1 if site_type == "foundation" else 0
        for lz, lx in cells:
            wx, wz = terrain_map.local_to_world(lz, lx)
            for dy in range(1 - dig, 6):
                editor.placeBlock((wx, floor_y + dy, wz), _AIR)
            bz, bx = lz - tz, lx - tx
            on_edge = bz in (0, pad - 1) or bx in (0, pad - 1)
            if site_type == "foundation":
                # Excavated pit: cobble footing trench around a dug-out floor.
                block = (
                    Block(rng.choice(_STONE_BRICKS))
                    if on_edge
                    else Block(rng.choice(("packed_mud", "coarse_dirt", "mud")))
                )
                editor.placeBlock((wx, floor_y - dig, wz), block)
            elif site_type == "stockpile":
                # Trodden storage yard, no footing.
                editor.placeBlock(
                    (wx, floor_y, wz),
                    Block(rng.choice(("dirt_path", "coarse_dirt", "gravel"))),
                )
            else:
                block = (
                    Block(rng.choice(_STONE_BRICKS))
                    if on_edge
                    else Block(rng.choice(_FLOOR_DIRT))
                )
                editor.placeBlock((wx, floor_y, wz), block)

        interior = [
            (lz, lx)
            for (lz, lx) in cells
            if 0 < lz - tz < pad - 1 and 0 < lx - tx < pad - 1
        ]
        rng.shuffle(interior)

        if site_type == "stone_shell":
            self._stage_stone_shell(editor, terrain_map, tz, tx, pad, floor_y, rng)
            props = [
                (rng.choice(_MATERIALS), rng.randint(1, 2)),
                (rng.choice(_MATERIALS), rng.randint(1, 2)),
                (Block("hay_block"), 1),
                (Block("crafting_table"), 1),
                (Block("scaffolding"), 1),
            ]
        elif site_type == "timber_frame":
            self._stage_timber_frame(editor, terrain_map, tz, tx, pad, floor_y, rng)
            props = [
                (Block("oak_log", states={"axis": "y"}), rng.randint(1, 2)),
                (Block("oak_planks"), rng.randint(1, 2)),
                (Block("barrel", states={"facing": "up"}), 1),
                (Block("crafting_table"), 1),
                (Block("stonecutter"), 1),
            ]
        elif site_type == "foundation":
            props = [
                (Block(rng.choice(_HEAPS)), rng.randint(1, 2)),
                (Block(rng.choice(_HEAPS)), 1),
                (Block("cobblestone"), rng.randint(1, 2)),
                (Block("barrel", states={"facing": "up"}), 1),
                (Block("oak_fence"), 1),
            ]
        else:  # stockpile
            props = [
                (rng.choice(_STACKABLE), rng.randint(2, 3)),
                (rng.choice(_STACKABLE), rng.randint(1, 3)),
                (rng.choice(_STACKABLE), rng.randint(1, 2)),
                (Block(rng.choice(_HEAPS)), rng.randint(1, 2)),
                (Block("scaffolding"), rng.randint(2, 3)),
            ]

        prop_y = floor_y + 1 - dig
        for (lz, lx), (block, stack) in zip(interior, props):
            wx, wz = terrain_map.local_to_world(lz, lx)
            for h in range(stack):
                editor.placeBlock((wx, prop_y + h, wz), block)

        # A couple of workers on site, named if we have a roster.  Only draw
        # residents who haven't been spawned elsewhere yet (e.g. by an earlier
        # site, or later by PopulationGenerator), so nobody appears twice.
        people: list["Person"] = []
        if lore is not None:
            unspawned = [p for p in lore.residents if not p.spawned]
            people = [p for p in unspawned if p.trade == "mason"] or unspawned[:2]
        for k in range(2):
            if not interior:
                break
            lz, lx = interior[-(k + 1)]
            wx, wz = terrain_map.local_to_world(lz, lx)
            person = people[k] if k < len(people) else None
            _summon_villager(
                editor,
                wx,
                prop_y,
                wz,
                name=person.name if person else None,
                profession=profession,
            )
            if person is not None:
                person.spawned = True

    # ------------------------------------------------------------------
    def _stage_stone_shell(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        tz: int,
        tx: int,
        pad: int,
        floor_y: int,
        rng: random.Random,
    ) -> None:
        """Corner scaffolding posts + a half-raised stone wall on two sides."""
        for cz, cx in ((0, 0), (0, pad - 1), (pad - 1, 0), (pad - 1, pad - 1)):
            wx, wz = terrain_map.local_to_world(tz + cz, tx + cx)
            for h in range(1, 5):
                editor.placeBlock((wx, floor_y + h, wz), _SCAFFOLD)

        wall_h = rng.randint(1, 2)
        for bx in range(pad):
            wx, wz = terrain_map.local_to_world(tz + 0, tx + bx)
            for h in range(1, wall_h + 1):
                editor.placeBlock(
                    (wx, floor_y + h, wz), Block(rng.choice(_STONE_BRICKS))
                )
        for bz in range(pad):
            wx, wz = terrain_map.local_to_world(tz + bz, tx + 0)
            for h in range(1, wall_h + 1):
                editor.placeBlock(
                    (wx, floor_y + h, wz), Block(rng.choice(_STONE_BRICKS))
                )

    def _stage_timber_frame(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        tz: int,
        tx: int,
        pad: int,
        floor_y: int,
        rng: random.Random,
    ) -> None:
        """Full-height log corner posts, top plates, half-planked walls and a
        scaffolding tower — a house skeleton going up."""
        post = Block("oak_log", states={"axis": "y"})
        post_h = 4
        corners = ((0, 0), (0, pad - 1), (pad - 1, 0), (pad - 1, pad - 1))
        for cz, cx in corners:
            wx, wz = terrain_map.local_to_world(tz + cz, tx + cx)
            for h in range(1, post_h + 1):
                editor.placeBlock((wx, floor_y + h, wz), post)

        # Top plates connecting the posts along two sides (a partial ring beam).
        beam_z = Block("oak_log", states={"axis": "z"})
        beam_x = Block("oak_log", states={"axis": "x"})
        for bx in range(1, pad - 1):
            wx, wz = terrain_map.local_to_world(tz + 0, tx + bx)
            editor.placeBlock((wx, floor_y + post_h, wz), beam_x)
        for bz in range(1, pad - 1):
            wx, wz = terrain_map.local_to_world(tz + bz, tx + 0)
            editor.placeBlock((wx, floor_y + post_h, wz), beam_z)

        # Half-planked infill on one side, rising course by course.
        wall_h = rng.randint(1, 2)
        for bx in range(1, pad - 1):
            wx, wz = terrain_map.local_to_world(tz + 0, tx + bx)
            for h in range(1, wall_h + 1):
                editor.placeBlock((wx, floor_y + h, wz), Block("oak_planks"))

        # Scaffolding tower against one open corner.
        sz, sx = pad - 1, pad - 1
        wx, wz = terrain_map.local_to_world(tz + sz, tx + sx - 1)
        for h in range(1, post_h + 1):
            editor.placeBlock((wx, floor_y + h, wz), _SCAFFOLD)
