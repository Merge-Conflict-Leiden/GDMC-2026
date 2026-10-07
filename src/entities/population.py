"""
This module handles villager population placement along the road network.

When a :class:`lore.Lore` roster is supplied, the settlement's named residents
are spawned as custom-named villagers with the profession matching their trade,
so the people the chronicle names can actually be found walking the streets.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

import numpy as np
from gdpc.editor import Editor

from terrain.terrain_types import TerrainMap
from utils import log

if TYPE_CHECKING:
    from lore.names import Lore, Person

_ROAD_CELLS_PER_VILLAGER = 20
_ROAD_CELLS_PER_CRITTER = 40  # ~one ambient animal per N dry road cells

# Climate-appropriate ambient wildlife.  These wander freely (unlike penned
# livestock) so small/jumpy species that would escape a fence — rabbits, goats,
# frogs — are welcome here.  Matched by biome substring, first hit wins, so the
# more specific keys are listed before the general ones; falls back to _DEFAULT.
_DEFAULT_FAUNA = ["cat", "bee", "rabbit"]
_CLIMATE_FAUNA: list[tuple[str, list[str]]] = [
    ("bamboo", ["parrot", "parrot", "cat"]),
    ("jungle", ["parrot", "parrot", "cat", "bee"]),
    ("mangrove", ["frog", "frog"]),
    ("swamp", ["frog", "frog", "cat"]),
    ("old_growth", ["wolf", "wolf", "rabbit"]),  # old-growth taigas
    ("taiga", ["wolf", "rabbit", "cat"]),
    # "cherry" before "grove": the cherry biome id is "cherry_grove".
    ("cherry", ["bee", "rabbit", "cat"]),
    ("grove", ["wolf", "goat", "rabbit"]),
    ("snowy", ["wolf", "rabbit", "goat"]),
    ("frozen", ["wolf", "rabbit", "goat"]),
    ("ice", ["wolf", "goat"]),
    ("peaks", ["goat", "goat"]),
    ("windswept", ["goat", "wolf"]),
    ("mountain", ["goat", "wolf"]),
    ("savanna", ["armadillo", "armadillo", "cat", "bee"]),
    ("badlands", ["armadillo", "rabbit"]),
    ("desert", ["rabbit", "cat"]),
    ("flower", ["bee", "bee", "rabbit"]),
    ("sunflower", ["bee", "rabbit"]),
    ("meadow", ["bee", "rabbit", "cat"]),
    ("plains", ["bee", "rabbit", "cat"]),
    ("forest", ["wolf", "rabbit", "bee", "cat"]),
]


def _climate_fauna(biome: str) -> list[str]:
    """Return the ambient-wildlife pool for a (lowercased) biome id."""
    for key, pool in _CLIMATE_FAUNA:
        if key in biome:
            return pool
    return _DEFAULT_FAUNA


def _summon_entity(
    editor: Editor,
    entity: str,
    wx: int,
    wy: int,
    wz: int,
    *,
    persist: bool = False,
) -> None:
    """Summon a plain mob, optionally flagged not to despawn."""
    nbt = " {PersistenceRequired:1b}" if persist else ""
    editor.runCommand(f"summon minecraft:{entity} {wx} {wy} {wz}{nbt}")


def _summon_villager(
    editor: Editor,
    wx: int,
    wy: int,
    wz: int,
    *,
    name: Optional[str] = None,
    profession: Optional[str] = None,
) -> None:
    """Summon a villager, optionally custom-named and with a set profession."""
    tags: list[str] = ["PersistenceRequired:1b"]
    if name:
        # MC 1.21.5 stores CustomName as a direct text component, so a bare
        # quoted string is the name itself; wrapping it in {"text":...} would
        # render the JSON literally.
        safe = name.replace("\\", "").replace('"', "").replace("'", "")
        tags.append(f"CustomName:'{safe}'")
        tags.append("CustomNameVisible:1b")
    if profession:
        tags.append(
            "VillagerData:{"
            f'profession:"minecraft:{profession}",'
            'level:2,type:"minecraft:plains"}'
        )
    nbt = "{" + ",".join(tags) + "}"
    editor.runCommand(f"summon minecraft:villager {wx} {wy} {wz} {nbt}")


class PopulationGenerator:
    """Spawn villagers on road cells — named residents first, then extras."""

    def generate(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        lore: Optional["Lore"] = None,
    ) -> None:
        feat = terrain_map.features
        road_map = terrain_map.road_map

        if feat is None or road_map is None:
            log("[Population] No features or road map, skipping.")
            return

        # Only spawn on LAND road cells.  A road cell over water is a bridge
        # span whose feat["height"] is the river bottom (OCEAN_FLOOR), so a
        # villager placed there would land underwater beneath the deck and be
        # stuck — filter those out.
        wm = terrain_map.water_mask

        def _is_water(lz: int, lx: int) -> bool:
            if wm is not None:
                return bool(wm[lz, lx])
            return float(feat[lz, lx]["water_pct"]) > 0.5

        road_cells = [
            (int(lz), int(lx))
            for lz, lx in zip(*np.where(road_map > 0))
            if not _is_water(int(lz), int(lx))
        ]
        if not road_cells:
            log("[Population] No dry road cells found, skipping.")
            return

        # Skip residents another system (e.g. ConstructionSiteGenerator's
        # masons) has already summoned into the world, so nobody is doubled.
        residents: list["Person"] = (
            [p for p in lore.residents if not p.spawned] if lore is not None else []
        )

        # Enough spawn points for every named resident plus the usual crowd.
        n_generic = max(1, len(road_cells) // _ROAD_CELLS_PER_VILLAGER)
        n_total = min(len(road_cells), len(residents) + n_generic)
        spawn_cells = random.sample(road_cells, n_total)

        named = 0
        for idx, (lz, lx) in enumerate(spawn_cells):
            wx, wz = terrain_map.local_to_world(int(lz), int(lx))
            # feat["height"] is the road walking surface (road block is one
            # below).  Spawn one block above it so villagers settle onto the
            # road rather than risk clipping into the surface.
            surface_y = int(feat[lz, lx]["height"]) + 1
            if idx < len(residents):
                person = residents[idx]
                _summon_villager(
                    editor,
                    wx,
                    surface_y,
                    wz,
                    name=person.name,
                    profession=person.profession,
                )
                person.spawned = True
                named += 1
            else:
                _summon_villager(editor, wx, surface_y, wz)

        log(
            f"[Population] Spawned {len(spawn_cells)} villagers "
            f"({named} named residents)."
        )

        # ── Wandering traders: a small, transient presence passing through ──
        # Kept deliberately sparse (0–2, scaled with town size) so the town
        # doesn't read as a trader convention.  Each brings a llama or two.
        road_set = set(road_cells)
        n_traders = min(2, 1 + len(road_cells) // 200) if random.random() < 0.8 else 0
        traders = 0
        for lz, lx in random.sample(road_cells, min(n_traders, len(road_cells))):
            wx, wz = terrain_map.local_to_world(int(lz), int(lx))
            wy = int(feat[lz, lx]["height"]) + 1
            _summon_entity(editor, "wandering_trader", wx, wy, wz)
            traders += 1
            for dz, dx in random.sample([(1, 0), (-1, 0), (0, 1), (0, -1)], 2):
                nz, nx = int(lz) + dz, int(lx) + dx
                if (nz, nx) in road_set:
                    nwx, nwz = terrain_map.local_to_world(nz, nx)
                    _summon_entity(
                        editor,
                        "trader_llama",
                        nwx,
                        int(feat[nz, nx]["height"]) + 1,
                        nwz,
                    )

        # ── Climate-appropriate ambient wildlife ───────────────────────────
        # A representative biome from the centre of the road network picks the
        # pool; a handful of critters are scattered (kept persistent so they
        # stay part of the scene rather than wandering off / despawning).
        mz = int(np.mean([c[0] for c in road_cells]))
        mx = int(np.mean([c[1] for c in road_cells]))
        mwx, mwz = terrain_map.local_to_world(mz, mx)
        try:
            biome = editor.getBiome((mwx, int(feat[mz, mx]["height"]) - 1, mwz)).lower()
        except Exception:
            biome = ""
        pool = _climate_fauna(biome)
        n_fauna = min(16, max(2, len(road_cells) // _ROAD_CELLS_PER_CRITTER))
        critters = 0
        for lz, lx in random.sample(road_cells, min(n_fauna, len(road_cells))):
            wx, wz = terrain_map.local_to_world(int(lz), int(lx))
            wy = int(feat[lz, lx]["height"]) + 1
            _summon_entity(editor, random.choice(pool), wx, wy, wz, persist=True)
            critters += 1

        log(
            f"[Population] Spawned {traders} wandering trader(s) and "
            f"{critters} ambient animal(s) (biome '{biome}')."
        )
