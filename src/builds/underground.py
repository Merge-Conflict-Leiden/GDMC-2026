"""
Underground structures — currently a crypt beneath the graveyard.

Few GDMC entries build anything underground (it has its own "best underground"
special award), so a hand-lit stone crypt under the churchyard is both a
differentiator and a natural extension of the named-dead narrative: the notable
departed named on the surface graves are entombed here, and the chronicle can
send readers down to find them.

The crypt is carved straight down into solid terrain (always safe — everything
below the surface is ground), enclosed in weathered stone, and reached by a
ladder hatch in the churchyard.  Everything is seeded from the site so a given
map always yields the same crypt.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

from gdpc.block import Block
from gdpc.editor import Editor

from terrain.terrain_types import TerrainMap
from utils import log, sign_nbt

if TYPE_CHECKING:
    from lore.names import Lore

_AIR = Block("minecraft:air")
# Weathered masonry pool — instantly "ancient", and a nod to the age/wear idea.
_STONE = [
    "stone_bricks",
    "stone_bricks",
    "cracked_stone_bricks",
    "mossy_stone_bricks",
    "chiseled_stone_bricks",
]
_LOOT = (
    "minecraft:chests/simple_dungeon",
    "minecraft:chests/abandoned_mineshaft",
)


class CryptGenerator:
    """Carve a lantern-lit stone crypt beneath the graveyard."""

    RW = 7  # interior width  (x)
    RL = 9  # interior length (z)
    RH = 4  # interior height
    CAP = 3  # solid earth left above the ceiling

    def generate(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        lore: Optional["Lore"] = None,
        *,
        pre_occupied: Optional[set[tuple[int, int]]] = None,
    ) -> bool:
        """Build the crypt.  Returns True if one was placed."""
        feat = terrain_map.features
        site = terrain_map.graveyard_site
        if feat is None or site is None:
            return False

        depth, width = feat.shape
        cz, cx = site
        ground = int(feat[cz, cx]["height"]) - 1  # top solid block of the surface
        floor_y = ground - (self.CAP + self.RH + 2)
        if floor_y < 5:
            log("[Crypt] Surface too low for a crypt, skipping.")
            return False

        ceiling_y = floor_y + self.RH + 1
        ix0, ix1 = cx - self.RW // 2, cx + self.RW // 2
        iz0, iz1 = cz - self.RL // 2, cz + self.RL // 2
        # Need room for the walls plus a little margin for the ladder hatch.
        if ix0 - 1 < 0 or ix1 + 1 >= width or iz0 - 1 < 0 or iz1 + 1 >= depth:
            log("[Crypt] Reserved footprint hits the build edge, skipping.")
            return False

        rng = random.Random(((cz & 0xFF) << 8) ^ (cx & 0xFF) ^ 0x0C)

        def w(lz: int, lx: int) -> tuple[int, int]:
            return terrain_map.local_to_world(lz, lx)

        def stone() -> Block:
            return Block(rng.choice(_STONE))

        # ── Shell: floor, ceiling, and boundary walls (interior hollowed) ────
        for lx in range(ix0 - 1, ix1 + 2):
            for lz in range(iz0 - 1, iz1 + 2):
                wx, wz = w(lz, lx)
                editor.placeBlock((wx, floor_y, wz), stone())
                editor.placeBlock((wx, ceiling_y, wz), stone())
                on_wall = lx in (ix0 - 1, ix1 + 1) or lz in (iz0 - 1, iz1 + 1)
                for y in range(floor_y + 1, ceiling_y):
                    editor.placeBlock((wx, y, wz), stone() if on_wall else _AIR)

        # ── Ladder hatch up to the churchyard ───────────────────────────────
        # Shaft in the interior cell against the north wall; the ladder clings
        # to that wall.  A trapdoor + stone rim frames the opening at the surface.
        sx, sz = cx, iz0
        for y in range(floor_y + 1, ground + 2):
            wx, wz = w(sz, sx)
            editor.placeBlock((wx, y, wz), _AIR)
            editor.placeBlock((wx, y, wz), Block("ladder", states={"facing": "south"}))
        wx, wz = w(sz, sx)
        editor.placeBlock(
            (wx, ground + 1, wz),
            Block(
                "oak_trapdoor",
                states={"half": "top", "facing": "south", "open": "false"},
            ),
        )
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0:
                    continue
                rx, rz = w(sz + dz, sx + dx)
                editor.placeBlock((rx, ground + 1, rz), stone())

        # ── Central founder plinth ──────────────────────────────────────────
        wx, wz = w(cz, cx)
        editor.placeBlock((wx, floor_y + 1, wz), Block("chiseled_stone_bricks"))
        editor.placeBlock((wx, floor_y + 2, wz), Block("soul_lantern"))

        # ── Tombs along the two long walls ──────────────────────────────────
        for lx in (ix0, ix1):
            for lz in range(iz0 + 1, iz1, 3):
                if (lx, lz) == (sx, sz):
                    continue
                tx, tz = w(lz, lx)
                editor.placeBlock((tx, floor_y + 1, tz), Block("chiseled_stone_bricks"))
                editor.placeBlock(
                    (tx, floor_y + 2, tz),
                    Block("smooth_stone_slab", states={"type": "bottom"}),
                )
                if rng.random() < 0.5:
                    editor.placeBlock(
                        (tx, floor_y + 3, tz),
                        Block("candle", states={"lit": "false", "candles": "1"}),
                    )

        # ── Hanging soul-lanterns for a cold blue glow ──────────────────────
        # Hung directly beneath the stone ceiling (no chain block — some server
        # builds reject minecraft:chain).
        for lx in (ix0 + 1, ix1 - 1):
            for lz in (iz0 + 2, iz1 - 2):
                hx, hz = w(lz, lx)
                editor.placeBlock(
                    (hx, ceiling_y - 1, hz),
                    Block("soul_lantern", states={"hanging": "true"}),
                )

        # ── Cobwebs in the corners for age ──────────────────────────────────
        for _ in range(6):
            lx = rng.choice([ix0, ix1, rng.randint(ix0, ix1)])
            lz = rng.randint(iz0, iz1)
            cxw, czw = w(lz, lx)
            editor.placeBlock((cxw, ceiling_y - 1, czw), Block("cobweb"))

        # ── "Wall of the dead": name plaques on the south wall ──────────────
        dead = list(lore.deceased) if lore is not None else []
        town = lore.settlement_name if lore is not None else "this town"
        for k in range(min(6, max(1, len(dead)))):
            lx = cx - 1 + (k % 3)
            row_y = floor_y + 2 + (k // 3)
            px, pz = w(iz1, lx)
            person = dead[k] if k < len(dead) else None
            if person is not None:
                given, _, surname = str(person.name).partition(" ")
                data = sign_nbt(["Here lies", given, surname, "of " + town])
            else:
                data = sign_nbt(["The forgotten", "dead of", town, ""])
            editor.placeBlock(
                (px, row_y, pz),
                Block("oak_wall_sign", states={"facing": "north"}, data=data),
            )

        # ── Loot chest at the far end ───────────────────────────────────────
        lx, lz = cx, iz1 - 1
        gx, gz = w(lz, lx)
        editor.placeBlock(
            (gx, floor_y + 1, gz),
            Block(
                "chest",
                states={"facing": "north"},
                data=f'{{LootTable:"{rng.choice(_LOOT)}"}}',
            ),
        )

        # Keep surface scatter off the hatch and its stone rim.
        if pre_occupied is not None:
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    pre_occupied.add((sz + dz, sx + dx))

        log(f"[Crypt] Carved a crypt beneath the graveyard at floor y={floor_y}.")
        return True
