"""
This module defines a generator for the MAZE sub-zone — an ornamental hedge
labyrinth planted as a pleasure garden near the keep (or, in a large city, as
a courtyard garden inside a residential quarter).

It is a deliberate "game within the game", with a reward chest at the heart.

Design
------
  * The pad is grown organically (BFS region-growing from the reserved site)
    across gently varying terrain, so the labyrinth hugs the district's real
    shape instead of being stamped as a square — hedges follow terrain like a
    proper landscaped garden.
  * A maze is carved with an iterative recursive-backtracker over the lattice
    of corridor cells that fit inside the organic pad, then lightly "braided":
    a few extra walls are knocked through, creating loops so larger mazes read
    as genuinely intricate rather than one long corridor.
  * Walls are biome-appropriate hedges (leaf blocks); corridors are dirt paths.
  * Big mazes get two or three lantern-lit entrances; the reward chest sits at
    the corridor cell that maximises the MINIMUM walking distance to every
    entrance, so the prize is a real journey no matter which door is used.

Everything is seeded from the site coordinates, so a given map always grows
the same maze.
"""

from __future__ import annotations

import random
from collections import deque
from typing import TYPE_CHECKING, Optional

import numpy as np
from gdpc.block import Block
from gdpc.editor import Editor

from builds._vegetation import clear_vegetation
from terrain.terrain_types import SubZone, TerrainMap
from utils import log, sign_nbt

if TYPE_CHECKING:
    from lore.names import Lore

_AIR = Block("minecraft:air")
_DIRT = Block("dirt")
_PATH = Block("dirt_path")
_LANTERN = Block("lantern")

# biome-key -> hedge leaf block (mirrors the orchard biome table).
_BIOME_HEDGE = {
    "cherry": "flowering_azalea_leaves",
    "jungle": "jungle_leaves",
    "savanna": "acacia_leaves",
    "badlands": "acacia_leaves",
    "taiga": "spruce_leaves",
    "snowy": "spruce_leaves",
    "frozen": "spruce_leaves",
    "mangrove": "mangrove_leaves",
    "birch": "birch_leaves",
    "default": "oak_leaves",
}

_LOOT_TABLES = (
    "minecraft:chests/simple_dungeon",
    "minecraft:chests/village/village_weaponsmith",
    "minecraft:chests/jungle_temple",
)

_CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Sign rotation → (dz, dx) the sign faces (where its reader stands).
_ROT_TO_DIR = {0: (1, 0), 4: (0, -1), 8: (-1, 0), 12: (0, 1)}
# Chest facing → (dz, dx) its front points at (where its opener stands).
_FACING_TO_DIR = {"south": (1, 0), "west": (0, -1), "north": (-1, 0), "east": (0, 1)}


def _biome_hedge(biome: str) -> str:
    for key, leaf in _BIOME_HEDGE.items():
        if key != "default" and key in biome:
            return leaf
    return _BIOME_HEDGE["default"]


class MazeGenerator:
    """Carve an organically shaped hedge labyrinth in the reserved district."""

    HEDGE_HEIGHT = 3
    CLEAR_HEIGHT = 4
    MIN_AREA = 90  # smallest organic pad worth a labyrinth (~10x10)
    MAX_AREA = 1600  # cap: a huge district never becomes wall-to-wall maze
    HEIGHT_BAND = 4  # max surface-height spread the pad may grow across
    BRAID_CHANCE = 0.08  # extra wall knock-throughs → loops in big mazes

    def generate(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        lore: Optional["Lore"] = None,
        *,
        pre_occupied: Optional[set[tuple[int, int]]] = None,
    ) -> bool:
        """Build every reserved maze.  Returns True if at least one was placed."""
        feat = terrain_map.features
        sub_map = terrain_map.sub_zone_map
        road_map = terrain_map.road_map_expanded

        if feat is None or sub_map is None:
            return False
        sites = terrain_map.maze_sites or (
            [terrain_map.maze_site] if terrain_map.maze_site is not None else []
        )
        if not sites:
            return False

        coords = np.argwhere(sub_map == int(SubZone.MAZE))
        if len(coords) == 0:
            return False

        depth, width = feat.shape
        excl: set[tuple[int, int]] = set(pre_occupied) if pre_occupied else set()

        def _on_road(lz: int, lx: int) -> bool:
            return (
                road_map is not None
                and 0 <= lz < road_map.shape[0]
                and 0 <= lx < road_map.shape[1]
                and road_map[lz, lx] != 0
            )

        plot = {
            (int(r[0]), int(r[1]))
            for r in coords
            if (int(r[0]), int(r[1])) not in excl
            and not _on_road(int(r[0]), int(r[1]))
            and float(feat[int(r[0]), int(r[1])]["water_pct"]) == 0.0
        }

        built = 0
        for site in sites:
            blob = self._build_one(editor, terrain_map, feat, plot, site, lore, excl)
            if blob is None:
                continue
            plot -= blob  # later mazes never grow into an earlier one
            excl |= blob
            if pre_occupied is not None:
                pre_occupied |= blob
            built += 1
        return built > 0

    def _build_one(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        feat: np.ndarray,
        plot: set[tuple[int, int]],
        site: tuple[int, int],
        lore: Optional["Lore"],
        excl: set[tuple[int, int]],
    ) -> Optional[set[tuple[int, int]]]:
        """Grow, carve and render one labyrinth around *site*.
        Returns the claimed pad cells, or None if nothing was built."""
        depth, width = feat.shape
        blob = self._grow_blob(plot, feat, site)
        if blob is None or len(blob) < self.MIN_AREA:
            log("[Maze] No workable pad grows in the reserved district, skipping.")
            return None

        cz, cx = site
        rng = random.Random((cz << 16) ^ cx)
        heights = [int(feat[lz, lx]["height"]) for lz, lx in blob]
        floor_y = int(np.median(heights)) - 1

        # Vegetation clear over the whole pad.
        clear_vegetation(
            list(blob), feat, depth, width, editor, terrain_map, protected=excl
        )

        # Biome-appropriate hedge.
        wcx, wcz = terrain_map.local_to_world(cz, cx)
        biome = editor.getBiome((wcx, floor_y, wcz)).lower()
        hedge = Block(
            _biome_hedge(biome), states={"persistent": "true", "distance": "1"}
        )

        carved = self._carve_maze(blob, rng)
        if carved is None:
            log("[Maze] Pad too fragmented to carve a labyrinth, skipping.")
            return None
        paths, cell_blocks = carved

        # Entrances: 1 for a small garden, up to 3 for a grand labyrinth.
        n_entrances = 1 if len(blob) < 400 else (2 if len(blob) < 900 else 3)
        entrances = self._open_entrances(
            blob, paths, cell_blocks, rng, n_entrances, feat, floor_y, excl
        )
        if not entrances:
            # Only possible when the corridor lattice touches no pad edge at
            # all — a sealed maze is useless, so this stays a hard bail.
            log("[Maze] No boundary wall adjacent to any corridor, skipping.")
            return None

        # Heart: the corridor cell maximising the minimum walking distance to
        # EVERY entrance — the prize is a real journey no matter which door
        # the solver picks.  A clearing is opened around it so it reads as a
        # reward room, under two constraints that keep the journey honest:
        # the pad rim stays hedge (the only perimeter gaps are the entrance
        # doorways), and a clearing cell may not knock through to a corridor
        # outside the clearing (nearby corridors are often walking-FAR — a
        # single knock-through would shortcut the maze right at the prize).
        boundary = {
            c
            for c in blob
            if any((c[0] + dz, c[1] + dx) not in blob for dz, dx in _CARDINALS)
        }
        heart = self._farthest_cell(paths, cell_blocks, entrances)
        ring = {
            (heart[0] + dz, heart[1] + dx) for dz in (-1, 0, 1) for dx in (-1, 0, 1)
        }
        for c in sorted(ring):
            if c == heart or c not in blob or c in boundary or c in paths:
                continue
            if any(
                (c[0] + dz, c[1] + dx) in paths and (c[0] + dz, c[1] + dx) not in ring
                for dz, dx in _CARDINALS
            ):
                continue
            paths.add(c)

        town = lore.settlement_name if lore is not None else "the keep"
        self._render(editor, terrain_map, feat, blob, paths, floor_y, hedge, entrances)
        self._place_reward(editor, terrain_map, blob, paths, heart, floor_y, rng, town)

        log(
            f"[Maze] Grew a {len(blob)}-cell organic labyrinth "
            f"({len(entrances)} entrance(s)) around ({cz},{cx})."
        )
        return blob

    # ------------------------------------------------------------------
    # Organic pad growth
    # ------------------------------------------------------------------
    def _grow_blob(
        self,
        plot: set[tuple[int, int]],
        feat: np.ndarray,
        site: tuple[int, int],
    ) -> Optional[set[tuple[int, int]]]:
        """BFS region-growing from the cell nearest the site, accepting plot
        cells while the pad's total height spread stays within HEIGHT_BAND and
        the area under MAX_AREA.  Returns the grown cell set (or None)."""
        if not plot:
            return None
        scz, scx = site
        start = min(plot, key=lambda c: (c[0] - scz) ** 2 + (c[1] - scx) ** 2)

        h0 = int(feat[start[0], start[1]]["height"])
        min_h = max_h = h0
        blob: set[tuple[int, int]] = {start}
        queue: deque = deque([start])
        while queue and len(blob) < self.MAX_AREA:
            lz, lx = queue.popleft()
            for dz, dx in _CARDINALS:
                nb = (lz + dz, lx + dx)
                if nb in blob or nb not in plot:
                    continue
                h = int(feat[nb[0], nb[1]]["height"])
                if max(max_h, h) - min(min_h, h) > self.HEIGHT_BAND:
                    continue
                min_h, max_h = min(min_h, h), max(max_h, h)
                blob.add(nb)
                queue.append(nb)
                if len(blob) >= self.MAX_AREA:
                    break
        return blob

    # ------------------------------------------------------------------
    # Maze carving (iterative recursive-backtracker over the organic pad)
    # ------------------------------------------------------------------
    def _carve_maze(
        self,
        blob: set[tuple[int, int]],
        rng: random.Random,
    ) -> Optional[tuple[set[tuple[int, int]], list[tuple[int, int]]]]:
        """
        Carve corridors into the pad.  Corridor cells sit on an odd lattice
        (2 blocks apart); a lattice cell is usable when its full 3x3 block
        neighbourhood lies inside the pad, so a solid hedge ring always
        remains around every corridor.  The spanning tree is carved over the
        largest connected component, then braided with a few extra
        knock-throughs so larger mazes contain loops.

        Returns (path blocks, carved lattice cell blocks) or None.
        """
        z0 = min(z for z, _ in blob)
        x0 = min(x for _, x in blob)

        def _usable(bz: int, bx: int) -> bool:
            return all(
                (bz + dz, bx + dx) in blob for dz in (-1, 0, 1) for dx in (-1, 0, 1)
            )

        lattice = {
            (lz, lx)
            for lz, lx in blob
            if (lz - z0) % 2 == 1 and (lx - x0) % 2 == 1 and _usable(lz, lx)
        }
        if len(lattice) < 9:
            return None

        def _neighbours(c: tuple[int, int]) -> list[tuple[int, int]]:
            return [
                (c[0] + 2 * dz, c[1] + 2 * dx)
                for dz, dx in _CARDINALS
                if (c[0] + 2 * dz, c[1] + 2 * dx) in lattice
            ]

        # Largest connected component of the lattice graph.
        best_comp: set[tuple[int, int]] = set()
        seen: set[tuple[int, int]] = set()
        for c in lattice:
            if c in seen:
                continue
            comp = {c}
            queue = deque([c])
            seen.add(c)
            while queue:
                cur = queue.popleft()
                for nb in _neighbours(cur):
                    if nb not in seen:
                        seen.add(nb)
                        comp.add(nb)
                        queue.append(nb)
            if len(comp) > len(best_comp):
                best_comp = comp
        if len(best_comp) < 9:
            return None

        # Recursive backtracker.
        paths: set[tuple[int, int]] = set()
        visited: set[tuple[int, int]] = set()
        start = rng.choice(sorted(best_comp))
        visited.add(start)
        paths.add(start)
        stack = [start]
        while stack:
            cur = stack[-1]
            nbrs = [
                nb for nb in _neighbours(cur) if nb in best_comp and nb not in visited
            ]
            if not nbrs:
                stack.pop()
                continue
            nb = rng.choice(nbrs)
            visited.add(nb)
            paths.add(nb)
            paths.add(((cur[0] + nb[0]) // 2, (cur[1] + nb[1]) // 2))  # wall through
            stack.append(nb)

        # Braiding: knock through a few remaining walls between carved cells
        # so the labyrinth gains loops (choices, escapes from dead ends).
        for c in sorted(visited):
            for nb in _neighbours(c):
                if nb not in visited or nb <= c:
                    continue
                wall = ((c[0] + nb[0]) // 2, (c[1] + nb[1]) // 2)
                if wall not in paths and rng.random() < self.BRAID_CHANCE:
                    paths.add(wall)

        return paths, sorted(visited)

    # ------------------------------------------------------------------
    # Entrances
    # ------------------------------------------------------------------
    def _open_entrances(
        self,
        blob: set[tuple[int, int]],
        paths: set[tuple[int, int]],
        cell_blocks: list[tuple[int, int]],
        rng: random.Random,
        n_entrances: int,
        feat: np.ndarray,
        floor_y: int,
        excl: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """
        Open up to ``n_entrances`` wall blocks on the pad boundary: a carved
        corridor cell whose neighbouring wall block sits on the pad edge (the
        block beyond it is outside the pad) becomes a doorway.  A doorway must
        actually be usable: the ground just outside it is dry, unoccupied and
        within a step of the corridor floor (falling back to a one-jump ledge,
        and as a last resort to any edge wall at all — walkability is never
        allowed to abandon the maze).  Openings are greedily spread as
        far apart as possible.  Returns the opened wall blocks (also added to
        ``paths``).
        """
        depth, width = feat.shape

        def _outside_ok(oz: int, ox: int, tolerance: int) -> bool:
            if not (0 <= oz < depth and 0 <= ox < width):
                return False
            if (oz, ox) in excl:
                return False
            cell = feat[oz, ox]
            if float(cell["water_pct"]) > 0.0:
                return False
            # Player stands on floor_y + 1 inside; outside surface must be
            # reachable without climbing a cliff or dropping off one.
            return abs(int(cell["height"]) - (floor_y + 1)) <= tolerance

        candidates: list[tuple[int, int]] = []
        relaxed: list[tuple[int, int]] = []
        geometric: list[tuple[int, int]] = []
        for bz, bx in cell_blocks:
            for dz, dx in _CARDINALS:
                wall = (bz + dz, bx + dx)
                outside = (bz + 2 * dz, bx + 2 * dx)
                if wall not in blob or wall in paths or outside in blob:
                    continue
                geometric.append(wall)
                if _outside_ok(outside[0], outside[1], 1):
                    candidates.append(wall)
                elif _outside_ok(outside[0], outside[1], 2):
                    relaxed.append(wall)
        if not candidates:
            candidates = relaxed
        if not candidates:
            # No comfortably walkable ground anywhere outside the rim — open
            # a doorway anyway rather than abandoning the whole maze.
            candidates = geometric
        if not candidates:
            return []

        rng.shuffle(candidates)
        chosen: list[tuple[int, int]] = [candidates[0]]
        for _ in range(n_entrances - 1):
            best = max(
                candidates,
                key=lambda c: min(
                    (c[0] - e[0]) ** 2 + (c[1] - e[1]) ** 2 for e in chosen
                ),
            )
            if best not in chosen and all(
                (best[0] - e[0]) ** 2 + (best[1] - e[1]) ** 2 >= 10**2 for e in chosen
            ):
                chosen.append(best)
        paths.update(chosen)
        return chosen

    # ------------------------------------------------------------------
    # Heart search
    # ------------------------------------------------------------------
    @staticmethod
    def _farthest_cell(
        paths: set[tuple[int, int]],
        cell_blocks: list[tuple[int, int]],
        entrances: list[tuple[int, int]],
    ) -> tuple[int, int]:
        """Corridor lattice cell maximising the minimum walking distance to
        every entrance (multi-source BFS over path blocks), so no doorway —
        main or secondary — ever opens a few steps from the reward."""
        dist: dict[tuple[int, int], int] = {e: 0 for e in entrances}
        queue: deque = deque(entrances)
        while queue:
            lz, lx = queue.popleft()
            for dz, dx in _CARDINALS:
                nb = (lz + dz, lx + dx)
                if nb in paths and nb not in dist:
                    dist[nb] = dist[(lz, lx)] + 1
                    queue.append(nb)
        reachable = [c for c in cell_blocks if c in dist]
        if not reachable:
            return cell_blocks[len(cell_blocks) // 2]
        return max(reachable, key=lambda c: dist[c])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        feat: np.ndarray,
        blob: set[tuple[int, int]],
        paths: set[tuple[int, int]],
        floor_y: int,
        hedge: Block,
        entrances: list[tuple[int, int]],
    ) -> None:
        for lz, lx in blob:
            wx, wz = terrain_map.local_to_world(lz, lx)

            # Support fill from natural ground up to the pad floor.
            nat_top = int(feat[lz, lx]["height"]) - 1
            for y in range(min(nat_top, floor_y), floor_y):
                editor.placeBlock((wx, y, wz), _DIRT)
            # Clear headroom above the floor.
            for dy in range(1, self.CLEAR_HEIGHT + 1):
                editor.placeBlock((wx, floor_y + dy, wz), _AIR)

            if (lz, lx) in paths:
                editor.placeBlock((wx, floor_y, wz), _PATH)
            else:
                editor.placeBlock((wx, floor_y, wz), _DIRT)
                for h in range(1, self.HEDGE_HEIGHT + 1):
                    editor.placeBlock((wx, floor_y + h, wz), hedge)

        # Lanterns on the hedge tops flanking every entrance opening.
        for ez, ex in entrances:
            for dz, dx in _CARDINALS:
                nb = (ez + dz, ex + dx)
                if nb in blob and nb not in paths:
                    wx, wz = terrain_map.local_to_world(nb[0], nb[1])
                    editor.placeBlock(
                        (wx, floor_y + self.HEDGE_HEIGHT + 1, wz), _LANTERN
                    )

    def _place_reward(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        blob: set[tuple[int, int]],
        paths: set[tuple[int, int]],
        heart: tuple[int, int],
        floor_y: int,
        rng: random.Random,
        town: str,
    ) -> None:
        """Lantern pillar + loot chest + payoff sign at the heart of the maze."""
        hz, hx = heart
        wx, wz = terrain_map.local_to_world(hz, hx)
        # Lantern pillar behind the chest.
        editor.placeBlock((wx, floor_y + 1, wz), Block("chiseled_stone_bricks"))
        editor.placeBlock((wx, floor_y + 2, wz), _LANTERN)

        # Chest on an adjacent corridor cell, facing the solver.
        chest_cell = next(
            (
                (hz + dz, hx + dx)
                for dz, dx in _CARDINALS
                if (hz + dz, hx + dx) in paths
            ),
            (hz, hx - 1),
        )
        loot = rng.choice(_LOOT_TABLES)
        cwx, cwz = terrain_map.local_to_world(chest_cell[0], chest_cell[1])
        # Open toward the corridor the solver arrives from, never into the
        # lantern pillar or a hedge.
        czl, cxl = chest_cell
        facing = "north"
        for pass_excl in ({(hz, hx)}, set()):
            found = next(
                (
                    f
                    for f, (dz, dx) in _FACING_TO_DIR.items()
                    if (czl + dz, cxl + dx) in paths
                    and (czl + dz, cxl + dx) not in pass_excl
                ),
                None,
            )
            if found is not None:
                facing = found
                break
        chest = Block(
            "chest",
            states={"facing": facing},
            data=f'{{LootTable:"{loot}"}}',
        )
        editor.placeBlock((cwx, floor_y + 1, cwz), chest)

        # Payoff sign in the clearing naming the labyrinth after the town.
        sign_cell = next(
            (
                (hz + dz, hx + dx)
                for dz, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))
                if (hz + dz, hx + dx) in paths and (hz + dz, hx + dx) != chest_cell
            ),
            None,
        )
        if sign_cell is not None:
            # Face an open corridor cell so the text never stares into a
            # hedge; prefer a cell the solver can actually stand on (not the
            # lantern pillar, not the chest).
            sz, sx = sign_cell
            rotation = 8
            for pass_excl in ({(hz, hx), chest_cell}, set()):
                found = next(
                    (
                        rot
                        for rot, (dz, dx) in _ROT_TO_DIR.items()
                        if (sz + dz, sx + dx) in paths
                        and (sz + dz, sx + dx) not in pass_excl
                    ),
                    None,
                )
                if found is not None:
                    rotation = found
                    break
            sign = Block(
                "oak_sign",
                states={"rotation": str(rotation)},
                data=sign_nbt(
                    ["You found", "the heart of", "the Labyrinth", f"of {town}"]
                ),
            )
            swx, swz = terrain_map.local_to_world(sign_cell[0], sign_cell[1])
            editor.placeBlock((swx, floor_y + 1, swz), sign)
