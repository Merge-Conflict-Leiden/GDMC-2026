"""
road_placer.py
--------------
Road block placement for the GDMC medieval city pipeline.

Reads terrain_map.road_map (populated by Phase 6) and places road blocks
in Minecraft. Each cell receives a uniformly random block from:
  rooted_dirt, coarse_dirt, packed_mud

Road class determines width after dilation:
  Primary   (3) — 5 cells wide (radius 2, full square)
  Secondary (2) — 3 cells wide (radius 1, full square)
  Tertiary  (1) — 3 cells wide (radius 1, full square)
"""

from __future__ import annotations

import logging
import random

import numpy as np
from gdpc.block import Block
from gdpc.editor import Editor

from blocks.block_processor import BlockProcessor
from consts import BUILD_DIR, BUILD_OUTPUT_DIR
from structures.structure_cache import StructureCache
from structures.structure_placer import StructurePlacer
from utils import sign_nbt

from . import _ROAD_BLOCKS
from .terrain_types import TerrainMap, ZoneType, _expand_road_map

logger = logging.getLogger(__name__)

_AIR = Block("minecraft:air")
_AIR_IDS = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}
# Bridge palettes: the deck and railing mix oak/spruce per cell (a plain list is
# placed with gdpc's random-per-block choice), giving a weathered, hand-built
# look rather than a single uniform material.
_BRIDGE_DECK = [
    Block("minecraft:spruce_planks"),
]
_BRIDGE_RAIL = [
    Block("minecraft:spruce_fence"),
]
# Support pillars pick ONE log per pillar (uniform up the whole column) but vary
# from pillar to pillar across the span.
_BRIDGE_PILLAR_PALETTE = [
    Block("minecraft:stripped_spruce_log", states={"axis": "y"}),
    Block("minecraft:oak_log", states={"axis": "y"}),
]
_BRIDGE_BASE = Block("minecraft:cobblestone")
_BRIDGE_PILLAR_SPACING = 3  # a support pillar every N edge cells, not a solid wall

# Signpost materials (mirrors the weathered stone-brick palette used elsewhere):
# a wall block as the post, a wooden sign on top.  stone_brick_wall is weighted
# so most posts are crisp with the occasional mossy/cracked/cobble accent.
_SIGN_WALL_BLOCKS = (
    "stone_brick_wall",
    "stone_brick_wall",
    "mossy_stone_brick_wall",
    "cobblestone_wall",
    "mossy_cobblestone_wall",
    "andesite_wall",
)
_SIGN_WOODS = ("oak", "spruce")


class RoadPlacer:
    """Place road blocks in Minecraft based on a completed TerrainMap."""

    def __init__(self, editor: Editor, terrain_map: TerrainMap) -> None:
        self._editor = editor
        self._tm = terrain_map

    def _is_water(self, lz: int, lx: int) -> bool:
        """
        Exact per-cell water test.

        features["water_pct"] is a 5x5 box-blur of the water mask — cells at
        the river edge read well below 1.0, so thresholding it misclassifies
        genuinely wet cells as land (and vice versa), leaving holes in bridge
        decks.  Use the exact mask when available.
        """
        wm = self._tm.water_mask
        if wm is not None:
            return bool(wm[lz, lx])
        feat = self._tm.features
        return float(feat[lz, lx]["water_pct"]) > 0.5

    def place_roads(self, protected: set[tuple[int, int]] | None = None) -> int:
        """
        Place road blocks for every road cell. Returns the total block count.

        *protected* cells (e.g. the castle bounding box, placed before roads)
        are skipped entirely — no road surface and, crucially, no air
        clearing, which would otherwise carve into structures standing on
        road cells.
        """
        road_map = self._tm.road_map
        if road_map is None:
            logger.warning(
                "RoadPlacer: terrain_map.road_map is None — run with build_roads=True first."
            )
            return 0

        features = self._tm.features
        if features is None:
            logger.warning("RoadPlacer: terrain_map.features is None.")
            return 0

        protected = protected or set()
        expanded = _expand_road_map(road_map)
        depth, width = expanded.shape
        count = 0
        bridge_count = 0
        _traced = 0  # number of placement traces emitted

        # NB: whole-tree vegetation clearing along the road corridor happens in
        # TerrainModifier.prepare_roads (before grading, while feat["height"]
        # still matches natural terrain).  The per-cell air column below is only
        # a headroom guarantee above the placed road surface.

        # Bridge decks: contiguous water runs along each road path become
        # simple spruce-plank bridges spanning at bank height.
        decks = self._compute_bridge_decks(features)

        for lz in range(depth):
            for lx in range(width):
                if expanded[lz, lx] == 0:
                    continue
                if (lz, lx) in protected:
                    continue

                wx, wz = self._tm.local_to_world(lz, lx)

                if self._is_water(lz, lx):
                    # feat["height"] of water cells is the river BOTTOM —
                    # place a plank deck at bank level instead of dirt.
                    deck_h = self._deck_for(lz, lx, decks)
                    if deck_h is None:
                        # Shoulder of a shore-hugging land road; leave water.
                        continue

                    # A bridge edge cell has at least one orthogonal neighbour
                    # outside the expanded road — these get a fence railing and
                    # a structural pillar dropping to the river bed.
                    is_edge = any(
                        not (0 <= lz + dz < depth and 0 <= lx + dx < width)
                        or expanded[lz + dz, lx + dx] == 0
                        for dz, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))
                    )

                    # Deck plank + headroom clearance
                    self._editor.placeBlock((wx, deck_h - 1, wz), _BRIDGE_DECK)
                    self._editor.placeBlock((wx, deck_h, wz), _AIR)
                    self._editor.placeBlock((wx, deck_h + 1, wz), _AIR)

                    if is_edge:
                        # Fence railing on the outer rail of the bridge (continuous)
                        self._editor.placeBlock((wx, deck_h, wz), _BRIDGE_RAIL)

                        # Support pillars every few cells along the span rather
                        # than a solid curtain: a cobblestone footing on the river
                        # bed, then a log column up to just below the deck.  Each
                        # pillar is a single log type (uniform up its height) but
                        # picked at random so pillars vary across the span.
                        if (lz + lx) % _BRIDGE_PILLAR_SPACING == 0:
                            pillar_block = random.choice(_BRIDGE_PILLAR_PALETTE)
                            river_y = int(features[lz, lx]["height"])
                            if river_y > 0:
                                self._editor.placeBlock(
                                    (wx, river_y - 1, wz), _BRIDGE_BASE
                                )
                            for py in range(river_y, deck_h - 1):
                                self._editor.placeBlock((wx, py, wz), pillar_block)

                    bridge_count += 1
                    continue

                h_raw = int(features[lz, lx]["height"])
                wy = h_raw - 1
                # Clear vegetation (trunks, leaves, tall plants) above the surface
                for dy in range(1, 20):
                    self._editor.placeBlock((wx, wy + dy, wz), _AIR)
                block = random.choice(_ROAD_BLOCKS)
                self._editor.placeBlock((wx, wy, wz), block)
                count += 1

                if _traced < 5:
                    logger.debug(
                        "ROAD PLACE #%d  local(%d,%d)  world(wx=%d wy=%d wz=%d)"
                        "  height_raw=%d  block=%s",
                        _traced + 1,
                        lz,
                        lx,
                        wx,
                        wy,
                        wz,
                        h_raw,
                        block.id,
                    )
                    _traced += 1

        logger.info(
            "RoadPlacer: placed %d road blocks, %d bridge deck blocks.",
            count,
            bridge_count,
        )
        self._place_step_edges(expanded, features, depth, width, protected)
        return count

    def _compute_bridge_decks(self, features: np.ndarray) -> dict[tuple[int, int], int]:
        """
        Map every water cell along a road path to its bridge deck height.

        Each contiguous run of water cells in a path spans at the height of
        the higher of its two bank cells, so the deck meets the road surface
        flush on at least one side.  Runs without any land anchor (degenerate
        all-water paths) are skipped.
        """
        network = self._tm.road_network
        if network is None:
            return {}

        depth, width = features.shape
        decks: dict[tuple[int, int], int] = {}

        def _wet(cell: tuple[int, int]) -> bool:
            cz, cx = cell
            if not (0 <= cz < depth and 0 <= cx < width):
                return False
            return self._is_water(cz, cx)

        for road in network.all_roads:
            path = [(int(z), int(x)) for z, x in road.path]
            n = len(path)
            i = 0
            while i < n:
                if not _wet(path[i]):
                    i += 1
                    continue
                j = i
                while j < n and _wet(path[j]):
                    j += 1
                bank_heights = []
                if i > 0:
                    pz, px = path[i - 1]
                    bank_heights.append(int(features[pz, px]["height"]))
                if j < n:
                    pz, px = path[j]
                    bank_heights.append(int(features[pz, px]["height"]))
                if bank_heights:
                    deck_h = max(bank_heights)
                    for cell in path[i:j]:
                        decks[cell] = max(decks.get(cell, deck_h), deck_h)
                i = j

        return decks

    @staticmethod
    def _deck_for(lz: int, lx: int, decks: dict[tuple[int, int], int]) -> int | None:
        """Deck height for an expanded road cell: nearest path-cell deck
        within the dilation radius (primary roads dilate by 2; +1 slack for
        diagonal path entries whose shoulders sit one cell further out)."""
        if not decks:
            return None
        best: int | None = None
        best_d = 99
        for dz in range(-3, 4):
            for dx in range(-3, 4):
                h = decks.get((lz + dz, lx + dx))
                if h is not None:
                    d = max(abs(dz), abs(dx))
                    if d < best_d:
                        best, best_d = h, d
        return best

    def _place_step_edges(
        self,
        expanded: np.ndarray,
        features: np.ndarray,
        depth: int,
        width: int,
        protected: set[tuple[int, int]],
    ) -> None:
        """
        Scatter spruce slabs and trapdoors at road elevation transitions.

        For each road cell that looks up toward a higher road cell:
          delta == 1  (gradual step) — higher chance, slabs slightly favoured
          delta >= 2  (steep rise)   — low chance, trapdoors heavily favoured
        """
        _DIRS = [
            (1, 0, "south"),
            (-1, 0, "north"),
            (0, 1, "east"),
            (0, -1, "west"),
        ]
        _TRAPDOOR = Block(
            "minecraft:spruce_trapdoor",
            {"half": "bottom", "open": "false", "facing": "north"},
        )
        _SLAB = Block("minecraft:spruce_slab", {"type": "bottom"})

        for lz in range(depth):
            for lx in range(width):
                if expanded[lz, lx] == 0:
                    continue
                if (lz, lx) in protected:
                    continue
                # Bridge cells: feat height is the river bottom, not the deck.
                if self._is_water(lz, lx):
                    continue

                h_here = int(features[lz, lx]["height"])
                wy_step = h_here  # one block above this cell's road surface

                for ddz, ddx, facing in _DIRS:
                    nlz, nlx = lz + ddz, lx + ddx
                    if not (0 <= nlz < depth and 0 <= nlx < width):
                        continue
                    if expanded[nlz, nlx] == 0:
                        continue
                    if self._is_water(nlz, nlx):
                        continue

                    delta = int(features[nlz, nlx]["height"]) - h_here
                    if delta <= 0:
                        continue  # only decorate on the lower side of a rise

                    if delta == 1:
                        p_trap, p_slab = 0.28, 0.38
                    else:  # steep: delta >= 2
                        p_trap, p_slab = 0.14, 0.04

                    r = random.random()
                    if r >= p_trap + p_slab:
                        continue

                    wx, wz = self._tm.local_to_world(lz, lx)
                    if r < p_trap:
                        block = Block(
                            "minecraft:spruce_trapdoor",
                            {"half": "bottom", "open": "false", "facing": facing},
                        )
                    else:
                        block = _SLAB
                    self._editor.placeBlock((wx, wy_step, wz), block)

    def place_lanterns(
        self,
        occupied: set[tuple[int, int]],
        *,
        spacing: int = 8,
    ) -> int:
        """
        Place lantern posts at intervals alongside road cells.

        A lantern is placed in an empty cell adjacent (orthogonally) to a road
        cell.  The post extends one cell further in the same direction so the
        arm of the lantern hangs away from the road.  Cells in *occupied*
        (buildings, wall, other roads) are skipped.

        Returns the number of lanterns placed.
        """
        road_map = self._tm.road_map
        features = self._tm.features
        if road_map is None or features is None:
            return 0

        _LANTERN_REL = "path/lantern.csv"
        cache = StructureCache(BUILD_DIR, BUILD_OUTPUT_DIR)
        placer = StructurePlacer(
            structure_cache=cache,
            block_processor=BlockProcessor(),
            block_palette=None,
            furniture_replacer=None,
        )
        try:
            cache.load_structure(_LANTERN_REL)  # pre-cache; raises if missing
        except Exception as exc:
            logger.warning("place_lanterns: could not load %s: %s", _LANTERN_REL, exc)
            return 0

        expanded = _expand_road_map(road_map)
        depth, width = road_map.shape
        placed = 0

        # Cardinal directions: (ddz, ddx, csv_direction)
        # csv_direction matches the lantern CSV orientation (arm extends in +z
        # at direction=0, rotating 90° per step).
        _CARDINALS = [
            (0, 1, 1),  # arm points +x
            (0, -1, 3),  # arm points -x
            (1, 0, 0),  # arm points +z
            (-1, 0, 2),  # arm points -z
        ]

        def _try_place(rz: int, rx: int) -> bool:
            """Attempt one lantern next to road cell (rz, rx)."""
            for ddz, ddx, direction in _CARDINALS:
                # Walk outward until we leave the expanded road area.
                # The first free cell is the post; one cell further is the arm.
                post_lz = post_lx = None
                for dist in range(1, 7):
                    plz = rz + ddz * dist
                    plx = rx + ddx * dist
                    if not (0 <= plz < depth and 0 <= plx < width):
                        break
                    if expanded[plz, plx] > 0:
                        continue  # still inside road surface
                    post_lz, post_lx = plz, plx
                    break

                if post_lz is None:
                    continue

                alz = post_lz + ddz  # arm: one cell further beyond post
                alx = post_lx + ddx

                if not (0 <= alz < depth and 0 <= alx < width):
                    continue
                if expanded[alz, alx] > 0:  # arm on another road
                    continue
                if (post_lz, post_lx) in occupied:
                    continue
                if (alz, alx) in occupied:
                    continue
                # No lantern posts in water (e.g. beside a bridge) — the
                # surface height there is the river bottom.
                if self._is_water(post_lz, post_lx):
                    continue

                wx, wz = self._tm.local_to_world(post_lz, post_lx)
                surface_y = int(features[post_lz, post_lx]["height"]) - 1

                # Clear pre-existing vegetation in BOTH lantern columns before
                # placing: the schematic is placed skip_air, so a worldgen
                # flower or tall grass would otherwise survive inside the
                # footprint — blocking the post's lower block or dangling
                # under the arm.
                awx, awz = self._tm.local_to_world(alz, alx)
                arm_surface_y = int(features[alz, alx]["height"]) - 1
                for dy in range(1, 5):
                    self._editor.placeBlock((wx, surface_y + dy, wz), Block("air"))
                    self._editor.placeBlock(
                        (awx, arm_surface_y + dy, awz), Block("air")
                    )

                with self._editor.pushTransform((wx, surface_y + 1, wz)):
                    placer.place_structure(
                        self._editor,
                        _LANTERN_REL,
                        direction=direction,
                        skip_air=True,
                    )

                # Register lantern cells so subsequent phases (misc) avoid them.
                occupied.add((post_lz, post_lx))
                occupied.add((alz, alx))
                return True  # one lantern per road cell
            return False

        # Walk each road path in order so spacing is measured ALONG the road.
        # Iterating np.where scan order instead clusters lanterns wherever
        # many road cells share the same array rows.
        lantern_road_cells: list[tuple[int, int]] = []
        min_sep_sq = spacing * spacing  # separation from other roads' lanterns

        road_network = self._tm.road_network
        roads = getattr(road_network, "all_roads", None) if road_network else None

        if roads:
            for road in roads:
                since = spacing  # allow a lantern near each road's start
                for rz, rx in road.path:
                    since += 1
                    if since < spacing:
                        continue
                    rz, rx = int(rz), int(rx)
                    # Keep distance from lanterns of other (crossing) roads.
                    if any(
                        (rz - pz) ** 2 + (rx - px) ** 2 < min_sep_sq
                        for pz, px in lantern_road_cells
                    ):
                        continue
                    if _try_place(rz, rx):
                        lantern_road_cells.append((rz, rx))
                        placed += 1
                        since = 0
        else:
            # Fallback when no RoadNetwork is available: array scan order.
            for step, (rz_raw, rx_raw) in enumerate(zip(*np.where(road_map > 0))):
                if step % spacing != 0:
                    continue
                if _try_place(int(rz_raw), int(rx_raw)):
                    placed += 1

        logger.info("place_lanterns: placed %d lantern posts.", placed)
        return placed

    def place_signposts(self, occupied: set[tuple[int, int]], lore=None) -> int:
        """
        Plant wayfinding signposts on the road at every city gate AND at major
        road intersections.

        Each is a single weathered wall post topped by a wooden sign.  Gate
        signs welcome travellers to the named town and its district; junction
        signs point to the market.  Placed on a road-edge cell so the
        carriageway stays passable.  The post + sign cell is verified clear
        (both blocks air) before placing; if something already occupies it,
        the next-nearest edge cell is tried instead.  Returns the count.
        """
        tm = self._tm
        road_map = tm.road_map
        features = tm.features
        if road_map is None or features is None:
            return 0

        expanded = _expand_road_map(road_map)
        depth, width = road_map.shape
        wall_cells = getattr(tm, "wall_cells", None) or set()
        town = lore.settlement_name if lore is not None else "Wayfarer's rest"
        town_center = tm.town_center

        _DIRS = {
            (1, 0): "South",
            (-1, 0): "North",
            (0, 1): "East",
            (0, -1): "West",
        }

        def _bearing_word(frm: tuple[int, int], to: tuple[int, int] | None) -> str:
            if to is None:
                return ""
            dz, dx = to[0] - frm[0], to[1] - frm[1]
            if dz == 0 and dx == 0:
                return ""
            if abs(dz) >= abs(dx):
                return _DIRS[(1 if dz > 0 else -1, 0)]
            return _DIRS[(0, 1 if dx > 0 else -1)]

        def _rotation_toward(frm: tuple[int, int], to: tuple[int, int] | None) -> int:
            if to is None:
                return 0
            dz, dx = to[0] - frm[0], to[1] - frm[1]
            if abs(dz) >= abs(dx):
                return 0 if dz > 0 else 8  # face south / north
            return 12 if dx > 0 else 4  # face east / west

        def _is_road_edge(z: int, x: int) -> bool:
            for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nz, nx = z + dz, x + dx
                if not (0 <= nz < depth and 0 <= nx < width) or expanded[nz, nx] == 0:
                    return True
            return False

        def _edge_near(
            tz: int, tx: int, taken: set[tuple[int, int]]
        ) -> tuple[int, int] | None:
            """Nearest road-edge (shoulder) cell to (tz, tx); any road cell as
            fallback.  Skips wall cells, water, and already-used spots."""
            first_any: tuple[int, int] | None = None
            for r in range(0, 13):
                ring: list[tuple[int, int]] = []
                for dz in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if max(abs(dz), abs(dx)) != r:
                            continue
                        z, x = tz + dz, tx + dx
                        if not (0 <= z < depth and 0 <= x < width):
                            continue
                        if expanded[z, x] == 0:
                            continue
                        if (z, x) in wall_cells or (z, x) in taken:
                            continue
                        if self._is_water(z, x):
                            continue
                        ring.append((z, x))
                if not ring:
                    continue
                ring.sort(key=lambda c: (c[0] - tz) ** 2 + (c[1] - tx) ** 2)
                if first_any is None:
                    first_any = ring[0]
                edges = [c for c in ring if _is_road_edge(*c)]
                if edges:
                    return edges[0]
            return first_any

        # ── Targets: gates first, then well-spaced road intersections ─────
        targets: list[tuple[tuple[int, int], str, int]] = []
        for gi, g in enumerate(tm.gate_sites or []):
            targets.append(((int(g[0]), int(g[1])), "gate", gi))

        rset = {(int(z), int(x)) for z, x in np.argwhere(road_map > 0)}
        junctions = [
            (z, x)
            for (z, x) in rset
            if sum(
                1
                for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if (z + dz, x + dx) in rset
            )
            >= 3
        ]
        _JCT_SEP = 18  # thin out clustered junction cells into one per crossroads
        picked: list[tuple[int, int]] = []
        for j in sorted(junctions):
            if all(
                (j[0] - p[0]) ** 2 + (j[1] - p[1]) ** 2 >= _JCT_SEP**2 for p in picked
            ):
                picked.append(j)
                targets.append((j, "junction", 0))

        if not targets:
            return 0

        def _spot_is_clear(z: int, x: int) -> bool:
            """Both the post and sign cell above it must be air in the world."""
            cwx, cwz = tm.local_to_world(z, x)
            cy = int(features[z, x]["height"])
            return (
                self._editor.getBlock((cwx, cy, cwz)).id in _AIR_IDS
                and self._editor.getBlock((cwx, cy + 1, cwz)).id in _AIR_IDS
            )

        # A bigger city justifies more wayfinding: scale the sign budget with
        # the number of urban districts (14 for a hamlet up to 24).
        n_urban = sum(
            1 for d in tm.districts if d.zone_type == ZoneType.URBAN and d.size > 0
        )
        _MAX_SIGNS = min(24, 14 + n_urban // 6)
        _MAX_SPOT_ATTEMPTS = 6
        placed = 0
        used: set[tuple[int, int]] = set()
        placed_pts: list[tuple[int, int]] = []

        for (tz, tx), kind, gi in targets:
            if placed >= _MAX_SIGNS:
                break

            # Try the nearest edge cell first; if it clusters with an already
            # placed sign, or the two blocks it needs aren't actually air (a
            # tree, an overhang, ...), fall back to the next-nearest one.
            spot = None
            rejected: set[tuple[int, int]] = set()
            for _ in range(_MAX_SPOT_ATTEMPTS):
                candidate = _edge_near(tz, tx, used | rejected)
                if candidate is None:
                    break
                if any(
                    (candidate[0] - p[0]) ** 2 + (candidate[1] - p[1]) ** 2 < 8**2
                    for p in placed_pts
                ) or not _spot_is_clear(*candidate):
                    rejected.add(candidate)
                    continue
                spot = candidate
                break
            if spot is None:
                continue

            plz, plx = spot
            wx, wz = tm.local_to_world(plz, plx)
            base_y = int(features[plz, plx]["height"])  # one above the road surface

            if kind == "gate":
                district = lore.district_name(gi) if lore is not None else "the town"
                lines = ["Welcome to", town, district]
            else:
                # Junction wayfinding: name the town, then point to the market
                # and the nearest gate with explicit cardinal bearings.
                lines = [town]
                market_b = _bearing_word(spot, town_center)
                lines.append(f"Market: {market_b}" if market_b else "Market")
                if tm.gate_sites:
                    g = min(
                        tm.gate_sites,
                        key=lambda s: (int(s[0]) - spot[0]) ** 2
                        + (int(s[1]) - spot[1]) ** 2,
                    )
                    gate_b = _bearing_word(spot, (int(g[0]), int(g[1])))
                    if gate_b:
                        lines.append(f"Gate: {gate_b}")

            # Single weathered wall post with the sign directly on top,
            # facing the town centre.
            self._editor.placeBlock(
                (wx, base_y, wz), Block(random.choice(_SIGN_WALL_BLOCKS))
            )
            wood = random.choice(_SIGN_WOODS)
            self._editor.placeBlock(
                (wx, base_y + 1, wz),
                Block(
                    f"{wood}_sign",
                    states={"rotation": str(_rotation_toward(spot, town_center))},
                    data=sign_nbt(lines),
                ),
            )
            used.add(spot)
            occupied.add(spot)
            placed_pts.append(spot)
            placed += 1

        logger.info(
            "place_signposts: placed %d signposts (gates + intersections).", placed
        )
        return placed
