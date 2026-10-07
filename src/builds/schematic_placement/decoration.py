"""
Decorative fill passes that run after the buildings are placed.

:class:`DecorationMixin` provides the scatter/garden/vignette methods of
:class:`~builds.schematic_placement.placer.SchematicBuildingPlacer`; it
relies on the host class for the editor, terrain map, and structure placer.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from gdpc.block import Block

from builds.garden import GardenGenerator
from furniture.loot import container_loot_data
from terrain.terrain_types import SubZone, ZoneType

from .model import Building
from .scoring import world_footprint

if TYPE_CHECKING:
    from gdpc.editor import Editor

    from structures.structure_placer import StructurePlacer
    from terrain.terrain_types import TerrainMap

logger = logging.getLogger(__name__)

_AIR = Block("minecraft:air")

# Per-schematic Y nudge applied only in place_misc_fillers.
# Negative = embed further into ground (e.g. well shaft).
_MISC_Y_OFFSETS: dict[str, int] = {
    # place_misc_fillers uses `height + embed` (not `height + embed - 1` like
    # _place_one), so the well needs an extra −1 to sit at the surface block.
    "misc/well.csv": -9,
}

_URBAN_GARDEN_PLANTS = [
    Block("minecraft:short_grass"),
    Block("minecraft:short_grass"),
    Block("minecraft:short_grass"),
    Block("minecraft:dandelion"),
    Block("minecraft:poppy"),
    Block("minecraft:cornflower"),
    Block("minecraft:oxeye_daisy"),
    Block("minecraft:allium"),
    Block("minecraft:azure_bluet"),
    Block("minecraft:white_tulip"),
    Block("minecraft:orange_tulip"),
    Block("minecraft:fern"),
    Block("minecraft:firefly_bush"),
    Block("minecraft:bush"),
    Block("minecraft:bush"),
]

_LEAF_LITTER_FACINGS = ("north", "south", "east", "west")
_LEAF_LITTER_CHANCE = 0.15

# Natural scatter: single-block plants placed across all zones.
_SCATTER_SINGLE: list[Block] = [
    Block("minecraft:short_grass"),
    Block("minecraft:short_grass"),
    Block("minecraft:short_grass"),
    Block("minecraft:short_grass"),
    Block("minecraft:fern"),
    Block("minecraft:fern"),
    Block("minecraft:bush"),
    Block("minecraft:bush"),
]
# Two-block tall plants (lower/upper halves placed separately).
_SCATTER_TALL: list[tuple[str, str]] = [
    ("minecraft:tall_grass", "minecraft:tall_grass"),
    ("minecraft:large_fern", "minecraft:large_fern"),
]


def _pick_urban_plant() -> Block:
    if random.random() < _LEAF_LITTER_CHANCE:
        return Block(
            "minecraft:leaf_litter",
            states={
                "facing": random.choice(_LEAF_LITTER_FACINGS),
                "segment_amount": str(random.randint(1, 4)),
            },
        )
    return random.choice(_URBAN_GARDEN_PLANTS)


class DecorationMixin:
    """Scatter/garden/vignette passes; mixed into SchematicBuildingPlacer."""

    # Provided by the host class (see SchematicBuildingPlacer.__init__).
    _editor: "Editor"
    _tm: "TerrainMap"
    _placer: "StructurePlacer"
    _garden_cells: set[tuple[int, int]]

    def _is_city_fringe(self, district) -> bool:
        """True for undesignated rural districts that touch the city (urban or
        wall) — the band where light decoration keeps the outskirts alive
        without ever reaching into deep wilderness."""
        if district.zone_type != ZoneType.RURAL or district.sub_zone != SubZone.NONE:
            return False
        districts = self._tm.districts
        return any(
            districts[nb].zone_type in (ZoneType.URBAN, ZoneType.WALL_PERIMETER)
            for nb in district.neighbours
        )

    def place_misc_fillers(
        self,
        buildings: list[Building],
        occupied: set[tuple[int, int]],
        *,
        urban_chance: float = 0.05,
        rural_chance: float = 0.02,
    ) -> int:
        """
        Scatter small decorative schematics (piles, carts) into leftover cells.

        Urban sub-zones (RESIDENTIAL, CIVIC, TOWN_CENTER) receive *urban_chance*
        per free cell.  Rural sub-zones (FARMLAND, ORCHARD, ANIMAL_PEN,
        FOREST_BUFFER) receive *rural_chance* per free cell.

        *occupied* is updated in-place with the footprints of placed schematics
        so subsequent calls (e.g. lantern placement) respect them.
        """
        _URBAN_ZONES = (SubZone.RESIDENTIAL, SubZone.CIVIC, SubZone.TOWN_CENTER)
        _RURAL_ZONES = (
            SubZone.FARMLAND,
            SubZone.ORCHARD,
            SubZone.ANIMAL_PEN,
            SubZone.FOREST_BUFFER,
        )

        feat = self._tm.features
        road_map = self._tm.road_map_expanded
        if feat is None or not buildings:
            return 0

        depth, width = feat.shape
        placed = 0

        for district in self._tm.districts:
            if district.sub_zone in _URBAN_ZONES:
                chance = urban_chance
            elif district.sub_zone in _RURAL_ZONES:
                chance = rural_chance
            elif self._is_city_fringe(district):
                # Undesignated rural land touching the city: a light scatter
                # of carts and piles so life doesn't stop dead at the last
                # farm.  Deep wilderness stays untouched.
                chance = rural_chance * 0.5
            else:
                continue
            if district.size == 0:
                continue

            for cell in district.cells:
                lz, lx = int(cell[0]), int(cell[1])
                if (lz, lx) in occupied:
                    continue
                if road_map is not None:
                    if (
                        0 <= lz < road_map.shape[0]
                        and 0 <= lx < road_map.shape[1]
                        and road_map[lz, lx] > 0
                    ):
                        continue
                # Skip steep terrain or water pools — piles look wrong there.
                if float(feat[lz, lx]["slope"]) > 2.5:
                    continue
                if float(feat[lz, lx]["water_pct"]) > 0.0:
                    continue
                if random.random() > chance:
                    continue

                b = random.choice(buildings)
                direction = random.randrange(4)
                fp = world_footprint((lz, lx), b.shadow, b.W, b.D, direction)
                if any(not (0 <= fz < depth and 0 <= fx < width) for fz, fx in fp):
                    continue
                if any((fz, fx) in occupied for fz, fx in fp):
                    continue
                # Verify entire footprint is on flat, dry ground and level
                # enough that the schematic won't float over lower cells.
                valid_fp = [
                    (fz, fx) for fz, fx in fp if 0 <= fz < depth and 0 <= fx < width
                ]
                if any(float(feat[fz, fx]["slope"]) > 2.5 for fz, fx in valid_fp):
                    continue
                if any(float(feat[fz, fx]["water_pct"]) > 0.0 for fz, fx in valid_fp):
                    continue
                fp_heights = [int(feat[fz, fx]["height"]) for fz, fx in valid_fp]
                if fp_heights and max(fp_heights) - min(fp_heights) > 0:
                    continue

                wx0, wz0 = self._tm.local_to_world(lz, lx)
                push_y = (
                    max(fp_heights) if fp_heights else int(feat[lz, lx]["height"])
                ) + _MISC_Y_OFFSETS.get(b.csv_path, 0)
                if (lz, lx) in self._garden_cells:
                    push_y += 1
                cell_did = (
                    int(self._tm.district_map[lz, lx])
                    if self._tm.district_map is not None
                    else -1
                )
                cell_district = (
                    self._tm.districts[cell_did]
                    if 0 <= cell_did < len(self._tm.districts)
                    else None
                )
                biome = (
                    cell_district.dominant_biome
                    if cell_district and cell_district.dominant_biome
                    else "plains"
                ).lower()

                with self._editor.pushTransform((wx0, push_y, wz0)):
                    self._placer.place_structure(
                        self._editor,
                        b.csv_path,
                        biome=biome,
                        direction=direction,
                        skip_air=True,
                    )
                # Misc schematics use repeating_command_block as structural
                # placeholders (e.g. well shaft). Replace them with air so
                # the shaft is hollow rather than filled with command blocks.
                for world_pos, _ in self._placer.command_block_positions:
                    self._editor.placeBlock(world_pos, _AIR)
                self._placer.command_block_positions.clear()

                for fz, fx in fp:
                    occupied.add((fz, fx))
                placed += 1

        logger.info("place_misc_fillers: placed %d decorations.", placed)
        return placed

    def place_gardens(
        self,
        occupied: set[tuple[int, int]],
        *,
        near_building_chance: float = 0.25,
        standalone_chance: float = 0.08,
    ) -> int:
        """
        Place circular farmland gardens (via GardenGenerator) in free cells.

        Two modes per candidate cell:
          near-building  — cell is close to 3+ occupied cells (likely a building
                           edge); the garden is placed at a comfortable distance
                           from the building cluster centre.
          standalone     — cell is in open space; garden placed on/near the cell.

        *occupied* is read but NOT modified — downstream generators (misc, plants)
        can still fill the ring of bare ground around each garden.
        """
        feat = self._tm.features
        road_map = self._tm.road_map_expanded
        if feat is None:
            return 0

        _GARDEN_ZONES = (
            SubZone.RESIDENTIAL,
            SubZone.CIVIC,
            SubZone.TOWN_CENTER,
            SubZone.FARMLAND,
            SubZone.ORCHARD,
        )
        depth, width = feat.shape
        gen = GardenGenerator()
        garden_occ: set[tuple[int, int]] = set()
        garden_centers: list[tuple[int, int]] = []
        placed = 0

        # Pre-build the allowed-cell set: every cell belonging to a garden zone.
        # Passing this to generate() ensures farmland never crosses into wall,
        # road, or non-garden zones even when _find_spot places the centre near
        # a boundary.
        allowed_cells: set[tuple[int, int]] = {
            (int(c[0]), int(c[1]))
            for d in self._tm.districts
            if d.sub_zone in _GARDEN_ZONES
            for c in d.cells
        }

        for district in self._tm.districts:
            if district.sub_zone not in _GARDEN_ZONES or district.size == 0:
                continue

            for cell in district.cells:
                lz, lx = int(cell[0]), int(cell[1])
                if (lz, lx) in occupied or (lz, lx) in garden_occ:
                    continue
                if not (0 <= lz < depth and 0 <= lx < width):
                    continue
                if road_map is not None and (
                    0 <= lz < road_map.shape[0]
                    and 0 <= lx < road_map.shape[1]
                    and road_map[lz, lx] > 0
                ):
                    continue
                if float(feat[lz, lx]["water_pct"]) > 0.0:
                    continue
                if float(feat[lz, lx]["slope"]) > 3.0:
                    continue

                adj = [
                    (lz + dz, lx + dx)
                    for dz in range(-3, 4)
                    for dx in range(-3, 4)
                    if (dz != 0 or dx != 0) and (lz + dz, lx + dx) in occupied
                ]
                combined = occupied | garden_occ

                if len(adj) >= 3:
                    if random.random() > near_building_chance:
                        continue
                    cz = sum(z for z, _ in adj) // len(adj)
                    cx = sum(x for _, x in adj) // len(adj)
                    span_z = max(z for z, _ in adj) - min(z for z, _ in adj) + 1
                    span_x = max(x for _, x in adj) - min(x for _, x in adj) + 1
                    success, glz, glx, grad = gen.generate(
                        self._editor,
                        self._tm,
                        cz,
                        cx,
                        combined,
                        building_lw=max(3, span_x),
                        building_ld=max(3, span_z),
                        allowed_cells=allowed_cells,
                    )
                else:
                    if random.random() > standalone_chance:
                        continue
                    success, glz, glx, grad = gen.generate(
                        self._editor,
                        self._tm,
                        lz,
                        lx,
                        combined,
                        allowed_cells=allowed_cells,
                    )

                if success:
                    placed += 1
                    garden_centers.append((glz, glx, grad))
                    for ddz in range(-10, 11):
                        for ddx in range(-10, 11):
                            garden_occ.add((glz + ddz, glx + ddx))

        # Export the exact farmland+ring footprint of each garden to occupied
        # so misc structures don't overwrite them.  The internal garden_occ
        # keeps the larger radius-10 spacing so gardens don't overlap each other.
        # Also record these cells so place_misc_fillers can bump push_y by 1
        # (farmland sits one block above the original terrain surface).
        self._garden_cells.clear()
        for gz, gx, gr in garden_centers:
            outer = (gr + 1) ** 2
            for dlz in range(-gr - 1, gr + 2):
                for dlx in range(-gr - 1, gr + 2):
                    if dlz * dlz + dlx * dlx <= outer:
                        occupied.add((gz + dlz, gx + dlx))
                        self._garden_cells.add((gz + dlz, gx + dlx))
        logger.info("place_gardens: placed %d circular gardens.", placed)
        return placed

    def place_urban_gardens(
        self,
        occupied: set[tuple[int, int]],
        *,
        chance: float = 0.22,
    ) -> int:
        """
        Scatter vegetation (flowers, grass, ferns) into free urban cells.

        Fills visual gaps between buildings with greenery.  Does NOT mark cells
        as occupied so lanterns and subsequent generators are unaffected.
        Only runs on flat, dry cells (slope ≤ 1.5, no water).
        """
        feat = self._tm.features
        road_map = self._tm.road_map_expanded
        if feat is None:
            return 0

        _GARDEN_ZONES = (SubZone.RESIDENTIAL, SubZone.CIVIC, SubZone.TOWN_CENTER)
        depth, width = feat.shape
        placed = 0

        for district in self._tm.districts:
            if district.sub_zone not in _GARDEN_ZONES or district.size == 0:
                continue

            for cell in district.cells:
                lz, lx = int(cell[0]), int(cell[1])
                if (lz, lx) in occupied:
                    continue
                if road_map is not None and (
                    0 <= lz < road_map.shape[0]
                    and 0 <= lx < road_map.shape[1]
                    and road_map[lz, lx] > 0
                ):
                    continue
                if float(feat[lz, lx]["slope"]) > 1.5:
                    continue
                if float(feat[lz, lx]["water_pct"]) > 0.0:
                    continue
                if random.random() > chance:
                    continue

                wx, wz = self._tm.local_to_world(lz, lx)
                surface_y = int(feat[lz, lx]["height"]) - 1

                # Clear 2 blocks above surface, then plant.
                for dy in range(1, 3):
                    self._editor.placeBlock((wx, surface_y + dy, wz), _AIR)
                self._editor.placeBlock(
                    (wx, surface_y + 1, wz),
                    _pick_urban_plant(),
                )
                placed += 1

        logger.info("place_urban_gardens: placed %d plants.", placed)
        return placed

    def place_natural_scatter(
        self,
        occupied: set[tuple[int, int]],
        *,
        chance: float = 0.12,
        tall_chance: float = 0.30,
    ) -> int:
        """
        Scatter grass, ferns, and small bushes across all free, flat, dry cells.

        Covers every zone — not just urban — so gaps between buildings, rural
        fields, and forest edges all get a light coat of ground cover.
        Does NOT mark cells as occupied.
        """
        feat = self._tm.features
        road_map = self._tm.road_map_expanded
        if feat is None:
            return 0

        depth, width = feat.shape
        placed = 0

        for lz in range(depth):
            for lx in range(width):
                if (lz, lx) in occupied:
                    continue
                if (
                    road_map is not None
                    and 0 <= lz < road_map.shape[0]
                    and 0 <= lx < road_map.shape[1]
                    and road_map[lz, lx] > 0
                ):
                    continue
                if float(feat[lz, lx]["water_pct"]) > 0.0:
                    continue
                if float(feat[lz, lx]["slope"]) > 2.0:
                    continue
                if random.random() > chance:
                    continue

                wx, wz = self._tm.local_to_world(lz, lx)
                surface_y = int(feat[lz, lx]["height"]) - 1

                if random.random() < tall_chance:
                    lower_id, upper_id = random.choice(_SCATTER_TALL)
                    self._editor.placeBlock((wx, surface_y + 1, wz), _AIR)
                    self._editor.placeBlock((wx, surface_y + 2, wz), _AIR)
                    self._editor.placeBlock(
                        (wx, surface_y + 1, wz),
                        Block(lower_id, states={"half": "lower"}),
                    )
                    self._editor.placeBlock(
                        (wx, surface_y + 2, wz),
                        Block(upper_id, states={"half": "upper"}),
                    )
                else:
                    self._editor.placeBlock((wx, surface_y + 1, wz), _AIR)
                    self._editor.placeBlock(
                        (wx, surface_y + 1, wz),
                        random.choice(_SCATTER_SINGLE),
                    )
                placed += 1

        logger.info("place_natural_scatter: placed %d plants.", placed)
        return placed

    def place_vignettes(
        self,
        occupied: set[tuple[int, int]],
        *,
        urban_chance: float = 0.035,
        rural_chance: float = 0.012,
        max_total: int = 28,
    ) -> int:
        """
        Scatter small hand-composed decorative "nooks" into leftover cells.

        Unlike ``place_misc_fillers`` (single schematics) and
        ``place_natural_scatter`` (single plants), each vignette is a little
        multi-block scene — a woodpile, a bench and lamp, a flower plot, a
        campfire with seats — that gives the settlement lived-in density
        (aesthetics).  Footprints are marked *occupied* so the plant scatter and
        lantern posts route around them.  Capped at *max_total* so the town
        reads as charmingly cluttered, not noisy.
        """
        feat = self._tm.features
        road_map = self._tm.road_map_expanded
        if feat is None:
            return 0
        depth, width = feat.shape

        _URBAN = (SubZone.RESIDENTIAL, SubZone.CIVIC, SubZone.TOWN_CENTER)
        _RURAL = (
            SubZone.FARMLAND,
            SubZone.ORCHARD,
            SubZone.ANIMAL_PEN,
            SubZone.FOREST_BUFFER,
        )
        _FLOWERS = [
            "poppy",
            "azure_bluet",
            "oxeye_daisy",
            "cornflower",
            "white_tulip",
            "red_tulip",
            "allium",
            "lily_of_the_valley",
        ]

        def _woodpile():
            ax = {"axis": "x"}
            foot = [(0, 0), (0, 1)]
            p = [
                (0, 0, 1, Block("oak_log", states=ax)),
                (0, 0, 2, Block("oak_log", states=ax)),
                (0, 1, 1, Block("oak_log", states=ax)),
            ]
            if random.random() < 0.5:
                p.append((0, 1, 2, Block("stripped_oak_log", states=ax)))
            return foot, p

        def _bench():
            f = random.choice(["north", "south", "east", "west"])
            st = {"facing": f, "half": "bottom"}
            # The two seat stairs must line up PERPENDICULAR to their facing so
            # they read as one continuous bench, not two stairs meeting nose to
            # nose.  north/south → seats run east-west (along x); east/west →
            # seats run north-south (along z).  The lamp post sits behind one end.
            if f in ("north", "south"):
                seat_a, seat_b, post = (0, 0), (0, 1), (1, 1)
            else:
                seat_a, seat_b, post = (0, 0), (1, 0), (1, 1)
            foot = [seat_a, seat_b, post]
            return foot, [
                (seat_a[0], seat_a[1], 1, Block("oak_stairs", states=st)),
                (seat_b[0], seat_b[1], 1, Block("oak_stairs", states=st)),
                (post[0], post[1], 1, Block("oak_fence")),
                (post[0], post[1], 2, Block("lantern")),
            ]

        def _flowerplot():
            foot = [(0, 0), (0, 1), (1, 0), (1, 1)]
            p = [(0, 0, 1, Block("oak_fence")), (0, 0, 2, Block("lantern"))]
            for cz, cx in ((0, 1), (1, 0), (1, 1)):
                p.append((cz, cx, 1, Block(random.choice(_FLOWERS))))
            return foot, p

        def _barrels():
            foot = [(0, 0), (0, 1)]
            # Street-corner barrels: humble outdoor storage (poor, no treasure).
            p = [
                (
                    0,
                    0,
                    1,
                    Block(
                        "barrel", data=container_loot_data("outdoor", is_barrel=True)
                    ),
                )
            ]
            if random.random() < 0.6:
                p.append(
                    (
                        0,
                        0,
                        2,
                        Block(
                            "barrel",
                            data=container_loot_data("outdoor", is_barrel=True),
                        ),
                    )
                )
            p.append((0, 1, 1, Block("decorated_pot")))
            return foot, p

        def _campfire():
            foot = [(0, 0), (0, 1), (1, 0), (1, 1)]
            p = [
                (0, 0, 1, Block("campfire", states={"lit": "true"})),
                (
                    0,
                    1,
                    1,
                    Block("oak_stairs", states={"facing": "west", "half": "bottom"}),
                ),
                (
                    1,
                    0,
                    1,
                    Block("oak_stairs", states={"facing": "east", "half": "bottom"}),
                ),
            ]
            if random.random() < 0.5:
                p.append((1, 1, 1, Block("oak_log", states={"axis": "y"})))
            return foot, p

        def _haystack():
            foot = [(0, 0), (0, 1)]
            return foot, [
                (0, 0, 1, Block("hay_block")),
                (0, 0, 2, Block("hay_block")),
                (0, 1, 1, Block("composter")),
            ]

        urban_builders = [_woodpile, _bench, _flowerplot, _barrels, _campfire]
        rural_builders = [_woodpile, _haystack, _campfire, _flowerplot]

        placed = 0
        for district in self._tm.districts:
            if placed >= max_total:
                break
            if district.sub_zone in _URBAN:
                chance, builders = urban_chance, urban_builders
            elif district.sub_zone in _RURAL:
                chance, builders = rural_chance, rural_builders
            elif self._is_city_fringe(district):
                # City fringe: the occasional campfire or woodpile just
                # outside the fields keeps the outskirts feeling travelled.
                chance, builders = rural_chance * 0.5, rural_builders
            else:
                continue
            if district.size == 0:
                continue

            for cell in district.cells:
                if placed >= max_total:
                    break
                lz, lx = int(cell[0]), int(cell[1])
                if (lz, lx) in occupied:
                    continue
                if (
                    road_map is not None
                    and 0 <= lz < road_map.shape[0]
                    and 0 <= lx < road_map.shape[1]
                    and road_map[lz, lx] > 0
                ):
                    continue
                if float(feat[lz, lx]["slope"]) > 2.0:
                    continue
                if float(feat[lz, lx]["water_pct"]) > 0.0:
                    continue
                if random.random() > chance:
                    continue

                foot, placements = random.choice(builders)()
                fcells = [(lz + dz, lx + dx) for dz, dx in foot]
                if any(not (0 <= fz < depth and 0 <= fx < width) for fz, fx in fcells):
                    continue
                if any((fz, fx) in occupied for fz, fx in fcells):
                    continue
                if any(
                    float(feat[fz, fx]["water_pct"]) > 0.0
                    or float(feat[fz, fx]["slope"]) > 2.0
                    for fz, fx in fcells
                ):
                    continue
                hs = [int(feat[fz, fx]["height"]) for fz, fx in fcells]
                if max(hs) - min(hs) > 0:
                    continue

                base = hs[0] - 1  # ground-surface Y shared by the whole footprint
                for fz, fx in fcells:
                    wx, wz = self._tm.local_to_world(fz, fx)
                    for dy in range(1, 4):
                        self._editor.placeBlock((wx, base + dy, wz), _AIR)
                for dz, dx, dy, block in placements:
                    wx, wz = self._tm.local_to_world(lz + dz, lx + dx)
                    self._editor.placeBlock((wx, base + dy, wz), block)
                for fz, fx in fcells:
                    occupied.add((fz, fx))
                placed += 1

        logger.info("place_vignettes: placed %d decorative nooks.", placed)
        return placed
