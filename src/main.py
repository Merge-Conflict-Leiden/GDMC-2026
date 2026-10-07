"""
This module contains the main entry point of the generator.
"""

import argparse
import logging

from gdpc.editor import Editor
from glm import ivec2

from builds import (
    AnimalPenGenerator,
    ConstructionSiteGenerator,
    CryptGenerator,
    FarmlandGenerator,
    FlockGenerator,
    GraveyardGenerator,
    HarborGenerator,
    MazeGenerator,
    OrchardGenerator,
    SchematicBuildingPlacer,
)
from chronicle import Chronicles
from entities import DisasterGenerator, PopulationGenerator
from lore import Lore
from terrain.road_placer import RoadPlacer
from terrain.terrain_modifier import TerrainModifier
from terrain.terrain_segmenter import TerrainSegmenter
from terrain.terrain_types import SubZone, ZoneType
from terrain.wall_placer import WallPlacer
from utils import rotate_offset, setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the settlement generator."""
    parser = argparse.ArgumentParser(description="GDMC settlement generator")
    parser.add_argument(
        "-v",
        action="count",
        default=2,
        dest="verbosity",
        help="Increase log verbosity; default is INFO, -v adds DEBUG",
    )
    args = parser.parse_args()
    setup_logging(verbosity=args.verbosity)

    editor = Editor(buffering=True, bufferLimit=2048)

    try:
        # Make the world more static during generation, and also protect the
        # eventually generated build
        editor.runCommand("gamerule minecraft:fire_spread_radius_around_player 0")
        editor.runCommand("gamerule minecraft:mob_griefing false")
        editor.runCommand("gamerule minecraft:block_drops false")
        editor.runCommand("gamerule minecraft:entity_drops false")

        build_rect = editor.getBuildArea()

        segmenter = TerrainSegmenter(editor, build_rect)
        terrain = segmenter.run(visualize=True, build_roads=True, build_wall=True)

        modifier = TerrainModifier(editor, terrain)
        sbp = SchematicBuildingPlacer(editor, terrain, modifier)

        # ── Phase A: terrain prep (smooth road heights, clear wall corridor) ──
        modifier.fill_terrain_holes()
        if terrain.road_network is not None:
            modifier.prepare_roads(terrain.road_network)
        if terrain.wall_layout is not None:
            modifier.prepare_wall(terrain.wall_layout)

        # Shared exclusion set, grows with each phase so later phases never
        # overwrite earlier structures.
        pre_occ: set[tuple[int, int]] = set()

        # ── Phase B: major structures (level terrain before road surface) ──
        # Pass pre_occ so the castle footprint is tracked for all later phases.
        # castle_cells holds the exact rotated bounding box of the placed
        # castle (no buffer) so place_roads skips those cells — the castle is
        # placed BEFORE the road surface, and the roads' air-clearing pass
        # would otherwise carve into any part of it standing on road cells.
        castle_cells: set[tuple[int, int]] = set()
        castle = sbp.load_file(
            "castle/castle_small.csv", category="castle", footprint_y=1
        )
        if castle is not None:
            results = sbp.place_at_subzone(
                SubZone.CASTLE,
                [castle],
                # y_offset=-1: the castle is loaded footprint_y=1, so its real
                # ground layer is schematic y=1.  Lowering by one seats that layer
                # flush with the leveled terrain (courtyard at grade, no plinth),
                # rather than one block above like the plinth-mounted civil builds.
                occupied=pre_occ,
                y_offset=-1,
                rotate_to_town=True,
                terrace=True,
                approach="jigsaw",
                furniture=True,
                # The CASTLE sub-zone spans several merged districts sized to
                # fit the 53x41 box plus breathing room (classifier grows it
                # to CASTLE_ZONE_EXTENT); search the whole union so the box
                # can sit anywhere inside the zone at the normal height cap.
                merge_districts=True,
                # Orient the gate by the quality of its future road
                # connection: face a road, prefer the side with the least
                # elevation to bridge, never face away over the keep.
                score_gate_road=True,
                # Small slide allowance for maps where zone growth was cut
                # short (urban-share cap).  Does not relax the height cap,
                # and placement may fail rather than being forced.
                expand_search=8,
            )
            if results:
                logger.info("Placed castle at %s", results[0].anchor)
                terrain.castle_entrance = results[0].entrance_local
                # mark_footprint only covers non-air y=0 cells; courtyards or
                # cells outside the castle Voronoi district are missed.  Block
                # the FULL rotated bounding box (every W×D cell) with a 2-cell
                # buffer so no building can ever clip into the castle.
                alz, alx = results[0].anchor
                direction = results[0].direction
                for sx in range(castle.W):
                    for sz in range(castle.D):
                        ox, oz = rotate_offset(sx, sz, castle.W, castle.D, direction)
                        castle_cells.add((alz + oz, alx + ox))
                        for dlz in range(-3, 4):
                            for dlx in range(-3, 4):
                                pre_occ.add((alz + oz + dlz, alx + ox + dlx))
            else:
                logger.error("Failed to place castle — its zone is handed to housing.")

        editor.flushBuffer()
        # Whatever the multi-district castle zone the keep does not occupy is
        # breathing room inside the walls: relabel it RESIDENTIAL so fill_urban
        # lines it with houses.  The placed bounding box (+ buffer) is already
        # in pre_occ, so neither houses nor the wall can ever clip the castle.
        for d in terrain.districts:
            if d.sub_zone == SubZone.CASTLE:
                d.sub_zone = SubZone.RESIDENTIAL

        # ── Phase C: road + wall surface blocks ──
        # Skip the castle bounding box: its doorstep track (laid in Phase B)
        # already bridges the entrance to the road network.
        RoadPlacer(editor, terrain).place_roads(protected=castle_cells)

        # The castle doorstep track was laid in Phase B; the wall must not
        # plant pikes on it (a skipped step leaves a small passage instead).
        pre_occ |= sbp.approach_cells

        if terrain.wall_layout is not None:
            _wall_count, wall_cells = WallPlacer(editor, terrain, modifier).place_wall(
                terrain.wall_layout, pre_occupied=pre_occ
            )
            # Record the physical wall so later approach-path A* (house and
            # barn doorstep tracks) routes around it instead of through it.
            terrain.wall_cells = wall_cells
            # Add every actual wall cell (plus a 2-cell buffer) to pre_occ so
            # no later phase can overwrite the physical wall blocks.
            for lz, lx in wall_cells:
                for dlz in range(-2, 3):
                    for dlx in range(-2, 3):
                        pre_occ.add((lz + dlz, lx + dlx))
        editor.flushBuffer()

        # ── Phase C2: Market square ──
        # Must run before fill_urban so TOWN_CENTER cells are pre-occupied.
        market = sbp.load_dir("market")
        fountains = [b for b in market if b.category == "fountain"]
        stands = [b for b in market if b.category == "stand"]
        chronicle_b = next((b for b in market if b.category == "chronicle"), None)
        sbp.place_market_square(fountains, stands, pre_occ, chronicle=chronicle_b)
        editor.flushBuffer()

        lectern_pos = None
        if (
            sbp.fountain_anchor is not None
            and sbp.fountain_building is not None
            and sbp.fountain_building.lectern_offset is not None
        ):
            lz, lx = sbp.fountain_anchor
            sx, sy, sz = sbp.fountain_building.lectern_offset
            ox, oz = rotate_offset(
                sx,
                sz,
                sbp.fountain_building.W,
                sbp.fountain_building.D,
                sbp.fountain_direction,
            )
            wx, wz = terrain.local_to_world(lz + oz, lx + ox)
            wy = sbp.fountain_floor_y + sy - 1
            lectern_pos = (wx, wy, wz)

        # ── Phase D: civil buildings (houses, professions, civic) ──
        # fill_urban merges road cells into pre_occ and returns the full
        # occupied set (pre_occ + roads + building footprints).
        civil = sbp.load_dir("civil")
        full_occ, building_counts = sbp.fill_urban(
            civil, y_offset=1, pre_occupied=pre_occ
        )

        # Protect jigsaw approach paths (castle, windmills, house doorsteps)
        # so rural generators and lantern posts never build on top of them.
        full_occ |= sbp.approach_cells

        # Shared naming/cast for this settlement.  Built now that the trades
        # actually placed are known; seeded from the build-area origin so the
        # same location always yields the same town, districts and people.
        villager_types, utility_buildings = sbp.summarize_inhabitants(building_counts)
        lore = Lore(seed=(terrain.x0 * 73856093) ^ (terrain.z0 * 19349663))
        n_urban_districts = sum(
            1 for d in terrain.districts if d.zone_type == ZoneType.URBAN and d.size > 0
        )
        lore.populate(
            villager_types,
            n_townsfolk=max(4, n_urban_districts // 3),
            n_deceased=max(4, n_urban_districts // 6),
        )

        # Lantern posts + wayfinding signposts (after buildings so occupancy is
        # complete).  Signposts also register the named districts into `lore`.
        road_placer = RoadPlacer(editor, terrain)
        road_placer.place_lanterns(full_occ)
        road_placer.place_signposts(full_occ, lore)
        editor.flushBuffer()

        # ── Phase E: rural generators ──
        # full_occ prevents farmland / pens / orchards from overwriting anything.

        windmill = sbp.load_file("farmland/windmill1.csv", category="windmill")
        if windmill is not None:
            sbp.place_at_sites(
                terrain.windmill_sites,
                [windmill],
                level_terrain=False,
                furniture=True,
                approach="jigsaw",
                occupied=full_occ,
            )
        editor.flushBuffer()

        # Windmill doorstep tracks were placed above — refresh the protected
        # approach-cell set before the rural generators run.
        full_occ |= sbp.approach_cells

        FarmlandGenerator().generate(editor, terrain, pre_occupied=full_occ)

        bx, by, bz = build_rect.offset.x, build_rect.offset.y, build_rect.offset.z
        bsx, bsy, bsz = build_rect.size.x, build_rect.size.y, build_rect.size.z
        editor.runCommand(
            f"kill @e[type=!minecraft:player,x={bx},y={by},z={bz},dx={bsx},dy={bsy},dz={bsz}]"
        )
        AnimalPenGenerator().generate(
            editor, terrain, pre_occupied=full_occ, path_cells=sbp.approach_cells
        )

        OrchardGenerator().generate(editor, terrain, pre_occupied=full_occ)
        # Capture what was *actually* built: a planner-scheduled harbor or maze
        # zone can turn out unbuildable and be skipped inside the generator, so
        # the chronicle must reflect the real placement result, not the plan.
        has_harbour = HarborGenerator().generate(editor, terrain, pre_occupied=full_occ)
        has_maze = MazeGenerator().generate(
            editor, terrain, lore, pre_occupied=full_occ
        )
        n_graves = GraveyardGenerator().generate(
            editor, terrain, lore, pre_occupied=full_occ
        )
        # A crypt beneath the churchyard entombs the same named dead (only if the
        # graveyard was actually placed).
        has_crypt = bool(n_graves) and CryptGenerator().generate(
            editor, terrain, lore, pre_occupied=full_occ
        )
        # Stage a couple of half-built lots (claims free urban lots BEFORE the
        # gap fillers below scatter gardens/decoration into them).
        n_construction = ConstructionSiteGenerator().generate(
            editor, terrain, lore, pre_occupied=full_occ
        )
        editor.flushBuffer()

        # ── Phase F: final gap fillers ──
        # Gardens run before misc so they claim contiguous patches first;
        # misc then scatters into whatever cells remain.  Densities are set
        # generously so the town reads as lived-in rather than freshly swept.
        misc = sbp.load_dir("misc")
        sbp.place_urban_gardens(full_occ, chance=0.30)
        sbp.place_gardens(full_occ, near_building_chance=0.30, standalone_chance=0.12)
        sbp.place_misc_fillers(misc, full_occ, urban_chance=0.09, rural_chance=0.035)
        # Hand-composed decorative "nooks" (claim footprints before the plant
        # scatter below fills the remaining gaps around them).
        sbp.place_vignettes(
            full_occ, urban_chance=0.06, rural_chance=0.02, max_total=40
        )
        sbp.place_natural_scatter(full_occ, chance=0.18)
        editor.flushBuffer()

        FlockGenerator().place_flocks(editor, terrain)
        editor.flushBuffer()

        disaster = DisasterGenerator().generate(editor, terrain)

        # ── Phase G: chronicles ──
        # villager_types / utility_buildings were computed in Phase D.  The Lore
        # cast, district names and the graveyard's dead are woven in so the book
        # names the same town, people and dead that appear in the world.
        chronicle = Chronicles(
            editor,
            editor.worldSlice,
            ivec2(terrain.x0, terrain.z0),
            terrain.width,
            terrain.depth,
            terrain.town_center,
            villager_types,
            disaster,
            has_harbour,
            utility_buildings,
            lectern_pos,
            building_counts=building_counts,
            castle_entrance=terrain.castle_entrance,
            settlement_name=lore.settlement_name,
            residents=lore.resident_labels(),
            districts=lore.district_labels(),
            deceased=lore.deceased_labels(),
            has_maze=has_maze,
            has_construction=n_construction > 0,
            has_crypt=has_crypt,
        )
        chronicle.place_chronicle()

        # Clean up possible droppings due to overlapping placement, or entity
        # cramming
        editor.runCommand("kill @e[type=item]")
        editor.runCommand("kill @e[type=experience_orb]")

        # Spawn villagers LAST — into the finished world, so no later block
        # placement can suffocate or displace them.  Flush first so every block
        # (including the chronicle lectern) is already placed before they drop in.
        editor.flushBuffer()
        PopulationGenerator().generate(editor, terrain, lore)

        logger.info("Generation complete.")

    except Exception:
        logger.exception("Error during generation.")
        raise
    finally:
        editor.flushBuffer()


if __name__ == "__main__":
    main()
