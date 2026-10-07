"""
Single unified placer for every CSV-based schematic building in the pipeline.

Replaces: CastlePlacer, WindmillPlacer, HousePlacer, CivilPlacer and the
partial civic placers (library, armory, tavern).

Core flow (shared by all buildings)
-------------------------------------
  1. load_dir / load_file  — parse CSV → Building descriptor
     • footprint  : y=0 non-air, non-jigsaw cells (actual 2D shadow)
     • jigsaw     : (sx, sz) of minecraft:jigsaw markers (road connection pts)
     • entrance   : door block or fence-gate centroid for rotation scoring
  2. place_at_subzone      — castle, civic landmarks (one per district)
  3. place_at_sites        — windmill (pre-determined site list, Y-embed)
  4. fill_urban            — civil buildings (quota-balanced zone fill)

Per-call knobs
--------------
  rotate_to_town  — entrance faces town_center (castle style)
  terrace         — wider levelled apron around the footprint (castle)
  y_embed_range   — (min, max) random vertical embed (windmill: -7..0)
  level_terrain   — skip levelling entirely (windmill on hills)
  approach        — "jigsaw" → packed_mud 1-cell track per jigsaw offset
                    "none"   → no road connection
  furniture       — call place_furniture() after placement (default True)
  clear_height    — blocks to clear above footprint before placing

The supporting pieces live in sibling modules:
  model       — Building/PlacedBuilding descriptors and CSV analysis
  palettes    — biome-driven foundation and building material palettes
  categories  — civil building categories, quotas, and weighted draws
  scoring     — footprint geometry and placement search/scoring
  decoration  — misc fillers, gardens, plant scatter, vignettes (mixin)
  market      — market square construction (mixin)
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import numpy as np
from gdpc.block import Block
from gdpc.editor import Editor

from blocks.block_processor import BlockProcessor
from builds._vegetation import (
    _AIR_NAMES,
    _GROUND_VEG_IDS,
    clear_vegetation,
    is_trunk_block,
)
from consts import BUILD_DIR, BUILD_OUTPUT_DIR
from furniture.furniture_placer import FurniturePlacer
from structures.structure_cache import StructureCache
from structures.structure_placer import StructurePlacer
from terrain.terrain_types import SubZone, TerrainMap
from utils import rotate_offset, seated_path_height

from .categories import (
    CATEGORY_CONFIG,
    CIVIC_PRIORITY,
    PROFESSION_CATS,
    CategoryConfig,
    available_civic,
    group_by_category,
    infer_category,
    weighted_category,
)
from .decoration import DecorationMixin
from .market import MarketSquareMixin
from .model import Building, PlacedBuilding, analyse
from .palettes import building_palette_for, foundation_palette_for
from .scoring import (
    MAX_HEIGHT_DIFF,
    _entrance_openness,
    bresenham_heights,
    dilated_cells,
    find_best_placement,
    find_fallback_placement,
    mark_footprint,
    road_occupied,
    world_footprint,
)

if TYPE_CHECKING:
    from terrain.terrain_modifier import TerrainModifier

logger = logging.getLogger(__name__)

_AIR = Block("minecraft:air")
_MUD = Block("minecraft:packed_mud")


class SchematicBuildingPlacer(DecorationMixin, MarketSquareMixin):
    """
    Unified placer for all CSV-based buildings.

    Usage in main.py:
        sbp = SchematicBuildingPlacer(editor, terrain, modifier)

        # Castle (its sub-zone spans several merged districts)
        castle = sbp.load_file("castle/castle_small.csv", category="castle")
        results = sbp.place_at_subzone(SubZone.CASTLE, [castle],
                      rotate_to_town=True, terrace=True,
                      approach="jigsaw", furniture=True,
                      merge_districts=True)
        if results:
            terrain.castle_entrance = results[0].entrance_local

        # Windmills
        windmill = sbp.load_file("farmland/tall_windmill.csv", category="windmill")
        sbp.place_at_sites(terrain.windmill_sites, [windmill],
                           y_embed_range=(-7, 0), level_terrain=False, furniture=True)

        # Civil buildings (houses, professions, civic)
        civil = sbp.load_dir("civil")
        sbp.fill_urban(civil)
    """

    def __init__(
        self,
        editor: Editor,
        terrain_map: TerrainMap,
        modifier: "TerrainModifier",
    ) -> None:
        self._editor = editor
        self._tm = terrain_map
        self._modifier = modifier
        self._garden_cells: set[tuple[int, int]] = set()
        self.fountain_anchor: tuple[int, int] | None = None
        self.fountain_direction: int = 0
        self.fountain_floor_y: int = 0
        self.fountain_building: "Building | None" = None

        # Every cell of every placed jigsaw approach path (castle, windmill,
        # house doorsteps).  Exposed via approach_cells so main.py can protect
        # these tracks from rural generators and hand them to the pen
        # generator for gate placement.
        self._approach_cells: set[tuple[int, int]] = set()
        self._cache = StructureCache(BUILD_DIR, BUILD_OUTPUT_DIR)
        self._placer = StructurePlacer(
            structure_cache=self._cache,
            block_processor=BlockProcessor(
                stochastic_removals={"amethyst_cluster": 0.7}
            ),
            block_palette=None,
            furniture_replacer=FurniturePlacer(),
        )

    @property
    def approach_cells(self) -> set[tuple[int, int]]:
        """Local (lz, lx) cells of all placed jigsaw approach paths."""
        return self._approach_cells

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_dir(
        self,
        subdir: str,
        *,
        category_fn=None,
    ) -> list[Building]:
        """
        Load all *.csv from builds/output/<subdir>/ as Building descriptors.
        *category_fn(stem) -> str* overrides the default prefix-based inference.
        """
        d = Path(BUILD_DIR) / BUILD_OUTPUT_DIR / subdir
        if not d.is_dir():
            logger.warning("SchematicBuildingPlacer.load_dir: %s not found.", d)
            return []

        cat_fn = category_fn or infer_category
        buildings: list[Building] = []
        for csv_path in sorted(d.glob("*.csv")):
            rel = f"{subdir}/{csv_path.name}"
            try:
                b = analyse(rel, csv_path, cat_fn(csv_path.stem))
                if b is not None:
                    buildings.append(b)
                    logger.info(
                        "load_dir: %-35s cat=%-14s W=%d D=%d jig=%d fp=%d",
                        csv_path.name,
                        b.category,
                        b.W,
                        b.D,
                        len(b.jigsaw_offsets),
                        len(b.footprint),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("load_dir: could not load %s: %s", rel, exc)

        logger.info("load_dir: loaded %d buildings from %s.", len(buildings), subdir)
        return buildings

    def load_file(
        self, rel_path: str, *, category: str = "building", footprint_y: int = 0
    ) -> Optional[Building]:
        """Load a single CSV relative to builds/output/."""
        csv_full = Path(BUILD_DIR) / BUILD_OUTPUT_DIR / rel_path
        if not csv_full.exists():
            logger.warning("load_file: %s not found.", csv_full)
            return None
        try:
            b = analyse(rel_path, csv_full, category, footprint_y=footprint_y)
            if b:
                logger.info(
                    "load_file: %s  W=%d D=%d jig=%d fp=%d",
                    rel_path,
                    b.W,
                    b.D,
                    len(b.jigsaw_offsets),
                    len(b.footprint),
                )
            return b
        except Exception as exc:  # noqa: BLE001
            logger.warning("load_file: could not load %s: %s", rel_path, exc)
            return None

    # ------------------------------------------------------------------
    # Placement modes
    # ------------------------------------------------------------------

    def place_at_subzone(
        self,
        subzone: SubZone,
        buildings: list[Building],
        *,
        count: int = 1,
        y_offset: int = 0,
        rotate_to_town: bool = False,
        terrace: bool = False,
        approach: str = "jigsaw",
        furniture: bool = True,
        clear_height: int = 30,
        max_height_diff: int = MAX_HEIGHT_DIFF,
        occupied: Optional[set[tuple[int, int]]] = None,
        expand_search: int = 0,
        relaxed_height_diff: Optional[int] = None,
        force: bool = False,
        merge_districts: bool = False,
        score_gate_road: bool = False,
    ) -> list[PlacedBuilding]:
        """
        Place up to *count* buildings in districts marked with *subzone*.
        Returns PlacedBuilding list (check entrance_local for castle gate).

        *expand_search* — when a district yields no valid placement, retry
        with its anchor set dilated outward by this many cells.  Anchors are
        the NW corner of the (possibly much larger) schematic box, so for a
        building bigger than its district — the castle — dilation is what
        lets the box slide until the district sits anywhere INSIDE it
        instead of only at its north-west corner.
        *relaxed_height_diff* — second-chance height cap for the expanded
        retry (terracing levels the footprint anyway, so a looser cap trades
        a taller skirt for actually getting a placement).
        *force* — when even the relaxed retry fails, fall back to a
        best-effort placement that ignores occupancy and height caps
        entirely (driest, flattest anchor wins).
        *merge_districts* — search the union of all matching districts as a
        single anchor set instead of each district separately.  For a zone
        deliberately spread over several districts (the castle), this lets
        the schematic box sit anywhere inside the whole zone.
        *score_gate_road* — rate every candidate orientation by the quality
        of the future gate→road approach (distance, elevation gap, terrain
        bumpiness along the connection, own-footprint crossings), so the
        entrance ends up facing a road on the side with the least elevation
        to bridge.
        """
        feat = self._tm.features
        if feat is None or not buildings:
            return []

        depth, width = feat.shape
        road_map = self._tm.road_map
        occ = occupied if occupied is not None else road_occupied(road_map)

        districts = sorted(
            [d for d in self._tm.districts if d.sub_zone == subzone and d.size > 0],
            key=lambda d: -d.centrality,
        )
        if merge_districts and len(districts) > 1:
            districts = [
                SimpleNamespace(
                    district_id=districts[0].district_id,
                    cells=np.concatenate([d.cells for d in districts]),
                )
            ]

        results: list[PlacedBuilding] = []
        for district in districts:
            if len(results) >= count:
                break

            result = find_best_placement(
                district,
                buildings,
                feat,
                road_map,
                occ,
                depth,
                width,
                prefer_town=rotate_to_town,
                tm=self._tm,
                max_height_diff=max_height_diff,
                # A merged zone has far more cells than one district; raise
                # the candidate budget so viable anchors are not cut off.
                max_candidates=1000 if merge_districts else 250,
                score_gate_road=score_gate_road,
            )
            if result is None and expand_search > 0:
                shim = SimpleNamespace(
                    cells=dilated_cells(district.cells, expand_search, depth, width)
                )
                caps = [max_height_diff]
                if relaxed_height_diff is not None:
                    caps.append(relaxed_height_diff)
                for cap in caps:
                    logger.info(
                        "place_at_subzone(%s): no fit in district %d — retrying "
                        "with anchors dilated by %d (max_height_diff=%d).",
                        subzone.name,
                        district.district_id,
                        expand_search,
                        cap,
                    )
                    result = find_best_placement(
                        shim,
                        buildings,
                        feat,
                        road_map,
                        occ,
                        depth,
                        width,
                        prefer_town=rotate_to_town,
                        tm=self._tm,
                        # The dilated anchor set is far larger than a plain
                        # district; a bigger candidate budget keeps viable
                        # anchors from being cut off by the search cap.
                        max_candidates=1000,
                        max_height_diff=cap,
                        score_gate_road=score_gate_road,
                    )
                    if result is not None:
                        break
            if result is None and force:
                logger.warning(
                    "place_at_subzone(%s): forcing placement in district %d — "
                    "ignoring occupancy and height caps.",
                    subzone.name,
                    district.district_id,
                )
                fallback_cells = (
                    dilated_cells(district.cells, max(expand_search, 8), depth, width)
                    if len(district.cells) > 0
                    else district.cells
                )
                result = find_fallback_placement(
                    SimpleNamespace(cells=fallback_cells),
                    buildings,
                    feat,
                    depth,
                    width,
                )
            if result is None:
                continue

            anchor, building, direction = result
            placed = self._place_one(
                anchor,
                building,
                direction,
                feat,
                depth,
                width,
                y_offset=y_offset,
                terrace=terrace,
                approach=approach,
                furniture=furniture,
                clear_height=clear_height,
                protected=occ,
            )
            mark_footprint(anchor, building, direction, occ)
            results.append(placed)

        return results

    def _best_site_direction(
        self,
        anchor: tuple[int, int],
        building: Building,
        feat: np.ndarray,
        depth: int,
        width: int,
    ) -> int:
        """
        Pick the rotation whose entrance faces the most open terrain.

        Site placements (windmills on hills) have a fixed anchor, so only
        the orientation is free.  Each of the four rotations is scored with
        the same entrance-openness measure the castle search uses, so the
        door never opens straight into a hillside.  Rotations that push the
        footprint off the map are rejected, and near-ties keep a random
        pick so windmills on open hilltops still vary in orientation.
        """
        scores: list[float] = []
        for direction in range(4):
            fp = world_footprint(
                anchor, building.footprint, building.W, building.D, direction
            )
            if not fp or any(
                not (0 <= lz < depth and 0 <= lx < width) for lz, lx in fp
            ):
                scores.append(float("-inf"))
                continue
            scores.append(
                _entrance_openness(anchor, building, direction, fp, feat, depth, width)
            )
        best = max(scores)
        if best == float("-inf"):
            return random.randrange(4)
        return random.choice([d for d in range(4) if scores[d] >= best - 0.25])

    def place_at_sites(
        self,
        sites: list[tuple[int, int]],
        buildings: list[Building],
        *,
        y_embed_range: tuple[int, int] = (0, 0),
        level_terrain: bool = True,
        approach: str = "none",
        furniture: bool = True,
        clear_height: int = 30,
        occupied: Optional[set[tuple[int, int]]] = None,
    ) -> int:
        """
        Place one randomly chosen building at each pre-determined site.
        *y_embed_range* gives (min, max) vertical offset below surface (windmill).
        *occupied* is updated in-place with each placed footprint (pad=1) so
        subsequent phases do not overwrite site structures.
        """
        feat = self._tm.features
        if feat is None or not buildings or not sites:
            return 0

        depth, width = feat.shape
        placed = 0

        for lz, lx in sites:
            if not (0 <= lz < depth and 0 <= lx < width):
                continue

            building = random.choice(buildings)
            direction = self._best_site_direction(
                (lz, lx), building, feat, depth, width
            )
            y_embed = random.randint(*y_embed_range)

            self._place_one(
                (lz, lx),
                building,
                direction,
                feat,
                depth,
                width,
                y_embed=y_embed,
                level_terrain=level_terrain,
                approach=approach,
                furniture=furniture,
                clear_height=clear_height,
                protected=occupied,
            )
            if occupied is not None:
                mark_footprint((lz, lx), building, direction, occupied)
            placed += 1
            logger.info(
                "place_at_sites: placed %s at local %s dir=%d embed=%d.",
                building.csv_path,
                (lz, lx),
                direction,
                y_embed,
            )

        return placed

    def fill_urban(
        self,
        buildings: list[Building],
        *,
        y_offset: int = 0,
        category_config: dict[str, CategoryConfig] = CATEGORY_CONFIG,
        furniture: bool = True,
        clear_height: int = 30,
        pre_occupied: Optional[set[tuple[int, int]]] = None,
    ) -> tuple[set[tuple[int, int]], dict[str, int]]:
        """
        Distribute civil buildings across TOWN_CENTER, CIVIC, and RESIDENTIAL
        districts using the quota model in *category_config*.

        *pre_occupied* (e.g. wall district cells) is merged with road cells to
        form the starting exclusion set.  The returned tuple is
        ``(occupied, counts)`` where *occupied* contains everything that was
        occupied after placement (pre_occupied + roads + building footprints)
        and *counts* maps each category name to the number of buildings placed.
        """
        feat = self._tm.features
        if feat is None or not buildings:
            return pre_occupied if pre_occupied is not None else set(), {}

        depth, width = feat.shape
        road_map = self._tm.road_map
        by_cat = group_by_category(buildings)
        counts: dict[str, int] = {}
        occupied = pre_occupied if pre_occupied is not None else set()
        occupied |= road_occupied(self._tm.road_map_expanded)

        kw = dict(y_offset=y_offset, furniture=furniture, clear_height=clear_height)

        # Civic priority queue is shared across all zone types so a category
        # placed in TOWN_CENTER is not duplicated in CIVIC or RESIDENTIAL.
        civic_queue = list(CIVIC_PRIORITY)

        # ── TOWN_CENTER ────────────────────────────────────────────────
        # The market square has already consumed most of this zone.  Any
        # remaining free cells first receive one civic landmark (pulled from
        # the shared queue), then residential buildings fill the rest.
        town_d = next(
            (
                d
                for d in self._tm.districts
                if d.sub_zone == SubZone.TOWN_CENTER and d.size > 0
            ),
            None,
        )
        if town_d is not None:
            available = available_civic(civic_queue, counts, category_config)
            if available:
                self._fill_civic_district(
                    town_d,
                    available,
                    by_cat,
                    counts,
                    category_config,
                    occupied,
                    feat,
                    road_map,
                    depth,
                    width,
                    max_buildings=1,
                    **kw,
                )
            self._fill_residential(
                town_d,
                by_cat,
                counts,
                category_config,
                occupied,
                feat,
                road_map,
                depth,
                width,
                **kw,
            )

        # ── CIVIC districts ───────────────────────────────────────────
        # Strategy: try civic buildings first, then ALWAYS fill the rest of
        # the district with residential buildings — a single landmark in a
        # large district otherwise leaves the civic ring feeling empty.
        # Track which civic categories still need a home so they can retry
        # in residential districts.
        districts = sorted(
            [
                d
                for d in self._tm.districts
                if d.sub_zone == SubZone.CIVIC and d.size > 0
            ],
            key=lambda d: -d.centrality,
        )
        for i, district in enumerate(districts, start=1):
            logger.info(
                f"filling district {i}/{len(districts)}: max_buildings={max(4, int(math.sqrt(district.size) / 0.9))}"
            )
            available = available_civic(civic_queue, counts, category_config)

            if available:
                self._fill_civic_district(
                    district,
                    available,
                    by_cat,
                    counts,
                    category_config,
                    occupied,
                    feat,
                    road_map,
                    depth,
                    width,
                    max_buildings=1,
                    **kw,
                )

            self._fill_residential(
                district,
                by_cat,
                counts,
                category_config,
                occupied,
                feat,
                road_map,
                depth,
                width,
                **kw,
            )

        # ── RESIDENTIAL districts ─────────────────────────────────────
        # Before the normal residential fill, attempt any civic buildings that
        # still haven't found a home.  Residential districts tend to be larger
        # so they may accommodate a church, library, etc. that didn't fit.
        for district in [
            d
            for d in self._tm.districts
            if d.sub_zone == SubZone.RESIDENTIAL and d.size > 0
        ]:
            remaining_civic = available_civic(civic_queue, counts, category_config)
            if remaining_civic:
                self._fill_civic_district(
                    district,
                    remaining_civic,
                    by_cat,
                    counts,
                    category_config,
                    occupied,
                    feat,
                    road_map,
                    depth,
                    width,
                    max_buildings=1,
                    **kw,
                )
            self._fill_residential(
                district,
                by_cat,
                counts,
                category_config,
                occupied,
                feat,
                road_map,
                depth,
                width,
                **kw,
            )

        logger.info("fill_urban: done. Placed counts: %s", counts)
        return occupied, counts

    @staticmethod
    def summarize_inhabitants(
        building_counts: dict[str, int],
    ) -> tuple[list[str], list[str]]:
        """Split placed-building counts into (villager professions, civic
        landmarks) for the chronicle narrative.

        Professions are trade buildings that imply a resident villager (butcher,
        mason, toolsmith, …); civic landmarks are the notable public buildings
        (townhall, church, library, …).  Plain houses and barns are neither.
        """
        professions = [
            c for c in building_counts if c in PROFESSION_CATS and c != "barn"
        ]
        landmarks = [c for c in building_counts if c in CIVIC_PRIORITY]
        return professions, landmarks

    # ------------------------------------------------------------------
    # Internal district helpers
    # ------------------------------------------------------------------

    def _fill_civic_district(
        self,
        district,
        preferred_cats: list[str],
        by_cat: dict[str, list[Building]],
        counts: dict[str, int],
        cfg: dict[str, CategoryConfig],
        occupied: set[tuple[int, int]],
        feat: np.ndarray,
        road_map: Optional[np.ndarray],
        depth: int,
        width: int,
        max_buildings: int = 1,
        y_offset: int = 0,
        furniture: bool = True,
        clear_height: int = 30,
    ) -> None:
        placed = 0
        for cat in preferred_cats:
            if placed >= max_buildings:
                break
            c = cfg.get(cat)
            if c and c.max_total != -1 and counts.get(cat, 0) >= c.max_total:
                continue
            pool = by_cat.get(cat, [])
            if not pool:
                continue
            building = random.choice(pool)
            result = find_best_placement(
                district,
                [building],
                feat,
                road_map,
                occupied,
                depth,
                width,
            )
            if result is None:
                continue
            anchor, b, direction = result
            self._place_one(
                anchor,
                b,
                direction,
                feat,
                depth,
                width,
                y_offset=y_offset,
                furniture=furniture,
                clear_height=clear_height,
                protected=occupied,
            )
            mark_footprint(anchor, b, direction, occupied)
            counts[cat] = counts.get(cat, 0) + 1
            placed += 1

    def _fill_residential(
        self,
        district,
        by_cat: dict[str, list[Building]],
        counts: dict[str, int],
        cfg: dict[str, CategoryConfig],
        occupied: set[tuple[int, int]],
        feat: np.ndarray,
        road_map: Optional[np.ndarray],
        depth: int,
        width: int,
        y_offset: int = 0,
        furniture: bool = True,
        clear_height: int = 30,
    ) -> None:
        house_pool = by_cat.get("house", [])
        max_buildings = max(
            4,
            int(math.sqrt(self._tm.width * self._tm.depth) * 0.4)
            // len(self._tm.districts),
        )
        placed = 0

        while placed < max_buildings:
            cat = weighted_category(by_cat, counts, cfg)
            pool = by_cat.get(cat, []) or house_pool
            if not pool:
                break
            result = find_best_placement(
                district,
                pool,
                feat,
                road_map,
                occupied,
                depth,
                width,
            )
            if result is None and house_pool and pool is not house_pool:
                # The drawn profession doesn't fit anywhere in this district;
                # retry with houses (usually smaller) before giving up so the
                # remaining space still gets filled.
                result = find_best_placement(
                    district,
                    house_pool,
                    feat,
                    road_map,
                    occupied,
                    depth,
                    width,
                )
            if result is None:
                break
            anchor, b, direction = result
            self._place_one(
                anchor,
                b,
                direction,
                feat,
                depth,
                width,
                y_offset=y_offset,
                furniture=furniture,
                clear_height=clear_height,
                protected=occupied,
            )
            mark_footprint(anchor, b, direction, occupied)
            counts[b.category] = counts.get(b.category, 0) + 1
            placed += 1

    # ------------------------------------------------------------------
    # Vegetation clearing
    # ------------------------------------------------------------------

    def _clear_vegetation(
        self,
        fp: list[tuple[int, int]],
        feat: np.ndarray,
        depth: int,
        width: int,
        protected: Optional[set[tuple[int, int]]] = None,
    ) -> None:
        clear_vegetation(
            fp, feat, depth, width, self._editor, self._tm, protected=protected
        )

    # ------------------------------------------------------------------
    # Core placement  (shared by all modes)
    # ------------------------------------------------------------------

    def _place_one(
        self,
        anchor: tuple[int, int],
        building: Building,
        direction: int,
        feat: np.ndarray,
        depth: int,
        width: int,
        *,
        y_embed: int = 0,
        y_offset: int = 0,
        level_terrain: bool = True,
        terrace: bool = False,
        approach: str = "jigsaw",
        furniture: bool = True,
        clear_height: int = 30,
        protected: Optional[set[tuple[int, int]]] = None,
    ) -> PlacedBuilding:
        fp = world_footprint(
            anchor, building.footprint, building.W, building.D, direction
        )
        fp = [(lz, lx) for lz, lx in fp if 0 <= lz < depth and 0 <= lx < width]

        # Determine biome and foundation block based on anchor's district.
        anchor_lz, anchor_lx = anchor
        did = (
            int(self._tm.district_map[anchor_lz, anchor_lx])
            if self._tm.district_map is not None
            else -1
        )
        district = (
            self._tm.districts[did] if 0 <= did < len(self._tm.districts) else None
        )
        biome = (
            district.dominant_biome
            if district and district.dominant_biome
            else "plains"
        ).lower()

        foundation_palette = foundation_palette_for(biome)

        # Clear vegetation before leveling so trees don't skew the floor height.
        # *protected* (the caller's occupied set) keeps the tree-box clear and
        # ground sweep off walls, approach paths, and earlier buildings.
        self._clear_vegetation(fp, feat, depth, width, protected)

        # Resolve entrance cell: jigsaw connector first, then door position.
        # This ensures the front of the building sits flush with the terrain.
        if building.jigsaw_offsets:
            sx, _, sz = building.jigsaw_offsets[0]
        else:
            sx, sz = building.door_x, building.door_z
        ox, oz = rotate_offset(sx, sz, building.W, building.D, direction)
        enz, enx = anchor[0] + oz, anchor[1] + ox
        entrance_cell: Optional[tuple[int, int]] = (
            (enz, enx) if 0 <= enz < depth and 0 <= enx < width else None
        )

        if level_terrain and fp:
            floor_y_raw = self._modifier.level_footprint(
                fp,
                fill_block=foundation_palette,
                strategy="entrance",
                entrance_cell=entrance_cell,
            )
        else:
            anchor_lz, anchor_lx = anchor
            floor_y_raw = int(feat[anchor_lz, anchor_lx]["height"]) + y_embed

        floor_y = floor_y_raw + y_offset

        if level_terrain and fp:
            # Base before walls: the base scan stops at the first solid
            # block, so a wall placed first would hide any void beneath its
            # own bottom from the scan.
            self._modifier.fill_footprint_base(
                fp, floor_y_raw, fill_block=foundation_palette
            )
            self._modifier.fill_footprint_walls(
                fp, floor_y_raw, fill_block=foundation_palette
            )
            self._modifier.make_building_apron(
                fp,
                floor_y_raw,
                fill_block=foundation_palette,
                radius=4 if terrace else 2,
                protected=protected,
                biome=biome,
            )
        elif fp:
            # No levelling (windmills on hills): still seal each footprint
            # column from the floor down to the first solid block, so the
            # structure gets a foundation instead of floating over dips.
            self._modifier.fill_footprint_base(
                fp, floor_y_raw, fill_block=foundation_palette
            )

        for lz, lx in fp:
            if protected and (lz, lx) in protected:
                continue
            wx, wz = self._tm.local_to_world(lz, lx)
            for dy in range(clear_height):
                self._editor.placeBlock((wx, floor_y + dy, wz), _AIR)

        wx0, wz0 = self._tm.local_to_world(anchor_lz, anchor_lx)

        self._placer.block_palette = building_palette_for(
            biome, building.category, building.csv_path
        )
        with self._editor.pushTransform((wx0, floor_y - 1, wz0)):
            self._placer.place_structure(
                self._editor,
                building.csv_path,
                biome=biome,
                direction=direction,
                skip_air=True,
            )

        if furniture:
            self._placer.place_furniture(
                self._editor,
                loot_category=building.category,
                loot_csv=building.csv_path,
            )
        else:
            # Command block positions from non-furniture placements (windmill,
            # castle) must not leak into subsequent furniture-enabled placements.
            self._placer.command_block_positions.clear()

        # ── Road connections ──────────────────────────────────────────
        entrance_local: Optional[tuple[int, int]] = None

        if approach == "jigsaw" and building.jigsaw_offsets:
            entrance_local = self._connect_jigsaw(
                anchor,
                building,
                direction,
                feat,
                depth,
                width,
                floor_y=floor_y,
                protected=protected,
            )

        return PlacedBuilding(
            anchor=anchor,
            direction=direction,
            floor_y=floor_y,
            entrance_local=entrance_local,
        )

    # ------------------------------------------------------------------
    # Approach path helpers
    # ------------------------------------------------------------------

    def _connect_jigsaw(
        self,
        anchor: tuple[int, int],
        building: Building,
        direction: int,
        feat: np.ndarray,
        depth: int,
        width: int,
        floor_y: int = 0,
        protected: Optional[set[tuple[int, int]]] = None,
    ) -> Optional[tuple[int, int]]:
        """Packed-mud 1-cell path from each jigsaw offset to nearest road."""
        from terrain.road_network import _astar

        road_map = self._tm.road_map
        if road_map is None:
            return None

        # Target the RENDERED road (expanded to full width), not the thin
        # centreline — otherwise the approach paves mud across the road shoulder
        # to reach the centre.  The nearest rendered-road cell is also usually
        # just a step or two away, so A* rarely fails.
        rme = self._tm.road_map_expanded
        if rme is None:
            rme = road_map
        road_cells = np.argwhere(rme > 0)
        if len(road_cells) == 0:
            return None

        # Prefer gentle routes over beelines: steep cells are surcharged so
        # the track contours around bumps — a little longer, but seated in
        # the terrain — instead of cutting straight across them.  (The base
        # A* cost already penalizes per-step height change; this additionally
        # steers the path off steep ground entirely.)
        slope_arr = feat["slope"].astype(np.float64)
        cost_mod = 1.0 + 0.35 * slope_arr * slope_arr
        road_heights = feat[road_cells[:, 0], road_cells[:, 1]]["height"].astype(
            np.float64
        )

        # Build the set of local cells occupied by this building so the
        # approach path cannot cut back through the footprint or its walls.
        footprint_local: set[tuple[int, int]] = set()
        for sx2, sz2 in building.footprint:
            ox2, oz2 = rotate_offset(sx2, sz2, building.W, building.D, direction)
            flz, flx = anchor[0] + oz2, anchor[1] + ox2
            if 0 <= flz < depth and 0 <= flx < width:
                footprint_local.add((flz, flx))

        # Cells under ANY part of this building (windmill blades, eaves,
        # towers).  The path may run through them — they are open at ground
        # level — but the air column above the track must not carve into the
        # overhang there.  Other buildings' shadows are already in *protected*.
        shadow_local: set[tuple[int, int]] = set()
        for sx2, sz2 in building.shadow:
            ox2, oz2 = rotate_offset(sx2, sz2, building.W, building.D, direction)
            slz, slx = anchor[0] + oz2, anchor[1] + ox2
            if 0 <= slz < depth and 0 <= slx < width:
                shadow_local.add((slz, slx))

        # The approach path must also route around the physical city wall
        # and around everything already placed (other buildings, the market,
        # lanterns).  Road cells stay traversable — they are the destination
        # and may legitimately appear inside *protected*.
        blocked: set[tuple[int, int]] = footprint_local | self._tm.wall_cells
        if protected:
            rme = self._tm.road_map_expanded
            for plz, plx in protected:
                if (
                    rme is not None
                    and 0 <= plz < rme.shape[0]
                    and 0 <= plx < rme.shape[1]
                    and rme[plz, plx] > 0
                ):
                    continue
                blocked.add((plz, plx))

        first_entrance = None
        for sx, sy, sz in building.jigsaw_offsets:
            ox, oz = rotate_offset(sx, sz, building.W, building.D, direction)
            lz = anchor[0] + oz
            lx = anchor[1] + ox
            if not (0 <= lz < depth and 0 <= lx < width):
                continue

            # World Y of the jigsaw block; path should sit one block below it.
            start_h = floor_y + sy - 2

            # Choose the target road cell by distance AND elevation: a
            # slightly farther road cell near the door's grade beats the
            # nearest one up or down a bank, so the track can run with the
            # terrain instead of ramping straight up to it.
            dists = np.sum((road_cells - np.array([[lz, lx]])) ** 2, axis=1)
            elev_gap = (road_heights - 1.0) - start_h
            target_scores = dists + 6.0 * elev_gap * elev_gap
            target = tuple(road_cells[int(np.argmin(target_scores))].tolist())

            # Start cell stays reachable even if it happens to share a local
            # cell with the footprint boundary or an occupied pad.
            exclude = blocked - {(lz, lx)}

            path = _astar(
                feat,
                self._tm.district_map,
                self._tm.districts,
                (lz, lx),
                target,
                bridge_set=set(),
                cost_mod=cost_mod,
                max_visits=6_000,
                depth=depth,
                width=width,
                exclude=exclude,
            )
            if path is None:
                # No route to the road network — the entry must still be
                # reachable from the ground, otherwise it is left floating
                # (frequent for windmills on unlevelled hills).  Build a
                # doorstep and a short ramp down to the local terrain.
                logger.warning(
                    "connect_jigsaw: no road path from %s for %s; "
                    "building fallback doorstep ramp.",
                    (lz, lx),
                    building.csv_path,
                )
                self._fallback_doorstep_ramp(
                    (lz, lx),
                    start_h,
                    footprint_local,
                    shadow_local,
                    blocked,
                    feat,
                    depth,
                    width,
                )
                continue

            if first_entrance is None:
                first_entrance = (lz, lx)

            # Split the path into the approach segment (non-road cells) and
            # detect the height of the first road cell we land on.
            approach_cells: list[tuple[int, int]] = []
            road_end_h: Optional[int] = None
            for plz, plx in path:
                if rme[plz, plx] > 0:  # reached the rendered road surface
                    road_end_h = int(feat[plz, plx]["height"]) - 1
                    break
                approach_cells.append((plz, plx))

            n = len(approach_cells)
            end_h = (
                road_end_h
                if road_end_h is not None
                else (
                    int(feat[approach_cells[-1][0], approach_cells[-1][1]]["height"])
                    - 1
                    if approach_cells
                    else floor_y
                )
            )

            step_heights = bresenham_heights(start_h, end_h, n)
            prev_h: Optional[int] = None
            for i, (plz, plx) in enumerate(approach_cells):
                # Keep the track on its even Bresenham grade where the
                # terrain allows: a minor ridge (≤ 2 blocks above the ramp)
                # is cut through — the air-clearing above carves the notch —
                # instead of humping the path over it.  Taller terrain lifts
                # the track to the surface, but never by more than 1 block
                # per cell: the doorstep is pinned one below the jigsaw and
                # every step after it stays walkable, carving through the
                # ground where the clamp would otherwise jump.  The reference
                # surface is seated toward the LOWER flanking terrain so a
                # track along a slope shoulder cuts into the ground instead
                # of riding the crest like a causeway.
                terrain_h = seated_path_height(feat, approach_cells, i, depth, width)
                if prev_h is None:
                    # Doorstep cell (the jigsaw itself): must match the entry
                    # exactly or the door ends up floating or buried.
                    h = start_h
                else:
                    h = (
                        step_heights[i]
                        if terrain_h - step_heights[i] <= 2
                        else terrain_h
                    )
                    h = max(min(h, prev_h + 1), prev_h - 1)
                self._pave_approach_cell(
                    plz, plx, h, terrain_h, (plz, plx) in shadow_local
                )
                prev_h = h

        return first_entrance

    def _pave_approach_cell(
        self,
        plz: int,
        plx: int,
        h: int,
        terrain_h: int,
        in_shadow: bool,
    ) -> None:
        """
        Place one packed-mud track cell at height *h* and open the air above.

        Hard clearing (any block → air) is limited to the walkable headroom,
        extended to open a terrain cutting outside the building's shadow.
        Above that only natural vegetation is removed, so the column never
        carves into the structure's own overhangs (windmill blades, roof
        eaves, tower flares) or into decorative persistent leaves.
        """
        wx, wz = self._tm.local_to_world(plz, plx)
        hard_top = h + 3 if in_shadow else max(h + 3, terrain_h + 2)
        for y in range(h + 1, hard_top + 1):
            self._editor.placeBlock((wx, y, wz), _AIR)
        for y in range(hard_top + 1, h + 20):
            blk = self._editor.getBlock((wx, y, wz))
            bid = blk.id or ""
            bname = bid.replace("minecraft:", "")
            if not bid or bname in _AIR_NAMES:
                continue
            is_leaf = (
                bname.endswith("_leaves") and blk.states.get("persistent") != "true"
            )
            clearable = is_leaf or bname in _GROUND_VEG_IDS
            if not in_shadow:
                # Away from the building a trunk over the track is a tree,
                # never our timber — brush it out with the canopy.
                clearable = clearable or is_trunk_block(bid)
            if clearable:
                self._editor.placeBlock((wx, y, wz), _AIR)
        self._editor.placeBlock((wx, h, wz), _MUD)
        self._modifier.fill_column_down(wx, h, wz, fill_block=[_MUD])
        self._approach_cells.add((plz, plx))

    def _fallback_doorstep_ramp(
        self,
        start: tuple[int, int],
        start_h: int,
        footprint_local: set[tuple[int, int]],
        shadow_local: set[tuple[int, int]],
        blocked: set[tuple[int, int]],
        feat: np.ndarray,
        depth: int,
        width: int,
        max_len: int = 12,
    ) -> None:
        """
        Doorstep + straight ramp down to the terrain when no road path exists.

        Marches outward (away from the footprint), dropping 1 block per cell
        from the doorstep until the track meets the ground, so a jigsaw entry
        is never left hanging in the air even without a road connection.
        """
        lz, lx = start

        # Outward = the 4-direction that leads away from the footprint,
        # preferring the one whose terrain is closest below the doorstep.
        best_dir: Optional[tuple[int, int]] = None
        best_score = float("inf")
        for dz, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nz, nx = lz + dz, lx + dx
            if not (0 <= nz < depth and 0 <= nx < width):
                continue
            if (nz, nx) in footprint_local or (nz, nx) in blocked:
                continue
            drop = start_h - (int(feat[nz, nx]["height"]) - 1)
            score = drop if drop >= 0 else 100 + abs(drop)  # never ramp uphill
            if score < best_score:
                best_score = score
                best_dir = (dz, dx)

        # Doorstep pad under the jigsaw — placed even when no ramp fits, so
        # the threshold column is always supported.
        terrain_h = int(feat[lz, lx]["height"]) - 1
        self._pave_approach_cell(lz, lx, start_h, terrain_h, (lz, lx) in shadow_local)

        if best_dir is None:
            return
        dz, dx = best_dir
        h = start_h
        for _ in range(max_len):
            lz, lx = lz + dz, lx + dx
            if not (0 <= lz < depth and 0 <= lx < width):
                break
            if (lz, lx) in footprint_local or (lz, lx) in blocked:
                break
            terrain_h = int(feat[lz, lx]["height"]) - 1
            if terrain_h >= h:  # ramp has met the ground
                break
            h -= 1
            self._pave_approach_cell(lz, lx, h, terrain_h, (lz, lx) in shadow_local)
            if h <= terrain_h + 1:
                break
