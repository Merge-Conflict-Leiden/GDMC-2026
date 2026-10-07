"""
classifier.py
-------------
Phase 4 — District classification (continuous suitability scoring)
Phase 5 — Special zone & landmark detection

Design principles (revised from original):
  - Urban fraction capped at ~22 % of buildable cells so the majority of the
    map remains rural — realistic for a medieval settlement that needs farms,
    pastures and forest to sustain itself.
  - Castle = the highest-elevation urban district.  The town grows outward
    and downward from this defensible anchor.
  - Town centre / market = the most-connected urban district close to the
    castle.  It sits between the keep and the main gate, exactly where a
    medieval market would develop.
  - Civic core & market quarter = grown by BFS from the castle and the town
    centre under a cell budget that is a *rising share* of the urban area.
    Bigger cities therefore dedicate proportionally more land to the core —
    like real medieval towns that gained cathedrals, guildhalls and extra
    market squares as they grew — while hamlets stay almost all houses.
  - Wall perimeter = only the outer-facing ring of rural districts (those
    whose majority of neighbours are urban or off-limits).  Inner-rural
    districts remain as farmland / orchard / pasture.
  - Rural sub-zones (FARMLAND, ORCHARD, ANIMAL_PEN, FOREST_BUFFER,
    WINDMILL_SITE) scale in count with the urban area so larger cities have
    proportionally more food production.
  - Harbor auto-detection uses coast-linearity as well as water fraction.
  - Bridge sites require genuine "opposite-bank" urban presence.
"""

from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np

from .feature_extractor import (
    WATER_BIOME_KEYWORDS,
    WATER_DIST_CUTOFF,
    roughness_norm,
    slope_norm,
)
from .terrain_types import (
    District,
    SubZone,
    SuperDistrict,
    TerrainMap,
    ZoneType,
)
from .voronoi import MIN_DISTRICT_SIZE

# ---------------------------------------------------------------------------
# Hard-override thresholds
# ---------------------------------------------------------------------------
HARD_WATER_OFF_LIMITS = 0.15  # water_pct > this → always OFF_LIMITS
HARD_SLOPE_OFF_LIMITS = 6.0  # mean_slope > this → always OFF_LIMITS
HARD_WATER_HARBOR_MIN = 0.25  # minimum water_pct to qualify as harbor

# ---------------------------------------------------------------------------
# Zone-fraction targets  (fraction of buildable, i.e. non-off-limits cells)
# ---------------------------------------------------------------------------
URBAN_FRACTION_TARGET = 0.22  # ~22 % urban keeps the city realistic in size
RURAL_FRACTION_MIN = 0.50  # at least 50 % of buildable area stays rural

# ---------------------------------------------------------------------------
# Suitability score weights
# ---------------------------------------------------------------------------
W_URBAN = dict(
    slope=0.35,
    roughness=0.25,
    water_pct=0.20,
    border=0.10,
    centrality=0.10,
)
W_RURAL = dict(
    slope=0.25,
    roughness=0.15,
    water_prox=0.25,
    adj_urban=0.20,
    tree_bonus=0.15,
)
W_HARBOR = dict(
    water_pct=0.50,
    adj_urban=0.30,
    coast_len=0.20,
)

# Additive weight applied to urban scores based on proximity to the
# proto-castle (highest-elevation buildable district).  This pulls the
# urban cluster toward the castle anchor so housing/civic buildings ring
# the keep rather than scattering to flat but peripheral terrain.
CASTLE_PROXIMITY_WEIGHT = 0.20

# The castle schematic box is ~53×41 while a single Voronoi district is only
# ~22×22, so the CASTLE sub-zone is grown over several adjacent urban
# districts until its bounding box can hold the schematic in either
# orientation plus breathing room between keep and wall.  Whatever the keep
# does not occupy is handed back to housing after placement (main.py Phase B).
CASTLE_ZONE_EXTENT = 53 + 2 * 6  # max schematic dimension + margin per side
# Never let the castle zone swallow the whole town: stop growing once it
# holds this share of all urban cells, even if the extent target is unmet.
CASTLE_ZONE_MAX_URBAN_SHARE = 0.45


# ---------------------------------------------------------------------------
# Phase 4: classification
# ---------------------------------------------------------------------------


def classify_districts(
    terrain_map: TerrainMap,
    *,
    min_size: int = MIN_DISTRICT_SIZE,
) -> None:
    """
    Classify every district and merge small fragments.
    Writes district.zone_type for all districts and builds terrain_map.zone_map.
    """
    districts = terrain_map.districts
    feat = terrain_map.features
    # -- Betweenness centrality in the district adjacency graph ---------------
    G = _build_adjacency_graph(districts)
    centrality = nx.betweenness_centrality(G, normalized=True, weight="weight")
    max_c = max(centrality.values()) if centrality else 1.0
    for d in districts:
        d.centrality = centrality.get(d.district_id, 0.0) / max(max_c, 1e-9)

    # -- Compute suitability scores -------------------------------------------
    n = len(districts)
    urban_scores = np.zeros(n, dtype=np.float64)
    rural_scores = np.zeros(n, dtype=np.float64)
    harbor_scores = np.zeros(n, dtype=np.float64)

    for d in districts:
        if d.size == 0:
            continue
        cz, cx = d.cells[:, 0], d.cells[:, 1]
        cell_feat = feat[cz, cx]
        sn = float(slope_norm(cell_feat).mean())
        rn = float(roughness_norm(cell_feat).mean())
        wp = float(d.water_pct)
        wd_n = float(1.0 - min(d.mean_water_dist / WATER_DIST_CUTOFF, 1.0))
        td = float(d.tree_density)
        border = 1.0 if d.is_border else 0.0
        cn = float(d.centrality)

        urban_scores[d.district_id] = (
            W_URBAN["slope"] * (1 - sn)
            + W_URBAN["roughness"] * (1 - rn)
            + W_URBAN["water_pct"] * (1 - wp)
            + W_URBAN["border"] * (1 - border)
            + W_URBAN["centrality"] * cn
        )

        rural_scores[d.district_id] = (
            W_RURAL["slope"] * (1 - sn) * 0.8
            + W_RURAL["roughness"] * (1 - rn) * 0.8
            + W_RURAL["water_prox"] * wd_n
            + W_RURAL["tree_bonus"] * (1 - td)
        )

        coast_len = _shared_edge_with_non_water(d, feat) / max(d.size, 1)
        harbor_scores[d.district_id] = W_HARBOR["water_pct"] * wp + W_HARBOR[
            "coast_len"
        ] * min(coast_len, 1.0)

    # -- Hard overrides: always off-limits ------------------------------------
    off_limits_ids: set[int] = set()
    for d in districts:
        if d.is_border:
            off_limits_ids.add(d.district_id)
        if d.water_pct > HARD_WATER_OFF_LIMITS:
            off_limits_ids.add(d.district_id)
        if d.mean_slope > HARD_SLOPE_OFF_LIMITS:
            off_limits_ids.add(d.district_id)
        if any(kw in d.dominant_biome for kw in WATER_BIOME_KEYWORDS):
            off_limits_ids.add(d.district_id)

    # -- Proto-castle anchor: pull urban toward the best castle candidate ------
    # Find the district most suitable for a castle (highest defensible ground,
    # reasonably central) from the buildable set, before zone assignment.
    # We then add a proximity bonus to every district's urban score so the
    # entire urban cluster forms AROUND the castle rather than scattering to
    # whatever terrain happens to be flattest elsewhere on the map.
    buildable = [
        d for d in districts if d.district_id not in off_limits_ids and d.size > 0
    ]

    if buildable:
        heights = [d.mean_height for d in buildable]
        min_h, max_h = min(heights), max(heights)
        h_range = max(max_h - min_h, 1.0)

        if h_range > 5.0:
            # Meaningful elevation: weight elevation heavily for the castle.
            proto_castle = max(
                buildable,
                key=lambda d: (
                    (d.mean_height - min_h) / h_range * 0.60 + d.centrality * 0.40
                ),
            )
        else:
            # Flat terrain: most-central buildable district anchors the city.
            proto_castle = max(buildable, key=lambda d: d.centrality)

        max_dist_sq = (
            max(_dist2(d.centroid, proto_castle.centroid) for d in buildable) or 1.0
        )

        for d in buildable:
            prox = 1.0 - _dist2(d.centroid, proto_castle.centroid) / max_dist_sq
            urban_scores[d.district_id] += CASTLE_PROXIMITY_WEIGHT * prox

    # -- Fraction-based urban assignment (cap at URBAN_FRACTION_TARGET) -------
    total_buildable_cells = max(sum(d.size for d in buildable), 1)
    urban_cell_target = int(total_buildable_cells * URBAN_FRACTION_TARGET)

    sorted_by_urban = sorted(buildable, key=lambda d: -urban_scores[d.district_id])
    urban_set: set[int] = set()
    urban_cells_assigned = 0
    for d in sorted_by_urban:
        if urban_cells_assigned >= urban_cell_target:
            break
        urban_set.add(d.district_id)
        urban_cells_assigned += d.size

    # -- Second pass: add rural/harbor adjacency-to-urban bonus ---------------
    for d in districts:
        bonus = 1.0 if any(nb in urban_set for nb in d.neighbours) else 0.0
        rural_scores[d.district_id] += W_RURAL["adj_urban"] * bonus
        harbor_scores[d.district_id] += W_HARBOR["adj_urban"] * bonus

    # -- Assign zone types ----------------------------------------------------
    for d in districts:
        did = d.district_id
        if did in off_limits_ids:
            if d.water_pct > HARD_WATER_HARBOR_MIN and any(
                nb in urban_set for nb in d.neighbours
            ):
                d.zone_type = ZoneType.HARBOR
                d.harbor_score = harbor_scores[did]
            else:
                d.zone_type = ZoneType.OFF_LIMITS
            continue

        if did in urban_set:
            d.zone_type = ZoneType.URBAN
            d.urban_score = urban_scores[did]
        else:
            d.zone_type = ZoneType.RURAL
            d.rural_score = rural_scores[did]

    # -- Merge small districts ------------------------------------------------
    _merge_small_districts(districts, terrain_map.district_map, min_size)

    # -- Rebuild zone_map -----------------------------------------------------
    dm = terrain_map.district_map
    zone_map = np.zeros(dm.shape, dtype=np.uint8)
    valid = dm >= 0
    zone_map[valid] = np.array(
        [int(districts[did].zone_type) for did in dm[valid].ravel()],
        dtype=np.uint8,
    )
    terrain_map.zone_map = zone_map

    # -- Super-districts (contiguous same-zone clusters) ----------------------
    terrain_map.super_districts = _build_super_districts(
        districts, terrain_map.district_map
    )


# ---------------------------------------------------------------------------
# Phase 5: special zone detection
# ---------------------------------------------------------------------------


def _grow_castle_zone(
    districts: list[District],
    castle_d: District,
    urban_ids: list[int],
) -> set[int]:
    """
    Absorb adjacent urban districts into the CASTLE sub-zone until its
    bounding box reaches CASTLE_ZONE_EXTENT on both axes.

    A single Voronoi district cannot hold the castle schematic, so the
    anchor search needs a zone the schematic box actually fits in (plus
    breathing room that later becomes housing).  Growth is greedy toward
    compactness: of all urban neighbours, the one whose centroid is closest
    to the seed district joins first.

    Returns the set of district ids that make up the zone.
    """
    castle_ids: set[int] = {castle_d.district_id}
    urban_total = sum(districts[i].size for i in urban_ids) or 1
    zone_cells = castle_d.size
    parts = [castle_d.cells] if castle_d.size > 0 else []

    def _extent_reached() -> bool:
        if not parts:
            return True
        allc = np.concatenate(parts)
        return bool(
            allc[:, 0].max() - allc[:, 0].min() + 1 >= CASTLE_ZONE_EXTENT
            and allc[:, 1].max() - allc[:, 1].min() + 1 >= CASTLE_ZONE_EXTENT
        )

    while (
        not _extent_reached() and zone_cells / urban_total < CASTLE_ZONE_MAX_URBAN_SHARE
    ):
        candidates = [
            districts[nb_id]
            for did in castle_ids
            for nb_id in districts[did].neighbours
            if nb_id not in castle_ids
            and districts[nb_id].zone_type == ZoneType.URBAN
            and districts[nb_id].size > 0
        ]
        if not candidates:
            break
        best = min(candidates, key=lambda d: _dist2(d.centroid, castle_d.centroid))
        best.sub_zone = SubZone.CASTLE
        castle_ids.add(best.district_id)
        zone_cells += best.size
        parts.append(best.cells)

    return castle_ids


def detect_special_zones(terrain_map: TerrainMap) -> None:
    """
    Detect landmark sites and assign sub-zones.
    Must be called after classify_districts().
    """
    districts = terrain_map.districts

    urban_ids = [d.district_id for d in districts if d.zone_type == ZoneType.URBAN]
    harbor_ids = [d.district_id for d in districts if d.zone_type == ZoneType.HARBOR]

    if not urban_ids:
        return

    # -- Castle: highest-elevation urban district (defensible high ground) ----
    # Slight slope penalty so a sheer cliff peak loses to a high plateau.
    castle_d = max(
        (districts[i] for i in urban_ids),
        key=lambda d: d.mean_height - 0.5 * d.mean_slope,
    )
    castle_d.sub_zone = SubZone.CASTLE
    # Grow the zone over adjacent urban districts until the castle schematic
    # (plus breathing room) actually fits — one district is far too small.
    castle_ids = _grow_castle_zone(districts, castle_d, urban_ids)
    castle_zone_cells = [
        districts[i].cells for i in castle_ids if districts[i].size > 0
    ]
    if castle_zone_cells:
        allc = np.concatenate(castle_zone_cells)
        # Zone centroid, not seed centroid: roads aim here, and the keep can
        # land anywhere inside the merged zone.
        terrain_map.castle_site = (
            int(round(float(allc[:, 0].mean()))),
            int(round(float(allc[:, 1].mean()))),
        )

    # -- Town centre: most connected urban district that is also near castle --
    # The market develops between the keep and the main gate — it needs high
    # pedestrian throughput (betweenness) and proximity to the castle.
    remaining_urban = [i for i in urban_ids if i not in castle_ids]
    town_d: District | None = None
    if remaining_urban:
        max_dist_sq = (
            max(
                _dist2(districts[i].centroid, castle_d.centroid)
                for i in remaining_urban
            )
            or 1.0
        )

        def _town_score(d: District) -> float:
            prox = 1.0 - _dist2(d.centroid, castle_d.centroid) / max_dist_sq
            return d.centrality * 0.60 + prox * 0.40

        town_d = max((districts[i] for i in remaining_urban), key=_town_score)
        town_d.sub_zone = SubZone.TOWN_CENTER
        if town_d.size > 0:
            cz = town_d.cells[:, 0].mean()
            cx = town_d.cells[:, 1].mean()
            terrain_map.town_center = (int(round(cz)), int(round(cx)))

    # -- Core budgets: bigger cities keep a bigger share for the core ---------
    # Small towns are almost all houses; as a medieval city grew, its centre
    # gained extra market squares, a cathedral, guildhalls and administrative
    # buildings, so the civic/market share of urban land RISES with city size
    # (saturating so the core never swallows the residential quarters).
    n_urban_live = sum(1 for i in urban_ids if districts[i].size > 0)
    urban_cells_total = sum(districts[i].size for i in urban_ids)
    core_growth = min(1.0, (n_urban_live / 64.0) ** 0.5)
    town_cell_budget = int(urban_cells_total * (0.02 + 0.04 * core_growth))
    civic_cell_budget = int(urban_cells_total * (0.12 + 0.16 * core_growth))
    # Even a hamlet has a church: budget at least one average district.
    avg_district_cells = urban_cells_total / max(n_urban_live, 1)
    civic_cell_budget = max(civic_cell_budget, int(avg_district_cells))

    assigned_subs = set(castle_ids)

    # -- Market quarter: grow TOWN_CENTER outward from the seed market --------
    # A single district suffices for a village square, but a large city's
    # market quarter spans several contiguous districts.
    if town_d is not None:
        assigned_subs.add(town_d.district_id)
        town_cells = town_d.size
        frontier = [town_d.district_id]
        seen: set[int] = {town_d.district_id} | castle_ids
        while frontier and town_cells < town_cell_budget:
            next_frontier: list[int] = []
            for did in frontier:
                for nb_id in districts[did].neighbours:
                    if nb_id in seen or town_cells >= town_cell_budget:
                        continue
                    seen.add(nb_id)
                    nb = districts[nb_id]
                    if nb.zone_type != ZoneType.URBAN:
                        continue
                    nb.sub_zone = SubZone.TOWN_CENTER
                    assigned_subs.add(nb_id)
                    town_cells += nb.size
                    next_frontier.append(nb_id)
            frontier = next_frontier

    # -- Civic core: grow CIVIC outward from the castle AND the market --------
    # Seeding from both keeps the core intact even when the keep sits on a
    # promontory with no urban neighbours of its own — the churches and
    # guildhalls then cluster around the market instead of vanishing (and the
    # wall, which rings CASTLE+CIVIC+TOWN_CENTER, still encloses a real core).
    frontier = sorted(castle_ids) + ([town_d.district_id] if town_d is not None else [])
    seen = set(frontier)
    civic_cells = 0
    while frontier and civic_cells < civic_cell_budget:
        next_frontier = []
        for did in frontier:
            for nb_id in districts[did].neighbours:
                if nb_id in seen:
                    continue
                seen.add(nb_id)
                nb = districts[nb_id]
                if nb.zone_type != ZoneType.URBAN:
                    continue
                # Expand through already-labelled core districts too, but
                # never relabel them.
                next_frontier.append(nb_id)
                if nb_id not in assigned_subs and civic_cells < civic_cell_budget:
                    nb.sub_zone = SubZone.CIVIC
                    assigned_subs.add(nb_id)
                    civic_cells += nb.size
        frontier = next_frontier

    # -- Residential: remaining urban districts --------------------------------
    for d in districts:
        if d.zone_type == ZoneType.URBAN and d.sub_zone == SubZone.NONE:
            d.sub_zone = SubZone.RESIDENTIAL

    # -- Wall perimeter & gate sites ------------------------------------------
    _assign_wall_perimeter(districts, terrain_map)

    # -- Rural sub-zones (scale with urban size) ------------------------------
    n_urban = sum(1 for i in urban_ids if districts[i].size > 0)
    _assign_rural_sub_zones(districts, n_urban)

    # -- Hedge maze: an ornamental labyrinth near the keep --------------------
    # Runs before the graveyard so the two claim different districts (the
    # graveyard only considers districts still left as NONE/FOREST_BUFFER).
    _detect_maze(districts, terrain_map)

    # -- Graveyard: one plot just outside the civic core ----------------------
    _detect_graveyard(districts, terrain_map)

    # -- Harbor dock sites (count capped, scaling with city size) -------------
    # Without a cap, every watery district touching the city becomes a dock,
    # which floods large coastal maps with harbors.  Keep the best-scoring
    # ones and demote the rest to OFF_LIMITS so roads, the harbor generator,
    # and the chronicle all see the same trimmed set.
    #
    # Boats need room to moor: prefer harbor districts whose in-district open
    # water (~water_pct × size) can hold a vessel plus berth clearance.  Only
    # filter when at least one roomy candidate exists, so cramped coastlines
    # still get their dock (the harbor generator then simply skips the boat).
    MIN_HARBOR_OPEN_WATER = 60
    roomy = [
        i
        for i in harbor_ids
        if districts[i].water_pct * districts[i].size >= MIN_HARBOR_OPEN_WATER
    ]
    if roomy and len(roomy) < len(harbor_ids):
        for did in set(harbor_ids) - set(roomy):
            districts[did].zone_type = ZoneType.OFF_LIMITS
        harbor_ids = roomy

    max_harbors = min(4, max(1, n_urban // 10))
    if len(harbor_ids) > max_harbors:
        harbor_ids = sorted(
            harbor_ids, key=lambda i: districts[i].harbor_score, reverse=True
        )
        for did in harbor_ids[max_harbors:]:
            districts[did].zone_type = ZoneType.OFF_LIMITS
        harbor_ids = harbor_ids[:max_harbors]

    for did in harbor_ids:
        d = districts[did]
        d.sub_zone = SubZone.HARBOR_DOCK
        if d.size > 0:
            cz = int(round(d.cells[:, 0].mean()))
            cx = int(round(d.cells[:, 1].mean()))
            terrain_map.harbor_sites.append((cz, cx))

    # -- Bridge sites ---------------------------------------------------------
    _detect_bridges(districts, terrain_map)

    # -- Windmill sites (scale with city size) --------------------------------
    _detect_windmill_sites(districts, terrain_map, n_urban)

    # -- Write sub_zone_map ---------------------------------------------------
    dm = terrain_map.district_map
    sub_map = np.zeros(dm.shape, dtype=np.uint8)
    valid = dm >= 0
    sub_map[valid] = np.array(
        [int(districts[did].sub_zone) for did in dm[valid].ravel()],
        dtype=np.uint8,
    )
    terrain_map.sub_zone_map = sub_map


# ---------------------------------------------------------------------------
# Internal helpers — Phase 4
# ---------------------------------------------------------------------------


def _build_adjacency_graph(districts: list[District]) -> nx.Graph:
    G = nx.Graph()
    for d in districts:
        G.add_node(d.district_id)
    for d in districts:
        for nb_id, count in d.neighbours.items():
            if nb_id > d.district_id:
                G.add_edge(d.district_id, nb_id, weight=count)
    return G


def _shared_edge_with_non_water(d: District, feat: np.ndarray) -> float:
    """Count cells in d that border a non-water cell (used for coast-length)."""
    if d.size == 0:
        return 0.0
    count = 0
    depth, width = feat.shape
    for cz, cx in d.cells:
        for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nz, nxx = int(cz) + dz, int(cx) + dx
            if 0 <= nz < depth and 0 <= nxx < width:
                if feat[nz, nxx]["water_pct"] < 0.1:
                    count += 1
                    break
    return float(count)


def _merge_small_districts(
    districts: list[District],
    district_map: np.ndarray,
    min_size: int,
) -> None:
    """Merge districts smaller than min_size into their most-similar neighbour."""
    changed = True
    while changed:
        changed = False
        for d in districts:
            if d.size == 0 or d.size >= min_size:
                continue
            if not d.neighbours:
                continue
            same_type = [
                nb_id
                for nb_id in d.neighbours
                if districts[nb_id].zone_type == d.zone_type
            ]
            best = (
                max(same_type, key=lambda i: d.neighbours[i])
                if same_type
                else max(d.neighbours, key=lambda i: districts[i].size)
            )
            target = districts[best]
            if d.size > 0:
                district_map[d.cells[:, 0], d.cells[:, 1]] = best
            target.cells = (
                np.vstack([target.cells, d.cells]) if target.size > 0 else d.cells
            )
            d.cells = np.empty((0, 2), dtype=np.int32)
            for nb_id, cnt in d.neighbours.items():
                if nb_id == best:
                    continue
                target.neighbours[nb_id] = target.neighbours.get(nb_id, 0) + cnt
                if nb_id < len(districts):
                    nb_d = districts[nb_id]
                    nb_d.neighbours[best] = nb_d.neighbours.get(best, 0) + cnt
                    nb_d.neighbours.pop(d.district_id, None)
            d.neighbours.clear()
            changed = True


def _build_super_districts(
    districts: list[District],
    _district_map: np.ndarray,
) -> list[SuperDistrict]:
    """Group contiguous same-typed districts via adjacency-graph BFS."""
    visited: set[int] = set()
    super_districts: list[SuperDistrict] = []
    sid = 0
    for d in districts:
        if d.district_id in visited or d.size == 0:
            continue
        zone = d.zone_type
        queue = deque([d.district_id])
        visited.add(d.district_id)
        cluster: list[int] = []
        while queue:
            cur = queue.popleft()
            cluster.append(cur)
            for nb_id in districts[cur].neighbours:
                if (
                    nb_id not in visited
                    and districts[nb_id].zone_type == zone
                    and districts[nb_id].size > 0
                ):
                    visited.add(nb_id)
                    queue.append(nb_id)
        super_districts.append(
            SuperDistrict(super_id=sid, zone_type=zone, district_ids=cluster)
        )
        sid += 1
    return super_districts


# ---------------------------------------------------------------------------
# Internal helpers — Phase 5
# ---------------------------------------------------------------------------


def _assign_wall_perimeter(
    districts: list[District],
    terrain_map: TerrainMap,
) -> None:
    """
    Wall surrounds only the inner fortified core (CASTLE + CIVIC + TOWN_CENTER).

    Layout from centre outward:
      inner core → wall ring → residential suburbs → rural outskirts → off-limits

    Residential districts are deliberate suburbs — historically the "faubourgs"
    that grew outside city walls, unfortified and vulnerable.  They need no wall.

    Water = natural defence.  Any candidate district whose entire outer face
    borders water / off-limits is skipped: the river or cliff provides the
    protection, so no masonry is needed there.  This creates realistic gaps in
    the wall ring at coastlines and river banks.

    Gate sites: wall districts with at least one non-water, non-core neighbour
    on the outer side — these are where roads punch through the wall.
    Gates are capped at 4, placed at the lowest-elevation candidates first
    (natural passes through the wall).
    Tower sites: wall corners with ≥ 2 wall neighbours.
    """
    inner_core_ids: set[int] = {
        d.district_id
        for d in districts
        if d.sub_zone in (SubZone.CASTLE, SubZone.CIVIC, SubZone.TOWN_CENTER)
    }
    off_ids: set[int] = {
        d.district_id
        for d in districts
        if d.zone_type in (ZoneType.OFF_LIMITS, ZoneType.HARBOR)
    }

    if not inner_core_ids:
        return

    wall_candidates: list[District] = []

    for d in districts:
        if d.district_id in inner_core_ids:
            continue
        # Must touch the inner core.
        if not any(nb in inner_core_ids for nb in d.neighbours):
            continue

        # Water-defence check: if every neighbour that is NOT in the inner core
        # is water/off-limits, the outer face is fully defended by terrain.
        # Skip those districts — no masonry wall needed.
        outer_non_water = [
            nb for nb in d.neighbours if nb not in inner_core_ids and nb not in off_ids
        ]
        if not outer_non_water:
            continue  # entirely water/cliff-defended on the outside

        d.zone_type = ZoneType.WALL_PERIMETER
        d.sub_zone = SubZone.NONE
        wall_candidates.append(d)

    # Gate sites: wall districts that face the outside world (suburbs / rural)
    # without water in the way.  Prefer the lowest-elevation candidates (natural
    # passes) and cap at 4 gates so we don't saturate the wall.
    wall_ids = {d.district_id for d in wall_candidates}

    gate_candidates = [
        d
        for d in wall_candidates
        if not any(nb in off_ids for nb in d.neighbours)  # not water-adjacent
        and any(
            nb not in inner_core_ids and nb not in wall_ids for nb in d.neighbours
        )  # has a real "outside" neighbour (suburb or rural)
    ]
    gate_candidates.sort(key=lambda d: d.mean_height)  # low passes first

    gate_added: set[int] = set()
    max_gates = max(2, min(4, len(gate_candidates)))
    for d in gate_candidates:
        if len(gate_added) >= max_gates:
            break
        # Spread gates: skip if another gate is already very close.
        cz = int(round(d.cells[:, 0].mean()))
        cx = int(round(d.cells[:, 1].mean()))
        too_close = any(
            _dist2((cz, cx), terrain_map.gate_sites[i]) < 30**2
            for i in range(len(terrain_map.gate_sites))
        )
        if too_close:
            continue
        terrain_map.gate_sites.append((cz, cx))
        d.sub_zone = SubZone.GATE_SITE
        gate_added.add(d.district_id)

    # Tower sites: wall corners / bends (≥ 2 wall neighbours).
    for d in wall_candidates:
        if d.sub_zone == SubZone.GATE_SITE:
            continue
        wall_nb_count = sum(1 for nb in d.neighbours if nb in wall_ids)
        d.sub_zone = SubZone.TOWER_SITE if wall_nb_count >= 2 else SubZone.NONE

    # Propagate WALL_PERIMETER into the zone_map (built later; update here too).
    zm = terrain_map.zone_map
    if zm is not None:
        for d in wall_candidates:
            if d.size > 0:
                zm[d.cells[:, 0], d.cells[:, 1]] = int(ZoneType.WALL_PERIMETER)


def _assign_rural_sub_zones(
    districts: list[District],
    n_urban_districts: int,
) -> None:
    """
    Assign fine-grained sub-zones within RURAL districts.

    Only districts within 2 hops of the urban/wall core are designated;
    districts farther out remain SubZone.NONE (natural terrain untouched).

    Sub-zone rules (priority order):
      1. FOREST_BUFFER   — tree_density > 0.35
      2. FARMLAND        — flat (slope < 2.0) and close to water (dist < 20)
      3. ORCHARD         — moderate slope and water proximity
      4. ANIMAL_PEN      — fallback, capped at n_urban_districts // 2
    """
    # Build 2-hop neighbourhood from urban/wall so only districts near the
    # city get designated; distant wilderness stays as natural terrain.
    urban_wall_ids: set[int] = {
        d.district_id
        for d in districts
        if d.zone_type in (ZoneType.URBAN, ZoneType.WALL_PERIMETER)
    }
    near_ids: set[int] = set()
    for d_id in urban_wall_ids:
        for nb1_id in districts[d_id].neighbours:
            if districts[nb1_id].zone_type == ZoneType.RURAL:
                near_ids.add(nb1_id)
                for nb2_id in districts[nb1_id].neighbours:
                    if districts[nb2_id].zone_type == ZoneType.RURAL:
                        near_ids.add(nb2_id)

    rural_districts = [
        d
        for d in districts
        if d.zone_type == ZoneType.RURAL
        and d.sub_zone == SubZone.NONE
        and d.size > 0
        and d.district_id in near_ids
    ]
    if not rural_districts:
        return

    farmland_quota = max(2, n_urban_districts * 2 // 3)
    orchard_quota = max(1, n_urban_districts // 4)
    animal_pen_quota = max(1, n_urban_districts // 2)
    farmland_assigned = orchard_assigned = animal_pen_assigned = 0

    def _farmland_score(d: District) -> float:
        return (1.0 - min(d.mean_slope / 4.0, 1.0)) * 0.6 + (
            1.0 - min(d.mean_water_dist / 30.0, 1.0)
        ) * 0.4

    rural_by_farmland = sorted(rural_districts, key=_farmland_score, reverse=True)

    for d in rural_by_farmland:
        if d.tree_density > 0.35:
            d.sub_zone = SubZone.FOREST_BUFFER
            continue

        if farmland_assigned < farmland_quota and _farmland_score(d) > 0.35:
            d.sub_zone = SubZone.FARMLAND
            farmland_assigned += 1
        elif orchard_assigned < orchard_quota and d.mean_slope < 3.0:
            d.sub_zone = SubZone.ORCHARD
            orchard_assigned += 1
        elif animal_pen_assigned < animal_pen_quota and d.mean_slope < 2.0:
            # Pens need genuinely level pasture: fences step but livestock
            # (and realism) don't climb hillsides.  Steeper districts stay
            # natural rather than wasting the pen quota on a cliff.
            d.sub_zone = SubZone.ANIMAL_PEN
            animal_pen_assigned += 1
        # else: leave as SubZone.NONE — natural terrain, no generator touches it


def _detect_bridges(
    districts: list[District],
    terrain_map: TerrainMap,
) -> None:
    """
    Detect narrow water corridors between urban (or rural) districts on
    opposite banks — these are candidates for bridge placement.
    """
    urban_or_wall = {
        d.district_id
        for d in districts
        if d.zone_type in (ZoneType.URBAN, ZoneType.WALL_PERIMETER, ZoneType.RURAL)
    }

    for d in districts:
        if d.zone_type != ZoneType.OFF_LIMITS or d.water_pct < 0.3 or d.size == 0:
            continue
        nb_urban = [nb for nb in d.neighbours if nb in urban_or_wall]
        if len(nb_urban) < 2:
            continue

        centroids = [
            districts[nb_id].centroid for nb_id in nb_urban if districts[nb_id].size > 0
        ]
        if len(centroids) < 2:
            continue

        all_z = [c[0] for c in centroids]
        all_x = [c[1] for c in centroids]
        span = max(max(all_z) - min(all_z), max(all_x) - min(all_x))
        d_span = max(
            d.cells[:, 0].max() - d.cells[:, 0].min(),
            d.cells[:, 1].max() - d.cells[:, 1].min(),
        )
        if span > d_span * 0.8:
            cz = int(round(d.cells[:, 0].mean()))
            cx = int(round(d.cells[:, 1].mean()))
            terrain_map.bridge_sites.append((cz, cx))
            d.sub_zone = SubZone.BRIDGE_SITE


def _detect_windmill_sites(
    districts: list[District],
    terrain_map: TerrainMap,
    n_urban_districts: int,
) -> None:
    """
    Pick windmill positions from within farmland districts.

    Windmills sit inside the wheat fields they serve, so the candidate pool is
    the set of FARMLAND districts themselves.  The district sub-zone is left as
    FARMLAND; the farmland generator already excludes a radius around each site
    so crops don't grow through the structure.  Elevation and low roughness
    break ties.  Count scales with city size.
    """
    candidates = [d for d in districts if d.sub_zone == SubZone.FARMLAND and d.size > 0]
    if not candidates:
        return

    max_h = max(d.mean_height for d in candidates)
    min_h = min(d.mean_height for d in candidates)
    h_range = max(max_h - min_h, 1.0)

    def _score(d: District) -> float:
        elev_norm = (d.mean_height - min_h) / h_range
        rough_penalty = min(d.mean_roughness / 12.0, 1.0)
        return elev_norm * 0.6 - rough_penalty * 0.4

    n_windmills = max(1, n_urban_districts // 16)
    sorted_cands = sorted(candidates, key=_score, reverse=True)

    # Greedy spread: each new windmill must be far enough from existing ones.
    min_windmill_sep_sq = 30.0**2
    placed: list[tuple[int, int]] = []

    for d in sorted_cands:
        if len(placed) >= n_windmills:
            break
        cz = int(round(d.cells[:, 0].mean()))
        cx = int(round(d.cells[:, 1].mean()))
        if all(_dist2((cz, cx), p) >= min_windmill_sep_sq for p in placed):
            terrain_map.windmill_sites.append((cz, cx))
            placed.append((cz, cx))


def _detect_maze(
    districts: list[District],
    terrain_map: TerrainMap,
) -> None:
    """
    Reserve one district for an ornamental hedge maze near the keep.

    A pleasure-garden labyrinth is a lord's folly, so we favour a flat, roomy
    rural district that borders the fortified core and sits as close to the
    castle as possible.  Only *free* districts (left NONE/FOREST_BUFFER by the
    rural pass) are eligible, so no farm, pen or orchard is displaced.
    """
    inner_core = {
        d.district_id
        for d in districts
        if d.sub_zone in (SubZone.CASTLE, SubZone.CIVIC, SubZone.TOWN_CENTER)
    }
    if not inner_core:
        return

    anchor = terrain_map.castle_site or terrain_map.town_center
    min_size = 120  # enough room for at least a ~9x9 maze pad

    def _borders_town(d: District) -> bool:
        return any(
            nb in inner_core or districts[nb].zone_type == ZoneType.WALL_PERIMETER
            for nb in d.neighbours
        )

    candidates = [
        d
        for d in districts
        if d.size >= min_size
        and d.zone_type == ZoneType.RURAL
        and d.sub_zone in (SubZone.NONE, SubZone.FOREST_BUFFER)
        and _borders_town(d)
    ]
    # A large city can afford a courtyard labyrinth: free residential
    # districts join the pool, so the maze may sit INSIDE the city as a
    # walled pleasure garden rather than always out among the fields.
    n_urban_live = sum(
        1 for d in districts if d.zone_type == ZoneType.URBAN and d.size > 0
    )
    if n_urban_live >= 12:
        candidates += [
            d
            for d in districts
            if d.size >= min_size
            and d.zone_type == ZoneType.URBAN
            and d.sub_zone == SubZone.RESIDENTIAL
        ]
    if not candidates:
        candidates = [
            d
            for d in districts
            if d.size >= min_size
            and d.zone_type == ZoneType.RURAL
            and d.sub_zone == SubZone.NONE
        ]
    if not candidates:
        return

    def _key(d: District) -> tuple[float, float]:
        near = _dist2(d.centroid, anchor) if anchor is not None else 0.0
        return (d.mean_slope, near)  # flattest first, then closest to the keep

    # A grand city can afford more than one pleasure garden: up to 3 mazes,
    # spaced apart so they read as separate follies.
    n_mazes = min(3, 1 + n_urban_live // 32)
    for d in sorted(candidates, key=_key):
        if len(terrain_map.maze_sites) >= n_mazes:
            break
        cz = int(round(d.cells[:, 0].mean()))
        cx = int(round(d.cells[:, 1].mean()))
        if any(_dist2((cz, cx), s) < 40**2 for s in terrain_map.maze_sites):
            continue
        d.sub_zone = SubZone.MAZE
        terrain_map.maze_sites.append((cz, cx))
    if terrain_map.maze_sites:
        terrain_map.maze_site = terrain_map.maze_sites[0]


def _detect_graveyard(
    districts: list[District],
    terrain_map: TerrainMap,
) -> None:
    """
    Reserve one small burial ground just outside the fortified core.

    Historically the churchyard/cemetery sits at the edge of town, near the
    civic buildings but off the busiest ground.  We pick a *free* rural district
    (one the rural sub-zone pass left as NONE or FOREST_BUFFER, so no farm, pen
    or orchard is displaced) that borders the inner core or the wall, preferring
    the flattest, smallest such district so the plot reads as a tidy yard.
    """
    inner_core = {
        d.district_id
        for d in districts
        if d.sub_zone in (SubZone.CASTLE, SubZone.CIVIC, SubZone.TOWN_CENTER)
    }
    if not inner_core:
        return

    def _borders_town(d: District) -> bool:
        return any(
            nb in inner_core or districts[nb].zone_type == ZoneType.WALL_PERIMETER
            for nb in d.neighbours
        )

    # A churchyard must be level ground — a steep district would drape
    # headstones down a hillside, so steep candidates are rejected outright
    # rather than merely deprioritised.
    MAX_GRAVEYARD_SLOPE = 2.0

    candidates = [
        d
        for d in districts
        if d.size > 0
        and d.zone_type == ZoneType.RURAL
        and d.sub_zone in (SubZone.NONE, SubZone.FOREST_BUFFER)
        and d.mean_slope <= MAX_GRAVEYARD_SLOPE
        and _borders_town(d)
    ]
    # Fallback: any free rural district anywhere (keeps a churchyard even on
    # maps where the core is ringed entirely by farmland/pens).
    if not candidates:
        candidates = [
            d
            for d in districts
            if d.size > 0
            and d.zone_type == ZoneType.RURAL
            and d.sub_zone == SubZone.NONE
            and d.mean_slope <= MAX_GRAVEYARD_SLOPE
        ]
    if not candidates:
        return

    # Larger cities bury more dead: up to 3 churchyards, spaced apart.  The
    # first (flattest, near the core) stays the primary site — the crypt is
    # dug beneath that one.
    n_urban_live = sum(
        1 for d in districts if d.zone_type == ZoneType.URBAN and d.size > 0
    )
    n_graveyards = min(3, 1 + n_urban_live // 24)
    for d in sorted(candidates, key=lambda d: (d.mean_slope, d.size)):
        if len(terrain_map.graveyard_sites) >= n_graveyards:
            break
        cz = int(round(d.cells[:, 0].mean()))
        cx = int(round(d.cells[:, 1].mean()))
        if any(_dist2((cz, cx), s) < 40**2 for s in terrain_map.graveyard_sites):
            continue
        d.sub_zone = SubZone.GRAVEYARD
        terrain_map.graveyard_sites.append((cz, cx))
    if terrain_map.graveyard_sites:
        terrain_map.graveyard_site = terrain_map.graveyard_sites[0]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
