"""
terrain_segmenter.py
--------------------
Top-level orchestrator: TerrainSegmenter.

Usage (with a live GDPC connection):
    from gdpc import Editor
    from terrain.terrain_segmenter import TerrainSegmenter

    editor = Editor()
    build_rect = editor.getBuildArea()
    segmenter  = TerrainSegmenter(editor, build_rect)
    terrain    = segmenter.run()

The run() method returns a fully-populated TerrainMap, which is the single
source of truth consumed by all downstream generation phases (road network,
building placement, wall construction, farm placement, etc.).

GDPC integration notes:
  - WorldSlice is fetched once and cached.  All height / biome / block
    lookups go through that single slice.
  - We use the MOTION_BLOCKING_NO_LEAVES heightmap type so trees are
    transparent (we detect them separately via block scans).
  - Biome fetching via GDPC 5.x uses editor.getBiome(); in older versions
    use worldSlice.getBiomeMap() if available, else fall back to None.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Optional

import numpy as np
from gdpc.editor import interface
from gdpc.vector_tools import ivec3

from .classifier import classify_districts, detect_special_zones
from .feature_extractor import ICE_BLOCK_NAMES, TREE_BLOCK_NAMES, extract_features
from .terrain_types import TerrainMap
from .voronoi import generate_districts

logger = logging.getLogger(__name__)


class TerrainSegmenter:
    """
    Full 5-phase terrain segmentation pipeline.

    Parameters
    ----------
    editor        : gdpc.Editor instance (may be None for offline testing).
    build_rect    : gdpc.Rect or anything with x, z, width, length attributes.
    rng_seed      : integer seed for reproducibility (default: 42).
    n_districts   : override automatic district count.
    border_buffer : cells from edge treated as border (default 2).
    blur_sigma    : Gaussian sigma for slope pre-smoothing (default 1.5).
    min_district_size : districts smaller than this are merged (default 30).
    """

    def __init__(
        self,
        editor=None,
        build_rect=None,
        *,
        rng_seed: int = 42,
        n_districts: Optional[int] = None,
        border_buffer: int = 2,
        blur_sigma: float = 1.5,
        min_district_size: int = 30,
    ):
        self._editor = editor
        self._rect = build_rect
        self._rng = random.Random(rng_seed)
        self._n_districts = n_districts
        self._border_buffer = border_buffer
        self._blur_sigma = blur_sigma
        self._min_district_size = min_district_size

        # Pre-fetched arrays (populated by _fetch_world_data)
        self._heightmap: Optional[np.ndarray] = None
        self._surface_heightmap: Optional[np.ndarray] = None
        self._biome_map: Optional[np.ndarray] = None
        self._biome_str_map: Optional[np.ndarray] = None
        self._water_mask: Optional[np.ndarray] = None
        self._tree_mask: Optional[np.ndarray] = None
        self._x0 = self._z0 = self._x1 = self._z1 = 0

    def run(
        self,
        *,
        visualize: bool = False,
        build_roads: bool = False,
        build_wall: bool = False,
    ) -> TerrainMap:
        """
        Execute all pipeline phases and return a populated TerrainMap.

        Parameters
        ----------
        visualize   : if True, open a matplotlib figure showing the segmentation
                      result (heightmap, zone map, sub-zone map, district map).
                      Requires matplotlib to be installed.
        build_roads : if True, run Phase 6 (road network generation) after the
                      terrain segmentation phases.  Populates terrain_map.road_map
                      and, if visualize is also True, overlays roads on the zone
                      map panel.
        build_wall  : if True, run Phase 7 (wall layout generation).  Populates
                      terrain_map.wall_layout with a WallLayout whose corner
                      positions satisfy the 5-block segment / 11×11 tower /
                      25-block gate placement constraints.
        """
        t0 = time.perf_counter()

        if self._heightmap is None:
            logger.info("Phase 1: fetching world data from Minecraft...")
            self._fetch_world_data()

        terrain_map = TerrainMap(
            x0=self._x0,
            z0=self._z0,
            x1=self._x1,
            z1=self._z1,
        )
        terrain_map.water_mask = self._water_mask
        terrain_map.surface_heightmap = self._surface_heightmap

        logger.info(
            "Build area: %d×%d blocks (%d total cells)",
            terrain_map.width,
            terrain_map.depth,
            terrain_map.width * terrain_map.depth,
        )

        logger.info("Phase 2: computing terrain features...")
        extract_features(
            terrain_map,
            self._heightmap,
            self._biome_map,
            self._water_mask,
            self._tree_mask,
            border_buffer=self._border_buffer,
            blur_sigma=self._blur_sigma,
        )
        terrain_map.biome_name_map = self._biome_str_map

        logger.info("Phase 3: generating Voronoi districts...")
        generate_districts(
            terrain_map,
            n_districts=self._n_districts,
            rng=self._rng,
        )
        logger.info("  → %d districts created.", len(terrain_map.districts))

        logger.info("Phase 4: classifying districts...")
        classify_districts(terrain_map, min_size=self._min_district_size)

        from .terrain_types import ZoneType

        counts = {zt: 0 for zt in ZoneType}
        for d in terrain_map.districts:
            if d.size > 0:
                counts[d.zone_type] += 1
        logger.info(
            "  → Zone counts: %s", {zt.name: v for zt, v in counts.items() if v > 0}
        )

        logger.info("Phase 5: detecting landmarks and special zones...")
        detect_special_zones(terrain_map)
        logger.info("  → Town center: %s", terrain_map.town_center)
        logger.info("  → Castle site: %s", terrain_map.castle_site)
        logger.info("  → Gate sites:  %s", terrain_map.gate_sites)
        logger.info("  → Harbor sites: %s", terrain_map.harbor_sites)
        logger.info("  → Bridge sites: %s", terrain_map.bridge_sites)
        logger.info("  → Windmill sites: %s", terrain_map.windmill_sites)

        if build_roads:
            logger.info("Phase 6: generating road network...")
            from .road_network import generate_roads as _gen_roads

            road_net = _gen_roads(terrain_map, rng=self._rng)
            terrain_map.road_network = road_net
            logger.info(
                "  → %d primary, %d secondary, %d tertiary roads.",
                len(road_net.primary_roads),
                len(road_net.secondary_roads),
                len(road_net.tertiary_roads),
            )

        if build_wall:
            logger.info("Phase 7: generating wall layout...")
            from .wall_layout import generate_wall_layout as _gen_wall

            wall = _gen_wall(terrain_map)
            terrain_map.wall_layout = wall
            if wall is not None:
                logger.info(
                    "  → %d wall corners, %d gate(s).",
                    wall.n,
                    sum(1 for g in wall.gate_t if g is not None),
                )
            else:
                logger.warning("  → Wall layout could not be generated.")

        elapsed = time.perf_counter() - t0
        logger.info("Terrain segmentation complete in %.2f s.", elapsed)

        if visualize:
            from .visualizer import plot_terrain_map

            plot_terrain_map(
                terrain_map,
                heightmap=self._heightmap,
                show=False,
                save_path="terrain.png",
            )

        return terrain_map

    def _fetch_world_data(self) -> None:
        """
        Fetch all required arrays from a live Minecraft world via GDPC.
        """
        box = self._rect
        x0, z0 = box.offset.x, box.offset.z
        x1 = x0 + box.size.x - 1
        z1 = z0 + box.size.z - 1

        self._x0, self._z0 = x0, z0
        self._x1, self._z1 = x1, z1
        depth = z1 - z0 + 1
        width = x1 - x0 + 1

        logger.info("  Fetching world data for build area...")
        world_slice = self._editor.loadWorldSlice(cache=True)

        # interface.getHeightmap() returns shape (dx, dz) = (width, depth) indexed [lx, lz].
        # Transpose to (depth, width) = [lz, lx] to match our array convention throughout.
        floor_arr = interface.getHeightmap(heightmapType="OCEAN_FLOOR_NO_PLANTS").T
        surface_arr = interface.getHeightmap(
            heightmapType="MOTION_BLOCKING_NO_PLANTS"
        ).T

        self._heightmap = floor_arr
        self._surface_heightmap = surface_arr
        self._water_mask = (surface_arr - floor_arr >= 1).astype(np.bool_)

        biome_raw = np.zeros((depth, width), dtype=np.uint16)
        biome_str = np.empty((depth, width), dtype=object)
        for lz in range(depth):
            for lx in range(width):
                wx, wz = x0 + lx, z0 + lz
                wy = int(self._heightmap[lz, lx])
                b = self._editor.getBiome(ivec3(wx, wy, wz))
                biome_raw[lz, lx] = hash(b) % 65535
                biome_str[lz, lx] = b
        self._biome_map = biome_raw
        self._biome_str_map = biome_str

        tree_mask = np.zeros((depth, width), dtype=np.bool_)
        logger.info("  Scanning surface for trees and ice...")
        for lz in range(depth):
            for lx in range(width):
                wy_surface = int(surface_arr[lz, lx]) - 1
                # getBlock takes (local_x, global_y, local_z); y needs no offset.
                # Ice is solid, so the heightmap reads it as ground — detect the
                # surface block itself and treat frozen water as water.
                surface_block = world_slice.getBlock((lx, wy_surface, lz))
                sname = (
                    surface_block.id
                    if hasattr(surface_block, "id")
                    else str(surface_block)
                )
                if sname in ICE_BLOCK_NAMES:
                    self._water_mask[lz, lx] = True
                    continue
                block_above = world_slice.getBlock((lx, wy_surface + 1, lz))
                bname_above = (
                    block_above.id if hasattr(block_above, "id") else str(block_above)
                )
                if bname_above in TREE_BLOCK_NAMES:
                    tree_mask[lz, lx] = True

        self._tree_mask = tree_mask
