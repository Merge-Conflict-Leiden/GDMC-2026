"""
This module defines a generator for flocks of birds.
"""

import logging
import math
import random
from typing import TYPE_CHECKING

import numpy as np
from gdpc import Editor

from blocks import BlockProcessor
from consts import BUILD_DIR, BUILD_OUTPUT_DIR
from structures import StructureCache, StructurePlacer
from terrain.terrain_types import ZoneType
from utils import rotate_offset

if TYPE_CHECKING:
    from terrain.terrain_types import TerrainMap

logger = logging.getLogger(__name__)


class FlockGenerator:
    """Place flocks of birds in V-formations above the terrain."""

    STRUCTURE_PATH = "bird/bird.csv"

    MIN_FLOCK_SIZE = 4
    MAX_FLOCK_SIZE = 7

    HEIGHT_ABOVE_TERRAIN_MIN = 40
    HEIGHT_ABOVE_TERRAIN_MAX = 65

    BASE_SPACING_MIN = 3
    BASE_SPACING_MAX = 5

    WING_SPREAD_MIN = 2
    WING_SPREAD_MAX = 3

    HORIZONTAL_JITTER = 1
    VERTICAL_JITTER = 2

    CURVE_FACTOR = 0.35

    def __init__(self):
        self.cache = StructureCache(BUILD_DIR, BUILD_OUTPUT_DIR)
        self.block_processor = BlockProcessor()
        self.placer = StructurePlacer(
            structure_cache=self.cache,
            block_processor=self.block_processor,
            block_palette=None,
            furniture_replacer=None,
        )

    def _place_bird(self, editor: Editor, direction: int, dx: int, dy: int, dz: int):
        with editor.pushTransform((dx, dy, dz)):
            self.placer.place_structure(
                editor, self.STRUCTURE_PATH, biome=None, direction=direction
            )

    def _place_flock(
        self,
        editor: Editor,
        direction: int,
        world_cx: int,
        world_cy: int,
        world_cz: int,
        *,
        flock_min: int | None = None,
        flock_max: int | None = None,
    ) -> None:
        """Place one flock centred at (world_cx, world_cy, world_cz)."""
        if world_cy + 1 >= 256:
            return

        lo = flock_min if flock_min is not None else self.MIN_FLOCK_SIZE
        hi = flock_max if flock_max is not None else self.MAX_FLOCK_SIZE
        flock_size = random.randint(lo, hi)
        base_spacing = random.randint(self.BASE_SPACING_MIN, self.BASE_SPACING_MAX)
        wing_spread = random.randint(self.WING_SPREAD_MIN, self.WING_SPREAD_MAX)

        with editor.pushTransform((world_cx, world_cy, world_cz)):
            # Lead bird
            self.placer.place_structure(
                editor, self.STRUCTURE_PATH, biome=None, direction=direction
            )

            placed = 1
            step = 1

            while placed < flock_size:
                forward = step * base_spacing
                curve = int(step * self.CURVE_FACTOR)

                for side in (-1, 1):
                    if placed >= flock_size:
                        break

                    dx = side * (step * wing_spread + curve)
                    dx += random.randint(
                        -self.HORIZONTAL_JITTER, self.HORIZONTAL_JITTER
                    )
                    dy = random.randint(0, self.VERTICAL_JITTER)
                    dz = forward + random.randint(
                        -self.HORIZONTAL_JITTER, self.HORIZONTAL_JITTER
                    )
                    dx, dz = rotate_offset(dx, dz, 1, 1, direction)

                    if world_cy + dy < 256:
                        self._place_bird(editor, direction, dx, dy, dz)

                    placed += 1

                step += 1

            # Trigger sculk sensors / bird animations
            editor.runCommand(
                "summon minecraft:splash_potion ~ ~ ~", position=(1, 3, 1)
            )

    def place_flocks(
        self,
        editor: Editor,
        terrain_map: "TerrainMap",
        *,
        num_flocks: int | None = None,
    ) -> int:
        """
        Spawn V-formations above and around the settlement.

        The flock count scales with CITY size (urban cell count), not raw map
        area: a hamlet gets a lone V or two while a metropolis draws many more
        scavenging flocks.  Map area only acts as a floor so an empty map
        still gets the occasional migrating V.  Most flocks anchor above the
        urban area itself; the rest roam the wider map.

        Headings are locally coherent: flocks are grouped into 1–3 migration
        streams and every flock follows the heading of its nearest stream, so
        neighbouring flocks always fly the same way (a small city keeps the
        old behaviour of one shared direction for all).

        Height is computed per flock from the local terrain maximum in a 16-cell
        radius so no flock clips into tall structures while also not floating
        unnecessarily high above flat terrain.
        """
        feat = terrain_map.features
        if feat is None:
            logger.warning("FlockGenerator: features not available, skipping birds.")
            return 0

        depth, width = feat.shape

        # City-size scale relative to the 128×128 baseline settlement
        # (~22 % urban of a 128×128 map ≈ 3600 urban cells → scale 1.0).
        zone_map = terrain_map.zone_map
        urban_pos: np.ndarray | None = None
        if zone_map is not None:
            urban_pos = np.argwhere(zone_map == int(ZoneType.URBAN))
            if len(urban_pos) == 0:
                urban_pos = None
        urban_cells = 0 if urban_pos is None else len(urban_pos)

        area_scale = math.sqrt(depth * width / (128 * 128))
        city_scale = math.sqrt(urban_cells / 3600.0)
        scale = max(city_scale, 0.5 * area_scale)

        if num_flocks is None:
            # Grows with the city, hard-capped so the sky never saturates:
            # baseline city → 2 flocks, 4× the urban area → 4, 16× → 8.
            num_flocks = max(1, min(12, round(2 * max(scale, 0.5))))
        # A V-formation stops reading as a V past a dozen birds — cap size.
        size_scale = min(max(scale, 1.0), 2.0)
        flock_min = max(3, round(self.MIN_FLOCK_SIZE * size_scale))
        flock_max = max(flock_min + 2, round(self.MAX_FLOCK_SIZE * size_scale))

        # Locally coherent headings: a few migration streams, each with its
        # own direction; every flock adopts the nearest stream's heading.
        n_streams = 1 if num_flocks <= 4 else (2 if num_flocks <= 8 else 3)
        stream_dirs = random.sample(range(4), n_streams)
        stream_anchors = [
            (random.uniform(0, depth), random.uniform(0, width))
            for _ in range(n_streams)
        ]

        placed = 0
        margin = max(15, min(depth, width) // 8)
        z_hi = max(margin, depth - margin - 1)
        x_hi = max(margin, width - margin - 1)
        city_bias = 0.7  # fraction of flocks anchored above the city

        for _ in range(num_flocks):
            if urban_pos is not None and random.random() < city_bias:
                # Above (or drifting near) the urban area.
                uz, ux = urban_pos[random.randrange(len(urban_pos))]
                lz = min(max(int(uz) + random.randint(-24, 24), margin), z_hi)
                lx = min(max(int(ux) + random.randint(-24, 24), margin), x_hi)
            else:
                lz = random.randint(margin, z_hi)
                lx = random.randint(margin, x_hi)

            d2 = [(lz - az) ** 2 + (lx - ax) ** 2 for az, ax in stream_anchors]
            direction = stream_dirs[d2.index(min(d2))]

            # Local terrain ceiling in a ~16-cell radius
            r = 16
            lz0 = max(0, lz - r)
            lz1 = min(depth, lz + r + 1)
            lx0 = max(0, lx - r)
            lx1 = min(width, lx + r + 1)
            local_max_y = int(feat["height"][lz0:lz1, lx0:lx1].max())

            clearance = random.randint(
                self.HEIGHT_ABOVE_TERRAIN_MIN, self.HEIGHT_ABOVE_TERRAIN_MAX
            )
            world_cy = local_max_y + clearance

            if world_cy >= 255:
                continue

            world_cx, world_cz = terrain_map.local_to_world(lz, lx)
            self._place_flock(
                editor,
                direction,
                world_cx,
                world_cy,
                world_cz,
                flock_min=flock_min,
                flock_max=flock_max,
            )
            placed += 1
            logger.debug(
                "FlockGenerator: flock %d at world(%d,%d,%d) dir=%d",
                placed,
                world_cx,
                world_cy,
                world_cz,
                direction,
            )

        logger.info(
            "FlockGenerator: placed %d flocks in %d stream(s) (directions=%s, "
            "urban_cells=%d).",
            placed,
            n_streams,
            stream_dirs,
            urban_cells,
        )
        return placed
