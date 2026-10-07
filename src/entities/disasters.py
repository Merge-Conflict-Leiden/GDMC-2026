"""
This module handles disaster entity spawning at the town centre.
"""

import random
from typing import Optional

from gdpc.editor import Editor

from terrain.terrain_types import TerrainMap
from utils import log

_DISASTER_TYPES = ["ender_dragon", "wither", "warden"]
_DISASTER_HEIGHT_OFFSET = {"ender_dragon": 80, "wither": 40, "warden": 1}
_DISASTER_CHANCE = 0.01


class DisasterGenerator:
    """Randomly spawn a disaster entity at the town centre (1% chance)."""

    def generate(self, editor: Editor, terrain_map: TerrainMap) -> Optional[str]:
        """
        Spawn a disaster entity at the town centre with 1% probability.

        Returns the entity type string if spawned, else None.
        """
        feat = terrain_map.features
        if feat is None or random.random() >= _DISASTER_CHANCE:
            return None

        disaster = random.choice(_DISASTER_TYPES)
        if terrain_map.town_center is not None:
            clz, clx = terrain_map.town_center
        else:
            depth, width = feat.shape
            clz, clx = depth // 2, width // 2

        wx, wz = terrain_map.local_to_world(clz, clx)
        surface_y = int(feat[clz, clx]["height"])
        spawn_y = surface_y + _DISASTER_HEIGHT_OFFSET[disaster]
        editor.runCommand("difficulty easy")
        editor.runCommand(f"summon minecraft:{disaster} {wx} {spawn_y} {wz}")
        log(f"[Disaster] {disaster} spawned at ({wx}, {spawn_y}, {wz})")
        return disaster
