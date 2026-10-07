"""
Block parsing, filtering, and biome-aware transformation utilities.

BlockProcessor turns one already-placed block into another (or into air)
so that hand-authored structures and procedurally generated terrain read
as if they were built from locally available materials.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Union

from gdpc.block import Block

_AIR = Block("minecraft:air")

# A single pattern's substitution rule: biome substring -> either a
# replacement block ID, a list of candidate IDs (one is chosen at
# random), or None to mean "leave the block as-is in this biome".
BiomeRuleMap = Dict[str, Union[Optional[str], List[str]]]
# category name -> {pattern -> BiomeRuleMap}
BiomePalettes = Dict[str, Dict[str, BiomeRuleMap]]


def _matches_pattern(block_id: str, pattern: str) -> bool:
    """
    Match a block ID against a pattern (prefix wildcard *_suffix or exact).

    :param block_id: The block ID to match
    :param pattern: The pattern to match
    :returns: Whether or not the pattern matches
    """
    if pattern.startswith("*"):
        return block_id.endswith(pattern[1:])
    return block_id == pattern


def _make_block(
    block_id: str,
    states: Optional[Dict[str, str]] = None,
    data: Optional[str] = None,
) -> Block:
    """
    Construct a Block, automatically adding persistent=true for leaf blocks.

    :param block_id: The block ID to make
    :param states: Optional block states to apply
    :param data: Optional SNBT block-entity data string
    :returns: Constructed Block
    """
    s = dict(states) if states else {}
    if "leaves" in block_id:
        s.setdefault("persistent", "true")
    return Block(block_id, states=s, data=data)


class BlockProcessor:
    """
    Parse block strings and adapt existing blocks to a target biome.

    Every public transformation method is a pure block-to-block (or
    block-to-air) mapping, which keeps this class reusable across
    structure placement, terrain passes, and decoration alike, and keeps
    generation reproducible when seeded (see ``rng`` below).
    """

    # Mossy cobblestone / stone-brick family, keyed by their vanilla ID.
    _MOSSY: Dict[str, str] = {
        "cobblestone": "mossy_cobblestone",
        "cobblestone_wall": "mossy_cobblestone_wall",
        "cobblestone_stairs": "mossy_cobblestone_stairs",
        "cobblestone_slab": "mossy_cobblestone_slab",
        "stone_bricks": "mossy_stone_bricks",
        "stone_brick_stairs": "mossy_stone_brick_stairs",
        "stone_brick_slab": "mossy_stone_brick_slab",
        "stone_brick_wall": "mossy_stone_brick_wall",
    }

    # Pattern-based palette for biome-dependent substitutions. Currently
    # covers leaves, whose IDs follow "<species>_leaves".
    DEFAULT_BIOME_PALETTES: BiomePalettes = {
        "leaves": {
            "*_leaves": {
                "cherry": "cherry_leaves",
                "jungle": "jungle_leaves",
                "birch": "birch_leaves",
                "snow": "spruce_leaves",
                "ice": "spruce_leaves",
                "grove": "spruce_leaves",
                "cold": "spruce_leaves",
                "frozen": "spruce_leaves",
                "taiga": "spruce_leaves",
                "savanna": "acacia_leaves",
                "badlands": "pale_oak_leaves",
                "pale": "pale_oak_leaves",
                "desert": None,
                "swamp": "mangrove_leaves",
                "dark_forest": "dark_oak_leaves",
            }
        },
    }

    def __init__(
        self,
        stochastic_removals: Optional[Dict[str, float]] = None,
        biome_replacements: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
        biome_palettes: Optional[BiomePalettes] = None,
        rng: Optional[random.Random] = None,
    ):
        """
        :param stochastic_removals: pattern -> probability of turning a
            matching block into air, independent of biome
        :param biome_replacements: block_id -> {biome substring ->
            replacement block ID or None}, for one-off overrides that
            don't fit the palette system
        :param biome_palettes: overrides ``DEFAULT_BIOME_PALETTES``
        :param rng: source of randomness. Defaults to a private
            ``random.Random()`` instance rather than the ``random``
            module's global state, so a caller can pass a seeded instance
            to make a generation run reproducible -- useful for GDMC
            submissions, where the same generator may be re-run against
            the same map during judging.
        """
        self.stochastic_removals = stochastic_removals or {}
        self.biome_replacements = biome_replacements or {}
        self.biome_palettes = biome_palettes or self.DEFAULT_BIOME_PALETTES
        self._rng = rng or random.Random()

        self._palette_index: Dict[str, BiomeRuleMap] = {}
        for palette in self.biome_palettes.values():
            for pattern, replacements in palette.items():
                self._palette_index.setdefault(pattern, {}).update(replacements)

    @staticmethod
    def build_block(
        block_id: str,
        states: Optional[Dict[str, str]] = None,
        data: Optional[str] = None,
    ) -> Block:
        """
        Construct a Block from its three CSV components.

        Leaves are made persistent by default.

        :param block_id: The block ID (e.g. ``minecraft:oak_sign``)
        :param states: Block state dict (e.g. ``{"facing": "south"}``)
        :param data: Optional SNBT block-entity data string
        :returns: Constructed Block
        """
        return _make_block(block_id, states, data)

    @staticmethod
    def parse_block(block_string: str, data: Optional[str] = None) -> Block:
        """
        Parse a combined block string into a Block object.

        Handles the legacy ``"minecraft:block[key=val,...]"`` format.
        Leaves are made persistent by default.

        :param block_string: The block string to parse
        :param data: Optional SNBT block-entity data string
        :returns: Parsed Block
        :raises ValueError: if a state entry isn't a ``key=value`` pair
        """
        states: Dict[str, str] = {}

        if "[" in block_string:
            block_string, state_str = block_string.split("[", 1)
            state_str = state_str.rstrip("]")
            for item in state_str.split(","):
                if "=" not in item:
                    raise ValueError(
                        f"Malformed block state {item!r} in "
                        f"{block_string!r}: expected 'key=value'"
                    )
                key, _, value = item.partition("=")
                states[key] = value

        return BlockProcessor.build_block(block_string, states, data)

    def apply_stochastic_filter(self, block: Block) -> Block:
        """
        Randomly remove certain blocks.

        Patterns can be exact IDs or suffix patterns like '_leaves'.

        :param block: The block to apply the filter to
        :returns: Air block if stochastically removed, else the unmodified block
        """
        if block.id is None:
            raise ValueError("Cannot filter a block with no ID")
        block_id = block.id.removeprefix("minecraft:")

        for pattern, probability in self.stochastic_removals.items():
            if _matches_pattern(block_id, pattern) and self._rng.random() < probability:
                return _AIR

        return block

    def randomize_candle_count(self, block: Block) -> Block:
        """
        Re-roll the cluster size of a candle block (1-4 candles).

        Candle COLOUR is a per-building choice made by the BlockPalette so
        clusters inside one build stay coordinated; the count is re-rolled
        uniformly per block here so repeated schematics don't show the exact
        same candle arrangement.  All other states (lit, waterlogged) and
        non-candle blocks are left untouched.

        :param block: The block to process
        :returns: The block with a randomized ``candles`` state if it is a
            candle, else the unmodified block
        """
        if block.id is None:
            raise ValueError("Cannot process a block with no ID")
        block_id = block.id.removeprefix("minecraft:")
        if block_id != "candle" and not block_id.endswith("_candle"):
            return block

        states = dict(block.states)
        states["candles"] = str(self._rng.randint(1, 4))
        return Block(block.id, states=states, data=block.data)

    def apply_biome_adjustments(self, block: Block, biome: Optional[str]) -> Block:
        """
        Apply biome-specific block substitutions.

        Precedence: stone weathering (moss / cracking), the pattern-based
        palette, then user-supplied per-block overrides.

        :param block: The block to apply biome adjustments to
        :param biome: The biome to adjust to
        :returns: The adjusted Block
        """
        if biome is None:
            return block
        if block.id is None:
            raise ValueError("Cannot adjust a block with no ID")

        block_id = block.id.removeprefix("minecraft:")
        biome = biome.lower()
        states = dict(block.states)
        data = block.data

        if ("jungle" in biome or "swamp" in biome) and block_id in self._MOSSY:
            if self._rng.random() < 0.35:
                return Block(self._MOSSY[block_id], states=states, data=data)

        if (
            "desert" in biome
            and block_id == "stone_bricks"
            and self._rng.random() < 0.20
        ):
            return Block("cracked_stone_bricks", states=states, data=data)

        for pattern, replacements in self._palette_index.items():
            if not _matches_pattern(block_id, pattern):
                continue
            for biome_key, repl in replacements.items():
                if biome_key not in biome:
                    continue
                if repl is None:
                    return block
                repl = repl if isinstance(repl, str) else self._rng.choice(repl)
                return _make_block(repl, states=states, data=data)

        for biome_key, replacement in self.biome_replacements.get(block_id, {}).items():
            if biome_key not in biome:
                continue
            if replacement is None:
                return block
            return _make_block(replacement, states=states, data=data)

        return block
