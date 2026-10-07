"""
Biome-driven material palettes for schematic buildings.

Two lookups, both keyed by substring match against the district's dominant
biome:
  foundation_palette_for — blocks used to level/underfill a footprint
  building_palette_for   — BlockPalette swaps applied to the schematic itself
"""

from __future__ import annotations

import random

from gdpc.block import Block

from blocks.block_palette import BlockPalette

# ---------------------------------------------------------------------------
# Building material palettes
# ---------------------------------------------------------------------------

# Blocks used as the base "accent material" group in civil schematics.
_BASE_ACCENT = ["terracotta", "brown_wool", "granite"]

_DEFAULT_BUILDING_PALETTE = BlockPalette(
    unidirectional_palettes={},
    bidirectional_palettes={"main": _BASE_ACCENT},
    block_to_palette={},
)

# ---------------------------------------------------------------------------
# Foundation palettes
# ---------------------------------------------------------------------------

_BIOME_FOUNDATIONS: dict[str, list[Block]] = {
    "cherry": [
        Block("minecraft:stone_bricks"),
        Block("minecraft:mossy_stone_bricks"),
        Block("minecraft:andesite"),
    ],
    "jungle": [
        Block("minecraft:mossy_cobblestone"),
        Block("minecraft:mossy_stone_bricks"),
        Block("minecraft:stone"),
    ],
    "taiga": [
        Block("minecraft:cobblestone"),
        Block("minecraft:stone"),
        Block("minecraft:andesite"),
    ],
    "snowy": [
        Block("minecraft:stone_bricks"),
        Block("minecraft:stone"),
        Block("minecraft:cobblestone"),
    ],
    "frozen": [
        Block("minecraft:stone_bricks"),
        Block("minecraft:stone"),
        Block("minecraft:cobblestone"),
    ],
    "savanna": [
        Block("minecraft:terracotta"),
        Block("minecraft:mud_bricks"),
        Block("minecraft:rooted_dirt"),
    ],
    "badlands": [
        Block("minecraft:terracotta"),
        Block("minecraft:orange_terracotta"),
        Block("minecraft:rooted_dirt"),
    ],
    "birch": [
        Block("minecraft:stone_bricks"),
        Block("minecraft:andesite"),
        Block("minecraft:mossy_stone_bricks"),
    ],
    "plains": [
        Block("minecraft:cobblestone"),
        Block("minecraft:stone_bricks"),
        Block("minecraft:andesite"),
    ],
    "desert": [
        Block("minecraft:sandstone"),
        Block("minecraft:smooth_sandstone"),
        Block("minecraft:red_sandstone"),
    ],
    "mesa": [
        Block("minecraft:terracotta"),
        Block("minecraft:orange_terracotta"),
        Block("minecraft:rooted_dirt"),
    ],
    "swamp": [
        Block("minecraft:mossy_cobblestone"),
        Block("minecraft:mud_bricks"),
        Block("minecraft:stone"),
    ],
    "ocean": [
        Block("minecraft:cobblestone"),
        Block("minecraft:prismarine"),
        Block("minecraft:stone"),
    ],
    "coast": [
        Block("minecraft:cobblestone"),
        Block("minecraft:sandstone"),
        Block("minecraft:stone"),
    ],
    "mountain": [
        Block("minecraft:stone"),
        Block("minecraft:andesite"),
        Block("minecraft:calcite"),
    ],
    "meadow": [
        Block("minecraft:stone_bricks"),
        Block("minecraft:andesite"),
        Block("minecraft:mossy_cobblestone"),
    ],
    "dark_forest": [
        Block("minecraft:cobblestone"),
        Block("minecraft:mossy_cobblestone"),
        Block("minecraft:stone_bricks"),
    ],
    "mushroom": [
        Block("minecraft:mycelium"),
        Block("minecraft:podzol"),
        Block("minecraft:coarse_dirt"),
    ],
    "default": [
        Block("minecraft:cobblestone"),
        Block("minecraft:stone"),
        Block("minecraft:stone_bricks"),
    ],
}


# Biome-specific overrides for houses: any key substring-matching the biome hint wins.
# To add a new biome, insert a new entry here.
_BIOME_BUILDING_PALETTES: dict[str, BlockPalette] = {
    "cherry": BlockPalette(
        unidirectional_palettes={
            "main": ["cherry_planks", "white_terracotta", "birch_planks"],
            "leaves": ["cherry_leaves"],
        },
        bidirectional_palettes={},
        block_to_palette={
            **{b: "main" for b in _BASE_ACCENT},
            **{"cherry_leaves": "leaves"},
        },
    ),
    # Sandy, dry materials for desert biomes.
    "desert": BlockPalette(
        unidirectional_palettes={
            "main": [
                "sandstone",
                "smooth_sandstone",
                "cut_sandstone",
                "chiseled_sandstone",
                "red_sandstone",
            ],
        },
        bidirectional_palettes={},
        block_to_palette={b: "main" for b in _BASE_ACCENT},
    ),
    # Cold, rocky materials for taiga and similar biomes.
    "taiga": BlockPalette(
        unidirectional_palettes={
            "main": [
                "andesite",
                "polished_andesite",
                "diorite",
                "stone_bricks",
                "cobblestone",
            ],
        },
        bidirectional_palettes={},
        block_to_palette={b: "main" for b in _BASE_ACCENT},
    ),
    # Sparse, utilitarian stone palette for snowy / frozen biomes.
    "snowy": BlockPalette(
        unidirectional_palettes={
            "main": [
                "stone_bricks",
                "cobblestone",
                "andesite",
                "cracked_stone_bricks",
            ],
        },
        bidirectional_palettes={},
        block_to_palette={b: "main" for b in _BASE_ACCENT},
    ),
    # Mossy and lush materials for jungle biomes.
    "jungle": BlockPalette(
        unidirectional_palettes={
            "main": [
                "mossy_cobblestone",
                "mossy_stone_bricks",
                "mud_bricks",
                "jungle_planks",
            ],
            "leaves": ["jungle_leaves"],
        },
        bidirectional_palettes={},
        block_to_palette={
            **{b: "main" for b in _BASE_ACCENT},
            **{"oak_leaves": "leaves", "spruce_leaves": "leaves"},
        },
    ),
    # Mud and mossy materials for swamp biomes.
    "swamp": BlockPalette(
        unidirectional_palettes={
            "main": [
                "mud_bricks",
                "packed_mud",
                "mossy_cobblestone",
                "mossy_stone_bricks",
            ],
            "leaves": ["mangrove_leaves"],
        },
        bidirectional_palettes={},
        block_to_palette={
            **{b: "main" for b in _BASE_ACCENT},
            **{"oak_leaves": "leaves", "spruce_leaves": "leaves"},
        },
    ),
}

# Biome-specific overrides for the castle category: any key substring-matching the biome hint wins.
# To add a new biome, insert a new entry here.
_CASTLE_BUILDING_PALLETS: dict[str, BlockPalette] = {
    "cherry": BlockPalette(
        unidirectional_palettes={
            "planks": ["stripped_cherry_wood", "cherry_planks", "birch_planks"],
            "trapdoor": ["cherry_trapdoor", "birch_trapdoor"],
            "logs": ["cherry_log", "stripped_cherry_log"],
            "stairs": ["cherry_stairs", "birch_stairs"],
            "slabs": ["cherry_slab", "birch_slab"],
            "tree": ["cherry_log"],
            "leaves": ["cherry_leaves"],
        },
        bidirectional_palettes={},
        block_to_palette={
            "oak_planks": "planks",
            "spruce_planks": "planks",
            "stripped_oak_wood": "planks",
            "spruce_trapdoor": "trapdoor",
            "oak_trapdoor": "trapdoor",
            "oak_log": "logs",
            "stripped_oak_log": "logs",
            "stripped_spruce_log": "logs",
            "spruce_stairs": "stairs",
            "oak_stairs": "stairs",
            "spruce_slab": "slabs",
            "oak_slab": "slabs",
            "oak_wood": "tree",
            "spruce_leaves": "leaves",
        },
    ),
}


def foundation_palette_for(biome: str) -> list[Block]:
    for key, palette in _BIOME_FOUNDATIONS.items():
        if key != "default" and key in biome:
            return palette
    return _BIOME_FOUNDATIONS["default"]


_ALL_CARPETS = [
    "white_carpet",
    "orange_carpet",
    "magenta_carpet",
    "light_blue_carpet",
    "yellow_carpet",
    "lime_carpet",
    "pink_carpet",
    "gray_carpet",
    "light_gray_carpet",
    "cyan_carpet",
    "purple_carpet",
    "blue_carpet",
    "brown_carpet",
    "green_carpet",
    "red_carpet",
    "black_carpet",
    "moss_carpet",
]

_CARPET_HUE_GROUPS = [
    ["pink_carpet", "magenta_carpet", "purple_carpet", "light_gray_carpet"],
    ["light_blue_carpet", "cyan_carpet", "blue_carpet", "white_carpet"],
    ["orange_carpet", "yellow_carpet", "brown_carpet", "red_carpet"],
    ["lime_carpet", "green_carpet", "moss_carpet", "cyan_carpet"],
    ["white_carpet", "light_gray_carpet", "gray_carpet", "black_carpet"],
]

_ALL_CANDLES = [
    "candle",
    "white_candle",
    "orange_candle",
    "magenta_candle",
    "light_blue_candle",
    "yellow_candle",
    "lime_candle",
    "pink_candle",
    "gray_candle",
    "light_gray_candle",
    "cyan_candle",
    "purple_candle",
    "blue_candle",
    "brown_candle",
    "green_candle",
    "red_candle",
    "black_candle",
]

_ALL_GLAZED_TERRACOTTA = [
    "white_glazed_terracotta",
    "orange_glazed_terracotta",
    "magenta_glazed_terracotta",
    "light_blue_glazed_terracotta",
    "yellow_glazed_terracotta",
    "lime_glazed_terracotta",
    "pink_glazed_terracotta",
    "gray_glazed_terracotta",
    "light_gray_glazed_terracotta",
    "cyan_glazed_terracotta",
    "purple_glazed_terracotta",
    "blue_glazed_terracotta",
    "brown_glazed_terracotta",
    "green_glazed_terracotta",
    "red_glazed_terracotta",
    "black_glazed_terracotta",
]


def building_palette_for(biome: str, category: str, csv_path: str = "") -> BlockPalette:
    if category == "castle":
        for key, palette in _CASTLE_BUILDING_PALLETS.items():
            if key in biome:
                base = palette
                break
        else:
            base = _DEFAULT_BUILDING_PALETTE
    else:
        for key, palette in _BIOME_BUILDING_PALETTES.items():
            if key in biome:
                base = palette
                break
        else:
            base = _DEFAULT_BUILDING_PALETTE

    # Pick one glazed terracotta colour for the whole placement so the
    # directional pattern tiles remain aligned across all blocks.
    chosen_terracotta = random.choice(_ALL_GLAZED_TERRACOTTA)
    extra_uni: dict[str, list[str]] = {"glazed_terracotta": [chosen_terracotta]}
    extra_b2p: dict[str, str] = {
        gt: "glazed_terracotta" for gt in _ALL_GLAZED_TERRACOTTA
    }

    # One candle colour (plain included) per building keeps its clusters
    # coordinated; cluster SIZE is re-rolled per block by
    # BlockProcessor.randomize_candle_count.
    chosen_candle = random.choice(_ALL_CANDLES)
    extra_uni["candle"] = [chosen_candle]
    extra_b2p.update({c: "candle" for c in _ALL_CANDLES})

    if "house_small" in csv_path:
        chosen_hue = random.choice(_CARPET_HUE_GROUPS)
        extra_uni["carpet"] = chosen_hue
        extra_b2p.update({c: "carpet" for c in _ALL_CARPETS})

    return BlockPalette(
        unidirectional_palettes={**dict(base.unidirectional_palettes), **extra_uni},
        bidirectional_palettes=dict(base.bidirectional_palettes),
        block_to_palette={**dict(base.block_to_palette), **extra_b2p},
    )
