"""
Context-aware container loot for chests and barrels.

Every chest/barrel we place used to spawn empty.  This module fills them with a
small, randomised set of items that (a) match the building/structure/zone they
sit in and (b) scale in value with the wealth of that place — a castle strongbox
should out-glitter a cottage cupboard, and a barrel on a random street corner
should hold next to nothing worth taking.

Design
------
* **Themes** decide *which* items can appear (a smith's chest holds ingots and
  tools; a fisherman's holds cod and prismarine).  Each theme is layered on top
  of a shared GENERIC pool so every container has some plausible odds-and-ends.
* **Tiers** decide *how good and how much*: the number of filled slots and the
  probability of drawing from each rarity bucket.  Rarity runs
  common → uncommon → rare → treasure, and the *treasure* odds are deliberately
  tiny (0 % for poor places, a few percent for a castle) so that across the many
  containers a settlement contains, a genuinely good find stays a rare thrill.

Only plain item stacks are emitted (id + count) — no enchantment/component NBT —
so the output is robust across game versions.  "Properly good" treasure is
expressed through inherently valuable *items* (diamonds, netherite scrap, a
diamond blade, a golden apple), not through enchantments.

The public entry point is :func:`container_loot_data`, which returns an SNBT
string ready to hand to ``Block(..., data=...)`` (the same mechanism the crypt
and maze already use), or ``None`` for a container that should stay empty.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

# An item pool entry: (item_id_without_namespace, min_count, max_count).
Item = Tuple[str, int, int]
# rarity -> list of items
Pool = Dict[str, List[Item]]

RARITIES = ("common", "uncommon", "rare", "treasure")

# Single chest / barrel both expose 27 slots.
_CONTAINER_SLOTS = 27


# ---------------------------------------------------------------------------
# Item pools
# ---------------------------------------------------------------------------
# Shared base merged into every theme so no container feels sterile.
_GENERIC: Pool = {
    "common": [
        ("stick", 2, 8),
        ("bread", 1, 4),
        ("coal", 1, 6),
        ("cobblestone", 4, 16),
        ("oak_planks", 4, 16),
        ("string", 1, 5),
        ("leather", 1, 3),
        ("apple", 1, 3),
        ("bone", 1, 4),
        ("flint", 1, 4),
        ("torch", 2, 8),
        ("clay_ball", 2, 6),
    ],
    "uncommon": [
        ("iron_nugget", 3, 9),
        ("iron_ingot", 1, 3),
        ("paper", 2, 6),
        ("book", 1, 2),
        ("arrow", 4, 12),
        ("bowl", 1, 3),
        ("cooked_beef", 1, 3),
        ("redstone", 2, 6),
        ("copper_ingot", 1, 4),
        ("glass_bottle", 1, 3),
    ],
    "rare": [
        ("gold_ingot", 1, 3),
        ("emerald", 1, 2),
        ("lapis_lazuli", 2, 6),
        ("iron_block", 1, 1),
        ("experience_bottle", 1, 3),
    ],
    "treasure": [
        ("diamond", 1, 2),
        ("golden_apple", 1, 1),
        ("emerald", 2, 4),
    ],
}

_THEMES: Dict[str, Pool] = {
    "generic": {},  # generic-only; base pool covers it
    "home": {
        "common": [
            ("bread", 2, 5),
            ("carrot", 2, 6),
            ("potato", 2, 6),
            ("wheat", 3, 9),
            ("egg", 1, 4),
            ("beetroot", 2, 6),
            ("cookie", 2, 6),
            ("wheat_seeds", 2, 6),
            ("white_wool", 1, 3),
        ],
        "uncommon": [
            ("milk_bucket", 1, 1),
            ("cooked_chicken", 1, 3),
            ("pumpkin_pie", 1, 2),
            ("honey_bottle", 1, 2),
            ("cooked_beef", 1, 3),
        ],
        "rare": [("golden_carrot", 1, 3)],
    },
    "kitchen": {
        "common": [
            ("bread", 2, 6),
            ("carrot", 3, 8),
            ("potato", 3, 8),
            ("beetroot", 2, 6),
            ("egg", 2, 5),
            ("cookie", 3, 8),
            ("mushroom_stew", 1, 1),
            ("wheat", 3, 9),
        ],
        "uncommon": [
            ("cake", 1, 1),
            ("cooked_beef", 2, 4),
            ("cooked_chicken", 2, 4),
            ("cooked_salmon", 1, 3),
            ("golden_carrot", 1, 2),
            ("honey_bottle", 1, 3),
            ("sweet_berries", 2, 6),
        ],
        "rare": [("golden_apple", 1, 1)],
        "treasure": [("enchanted_golden_apple", 1, 1)],
    },
    "farm": {
        "common": [
            ("wheat", 4, 16),
            ("wheat_seeds", 4, 12),
            ("hay_block", 1, 4),
            ("carrot", 3, 9),
            ("potato", 3, 9),
            ("beetroot_seeds", 3, 9),
            ("egg", 2, 6),
            ("bone_meal", 2, 8),
            ("pumpkin", 1, 3),
            ("melon_slice", 2, 6),
            ("white_wool", 1, 4),
        ],
        "uncommon": [
            ("apple", 2, 6),
            ("lead", 1, 2),
            ("shears", 1, 1),
            ("golden_carrot", 1, 2),
            ("hay_block", 3, 6),
        ],
        "rare": [("golden_carrot", 3, 5), ("saddle", 1, 1)],
    },
    "fish": {
        "common": [
            ("cod", 2, 6),
            ("salmon", 2, 6),
            ("string", 2, 8),
            ("kelp", 3, 9),
            ("ink_sac", 1, 4),
            ("bamboo", 2, 8),
        ],
        "uncommon": [
            ("cooked_cod", 2, 5),
            ("cooked_salmon", 2, 5),
            ("prismarine_shard", 1, 4),
            ("prismarine_crystals", 1, 3),
            ("tropical_fish", 1, 3),
            ("pufferfish", 1, 2),
            ("fishing_rod", 1, 1),
            ("nautilus_shell", 1, 1),
        ],
        "rare": [("nautilus_shell", 2, 3), ("emerald", 1, 2)],
        "treasure": [("heart_of_the_sea", 1, 1), ("trident", 1, 1)],
    },
    "fletcher": {
        "common": [
            ("arrow", 6, 16),
            ("feather", 2, 8),
            ("string", 2, 8),
            ("flint", 2, 6),
            ("stick", 4, 12),
        ],
        "uncommon": [
            ("bow", 1, 1),
            ("crossbow", 1, 1),
            ("arrow", 12, 24),
            ("spectral_arrow", 2, 6),
        ],
        "rare": [("tipped_arrow", 2, 5), ("emerald", 1, 2)],
    },
    "leather": {
        "common": [
            ("leather", 2, 6),
            ("rabbit_hide", 2, 6),
            ("string", 2, 6),
        ],
        "uncommon": [
            ("leather_helmet", 1, 1),
            ("leather_chestplate", 1, 1),
            ("leather_boots", 1, 1),
            ("saddle", 1, 1),
        ],
        "rare": [("emerald", 1, 3)],
    },
    "smith": {
        "common": [
            ("coal", 3, 9),
            ("iron_nugget", 4, 12),
            ("flint", 2, 6),
            ("cobblestone", 8, 16),
            ("gravel", 4, 12),
            ("raw_iron", 2, 6),
        ],
        "uncommon": [
            ("iron_ingot", 2, 5),
            ("iron_pickaxe", 1, 1),
            ("iron_axe", 1, 1),
            ("iron_sword", 1, 1),
            ("iron_shovel", 1, 1),
            ("iron_helmet", 1, 1),
            ("copper_ingot", 3, 8),
            ("redstone", 3, 9),
        ],
        "rare": [
            ("gold_ingot", 2, 4),
            ("diamond", 1, 1),
            ("iron_block", 1, 2),
        ],
        "treasure": [
            ("diamond", 2, 3),
            ("diamond_pickaxe", 1, 1),
            ("diamond_sword", 1, 1),
            ("netherite_scrap", 1, 1),
        ],
    },
    "mason": {
        "common": [
            ("stone", 6, 16),
            ("cobblestone", 8, 16),
            ("clay_ball", 4, 12),
            ("stone_bricks", 4, 12),
            ("gravel", 4, 12),
            ("sand", 4, 12),
        ],
        "uncommon": [
            ("bricks", 2, 8),
            ("smooth_stone", 4, 10),
            ("terracotta", 2, 8),
            ("chiseled_stone_bricks", 1, 4),
        ],
        "rare": [("emerald", 1, 2), ("quartz", 3, 8)],
    },
    "scholar": {
        "common": [
            ("paper", 4, 12),
            ("book", 2, 5),
            ("ink_sac", 1, 4),
            ("feather", 1, 5),
            ("stick", 2, 6),
        ],
        "uncommon": [
            ("writable_book", 1, 2),
            ("map", 1, 2),
            ("compass", 1, 1),
            ("bookshelf", 1, 3),
            ("glass_bottle", 1, 3),
            ("spyglass", 1, 1),
        ],
        "rare": [
            ("experience_bottle", 2, 4),
            ("emerald", 1, 2),
            ("lapis_lazuli", 4, 9),
        ],
    },
    "church": {
        "common": [
            ("candle", 1, 4),
            ("white_candle", 1, 2),
            ("glowstone_dust", 1, 4),
            ("bread", 1, 3),
            ("glass_bottle", 1, 3),
        ],
        "uncommon": [
            ("glowstone", 1, 3),
            ("gold_nugget", 3, 9),
            ("golden_carrot", 1, 2),
            ("honey_bottle", 1, 2),
            ("ender_pearl", 1, 1),
        ],
        "rare": [
            ("gold_ingot", 2, 4),
            ("emerald", 1, 3),
            ("golden_apple", 1, 1),
        ],
        "treasure": [
            ("enchanted_golden_apple", 1, 1),
            ("gold_block", 1, 1),
        ],
    },
    "market": {
        "common": [
            ("wheat", 3, 9),
            ("carrot", 2, 6),
            ("egg", 2, 6),
            ("leather", 1, 4),
            ("paper", 2, 6),
            ("sugar", 2, 6),
            ("white_wool", 1, 4),
        ],
        "uncommon": [
            ("emerald", 1, 2),
            ("gold_nugget", 4, 12),
            ("copper_ingot", 2, 6),
            ("book", 1, 3),
            ("glass", 2, 6),
        ],
        "rare": [
            ("emerald", 2, 5),
            ("gold_ingot", 2, 4),
            ("diamond", 1, 1),
        ],
        "treasure": [
            ("emerald_block", 1, 1),
            ("diamond", 1, 2),
        ],
    },
    "castle": {
        "common": [
            ("bread", 2, 5),
            ("arrow", 6, 16),
            ("coal", 3, 9),
            ("iron_nugget", 4, 12),
            ("cobblestone", 8, 16),
            ("torch", 4, 12),
        ],
        "uncommon": [
            ("iron_ingot", 2, 6),
            ("iron_sword", 1, 1),
            ("iron_chestplate", 1, 1),
            ("iron_helmet", 1, 1),
            ("iron_leggings", 1, 1),
            ("iron_boots", 1, 1),
            ("shield", 1, 1),
            ("crossbow", 1, 1),
            ("arrow", 8, 24),
            ("cooked_beef", 2, 5),
            ("golden_carrot", 2, 5),
        ],
        "rare": [
            ("gold_ingot", 2, 5),
            ("diamond", 1, 2),
            ("emerald", 2, 5),
            ("gold_block", 1, 1),
            ("diamond_helmet", 1, 1),
        ],
        "treasure": [
            ("diamond", 2, 4),
            ("diamond_sword", 1, 1),
            ("diamond_chestplate", 1, 1),
            ("netherite_scrap", 1, 2),
            ("enchanted_golden_apple", 1, 1),
            ("gold_block", 1, 2),
        ],
    },
}


# ---------------------------------------------------------------------------
# Tiers (wealth)
# ---------------------------------------------------------------------------
class _Tier:
    __slots__ = ("slots", "weights")

    def __init__(
        self,
        slots: Tuple[int, int],
        weights: Dict[str, float],
    ):
        self.slots = slots  # (min, max) filled slots
        self.weights = weights  # rarity -> selection weight


# Treasure weights are intentionally tiny: with a whole settlement of chests, a
# real find should stay uncommon.  Poor places never yield treasure at all.
_TIERS: Dict[str, _Tier] = {
    "poor": _Tier(
        (0, 3),
        {"common": 84.0, "uncommon": 14.0, "rare": 2.0, "treasure": 0.0},
    ),
    "modest": _Tier(
        (1, 5),
        {"common": 66.0, "uncommon": 27.0, "rare": 6.6, "treasure": 0.4},
    ),
    "rich": _Tier(
        (2, 6),
        {"common": 50.0, "uncommon": 33.0, "rare": 15.5, "treasure": 1.5},
    ),
    "noble": _Tier(
        (3, 8),
        {"common": 38.0, "uncommon": 36.0, "rare": 22.5, "treasure": 3.5},
    ),
}


# ---------------------------------------------------------------------------
# Category → (theme, tier)
# ---------------------------------------------------------------------------
# `house` is refined by schematic filename (small/medium/large) in _resolve.
_CONTEXT: Dict[str, Tuple[str, str]] = {
    "house": ("home", "modest"),
    "inn": ("kitchen", "modest"),
    "butcher": ("kitchen", "modest"),
    "barn": ("farm", "poor"),
    "shepherd": ("farm", "poor"),
    "windmill": ("farm", "modest"),
    "fisherman": ("fish", "modest"),
    "fletcher": ("fletcher", "modest"),
    "leatherworker": ("leather", "modest"),
    "weaponsmith": ("smith", "rich"),
    "toolsmith": ("smith", "rich"),
    "mason": ("mason", "modest"),
    "cartographer": ("scholar", "modest"),
    "library": ("scholar", "rich"),
    "school": ("scholar", "modest"),
    "postal": ("scholar", "modest"),
    "chronicle": ("scholar", "modest"),
    "church": ("church", "rich"),
    "cleric": ("church", "modest"),
    "townhall": ("market", "rich"),
    "bank": ("market", "rich"),
    "stand": ("market", "modest"),
    "fountain": ("market", "modest"),
    "castle": ("castle", "noble"),
    "harbor": ("fish", "modest"),
    "misc": ("generic", "poor"),
    "outdoor": ("generic", "poor"),  # street-corner decoration barrels/chests
}

_DEFAULT_CONTEXT = ("generic", "modest")

# Merged (generic + theme) pools, built once per theme on first use.
_merged_cache: Dict[str, Pool] = {}


def _merged_pool(theme: str) -> Pool:
    cached = _merged_cache.get(theme)
    if cached is not None:
        return cached
    theme_pool = _THEMES.get(theme, {})
    merged: Pool = {}
    for rarity in RARITIES:
        merged[rarity] = list(_GENERIC.get(rarity, [])) + list(
            theme_pool.get(rarity, [])
        )
    _merged_cache[theme] = merged
    return merged


def _resolve(category: Optional[str], csv_path: Optional[str]) -> Tuple[str, str]:
    theme, tier = _CONTEXT.get(category or "", _DEFAULT_CONTEXT)
    if category == "house":
        stem = (csv_path or "").lower()
        if "small" in stem:
            tier = "poor"
        elif "large" in stem:
            tier = "rich"
        else:
            tier = "modest"
    return theme, tier


def _pick_item(pool: Pool, weights: Dict[str, float], rng: Any) -> Item:
    buckets = [r for r in RARITIES if pool.get(r) and weights.get(r, 0.0) > 0.0]
    bucket = rng.choices(buckets, weights=[weights[r] for r in buckets])[0]
    return rng.choice(pool[bucket])


def container_loot_data(
    category: Optional[str],
    csv_path: Optional[str] = None,
    rng: Optional[Any] = None,
    *,
    is_barrel: bool = False,
) -> Optional[str]:
    """
    Build the block-entity ``data`` SNBT that fills a chest/barrel with items
    matching *category* (and *csv_path* for house sizing).

    Returns an SNBT string like ``{Items:[{Slot:0b,id:"minecraft:bread",count:3}]}``
    ready for ``Block(..., data=...)``, or ``None`` when the container should be
    left empty (a natural, common outcome for humble places).

    Barrels read as provisions/storage: they never hold treasure and lean a
    little more toward mundane goods than a chest in the same building.
    """
    rng = rng or random
    theme, tier_name = _resolve(category, csv_path)
    tier = _TIERS.get(tier_name, _TIERS["modest"])
    pool = _merged_pool(theme)

    weights = dict(tier.weights)
    if is_barrel:
        # A barrel is a larder, not a strongbox — no jackpots, more staples.
        weights["treasure"] = 0.0
        weights["common"] = weights.get("common", 0.0) * 1.5

    n = rng.randint(*tier.slots)
    if n <= 0:
        return None

    n = min(n, _CONTAINER_SLOTS)
    slots = rng.sample(range(_CONTAINER_SLOTS), n)
    entries: List[str] = []
    for slot in slots:
        item_id, lo, hi = _pick_item(pool, weights, rng)
        count = rng.randint(lo, hi)
        entries.append(f'{{Slot:{slot}b,id:"minecraft:{item_id}",count:{count}}}')

    if not entries:
        return None
    return "{Items:[" + ",".join(entries) + "]}"
