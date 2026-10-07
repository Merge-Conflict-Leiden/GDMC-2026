"""
Category & quota configuration for the civil building balance.

Maps schematic filename prefixes to categories, defines per-category quotas
and residential draw weights, and provides the weighted category draw used
by the urban fill.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from terrain.terrain_types import SubZone

from .model import Building


@dataclass(frozen=True)
class CategoryConfig:
    preferred_subzones: tuple[SubZone, ...]
    max_total: int  # -1 = unlimited
    residential_weight: float  # weight in the house/profession random draw


CATEGORY_CONFIG: dict[str, CategoryConfig] = {
    "townhall": CategoryConfig((SubZone.TOWN_CENTER, SubZone.CIVIC), 1, 0.0),
    "church": CategoryConfig((SubZone.CIVIC,), 1, 0.0),
    "school": CategoryConfig((SubZone.CIVIC,), 1, 0.0),
    "postal": CategoryConfig((SubZone.CIVIC,), 1, 0.0),
    "bank": CategoryConfig((SubZone.CIVIC, SubZone.TOWN_CENTER), 1, 0.0),
    "library": CategoryConfig((SubZone.CIVIC,), 2, 0.0),
    "inn": CategoryConfig((SubZone.CIVIC, SubZone.RESIDENTIAL), 2, 0.0),
    "butcher": CategoryConfig((SubZone.RESIDENTIAL,), 2, 0.30),
    "cartographer": CategoryConfig((SubZone.RESIDENTIAL,), 1, 0.20),
    "cleric": CategoryConfig((SubZone.RESIDENTIAL, SubZone.CIVIC), 2, 0.20),
    "fisherman": CategoryConfig((SubZone.RESIDENTIAL,), 2, 0.20),
    "fletcher": CategoryConfig((SubZone.RESIDENTIAL,), 2, 0.20),
    "leatherworker": CategoryConfig((SubZone.RESIDENTIAL,), 2, 0.20),
    "mason": CategoryConfig((SubZone.RESIDENTIAL,), 2, 0.20),
    "shepherd": CategoryConfig((SubZone.RESIDENTIAL,), 2, 0.15),
    "toolsmith": CategoryConfig((SubZone.RESIDENTIAL,), 2, 0.25),
    "weaponsmith": CategoryConfig((SubZone.RESIDENTIAL,), 1, 0.15),
    "barn": CategoryConfig((SubZone.RESIDENTIAL,), 3, 0.20),
    "house": CategoryConfig((SubZone.RESIDENTIAL,), -1, 1.00),
}

CIVIC_PRIORITY = ["townhall", "church", "library", "school", "inn", "bank", "postal"]
PROFESSION_CATS = [
    c
    for c, cfg in CATEGORY_CONFIG.items()
    if cfg.residential_weight > 0 and c != "house"
]

# Longest prefix first so "house_" beats bare matches.
_PREFIX_TO_CAT: list[tuple[str, str]] = sorted(
    [
        ("house_", "house"),
        ("barn_", "barn"),
        ("barn", "barn"),
        ("butcher", "butcher"),
        ("cartographer", "cartographer"),
        ("church", "church"),
        ("cleric", "cleric"),
        ("fisherman", "fisherman"),
        ("fletcher", "fletcher"),
        ("fountain", "fountain"),
        ("inn", "inn"),
        ("leatherworker", "leatherworker"),
        ("library", "library"),
        ("masonery", "mason"),
        ("postal", "postal"),
        ("school", "school"),
        ("shepherd", "shepherd"),
        ("stand", "stand"),
        ("toolsmith", "toolsmith"),
        ("townhall", "townhall"),
        ("weaponsmith", "weaponsmith"),
        ("chronicle", "chronicle"),
        ("bank", "bank"),
    ],
    key=lambda t: -len(t[0]),
)


def infer_category(stem: str) -> str:
    s = stem.lower()
    for prefix, cat in _PREFIX_TO_CAT:
        if s.startswith(prefix):
            return cat
    return "house"


def group_by_category(buildings: list[Building]) -> dict[str, list[Building]]:
    out: dict[str, list[Building]] = {}
    for b in buildings:
        out.setdefault(b.category, []).append(b)
    return out


def available_civic(
    queue: list[str],
    counts: dict[str, int],
    cfg: dict[str, CategoryConfig],
) -> list[str]:
    return [
        c
        for c in queue
        if c in cfg and (cfg[c].max_total == -1 or counts.get(c, 0) < cfg[c].max_total)
    ]


def weighted_category(
    by_cat: dict[str, list[Building]],
    counts: dict[str, int],
    cfg: dict[str, CategoryConfig],
) -> str:
    if random.random() < 0.65:
        return "house"
    pool = [
        (cat, cfg[cat].residential_weight)
        for cat in PROFESSION_CATS
        if by_cat.get(cat)
        and (cfg[cat].max_total == -1 or counts.get(cat, 0) < cfg[cat].max_total)
    ]
    if not pool:
        return "house"
    total = sum(w for _, w in pool)
    r = random.uniform(0, total)
    cumul = 0.0
    for cat, w in pool:
        cumul += w
        if r <= cumul:
            return cat
    return pool[-1][0]
