"""
lore/names.py
-------------
Deterministic naming and a shared "cast" for the settlement.

One :class:`Lore` instance is created per generation and shared by every system
that needs consistent proper nouns:

  * the settlement name (used on gate signs and in the chronicle),
  * district names (gate signposts + chronicle),
  * a roster of named residents — living villagers spawned into the world and
    the dead resting in the graveyard — all of whom the chronicle can name.

Everything is drawn from a seeded :class:`random.Random` so the same build area
always yields the same town, people and epitaphs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Word pools
# ---------------------------------------------------------------------------

_SETTLEMENT_PREFIX = [
    "Ash",
    "Black",
    "Bright",
    "Cold",
    "Elder",
    "Fox",
    "Grey",
    "Green",
    "Hollow",
    "Iron",
    "Mill",
    "Moss",
    "North",
    "Oak",
    "Raven",
    "Red",
    "Silver",
    "Stone",
    "Thorn",
    "West",
    "Wild",
    "Wind",
    "Wolf",
    "Yew",
]
_SETTLEMENT_SUFFIX = [
    "brook",
    "burgh",
    "crest",
    "dale",
    "field",
    "ford",
    "gate",
    "hallow",
    "haven",
    "hold",
    "mere",
    "moor",
    "reach",
    "ridge",
    "stead",
    "ton",
    "vale",
    "watch",
    "well",
    "wick",
    "wood",
    "wharf",
]

_DISTRICT_ADJ = [
    "Old",
    "New",
    "Upper",
    "Lower",
    "Great",
    "Little",
    "High",
    "Nether",
    "East",
    "West",
    "North",
    "South",
    "Market",
    "King's",
    "Queen's",
    "Guild",
    "Temple",
    "Abbey",
    "Castle",
    "Garrison",
    "Harbour",
    "Mill",
    "Weavers'",
    "Fishers'",
    "Potters'",
    "Millers'",
    "Coopers'",
    "Tanners'",
    "Smiths'",
    "Shepherds'",
    "Bakers'",
    "Brewers'",
    "Masons'",
    "Merchants'",
    "Saddlers'",
    "Elm",
    "Oak",
    "Willow",
    "Thorn",
    "Broad",
    "Crooked",
    "Silver",
    "Golden",
]
_DISTRICT_NOUN = [
    "Quarter",
    "Row",
    "End",
    "Ward",
    "Gate",
    "Rise",
    "Green",
    "Cross",
    "Yard",
    "Walk",
    "Lane",
    "Hill",
    "Close",
    "Court",
    "Wynd",
    "Steps",
    "Bridge",
    "Bank",
    "Mews",
    "Terrace",
]
# Saints for street/gate dedications ("St. Cuthbert's Gate").
_SAINTS = [
    "Cuthbert",
    "Dunstan",
    "Aldhelm",
    "Swithin",
    "Edmund",
    "Werburgh",
    "Botolph",
    "Oswald",
    "Milburga",
    "Wystan",
]

_GIVEN = [
    "Aldric",
    "Alwin",
    "Anselm",
    "Bram",
    "Cedric",
    "Cerdic",
    "Colwin",
    "Cuthbert",
    "Dunstan",
    "Edmund",
    "Edric",
    "Fenwick",
    "Godric",
    "Hamon",
    "Harald",
    "Leofric",
    "Odo",
    "Osric",
    "Ralf",
    "Roderic",
    "Rowan",
    "Swithin",
    "Theobald",
    "Ulric",
    "Wat",
    "Wilfred",
    "Wymar",
    "Yorick",
    "Aldous",
    "Bertram",
    "Gervase",
    "Nigel",
    "Reynold",
    "Warin",
    "Aldith",
    "Alys",
    "Avice",
    "Beatrix",
    "Cwen",
    "Edith",
    "Elga",
    "Ellyn",
    "Godgifu",
    "Hawise",
    "Hilda",
    "Ingrid",
    "Isolde",
    "Katla",
    "Mabel",
    "Maud",
    "Nesta",
    "Petra",
    "Rhiannon",
    "Rowena",
    "Sigrid",
    "Thora",
    "Wilda",
    "Winifred",
    "Emeline",
    "Sabine",
]
_SURNAME = [
    "Ashdown",
    "Blackwood",
    "Brewer",
    "Bywater",
    "Carter",
    "Chandler",
    "Cooper",
    "Dale",
    "Elmwood",
    "Fairwind",
    "Fletcher",
    "Fowler",
    "Greaves",
    "Hale",
    "Holloway",
    "Ironwright",
    "Langley",
    "Marsh",
    "Mason",
    "Mercer",
    "Millward",
    "Netherby",
    "Norcross",
    "Oakes",
    "Pemberton",
    "Pike",
    "Quill",
    "Ravenswood",
    "Reed",
    "Ryder",
    "Sadler",
    "Selby",
    "Shaw",
    "Slade",
    "Stone",
    "Tanner",
    "Thatcher",
    "Thorne",
    "Underhill",
    "Vance",
    "Wainwright",
    "Weaver",
    "Whitlock",
    "Woolcott",
    "Yardley",
    "Yarrow",
    "Ackworth",
    "Barrow",
    "Coldwell",
    "Denholm",
    "Garland",
    "Harrow",
    "Ivory",
    "Norwood",
    "Prentice",
    "Swanwick",
]

_EPITAPHS = [
    "Rest well",
    "Gone home",
    "At peace",
    "Long missed",
    "Ever kind",
    "Sorely missed",
    "Taken young",
    "A good soul",
    "Beloved",
    "Never forgot",
]

# Our building-category key -> human-readable trade noun (chronicle text).
_TRADE_LABEL = {
    "butcher": "butcher",
    "cartographer": "cartographer",
    "cleric": "cleric",
    "fisherman": "fisher",
    "fletcher": "bowyer",
    "leatherworker": "tanner",
    "mason": "mason",
    "shepherd": "shepherd",
    "toolsmith": "smith",
    "weaponsmith": "weaponsmith",
    "library": "librarian",
}

# Our building-category key -> Minecraft villager profession id (for /summon).
_MC_PROFESSION = {
    "butcher": "butcher",
    "cartographer": "cartographer",
    "cleric": "cleric",
    "fisherman": "fisherman",
    "fletcher": "fletcher",
    "leatherworker": "leatherworker",
    "mason": "mason",
    "shepherd": "shepherd",
    "toolsmith": "toolsmith",
    "weaponsmith": "weaponsmith",
    "library": "librarian",
}


@dataclass
class Person:
    """A named inhabitant, living or dead."""

    name: str  # full name, e.g. "Fenwick Ashdown"
    trade: Optional[str] = None  # our category key, e.g. "mason" (None = townsfolk)
    trade_label: Optional[str] = None  # readable trade, e.g. "mason"
    profession: Optional[str] = None  # MC villager profession id (None = generic)
    deceased: bool = False
    epitaph: Optional[str] = None
    # Set once an entity for this Person has actually been /summon'd, so other
    # systems drawing from the shared roster don't spawn them a second time.
    spawned: bool = False
    # Set once a headstone names this (deceased) Person, so a second graveyard
    # never repeats a name already carved in the first.
    buried: bool = False


class Lore:
    """Deterministic name generator + shared roster for one settlement."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)
        self.settlement_name: str = self.rng.choice(
            _SETTLEMENT_PREFIX
        ) + self.rng.choice(_SETTLEMENT_SUFFIX)
        self._district_cache: dict[object, str] = {}
        self.residents: list[Person] = []
        self.deceased: list[Person] = []

    # ------------------------------------------------------------------
    # Names
    # ------------------------------------------------------------------
    def district_name(self, key: object) -> str:
        """Return a stable, varied, unique district name for *key* — an
        adjective+noun ("Old Quarter"), a family's row ("Weaver's Row"), or a
        saint's dedication ("St. Cuthbert's Gate")."""
        if key not in self._district_cache:
            used = set(self._district_cache.values())
            name = self._make_district_name()
            for _ in range(20):
                if name not in used:
                    break
                name = self._make_district_name()
            self._district_cache[key] = name
        return self._district_cache[key]

    def _make_district_name(self) -> str:
        noun = self.rng.choice(_DISTRICT_NOUN)
        r = self.rng.random()
        if r < 0.55:
            return f"{self.rng.choice(_DISTRICT_ADJ)} {noun}"
        if r < 0.80:
            return f"{self.rng.choice(_SURNAME)}'s {noun}"
        return f"St. {self.rng.choice(_SAINTS)}'s {noun}"

    def _unique_name(self, used: set[str]) -> str:
        name = f"{self.rng.choice(_GIVEN)} {self.rng.choice(_SURNAME)}"
        for _ in range(50):
            if name not in used:
                break
            name = f"{self.rng.choice(_GIVEN)} {self.rng.choice(_SURNAME)}"
        used.add(name)
        return name

    # ------------------------------------------------------------------
    # Roster
    # ------------------------------------------------------------------
    def populate(
        self,
        professions: list[str],
        *,
        n_townsfolk: int = 4,
        n_deceased: int = 4,
    ) -> "Lore":
        """Build the living + deceased roster.

        *professions* is the list of trade-building categories actually placed
        (from :meth:`SchematicBuildingPlacer.summarize_inhabitants`).  Each gets
        one named tradesperson; *n_townsfolk* generic townsfolk and *n_deceased*
        of the departed round out the cast — callers scale both with city size.
        The chronicle only quotes a capped subset (see resident_labels /
        deceased_labels), so a large roster never bloats the book: extra named
        folk simply walk the streets and rest in the churchyard.
        """
        used: set[str] = set()
        for cat in professions:
            self.residents.append(
                Person(
                    name=self._unique_name(used),
                    trade=cat,
                    trade_label=_TRADE_LABEL.get(cat),
                    profession=_MC_PROFESSION.get(cat),
                )
            )
        for _ in range(max(0, n_townsfolk)):
            self.residents.append(Person(name=self._unique_name(used)))
        for _ in range(max(0, n_deceased)):
            self.deceased.append(
                Person(
                    name=self._unique_name(used),
                    deceased=True,
                    epitaph=self.rng.choice(_EPITAPHS),
                )
            )
        return self

    # ------------------------------------------------------------------
    # Chronicle-facing summaries (plain strings, no dependency on this module)
    # ------------------------------------------------------------------
    def resident_labels(self, limit: int = 8) -> list[str]:
        """Readable "Name, the trade" strings for the living, tradespeople first."""
        ordered = [p for p in self.residents if p.trade_label] + [
            p for p in self.residents if not p.trade_label
        ]
        out: list[str] = []
        for p in ordered[:limit]:
            out.append(f"{p.name}, the {p.trade_label}" if p.trade_label else p.name)
        return out

    def deceased_labels(self, limit: int = 6) -> list[str]:
        return [p.name for p in self.deceased[:limit]]

    def district_labels(self) -> list[str]:
        """All district names handed out so far (order they were first requested)."""
        return list(self._district_cache.values())
