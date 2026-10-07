"""
This module defines a helper for furniture/interior placement.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from gdpc import Block, Editor
from gdpc.vector_tools import ivec3

from blocks import BlockProcessor
from furniture.loot import container_loot_data

Placement = Tuple[ivec3, Block]
FacingMap = Dict[ivec3, str]

AIR_BLOCKS = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}
_AIR = Block("minecraft:air")

# All available pottery sherds; "brick" is doubled so most faces stay plain.
_POTTERY_SHERDS = [
    "minecraft:brick",
    "minecraft:brick",
    "minecraft:arms_up_pottery_sherd",
    "minecraft:blade_pottery_sherd",
    "minecraft:brewer_pottery_sherd",
    "minecraft:burn_pottery_sherd",
    "minecraft:danger_pottery_sherd",
    "minecraft:explorer_pottery_sherd",
    "minecraft:friend_pottery_sherd",
    "minecraft:heart_pottery_sherd",
    "minecraft:heartbreak_pottery_sherd",
    "minecraft:howl_pottery_sherd",
    "minecraft:miner_pottery_sherd",
    "minecraft:mourner_pottery_sherd",
    "minecraft:plenty_pottery_sherd",
    "minecraft:prize_pottery_sherd",
    "minecraft:shelter_pottery_sherd",
    "minecraft:skull_pottery_sherd",
    "minecraft:snort_pottery_sherd",
]

# All leaf block types (persistent crown on top of furniture stacks).
_LEAF_BLOCKS = [
    "oak_leaves",
    "spruce_leaves",
    "birch_leaves",
    "jungle_leaves",
    "acacia_leaves",
    "dark_oak_leaves",
    "mangrove_leaves",
    "cherry_leaves",
    "azalea_leaves",
    "flowering_azalea_leaves",
    "pale_oak_leaves",
]

# All 17 candle variants (plain + 16 dye colours).
_CANDLE_COLORS = [
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

# Amethyst growth stages placeable on top of furniture with a random facing.
_AMETHYST_CRYSTALS = [
    "small_amethyst_bud",
    "medium_amethyst_bud",
    "large_amethyst_bud",
    "amethyst_cluster",
]

# Skull blocks placeable on top of furniture; rotation 0-15 is chosen randomly.
_SKULL_BLOCKS = [
    "skeleton_skull",
    "wither_skeleton_skull",
    "zombie_head",
    "creeper_head",
    "piglin_head",
    "dragon_head",
]

# All vanilla potted-plant block IDs (placed on top of furniture stacks).
_POTTED_PLANTS = [
    "potted_oak_sapling",
    "potted_spruce_sapling",
    "potted_birch_sapling",
    "potted_jungle_sapling",
    "potted_acacia_sapling",
    "potted_dark_oak_sapling",
    "potted_cherry_sapling",
    "potted_fern",
    "potted_dandelion",
    "potted_poppy",
    "potted_blue_orchid",
    "potted_allium",
    "potted_azure_bluet",
    "potted_red_tulip",
    "potted_orange_tulip",
    "potted_white_tulip",
    "potted_pink_tulip",
    "potted_oxeye_daisy",
    "potted_cornflower",
    "potted_lily_of_the_valley",
    "potted_dead_bush",
    "potted_cactus",
    "potted_bamboo",
    "potted_azalea_bush",
    "potted_flowering_azalea_bush",
    "potted_torchflower",
]


@dataclass
class StackInfo:
    x: int
    z: int
    y_values: List[int]

    @property
    def height(self) -> int:
        return len(self.y_values)

    @property
    def bottom_y(self) -> int:
        return self.y_values[0]

    @property
    def top_y(self) -> int:
        return self.y_values[-1]

    @property
    def base_pos(self) -> ivec3:
        return ivec3(self.x, self.bottom_y, self.z)


class FurniturePlacer:
    def __init__(
        self,
        stackable_blocks: Optional[List[str]] = None,
        single_furniture: Optional[List[str]] = None,
        blocks_with_facing: Optional[Set[str]] = None,
    ):
        self.stackable_blocks = stackable_blocks or [
            "barrel",
            "chest",
            "bookshelf",
            "chiseled_bookshelf",
            "furnace",
            "blast_furnace",
            "smoker",
        ]

        self.single_furniture = single_furniture or [
            "loom",
            "lectern",
            "cauldron",
            "anvil",
            "grindstone",
            "stonecutter",
            "decorated_pot",
            "brewing_stand",
            "smithing_table",
            "fletching_table",
            "crafting_table",
            "lodestone",
            "enchanting_table",
            "scaffolding",
        ]

        self.support_blocks = [
            "spruce_planks",
            "spruce_stairs[half=top]",
            "spruce_slab[type=top]",
            "scaffolding",
        ]

        # NB: "iron_chain" is not a real block id (the block is "chain"), and
        # this server rejects "chain" too — so hanging supports use iron_bars.
        self.hanging_blocks = [
            "iron_bars",
        ]

        self.stack_decorations = ["cake"] + _POTTED_PLANTS

        self.hanging_compatible = {
            "barrel",
            "chest",
            "blast_furnace",
            "smoker",
        }

        self.blocks_with_facing = blocks_with_facing or {
            "barrel",
            "chest",
            "furnace",
            "blast_furnace",
            "smoker",
            "loom",
            "lectern",
            "stonecutter",
            "cartography_table",
            "chiseled_bookshelf",
        }

        # Loot context for the structure currently being furnished.  Set per
        # call in replace_command_blocks so chests/barrels are filled with items
        # matching the building they sit in (see furniture.loot).
        self._loot_category: Optional[str] = None
        self._loot_csv: Optional[str] = None

    def _is_air(self, block_id: str) -> bool:
        return block_id in AIR_BLOCKS

    def _is_on_ground(self, editor: Editor, pos: ivec3) -> bool:
        x, y, z = pos
        return not self._is_air(editor.getBlock((x, y - 1, z)).id)

    def _is_hemmed_in(self, editor: Editor, pos: ivec3) -> bool:
        x, y, z = pos
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if self._is_air(editor.getBlock((x + dx, y, z + dz)).id):
                return False
        return True

    def _find_ceiling(
        self, editor: Editor, pos: ivec3, max_height: int = 8
    ) -> Optional[int]:
        x, y, z = pos
        for i in range(1, max_height):
            if not self._is_air(editor.getBlock((x, y + i, z)).id):
                return y + i
        return None

    def _make_candle(self) -> Block:
        block = Block(random.choice(_CANDLE_COLORS))
        block.states["lit"] = random.choice(["true", "false"])
        block.states["candles"] = str(random.randint(1, 4))
        return block

    def _make_sea_pickle(self) -> Block:
        block = Block("sea_pickle")
        block.states["pickles"] = str(random.randint(2, 4))
        block.states["waterlogged"] = "false"
        return block

    def _make_cauldron(self) -> Block:
        level = random.randint(0, 3)

        if level == 0:
            return Block("cauldron")

        if random.random() < 0.65:
            block = Block("water_cauldron")
            block.states["level"] = str(level)
        else:
            block = Block("lava_cauldron")
        return block

    @staticmethod
    def _cardinal(facing: Optional[str]) -> Optional[str]:
        """Return facing unchanged if cardinal, otherwise pick a random cardinal."""
        if facing in ("north", "south", "east", "west"):
            return facing
        if facing is not None:
            return random.choice(["north", "south", "east", "west"])
        return None

    def _make_decorated_pot(self, facing: Optional[str]) -> Block:
        sherds = [random.choice(_POTTERY_SHERDS) for _ in range(4)]
        data = '{sherds:["' + '","'.join(sherds) + '"]}'
        block = Block("decorated_pot", data=data)
        cardinal = self._cardinal(facing)
        if cardinal:
            block.states["facing"] = cardinal
        return block

    def _make_furniture_block(self, furniture: str, facing: Optional[str]) -> Block:
        if furniture == "cauldron":
            return self._make_cauldron()
        if furniture == "decorated_pot":
            return self._make_decorated_pot(facing)

        block = Block(furniture)
        if facing and furniture in self.blocks_with_facing:
            block.states["facing"] = self._cardinal(facing)

        if furniture in ("chest", "barrel"):
            # Fill the container with context-appropriate loot rather than
            # leaving it empty.  None means "leave empty" (a common outcome).
            loot = container_loot_data(
                self._loot_category,
                self._loot_csv,
                is_barrel=(furniture == "barrel"),
            )
            if loot is not None:
                block.data = loot

        if furniture == "grindstone":
            block.states["face"] = "floor"
        elif furniture == "anvil":
            roll = random.random()
            if roll < 0.33:
                block = Block("chipped_anvil")
            elif roll < 0.66:
                block = Block("damaged_anvil")
        elif furniture == "chiseled_bookshelf":
            for i in range(6):
                block.states[f"slot_{i}_occupied"] = random.choice(["true", "false"])
        return block

    def _make_amethyst(self) -> Block:
        block = Block(random.choice(_AMETHYST_CRYSTALS))
        block.states["facing"] = "up"
        return block

    def _make_skull(self) -> Block:
        block = Block(random.choice(_SKULL_BLOCKS))
        block.states["rotation"] = str(random.randint(0, 15))
        return block

    def _top_decoration(self) -> Optional[Block]:
        if random.random() > 0.8:
            return None

        roll = random.random()
        if roll < 0.25:
            return self._make_candle()
        elif roll < 0.50:
            return Block(random.choice(self.stack_decorations))
        elif roll < 0.65:
            return self._make_sea_pickle()
        elif roll < 0.82:
            return self._make_skull()
        else:
            return self._make_amethyst()

    def _hanging_support_placements(
        self,
        editor: Editor,
        stack: StackInfo,
        furniture: str,
        facing: Optional[str],
    ) -> List[Placement]:
        if furniture not in self.hanging_compatible:
            return []

        if stack.height < 2:
            return []

        ceiling = self._find_ceiling(editor, stack.base_pos)
        if ceiling is None:
            return []

        x, z = stack.x, stack.z

        hang_y = random.choice(stack.y_values[1:])

        support = random.choice(self.hanging_blocks)

        placements: List[Placement] = []

        for y in stack.y_values:
            placements.append((ivec3(x, y, z), _AIR))

        placements.append(
            (ivec3(x, hang_y, z), self._make_furniture_block(furniture, facing))
        )

        for y in range(hang_y + 1, ceiling):
            placements.append((ivec3(x, y, z), Block(support)))

        return placements

    @staticmethod
    def _find_clusters(positions: List[ivec3]) -> List[List[ivec3]]:
        pos_set = set(positions)
        visited: Set[ivec3] = set()
        clusters: List[List[ivec3]] = []

        for start in positions:
            if start in visited:
                continue

            stack = [start]
            cluster: List[ivec3] = []

            while stack:
                pos = stack.pop()
                if pos in visited:
                    continue

                visited.add(pos)
                cluster.append(pos)

                x, y, z = pos
                for dx, dy, dz in (
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1),
                ):
                    n = ivec3(x + dx, y + dy, z + dz)
                    if n in pos_set and n not in visited:
                        stack.append(n)

            clusters.append(cluster)

        return clusters

    @staticmethod
    def _get_vertical_stacks(cluster: List[ivec3]) -> Dict[Tuple[int, int], StackInfo]:
        raw: Dict[Tuple[int, int], List[int]] = {}

        for pos in cluster:
            raw.setdefault((pos.x, pos.z), []).append(pos.y)

        return {(x, z): StackInfo(x, z, sorted(ys)) for (x, z), ys in raw.items()}

    def _make_support_block(self, block_string: str, facing: Optional[str]) -> Block:
        block = BlockProcessor.parse_block(block_string)

        if block.id.endswith("_stairs") and facing:
            block.states["facing"] = facing

        return block

    def _generate_pile(
        self,
        stacks: Dict[Tuple[int, int], StackInfo],
        section: Set[Tuple[int, int]],
        facing_map: FacingMap,
    ) -> List[Placement]:
        placements: List[Placement] = []
        furniture = random.choice(self.stackable_blocks)
        support_block = random.choice(self.support_blocks)

        for x, z in section:
            stack = stacks[(x, z)]
            height = random.randint(1, stack.height)
            facing = facing_map.get(ivec3(x, stack.bottom_y, z))

            for i, y in enumerate(stack.y_values):
                pos = ivec3(x, y, z)

                if height > 1 and i == 0 and random.random() < 0.3:
                    placements.append(
                        (pos, self._make_support_block(support_block, facing))
                    )
                elif i < height:
                    placements.append(
                        (pos, self._make_furniture_block(furniture, facing))
                    )
                else:
                    placements.append((pos, _AIR))

            if height > 1:
                top_y = stack.y_values[height - 1]
                dec = self._top_decoration()
                if dec and furniture != "chest":
                    placements.append((ivec3(x, top_y, z), dec))

        return placements

    def _place_single_stack(
        self,
        editor: Editor,
        stack: StackInfo,
        facing: Optional[str],
        preferred_furniture: Optional[str],
    ) -> Tuple[List[Placement], Optional[str]]:
        placements: List[Placement] = []
        x, z = stack.x, stack.z
        on_ground = self._is_on_ground(editor, stack.base_pos)

        # Ground-level potted plant: placed directly on the floor only when
        # at least one horizontal neighbour is air (so it's not hidden in a corner).
        if (
            on_ground
            and random.random() < 0.18
            and not self._is_hemmed_in(editor, stack.base_pos)
        ):
            placements.append((stack.base_pos, Block(random.choice(_POTTED_PLANTS))))
            for y in stack.y_values[1:]:
                placements.append((ivec3(x, y, z), _AIR))
            return placements, None

        # Special 3-high patterns (small independent chance each).
        if on_ground and stack.height == 3:
            roll = random.random()
            y0, y1, y2 = stack.y_values

            if roll < 0.07:
                # Scaffolding base with a two-block stalagmite growing up.
                dripstone_base = Block("pointed_dripstone")
                dripstone_base.states["vertical_direction"] = "up"
                dripstone_base.states["thickness"] = "frustum"
                dripstone_tip = Block("pointed_dripstone")
                dripstone_tip.states["vertical_direction"] = "up"
                dripstone_tip.states["thickness"] = "tip"
                placements += [
                    (ivec3(x, y0, z), Block("scaffolding")),
                    (ivec3(x, y1, z), dripstone_base),
                    (ivec3(x, y2, z), dripstone_tip),
                ]
                return placements, None

            elif roll < 0.14:
                # Decorated pot → potted bamboo → leaf crown.
                leaves = Block(random.choice(_LEAF_BLOCKS))
                leaves.states["persistent"] = "true"
                leaves.states["distance"] = "1"
                placements += [
                    (ivec3(x, y0, z), self._make_decorated_pot(facing)),
                    (ivec3(x, y1, z), Block("potted_bamboo")),
                    (ivec3(x, y2, z), leaves),
                ]
                return placements, None

            elif roll < 0.21:
                # Composter with two leaf blocks of the same type above it —
                # looks like a bush or shrub growing out of a planter.
                leaf_id = random.choice(_LEAF_BLOCKS)
                leaf = Block(leaf_id)
                leaf.states["persistent"] = "true"
                leaf.states["distance"] = "1"
                placements += [
                    (ivec3(x, y0, z), Block("composter", states={"level": "0"})),
                    (ivec3(x, y1, z), leaf),
                    (ivec3(x, y2, z), leaf),
                ]
                return placements, None

        # Special 2-high patterns.
        if on_ground and stack.height == 2 and random.random() < 0.08:
            y0, y1 = stack.y_values
            azalea = random.choice(
                ["potted_azalea_bush", "potted_flowering_azalea_bush"]
            )
            placements += [
                (ivec3(x, y0, z), self._make_decorated_pot(facing)),
                (ivec3(x, y1, z), Block(azalea)),
            ]
            return placements, None

        if preferred_furniture and random.random() < 0.4:
            furniture = preferred_furniture
        elif on_ground and random.random() < 0.6:
            furniture = random.choice(self.stackable_blocks)
        else:
            furniture = random.choice(self.single_furniture)

        if on_ground and furniture in self.stackable_blocks:
            height = random.randint(1, stack.height)
            top_y = None

            for i, y in enumerate(stack.y_values):
                pos = ivec3(x, y, z)

                if i < height:
                    roll = random.random()
                    if roll < 0.5:
                        to_place = furniture
                    elif roll < 0.75:
                        to_place = random.choice(self.stackable_blocks)
                    else:
                        to_place = random.choice(self.single_furniture)
                    placements.append(
                        (pos, self._make_furniture_block(furniture, facing))
                    )
                    top_y = y
                    if to_place in self.single_furniture:
                        break
                else:
                    placements.append((pos, _AIR))

            if (
                top_y is not None
                and furniture in self.stackable_blocks
                and furniture != "chest"
            ):
                dec = self._top_decoration()
                if dec is not None:
                    if self._is_air(editor.getBlock((x, top_y + 1, z)).id):
                        placements.append((ivec3(x, top_y + 1, z), dec))
                    elif top_y - stack.bottom_y >= 2:
                        placements.append((ivec3(x, top_y, z), dec))

            if random.random() < 0.15:
                placements.extend(
                    self._hanging_support_placements(editor, stack, furniture, facing)
                )

        else:
            placements.append(
                (stack.base_pos, self._make_furniture_block(furniture, facing))
            )
            for y in stack.y_values[1:]:
                placements.append((ivec3(x, y, z), _AIR))

        return placements, furniture

    def replace_command_blocks(
        self,
        editor: Editor,
        command_block_data: List[Tuple[ivec3, str]],
        loot_category: Optional[str] = None,
        loot_csv: Optional[str] = None,
    ) -> int:
        # Remember what we're furnishing so chest/barrel loot matches the
        # building (see _make_furniture_block / furniture.loot).
        self._loot_category = loot_category
        self._loot_csv = loot_csv
        positions: List[ivec3] = [p[0] for p in command_block_data]
        facing_map: FacingMap = {pos: facing for pos, facing in command_block_data}

        clusters = self._find_clusters(positions)
        processed: Set[ivec3] = set()
        total = 0

        for cluster in clusters:
            stacks = self._get_vertical_stacks(cluster)
            stack_keys = list(stacks.keys())

            pile_section: Set[Tuple[int, int]] = set()
            if len(stack_keys) >= 3 and random.random() < 0.25:
                n = random.randint(2, len(stack_keys))
                pile_section = set(random.sample(stack_keys, n))

            if pile_section:
                placements = self._generate_pile(stacks, pile_section, facing_map)
                for pos, block in placements:
                    if pos in processed:
                        continue
                    editor.placeBlock(pos, block)
                    processed.add(pos)
                    total += 1

            preferred_furniture: Optional[str] = None

            for x_z_key, stack in stacks.items():
                if x_z_key in pile_section:
                    continue

                facing = facing_map.get(ivec3(stack.x, stack.bottom_y, stack.z))

                if random.random() < 0.75:
                    placements, used = self._place_single_stack(
                        editor, stack, facing, preferred_furniture
                    )

                    if preferred_furniture is None and random.random() < 0.5:
                        preferred_furniture = used
                else:
                    placements = [
                        (ivec3(stack.x, y, stack.z), _AIR) for y in stack.y_values
                    ]

                for pos, block in placements:
                    if pos in processed:
                        continue
                    editor.placeBlock(pos, block)
                    processed.add(pos)
                    total += 1

        for pos in positions:
            if pos not in processed:
                editor.placeBlock(pos, _AIR)
                total += 1

        return total
