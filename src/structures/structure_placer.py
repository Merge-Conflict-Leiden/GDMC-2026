"""
This module defines the structure placement handler.
"""

import ast
import logging
from typing import List, Optional, Tuple

from gdpc.editor import Editor
from gdpc.transform import rotatedBoxTransform
from gdpc.vector_tools import Box
from pyglm.glm import ivec3

from blocks.block_palette import BlockPalette
from blocks.block_processor import BlockProcessor
from consts import FACING_ORDER
from furniture.furniture_placer import FurniturePlacer

logger = logging.getLogger(__name__)


def _rotate_facing(facing: str, direction: int) -> str:
    """
    Rotate the initial facing of a block, provided a rotation direction.

    :param facing: The initial facing as provided by the CSV-file (/schematic)
    :param direction: The direction to which the block should be rotated
    :return: The corrected facing of the block
    """
    if facing not in FACING_ORDER:
        return facing

    idx = FACING_ORDER.index(facing)
    return FACING_ORDER[(idx + direction) % 4]


class StructurePlacer:
    """Handle placement of CSV-based structures into the world."""

    def __init__(
        self, structure_cache, block_processor, block_palette, furniture_replacer
    ):
        self.cache = structure_cache
        self.block_processor: BlockProcessor = block_processor
        self.block_palette: BlockPalette = block_palette
        self.furniture_replacer: FurniturePlacer = furniture_replacer
        self.command_block_positions: List[Tuple[ivec3, str]] = []

    def place_structure(
        self,
        editor: Editor,
        csv_path: str,
        biome: Optional[str] = "plains",
        direction: int = 0,
        translation: Optional[ivec3] = ivec3(0, 0, 0),
        skip_air: bool = False,
        skip_blocks: frozenset[str] = frozenset(),
        check_occupied: frozenset[str] = frozenset(),
    ) -> Tuple[int, int]:
        """
        Place the specified structure.

        To a large extent, it is the caller's responsibility to ensure that the
        correct transforms have been pushed to the editor instance to allow for
        placement at the correct location.

        For substructure placement, the translation parameter can be optionally
        provided.

        :param editor: Editor instance
        :param csv_path: Relative path to the CSV file of structure
        :param biome: Biome to be considered for this structure
        :param direction: Optional direction to rotate default orientation to
        :param translation: Optional offset to translate placement position with
        :return: Tuple of (height, width) of structure
        """
        df, width, height, depth = self.cache.load_structure(csv_path)

        box = Box((0, 0, 0), (width, height, depth))

        with editor.pushTransform(translation):
            rotation_transform = rotatedBoxTransform(box, direction)
            with editor.pushTransform(rotation_transform):
                for row in df.itertuples(index=False):
                    pos = None
                    try:
                        pos = ast.literal_eval(row.Position)
                        states = (
                            ast.literal_eval(row.States)
                            if isinstance(row.States, str)
                            else {}
                        )
                        data = row.Data if isinstance(row.Data, str) else None
                        block = self.block_processor.build_block(
                            row.BlockID, states, data
                        )

                        assert block.id is not None
                        if skip_air and block.id.split(":")[-1] in (
                            "air",
                            "cave_air",
                            "void_air",
                        ):
                            continue
                        # Jigsaw blocks are placement metadata (entrance and
                        # approach-path connectors, read from the CSV by the
                        # schematic loaders) — never place them in the world.
                        if "jigsaw" in block.id:
                            continue
                        if skip_blocks and any(
                            block.id.startswith(p) for p in skip_blocks
                        ):
                            continue
                        if check_occupied and any(
                            block.id.startswith(p) for p in check_occupied
                        ):
                            existing = editor.getBlock(pos)
                            if existing.id not in (
                                None,
                                "minecraft:air",
                                "minecraft:cave_air",
                                "minecraft:void_air",
                            ):
                                continue
                        if "repeating_command_block" in block.id:
                            world_pos = editor.transform.apply(pos)

                            facing = block.states.get("facing")

                            assert facing is not None
                            facing = _rotate_facing(facing, direction)

                            self.command_block_positions.append((world_pos, facing))

                        if self.block_palette is not None:
                            block = self.block_palette.apply_palette(block)

                        block = self.block_processor.randomize_candle_count(block)
                        block = self.block_processor.apply_stochastic_filter(block)
                        if biome is not None:
                            block = self.block_processor.apply_biome_adjustments(
                                block, biome
                            )

                        editor.placeBlock(pos, block)

                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning("Failed to place block at %s: %s", pos, e)
                        continue

        return height, width

    def place_furniture(
        self,
        editor: Editor,
        loot_category: Optional[str] = None,
        loot_csv: Optional[str] = None,
    ) -> int:
        """
        Place the furniture within the structure.

        :param editor: The current editor instance
        :param loot_category: Building category driving chest/barrel loot theme
        :param loot_csv: Schematic path (used to size house loot by small/large)
        :return: The total number of placed furniture pieces
        """
        if not self.command_block_positions:
            return 0

        replaced = self.furniture_replacer.replace_command_blocks(
            editor,
            self.command_block_positions,
            loot_category=loot_category,
            loot_csv=loot_csv,
        )

        self.command_block_positions = []

        return replaced
