"""
This module defines a helper class for block palette management.
"""

import random
from typing import Dict, List

from gdpc.block import Block


class BlockPalette:
    """Manage material palettes for different structure components."""

    def __init__(
        self,
        unidirectional_palettes: Dict[str, List[str]],
        bidirectional_palettes: Dict[str, List[str]],
        block_to_palette: Dict[str, str],
    ):
        """
        Initialize the BlockPalette class with given palettes and block mappings.

        :param unidirectional_palettes: A mapping of palette name to list of block IDs for unidirectional replacements
        :param bidirectional_palettes: A mapping of palette name to list of block IDs for bidirectional replacements
        :param block_to_palette: A mapping of block ID to palette name for determining which palette to apply
        """
        self.unidirectional_palettes = unidirectional_palettes
        self.bidirectional_palettes = bidirectional_palettes
        self.block_to_palette = block_to_palette

        self.block_to_bidirectional = {
            block: palette
            for palette, blocks in self.bidirectional_palettes.items()
            for block in blocks
        }

    def apply_palette(self, block: Block) -> Block:
        """
        Replace a block with a random variant from its palette, if applicable.

        :param block: The original block to potentially replace
        :return: The potentially replaced block based on the palette configuration
        """
        block_id = block.id.removeprefix("minecraft:")

        palette = self.block_to_palette.get(block_id)
        if palette is not None and palette in self.unidirectional_palettes:
            replacement = random.choice(self.unidirectional_palettes[palette])
            return Block(replacement, block.states)

        palette = self.block_to_bidirectional.get(block_id)
        if palette is not None and palette in self.bidirectional_palettes:
            replacement = random.choice(self.bidirectional_palettes[palette])
            return Block(replacement, block.states)

        return block
