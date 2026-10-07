"""
This module defines a dataclass for build configuration.
"""

from dataclasses import dataclass, field


@dataclass
class BuildConfig:
    """Configuration parameters for structure generation."""

    width: int
    depth: int

    clear_height: int

    stochastic_removals: dict = field(default_factory=dict)
    biome_replacements: dict = field(default_factory=dict)

    unidirectional_palettes: dict = field(default_factory=dict)
    bidirectional_palettes: dict = field(default_factory=dict)
    block_to_palette: dict = field(default_factory=dict)
