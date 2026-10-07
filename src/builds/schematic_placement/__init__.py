"""
Unified placement of CSV-based schematic buildings.

Split across modules, class SchematicBuildingPlacer is the public entry point.
"""

from .model import Building, PlacedBuilding, analyse
from .placer import SchematicBuildingPlacer

__all__ = [
    "Building",
    "PlacedBuilding",
    "SchematicBuildingPlacer",
    "analyse",
]
