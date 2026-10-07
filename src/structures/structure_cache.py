"""
This module defines the structure caching system.
"""

import ast
from pathlib import Path
from typing import Tuple

import pandas as pd


def _calculate_dimensions(df: pd.DataFrame, csv_path: str) -> Tuple[int, int, int]:
    """
    Calculate the dimensions of a structure from its dataframe.

    :param df: Dataframe of structure
    :param csv_path: Relative path to the CSV file of structure
    :return: Tuple of (width, height, depth)
    """
    try:
        positions = df["Position"].map(ast.literal_eval)
        coords = pd.DataFrame(positions.tolist(), columns=["x", "y", "z"])
    except (ValueError, SyntaxError, TypeError) as e:
        raise ValueError(f"Invalid position format in {csv_path}") from e

    width = int(coords["x"].max()) + 1
    height = int(coords["y"].max()) + 1
    depth = int(coords["z"].max()) + 1

    return width, height, depth


# TODO: Add heading to ensure direction of placed object is general for all
class StructureCache:
    """Cache loaded CSV structures."""

    def __init__(self, build_dir: str, build_output_dir: str):
        self.base_path = Path(build_dir) / build_output_dir
        self._cache: dict[str, Tuple[pd.DataFrame, int, int, int]] = {}

    def load_structure(self, csv_path: str) -> Tuple[pd.DataFrame, int, int, int]:
        """
        Load a structure from a CSV file with caching.

        :param csv_path: Relative path to the CSV file of structure
        :return: Tuple of (dataframe, width, height, depth) of structure
        """
        if csv_path in self._cache:
            return self._cache[csv_path]

        full_path = self.base_path / csv_path
        if not full_path.exists():
            raise FileNotFoundError(f"Structure file not found: {full_path}")

        df = pd.read_csv(full_path)
        if df.empty:
            raise ValueError(f"Structure file is empty: {csv_path}")

        width, height, depth = _calculate_dimensions(df, csv_path)

        result = (df, width, height, depth)
        self._cache[csv_path] = result
        return result

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
