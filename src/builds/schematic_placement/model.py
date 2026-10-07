"""
Schematic descriptors and CSV analysis.

:class:`Building` and :class:`PlacedBuilding` are the data structures shared
by every placement mode; :func:`analyse` parses a CSV schematic into a
Building descriptor.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class Building:
    """Universal descriptor for a CSV-based schematic."""

    csv_path: str
    category: str
    W: int
    D: int
    H: int
    footprint: frozenset[tuple[int, int]]  # (sx, sz) at y=0, excl. jigsaw/air
    shadow: frozenset[tuple[int, int]]  # (sx, sz) non-air at ANY y (⊇ footprint)
    jigsaw_offsets: list[tuple[int, int, int]]  # (sx, sy, sz) of jigsaw blocks
    door_face: str  # 'north'|'south'|'east'|'west'
    door_x: int  # entrance centroid in schematic space
    door_z: int
    lectern_offset: tuple[int, int, int] | None = None


@dataclass
class PlacedBuilding:
    """Result returned by placement methods."""

    anchor: tuple[int, int]  # local (lz, lx)
    direction: int
    floor_y: int
    entrance_local: Optional[tuple[int, int]]  # local (lz, lx) of entrance cell


def analyse(
    rel_path: str, csv_path: Path, category: str, footprint_y: int = 0
) -> Optional[Building]:
    """
    Parse a CSV schematic into a Building descriptor.

    Entrance detection order:
      1. Door blocks (half=lower)
      2. Fence gate blocks (open or closed)
      3. minecraft:jigsaw offsets[0]
      4. Fallback: centre of south (high-z) face
    """
    df = pd.read_csv(csv_path).dropna(subset=["BlockID"])
    if df.empty:
        return None
    return building_from_df(rel_path, df, category, footprint_y)


def building_from_df(
    rel_path: str, df: pd.DataFrame, category: str, footprint_y: int = 0
) -> Optional[Building]:
    """
    Build a :class:`Building` descriptor from an in-memory schematic frame.
    The df-parsing core of :func:`analyse`, kept separate so a frame can be
    analysed without re-reading it from disk.
    """
    if df.empty:
        return None

    positions = df["Position"].map(ast.literal_eval)
    coords = pd.DataFrame(positions.tolist(), columns=["x", "y", "z"])
    W = int(coords["x"].max()) + 1
    D = int(coords["z"].max()) + 1
    H = int(coords["y"].max()) + 1

    jig_mask = df["BlockID"].str.contains("jigsaw", case=False, na=False)
    air_mask = df["BlockID"].str.contains(r"\bair\b", case=False, na=False)

    # Footprint: solid, non-jigsaw blocks at footprint_y.
    y0_solid = (coords["y"] == footprint_y) & ~jig_mask & ~air_mask
    if y0_solid.sum() > 0:
        xs = coords.loc[y0_solid, "x"].astype(int)
        zs = coords.loc[y0_solid, "z"].astype(int)
        footprint: frozenset[tuple[int, int]] = frozenset(zip(xs.tolist(), zs.tolist()))
    else:
        footprint = frozenset((x, z) for x in range(W) for z in range(D))

    # Shadow: solid columns at ANY y — the true horizontal extent including
    # roof/canopy overhangs.  Used for collision checks so overhanging blocks
    # cannot be stamped over previously placed structures.
    solid = ~jig_mask & ~air_mask
    sxs = coords.loc[solid, "x"].astype(int)
    szs = coords.loc[solid, "z"].astype(int)
    shadow = frozenset(zip(sxs.tolist(), szs.tolist())) | footprint

    # Jigsaw positions.
    jig_coords = coords[jig_mask]
    jigsaw_offsets = [
        (int(r.x), int(r.y), int(r.z)) for r in jig_coords.itertuples(index=False)
    ]

    # Entrance: doors → fence gates → jigsaw → fallback.
    door_mask = df["BlockID"].str.contains("_door", case=False, na=False) & ~df[
        "BlockID"
    ].str.contains("trapdoor", case=False, na=False)
    lower_mask = door_mask & df["States"].str.contains(
        "'half'.*'lower'", case=False, na=False
    )
    if not lower_mask.any():
        lower_mask = door_mask

    gate_mask = df["BlockID"].str.contains("fence_gate", case=False, na=False)

    if lower_mask.any():
        dc = coords[lower_mask]
        min_y = dc["y"].min()
        gd = dc[dc["y"] <= min_y + 1]
        dx = int(round(gd["x"].mean()))
        dz = int(round(gd["z"].mean()))
    elif gate_mask.any():
        gc = coords[gate_mask]
        min_y = gc["y"].min()
        gg = gc[gc["y"] <= min_y + 1]
        dx = int(round(gg["x"].mean()))
        dz = int(round(gg["z"].mean()))
    elif jigsaw_offsets:
        dx, _, dz = jigsaw_offsets[0]
    else:
        dx, dz = W // 2, D - 1

    door_face = _door_face(dx, dz, W, D)

    lectern_mask = df["BlockID"] == "minecraft:lectern"
    lectern_offset: tuple[int, int, int] | None = None
    if lectern_mask.any():
        row = coords[lectern_mask].iloc[0]
        lectern_offset = (int(row.x), int(row.y), int(row.z))

    return Building(
        csv_path=rel_path,
        category=category,
        W=W,
        D=D,
        H=H,
        footprint=footprint,
        shadow=shadow,
        jigsaw_offsets=jigsaw_offsets,
        door_face=door_face,
        door_x=dx,
        door_z=dz,
        lectern_offset=lectern_offset,
    )


def _door_face(door_x: int, door_z: int, W: int, D: int) -> str:
    dists = {
        "north": door_z,
        "south": D - 1 - door_z,
        "west": door_x,
        "east": W - 1 - door_x,
    }
    return min(dists, key=dists.__getitem__)
