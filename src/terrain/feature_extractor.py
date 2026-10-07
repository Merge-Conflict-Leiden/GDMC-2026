"""
feature_extractor.py
--------------------
Phase 1 & 2 of the terrain segmentation pipeline.

Phase 1 — Raw data acquisition:
    Accepts pre-fetched numpy arrays (heightmap, biome_map, water_mask,
    surface_blocks) so the module stays testable without a live Minecraft
    connection.  The caller (main pipeline) is responsible for fetching via
    GDPC.

Phase 2 — Per-cell feature computation:
    Sobel slope, roughness (rolling std-dev), water distance transform,
    local water fraction, tree density, border mask.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    uniform_filter,
)
from skimage.filters import sobel

from .terrain_types import FEATURE_DTYPE, TerrainMap

# ---------------------------------------------------------------------------
# Minecraft block IDs / name fragments considered as "water"
# ---------------------------------------------------------------------------
WATER_BLOCK_NAMES = frozenset(
    {
        "minecraft:water",
        "minecraft:flowing_water",
        "minecraft:lava",  # also off-limits
        "minecraft:flowing_lava",
        "minecraft:kelp",
        "minecraft:seagrass",
        "minecraft:tall_seagrass",
    }
)

# Ice is a SOLID block, so frozen water reads as flat buildable ground on the
# heightmap (surface − floor == 0).  These are detected separately and folded
# into the water mask so frozen lakes/rivers are off-limits like open water.
ICE_BLOCK_NAMES = frozenset(
    {
        "minecraft:ice",
        "minecraft:packed_ice",
        "minecraft:blue_ice",
        "minecraft:frosted_ice",
    }
)

# Log-like blocks → tree density.  Canonical list, imported by the segmenter so
# tree detection during world scanning matches what downstream tree_density
# scores expect.  Includes bamboo and mushroom stems so jungle and mushroom-
# field biomes are correctly identified as "treed" terrain.
TREE_BLOCK_NAMES = frozenset(
    {
        "minecraft:oak_log",
        "minecraft:birch_log",
        "minecraft:spruce_log",
        "minecraft:jungle_log",
        "minecraft:acacia_log",
        "minecraft:dark_oak_log",
        "minecraft:mangrove_log",
        "minecraft:cherry_log",
        "minecraft:pale_oak_log",
        "minecraft:oak_wood",
        "minecraft:birch_wood",
        "minecraft:spruce_wood",
        "minecraft:jungle_wood",
        "minecraft:acacia_wood",
        "minecraft:dark_oak_wood",
        "minecraft:mangrove_wood",
        "minecraft:cherry_wood",
        "minecraft:pale_oak_wood",
        "minecraft:bamboo",
        "minecraft:bamboo_block",
        "minecraft:mushroom_stem",
    }
)

# Biome name substrings that imply ocean / river — used as an additional water hint
WATER_BIOME_KEYWORDS = frozenset({"ocean", "river"})

# Sobel roughness window (half-width in cells)
SLOPE_WINDOW = 1  # 3×3 Sobel kernel (built into skimage.filters.sobel)
ROUGHNESS_WINDOW = 7  # std-dev window for height variance
WATER_DIST_CUTOFF = 48.0  # cells beyond this get clamped to 1.0 (normalised)
TREE_DENSITY_WINDOW = 5  # for local log-block fraction


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features(
    terrain_map: TerrainMap,
    heightmap: np.ndarray,  # [z, x] int16 — surface Y
    biome_map: np.ndarray,  # [z, x] uint16 — dominant Minecraft biome ID
    water_mask: np.ndarray,  # [z, x] bool   — True where water/lava at surface
    tree_mask: np.ndarray,  # [z, x] bool   — True where log block at surface+1..+5
    *,
    border_buffer: int = 2,  # cells from build-area edge counted as "border"
    blur_sigma: float = 1.5,  # Gaussian sigma applied before slope for smoothness
) -> np.ndarray:
    """
    Compute the FEATURE_DTYPE structured array from raw maps.

    Returns the array AND writes it into terrain_map.features in-place.

    Parameters
    ----------
    terrain_map   : TerrainMap with x0,z0,x1,z1 already set.
    heightmap     : surface Y values, shape (depth, width).
    biome_map     : dominant biome ID per cell, same shape.
    water_mask    : True where surface is water/lava, same shape.
    tree_mask     : True where log block detected, same shape.
    border_buffer : cells from the build-area edge marked as is_border.
    blur_sigma    : Gaussian blur sigma applied to heightmap before Sobel.
    """
    depth, width = heightmap.shape
    assert biome_map.shape == (depth, width), "biome_map shape mismatch"
    assert water_mask.shape == (depth, width), "water_mask shape mismatch"
    assert tree_mask.shape == (depth, width), "tree_mask shape mismatch"

    feat = np.zeros((depth, width), dtype=FEATURE_DTYPE)

    # -- Height -----------------------------------------------------------------
    feat["height"] = heightmap.astype(np.int16)

    # -- Slope (Sobel on Gaussian-blurred heightmap) ---------------------------
    # Using float64 internally for precision; Sobel returns float64.
    h_float = heightmap.astype(np.float64)
    h_blur = gaussian_filter(h_float, sigma=blur_sigma)
    slope = sobel(h_blur)  # magnitude of gradient (blocks/cell)
    feat["slope"] = slope.astype(np.float32)

    # -- Roughness (height std-dev in 7×7 window) ------------------------------
    # We compute E[X²] - E[X]² via two uniform_filter passes (O(n), fast).
    h_mean_sq = uniform_filter(h_float**2, size=ROUGHNESS_WINDOW)
    h_sq_mean = uniform_filter(h_float, size=ROUGHNESS_WINDOW) ** 2
    variance = np.maximum(0.0, h_mean_sq - h_sq_mean)  # numerical guard
    feat["roughness"] = np.sqrt(variance).astype(np.float32)

    # -- Water proximity -------------------------------------------------------
    # distance_transform_edt: distance in cells to nearest False pixel.
    # "False" in our input = water cell → we want distance from water.
    not_water = ~water_mask
    raw_dist = distance_transform_edt(not_water).astype(np.float32)
    feat["water_dist"] = raw_dist

    # -- Local water fraction (5×5 window) ------------------------------------
    # uniform_filter on bool (as float) gives local mean = fraction.
    local_water = uniform_filter(
        water_mask.astype(np.float32), size=5, mode="constant", cval=0.0
    )
    feat["water_pct"] = local_water

    # -- Tree density (5×5 window) --------------------------------------------
    local_tree = uniform_filter(
        tree_mask.astype(np.float32),
        size=TREE_DENSITY_WINDOW,
        mode="constant",
        cval=0.0,
    )
    feat["tree_density"] = local_tree

    # -- Border mask -----------------------------------------------------------
    border = np.zeros((depth, width), dtype=np.bool_)
    if border_buffer > 0:
        border[:border_buffer, :] = True
        border[-border_buffer:, :] = True
        border[:, :border_buffer] = True
        border[:, -border_buffer:] = True
    feat["is_border"] = border

    # -- Biome -----------------------------------------------------------------
    feat["biome_id"] = biome_map.astype(np.uint16)

    terrain_map.features = feat
    return feat


# ---------------------------------------------------------------------------
# Normalised feature accessors  (used in Phase 4 scoring)
# ---------------------------------------------------------------------------


def normalise(arr: np.ndarray, lo: float = None, hi: float = None) -> np.ndarray:
    """Min-max normalise to [0, 1]. Clips at lo/hi if provided."""
    lo = lo if lo is not None else float(arr.min())
    hi = hi if hi is not None else float(arr.max())
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr.clip(lo, hi) - lo) / (hi - lo)).astype(np.float32)


def slope_norm(feat: np.ndarray, max_slope: float = 8.0) -> np.ndarray:
    """Normalised slope: 0 = flat, 1 = very steep (≥ max_slope blocks/cell)."""
    return normalise(feat["slope"], 0.0, max_slope)


def roughness_norm(feat: np.ndarray, max_roughness: float = 12.0) -> np.ndarray:
    return normalise(feat["roughness"], 0.0, max_roughness)


def water_dist_norm(feat: np.ndarray) -> np.ndarray:
    """Normalised distance to water: 0 = on water, 1 = far away."""
    return normalise(feat["water_dist"], 0.0, WATER_DIST_CUTOFF)


def build_water_mask_from_block_names(
    surface_block_names: np.ndarray,  # [z, x] object/str array of block IDs
) -> np.ndarray:
    """
    Helper: construct a bool water_mask from a string block-name array.
    This is what you'd call after fetching with GDPC's WorldSlice.
    """
    mask = np.zeros(surface_block_names.shape, dtype=np.bool_)
    for name in WATER_BLOCK_NAMES:
        mask |= surface_block_names == name
    return mask


def build_tree_mask_from_block_names(
    block_names_at_height_plus_1: np.ndarray,
) -> np.ndarray:
    """Helper: detect log blocks just above surface."""
    mask = np.zeros(block_names_at_height_plus_1.shape, dtype=np.bool_)
    for name in TREE_BLOCK_NAMES:
        mask |= block_names_at_height_plus_1 == name
    return mask
