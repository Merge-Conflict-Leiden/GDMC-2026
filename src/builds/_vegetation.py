"""
_vegetation.py
--------------
Shared vegetation-clearing utility for building placement and farmland generation.

Strategy (three-phase):
  1. Trunk scan  — BFS from the footprint; detect trunk bases in ALL cells
                   (including inside the footprint).  Protected columns (occupied
                   by earlier buildings/walls) are skipped because their logs are
                   building blocks, not trees.
  2. Box clear   — for every detected trunk, clear the species-appropriate
                   bounding box from terrain upward.  No separate REMOVE_RADIUS
                   gate: every trunk found within SEARCH_RADIUS is removed.
  3. Residual    — scan every cell in the BFS range with early-exit per column;
                   clear any leaf block whose `persistent` state is NOT "true".
                   Natural leaves are always persistent=false (set by the game);
                   our placer always sets persistent=true, so this is an exact
                   discriminator that safely ignores building leaves and never
                   clears anything inside a protected column that belongs to us.
  4. Ground sweep — probabilistic near-footprint removal of small vegetation.

Floating-leaf / floating-log sources addressed:
  * Trees rooted inside the footprint (d=0) — now scanned in Phase 1.
  * Trunks at d=8–10 previously discarded by REMOVE_RADIUS — now kept.
  * Canopy from trunks beyond SEARCH_RADIUS — caught by Phase 3 leaf scan.
  * Natural leaves above protected building columns — cleared by Phase 3
    (persistent=false check prevents touching building leaves there).
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
from gdpc.block import Block
from gdpc.editor import Editor

from terrain.terrain_types import TerrainMap

_AIR = Block("minecraft:air")

# --- Tuning constants ---
# Trees are removed as whole units by following their connected logs, then
# orphaned canopy is deleted by replaying Minecraft's leaf-decay rule.  This is
# bounded by *which trunks we pick* (bases inside the build region), never by
# connectivity — so a jungle is never nuked: neighbouring trees keep their
# trunks and therefore their leaf support.
_VEG_MARGIN = 3  # footprint dilation: trunks based this close are removed
_VEG_LEAF_SUPPORT_R = 6  # MC leaf-decay radius: a leaf survives within this of a log
_VEG_CANOPY_MAX = 40  # Y blocks above ground scanned for logs/leaves
_VEG_MAX_H_SPREAD = 9  # cap: don't follow a branch further than this from its base
_VEG_TOTAL_LOG_CAP = 20000  # global safety cap on logs removed in one call
# Ground-feather removal probability by ring distance (1..margin) from the plot.
_VEG_GROUND_PROBS = (1.0, 0.80, 0.55, 0.25)

_AIR_NAMES = frozenset({"air", "cave_air", "void_air"})

_GROUND_VEG_IDS = frozenset(
    {
        "short_grass",
        "grass",
        "tall_grass",
        "fern",
        "large_fern",
        "dead_bush",
        "vine",
        "twisting_vines",
        "twisting_vines_plant",
        "weeping_vines",
        "weeping_vines_plant",
        "cave_vines",
        "cave_vines_plant",
        "hanging_roots",
        "bamboo",
        "bamboo_sapling",
        "sugar_cane",
        "sweet_berry_bush",
        "firefly_bush",
        "glow_berries",
        "azalea",
        "flowering_azalea",
        "mangrove_propagule",
        "mangrove_roots",
        "muddy_mangrove_roots",
        "dandelion",
        "poppy",
        "blue_orchid",
        "allium",
        "azure_bluet",
        "white_tulip",
        "orange_tulip",
        "pink_tulip",
        "red_tulip",
        "oxeye_daisy",
        "cornflower",
        "lily_of_the_valley",
        "wither_rose",
        "sunflower",
        "lilac",
        "rose_bush",
        "peony",
        "spore_blossom",
        "cactus",
        "pink_petals",
        "torchflower",
        "pitcher_plant",
        "moss_carpet",
        "big_dripleaf",
        "big_dripleaf_stem",
        "small_dripleaf",
        "red_mushroom",
        "brown_mushroom",
    }
)


def is_trunk_block(block_id: str) -> bool:
    b = block_id.replace("minecraft:", "")
    return (
        b.endswith("_log")
        or b.endswith("_wood")
        or b in ("bamboo", "bamboo_block", "mushroom_stem")
    )


def _is_bulk_tree(block_id: str) -> bool:
    """Solid tree body followed by the whole-tree removal: trunk blocks plus
    giant-mushroom caps (which, like trunks, don't decay and must be removed by
    connectivity — leaves are handled separately by support-based decay)."""
    b = block_id.replace("minecraft:", "")
    return is_trunk_block(block_id) or b in (
        "red_mushroom_block",
        "brown_mushroom_block",
    )


def is_log(block_id: str) -> bool:
    return is_trunk_block(block_id)


def is_clearable_veg(block_id: str) -> bool:
    b = block_id.replace("minecraft:", "")
    return (
        b.endswith("_leaves")
        or b.endswith("_log")
        or b.endswith("_wood")
        or b.endswith("_sapling")
        or b
        in (
            "bamboo",
            "bamboo_block",
            "mushroom_stem",
            "red_mushroom_block",
            "brown_mushroom_block",
        )
        or b in _GROUND_VEG_IDS
    )


def clear_vegetation(
    fp: list[tuple[int, int]],
    feat: np.ndarray,
    depth: int,
    width: int,
    editor: Editor,
    terrain_map: TerrainMap,
    protected: set[tuple[int, int]] | None = None,
) -> None:
    """
    Clear trees and ground vegetation around (and including) a footprint.

    Parameters
    ----------
    fp          : local (lz, lx) cells forming the footprint.
    feat        : FEATURE_DTYPE structured array, shape (depth, width).
    depth/width : map dimensions.
    editor      : GDPC Editor instance.
    terrain_map : TerrainMap (for local_to_world).
    protected   : cells occupied by earlier buildings, walls, or roads.
                  Their columns are never used as trunk bases and are never
                  removed, so building timber/leaves are always safe.

    Approach
    --------
    Trees are removed as *whole units*: a trunk whose base sits in the build
    region is followed through its connected logs (any height, 2×2 bases and
    branches included) and deleted; then Minecraft's leaf-decay rule is replayed
    so canopy orphaned by the removed trunks disappears while neighbouring trees
    — which keep their trunks and therefore their leaf support — stay intact.
    Selection is bounded by *which trunk bases we pick*, never by connectivity,
    so a dense forest is never over-cleared.
    """
    fp_set = set(fp)
    if protected is None:
        protected = set()

    wm = terrain_map.water_mask

    def _is_water(lz: int, lx: int) -> bool:
        if wm is not None:
            return bool(wm[lz, lx])
        return float(feat[lz, lx]["water_pct"]) > 0.5

    def _in_bounds(lz: int, lx: int) -> bool:
        return 0 <= lz < depth and 0 <= lx < width

    # ── Region: footprint dilated by the margin ──────────────────────────────
    region: set[tuple[int, int]] = set()
    for lz, lx in fp_set:
        for dz in range(-_VEG_MARGIN, _VEG_MARGIN + 1):
            for dx in range(-_VEG_MARGIN, _VEG_MARGIN + 1):
                if _in_bounds(lz + dz, lx + dx):
                    region.add((lz + dz, lx + dx))

    # ── 1. Trunk bases inside the region ─────────────────────────────────────
    # feat["height"] is the first cell above ground (the NO_PLANTS heightmap
    # stays at ground level even under a canopy), which is exactly where a trunk
    # base log sits.  Protected and water columns are skipped.
    bases: list[tuple[int, int, int]] = []
    for lz, lx in region:
        if (lz, lx) in protected or _is_water(lz, lx):
            continue
        wx, wz = terrain_map.local_to_world(lz, lx)
        gy = int(feat[lz, lx]["height"])
        if _is_bulk_tree(editor.getBlock((wx, gy, wz)).id or ""):
            bases.append((wx, gy, wz))

    # ── 2. Remove whole trees by connected logs (capped, protected-safe) ─────
    # 26-neighbour flood over bulk-tree blocks catches tall trunks, 2×2 jungle
    # bases and angled branches.  The per-base horizontal cap stops a touching
    # branch cascading into a neighbour; protected columns are never eaten.
    removed: set[tuple[int, int, int]] = set()
    removed_cols: set[tuple[int, int]] = set()
    for bx, by, bz in bases:
        if (bx, by, bz) in removed:
            continue
        stack = [(bx, by, bz)]
        while stack:
            x, y, z = stack.pop()
            if (x, y, z) in removed:
                continue
            if abs(x - bx) > _VEG_MAX_H_SPREAD or abs(z - bz) > _VEG_MAX_H_SPREAD:
                continue
            lz, lx = terrain_map.world_to_local(x, z)
            if not _in_bounds(lz, lx) or (lz, lx) in protected:
                continue
            if not _is_bulk_tree(editor.getBlock((x, y, z)).id or ""):
                continue
            removed.add((x, y, z))
            removed_cols.add((lz, lx))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx or dy or dz:
                            stack.append((x + dx, y + dy, z + dz))
            if len(removed) >= _VEG_TOTAL_LOG_CAP:
                break
        if len(removed) >= _VEG_TOTAL_LOG_CAP:
            break
    for p in removed:
        editor.placeBlock(p, _AIR)

    # ── 3. Support-based leaf decay (replays Minecraft's rule) ───────────────
    # A persistent=false leaf survives iff a SURVIVING log is within the decay
    # radius.  Neighbouring trees keep their trunks, so their canopy is kept;
    # only leaves orphaned by the trees we removed vanish.  This is what makes
    # whole-tree removal safe in a jungle — no floaters, no biome nuke.
    if removed_cols:
        search: set[tuple[int, int]] = set()
        for lz, lx in removed_cols:
            for dz in range(-_VEG_LEAF_SUPPORT_R, _VEG_LEAF_SUPPORT_R + 1):
                for dx in range(-_VEG_LEAF_SUPPORT_R, _VEG_LEAF_SUPPORT_R + 1):
                    if _in_bounds(lz + dz, lx + dx):
                        search.add((lz + dz, lx + dx))

        surviving_logs: set[tuple[int, int, int]] = set()
        candidate_leaves: list[tuple[int, int, int]] = []
        for lz, lx in search:
            wx, wz = terrain_map.local_to_world(lz, lx)
            gy = int(feat[lz, lx]["height"])
            # Scan the full canopy height — no air-stop: a tree's canopy sits
            # above a bare-trunk/air gap (worse once we punch out the trunk), so
            # early-stopping on air would miss it.
            for dy in range(_VEG_CANOPY_MAX):
                y = gy + dy
                blk = editor.getBlock((wx, y, wz))
                bid = blk.id or ""
                bname = bid.replace("minecraft:", "")
                if not bid or bname in _AIR_NAMES:
                    continue
                if _is_bulk_tree(bid):
                    if (wx, y, wz) not in removed:
                        surviving_logs.add((wx, y, wz))
                elif (
                    bname.endswith("_leaves") and blk.states.get("persistent") != "true"
                ):
                    candidate_leaves.append((wx, y, wz))

        # Dilate surviving logs into a "supported" volume, then drop the rest.
        supported: set[tuple[int, int, int]] = set()
        r = _VEG_LEAF_SUPPORT_R
        for x, y, z in surviving_logs:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        supported.add((x + dx, y + dy, z + dz))
        for p in candidate_leaves:
            if p not in supported:
                editor.placeBlock(p, _AIR)

    # ── 4. Footprint airspace trim ───────────────────────────────────────────
    # Clear anything still standing directly over the build footprint: the
    # branches of an outside tree that overhang the roof (that tree keeps its
    # trunk and the rest of its canopy — it is merely brushed back at the wall),
    # plus any leftover ground cover in the plot itself.
    for lz, lx in fp_set:
        if (lz, lx) in protected or _is_water(lz, lx):
            continue
        wx, wz = terrain_map.local_to_world(lz, lx)
        gy = int(feat[lz, lx]["height"])
        # Full-height scan (no air-stop): overhanging canopy sits above an air gap.
        for dy in range(_VEG_CANOPY_MAX):
            y = gy + dy
            blk = editor.getBlock((wx, y, wz))
            bid = blk.id or ""
            bname = bid.replace("minecraft:", "")
            if not bid or bname in _AIR_NAMES:
                continue
            is_leaf = (
                bname.endswith("_leaves") and blk.states.get("persistent") != "true"
            )
            if is_leaf or _is_bulk_tree(bid) or bname in _GROUND_VEG_IDS:
                editor.placeBlock((wx, y, wz), _AIR)

    # ── 5. Ground-cover feather over the margin ──────────────────────────────
    # Light probabilistic sweep of small plants in the margin ring so the
    # clearing fades into the terrain instead of ending on a hard line.
    fdist: dict[tuple[int, int], int] = {c: 0 for c in fp_set}
    fq: deque[tuple[int, int]] = deque(fp_set)
    while fq:
        lz, lx = fq.popleft()
        d = fdist[(lz, lx)]
        if d >= _VEG_MARGIN:
            continue
        for ddz, ddx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (lz + ddz, lx + ddx)
            if nb not in fdist and _in_bounds(nb[0], nb[1]):
                fdist[nb] = d + 1
                fq.append(nb)
    for (lz, lx), d in fdist.items():
        if d == 0 or (lz, lx) in protected or _is_water(lz, lx):
            continue
        if random.random() > _VEG_GROUND_PROBS[d - 1]:
            continue
        wx, wz = terrain_map.local_to_world(lz, lx)
        gy = int(feat[lz, lx]["height"])
        for dy in range(3):
            blk = editor.getBlock((wx, gy + dy, wz))
            bid = blk.id or ""
            if _is_bulk_tree(bid):
                break
            bname = bid.replace("minecraft:", "")
            is_leaf = (
                bname.endswith("_leaves") and blk.states.get("persistent") != "true"
            )
            if is_leaf or bname in _GROUND_VEG_IDS:
                editor.placeBlock((wx, gy + dy, wz), _AIR)
