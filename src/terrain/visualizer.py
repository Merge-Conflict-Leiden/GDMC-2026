"""
Matplotlib 2x2 plot visualization of the terrain segmentation output.
"""

from __future__ import annotations

import numpy as np

from .terrain_types import SubZone, TerrainMap, ZoneType

# ZoneType -> (hex color, display label)
_ZONE_META: dict[ZoneType, tuple[str, str]] = {
    ZoneType.UNCLASSIFIED: ("#aaaaaa", "Unclassified"),
    ZoneType.OFF_LIMITS: ("#333333", "Off-limits"),
    ZoneType.WALL_PERIMETER: ("#8B4513", "Wall"),
    ZoneType.RURAL: ("#4CAF50", "Rural"),
    ZoneType.HARBOR: ("#2196F3", "Harbor"),
    ZoneType.URBAN: ("#FF5722", "Urban"),
}

# SubZone -> hex colour
_SUB_ZONE_COLOR: dict[SubZone, str] = {
    SubZone.NONE: "#aaaaaa",
    SubZone.FARMLAND: "#a5d6a7",
    SubZone.ORCHARD: "#66bb6a",
    SubZone.ANIMAL_PEN: "#81c784",
    SubZone.FOREST_BUFFER: "#1b5e20",
    SubZone.TOWN_CENTER: "#e91e63",
    SubZone.CIVIC: "#9c27b0",
    SubZone.RESIDENTIAL: "#ff7043",
    SubZone.CASTLE: "#b71c1c",
    SubZone.GATE_SITE: "#5d4037",
    SubZone.TOWER_SITE: "#6d4c41",
    SubZone.HARBOR_DOCK: "#0288d1",
    SubZone.BRIDGE_SITE: "#00acc1",
}

# Special-site marker style: (marker, colour, size, label)
_SITE_MARKERS = [
    ("town_center", "*", "gold", 12, "Town center"),
    ("castle_site", "D", "crimson", 8, "Castle"),
    ("gate_sites", "^", "sienna", 7, "Gate"),
    ("harbor_sites", "o", "deepskyblue", 7, "Harbor"),
    ("bridge_sites", "s", "cyan", 6, "Bridge"),
    ("windmill_sites", "P", "yellow", 7, "Windmill"),
]


def plot_terrain_map(
    terrain_map: TerrainMap,
    heightmap: np.ndarray | None = None,
    *,
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """
    Render a 4-panel figure showing the terrain segmentation result.

    Panels
    ------
    1. Heightmap  — raw elevation (skipped if heightmap is None).
    2. Zone map   — coarse ZoneType classification with special-site markers.
    3. Sub-zone map — fine SubZone classification.
    4. District map — Voronoi cells with seed points.

    Parameters
    ----------
    terrain_map : fully-populated TerrainMap returned by TerrainSegmenter.run().
    heightmap   : (depth, width) int16 elevation array; pass self._heightmap
                  from TerrainSegmenter if available.
    show        : call plt.show() when done (set False when saving only).
    save_path   : optional file path to save the figure (e.g. "terrain.png").
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install it with: pip install matplotlib"
        ) from exc

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Terrain Segmentation", fontsize=14, fontweight="bold")
    ax_height, ax_zone, ax_sub, ax_district = axes.flat

    # World-coordinate extent for all imshow calls.
    # Format: [left, right, bottom, top] with origin="upper":
    #   x-axis → world X  (left = x0, right = x1+1)
    #   y-axis → world Z  (top  = z0, bottom = z1+1 because origin="upper" flips y)
    x0, z0 = terrain_map.x0, terrain_map.z0
    x1, z1 = terrain_map.x1, terrain_map.z1
    extent = [x0, x1 + 1, z1 + 1, z0]

    # -- Panel 1: heightmap ---------------------------------------------------
    if heightmap is not None:
        im = ax_height.imshow(heightmap, cmap="terrain", origin="upper", extent=extent)

        if terrain_map.water_mask is not None:
            _overlay_water(ax_height, terrain_map.water_mask, extent=extent)

        fig.colorbar(im, ax=ax_height, label="Y level", shrink=0.8)
    else:
        ax_height.text(
            0.5,
            0.5,
            "Heightmap not available",
            ha="center",
            va="center",
            transform=ax_height.transAxes,
        )
    ax_height.set_title("Heightmap")

    # -- Panel 2: zone map ----------------------------------------------------
    if terrain_map.zone_map is not None:
        zone_vals = sorted(int(z) for z in ZoneType)
        zone_colors = [_ZONE_META[ZoneType(v)][0] for v in zone_vals]
        zone_cmap = ListedColormap(zone_colors)
        remap_zone = {v: i for i, v in enumerate(zone_vals)}
        zone_img = _remap(terrain_map.zone_map, remap_zone)
        ax_zone.imshow(
            zone_img,
            cmap=zone_cmap,
            vmin=0,
            vmax=len(zone_vals) - 1,
            origin="upper",
            extent=extent,
        )

        if terrain_map.water_mask is not None:
            _overlay_water(ax_zone, terrain_map.water_mask, extent=extent)

        if terrain_map.district_map is not None:
            _overlay_borders(ax_zone, terrain_map.district_map, extent=extent)

        # Road overlay: drawn before special-site markers so markers sit on top
        if terrain_map.road_map is not None:
            overlay = np.zeros((*terrain_map.road_map.shape, 4), dtype=np.float32)
            overlay[terrain_map.road_map == 3] = [
                1.00,
                1.00,
                1.00,
                0.90,
            ]  # primary:   white
            overlay[terrain_map.road_map == 2] = [
                0.85,
                0.85,
                0.85,
                0.80,
            ]  # secondary: light grey
            overlay[terrain_map.road_map == 1] = [
                0.60,
                0.45,
                0.20,
                0.70,
            ]  # tertiary:  dirt brown
            ax_zone.imshow(overlay, origin="upper", extent=extent)

        _plot_special_sites(ax_zone, terrain_map, x0=x0, z0=z0)

        legend_patches = [
            mpatches.Patch(
                color=_ZONE_META[ZoneType(v)][0], label=_ZONE_META[ZoneType(v)][1]
            )
            for v in zone_vals
        ]
        ax_zone.legend(
            handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.85
        )
    ax_zone.set_title("Zone Map")

    # -- Panel 3: sub-zone map ------------------------------------------------
    if terrain_map.sub_zone_map is not None:
        present_subs = sorted(set(terrain_map.sub_zone_map.ravel().tolist()))
        sub_colors = [_SUB_ZONE_COLOR.get(SubZone(v), "#ffffff") for v in present_subs]
        sub_cmap = ListedColormap(sub_colors)
        remap_sub = {v: i for i, v in enumerate(present_subs)}
        sub_img = _remap(terrain_map.sub_zone_map, remap_sub)
        ax_sub.imshow(
            sub_img,
            cmap=sub_cmap,
            vmin=0,
            vmax=len(present_subs) - 1,
            origin="upper",
            extent=extent,
        )

        sub_patches = [
            mpatches.Patch(
                color=_SUB_ZONE_COLOR.get(SubZone(v), "#ffffff"), label=SubZone(v).name
            )
            for v in present_subs
            if SubZone(v) != SubZone.NONE
        ]
        if sub_patches:
            ax_sub.legend(
                handles=sub_patches,
                loc="upper right",
                fontsize=6,
                framealpha=0.85,
                ncol=max(1, len(sub_patches) // 8),
            )
    ax_sub.set_title("Sub-zone Map")

    # -- Panel 4: district map ------------------------------------------------
    if terrain_map.district_map is not None:
        dmap = terrain_map.district_map.astype(np.float32)
        dmap[dmap < 0] = np.nan
        ax_district.imshow(
            dmap, cmap="tab20b", origin="upper", interpolation="nearest", extent=extent
        )
        for d in terrain_map.districts:
            if d.size > 0:
                sz, sx = d.seed
                ax_district.plot(sx + x0, sz + z0, "k.", markersize=2, alpha=0.5)
    ax_district.set_title(f"Districts ({len(terrain_map.districts)} Voronoi cells)")

    # Shared axis labels — world coordinates
    for ax in axes.flat:
        ax.set_xlabel("X (world)", fontsize=8)
        ax.set_ylabel("Z (world)", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _overlay_water(ax, water_mask: np.ndarray, *, extent=None) -> None:
    """
    Semi-transparent blue striped water overlay.
    """
    h, w = water_mask.shape
    yy, xx = np.indices((h, w))

    # diagonal hatch pattern
    stripes = ((xx + yy) % 8) < 3

    overlay = np.zeros((h, w, 4), dtype=np.float32)

    # light blue tint everywhere water exists
    overlay[..., 0] = 0.15
    overlay[..., 1] = 0.45
    overlay[..., 2] = 1.00
    overlay[..., 3] = water_mask.astype(np.float32) * 0.18

    ax.imshow(overlay, origin="upper", extent=extent)

    # darker blue stripes
    stripe_overlay = np.zeros((h, w, 4), dtype=np.float32)
    stripe_overlay[..., 1] = 0.35
    stripe_overlay[..., 2] = 1.00
    stripe_overlay[..., 3] = (
        water_mask.astype(np.float32) * stripes.astype(np.float32) * 0.35
    )

    ax.imshow(stripe_overlay, origin="upper", extent=extent)


def _remap(arr: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Remap integer array values through mapping (unmapped values → 0)."""
    out = np.zeros_like(arr, dtype=np.int32)
    for src, dst in mapping.items():
        out[arr == src] = dst
    return out


def _overlay_borders(
    ax, district_map: np.ndarray, *, extent: list | None = None
) -> None:
    """Draw translucent white lines at district boundaries."""
    border = np.zeros(district_map.shape, dtype=np.float32)
    border[:-1, :] += district_map[:-1, :] != district_map[1:, :]
    border[:, :-1] += district_map[:, :-1] != district_map[:, 1:]
    border = np.clip(border, 0, 1)
    ax.imshow(
        border, cmap="binary_r", alpha=border * 0.45, origin="upper", extent=extent
    )


def _plot_special_sites(
    ax, terrain_map: TerrainMap, *, x0: int = 0, z0: int = 0
) -> None:
    """Overlay special-site markers in world coordinates."""
    legend_handles = []
    try:
        import matplotlib.lines as mlines  # noqa: F401  (import confirms matplotlib present)
    except ImportError:
        return

    for attr, marker, color, ms, label in _SITE_MARKERS:
        sites = getattr(terrain_map, attr, None)
        if sites is None:
            continue
        if isinstance(sites, tuple):
            sites = [sites]
        first = True
        for lz, lx in sites:
            # Convert local (z, x) → world coordinates for the plot call.
            wx, wz = lx + x0, lz + z0
            lbl = label if first else "_nolegend_"
            h = ax.plot(
                wx,
                wz,
                marker,
                color=color,
                markersize=ms,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=lbl,
                zorder=5,
            )
            if first:
                legend_handles.append(h[0])
                first = False

    if legend_handles:
        existing = ax.get_legend()
        all_handles = (
            list(existing.legend_handles) + legend_handles
            if existing
            else legend_handles
        )
        all_labels = [h.get_label() for h in all_handles]
        ax.legend(
            all_handles, all_labels, loc="upper right", fontsize=7, framealpha=0.85
        )
