#!/usr/bin/env python3
"""
plot_subtraction_3d_regions_v2.py

Plot true 3D subtraction regions using:
- region_voxels.csv
- region_summary.csv
- rerun_candidates.csv

Adapted from plot_conjunction_3d_regions_v4.py.

Difference from conjunction version
-----------------------------------
Subtraction regions are separated by sign:

positive:
    A - B > 0
    shown as A > B

negative:
    A - B < 0
    shown as B > A

New behavior compared with old subtraction v1
---------------------------------------------
1. Uses true region voxel membership from region_voxels.csv.
2. Keeps combined plot with all selected regions together.
3. Also makes separate plots for each selected region.
4. Supports boundary-only plotting.
5. Supports full-grid and zoomed versions.
6. Uses defaults:
   voxel_size = 10
   voxel_alpha = 0.45
   candidate_size_scale = 1.5

Axes
----
x = NoiseStd
y = Percent
z = Layer

Typical usage
-------------
python plot_subtraction_3d_regions_v2.py \
  --region_voxels_csv ./tfce_results/sub3d_targetpairs_region_voxels.csv \
  --region_csv ./tfce_results/sub3d_targetpairs_region_summary.csv \
  --rerun_csv ./tfce_results/sub3d_targetpairs_rerun_candidates.csv \
  --outdir ./tfce_results/sub3d_targetpairs_3dplots_v2 \
  --pairs Phonemic_minus_Neologism Semantic_minus_Neologism \
  --top_regions 0 \
  --boundary_only \
  --also_full_grid
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# --------------------------------------------------
# Basic helpers
# --------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(s: str) -> str:
    out = str(s)
    for ch in [" ", "/", "\\", ":", ";", ",", ">", "<", "|", "(", ")"]:
        out = out.replace(ch, "_")
    return out


def pretty_contrast_title(pair_name: str, sign: str) -> str:
    """
    Example:
        pair_name = Phonemic_minus_Neologism

    positive:
        Phonemic > Neologism

    negative:
        Neologism > Phonemic
    """
    if "_minus_" not in pair_name:
        return f"{pair_name} ({sign})"

    a, b = pair_name.split("_minus_", 1)

    if sign == "positive":
        return f"{a} > {b}"
    elif sign == "negative":
        return f"{b} > {a}"
    else:
        return f"{pair_name} ({sign})"


def region_color_list(n: int):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def make_point_legend():
    return [
        Line2D(
            [0], [0],
            marker="*",
            color="black",
            linestyle="None",
            markersize=11,
            label="Peak",
        ),
        Line2D(
            [0], [0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            linestyle="None",
            markersize=8,
            label="Centroid",
        ),
        Line2D(
            [0], [0],
            marker="x",
            color="black",
            linestyle="None",
            markersize=7,
            label="Support",
        ),
    ]


def get_region_size_from_summary(region_summary_pair_sign: pd.DataFrame, region_rank: int):
    if region_summary_pair_sign is None or region_summary_pair_sign.empty:
        return None

    row = region_summary_pair_sign[
        region_summary_pair_sign["region_rank"] == region_rank
    ]

    if row.empty:
        return None

    row = row.iloc[0]

    for c in ["size_voxels", "region_size", "size"]:
        if c in row.index:
            try:
                return int(row[c])
            except Exception:
                return row[c]

    return None


def make_region_legend(region_ranks, region_summary_pair_sign, colors):
    handles = []

    for rr, color in zip(region_ranks, colors):
        size = get_region_size_from_summary(region_summary_pair_sign, rr)

        if size is None:
            label = f"Region {rr}"
        else:
            label = f"Region {rr} (n={size})"

        handles.append(
            Line2D(
                [0], [0],
                marker="s",
                color=color,
                markerfacecolor=color,
                linestyle="None",
                markersize=9,
                label=label,
            )
        )

    return handles


# --------------------------------------------------
# Column normalization
# --------------------------------------------------

def normalize_region_voxels_columns(region_voxels_df: pd.DataFrame):
    required = [
        "pair_name",
        "sign",
        "region_rank",
        "Layer",
        "NoiseStd",
        "Percent",
        "idx_layer",
        "idx_noise",
        "idx_percent",
    ]

    missing = [c for c in required if c not in region_voxels_df.columns]

    if missing:
        raise ValueError(
            f"region_voxels_csv is missing required columns: {missing}. "
            f"Available columns: {list(region_voxels_df.columns)}"
        )

    num_cols = [
        "region_rank",
        "Layer",
        "NoiseStd",
        "Percent",
        "idx_layer",
        "idx_noise",
        "idx_percent",
    ]

    for c in num_cols:
        region_voxels_df[c] = pd.to_numeric(region_voxels_df[c], errors="coerce")

    region_voxels_df["sign"] = region_voxels_df["sign"].astype(str).str.strip()

    region_voxels_df = region_voxels_df.dropna(
        subset=[
            "pair_name",
            "sign",
            "region_rank",
            "Layer",
            "NoiseStd",
            "Percent",
            "idx_layer",
            "idx_noise",
            "idx_percent",
        ]
    ).copy()

    region_voxels_df["region_rank"] = region_voxels_df["region_rank"].astype(int)
    region_voxels_df["idx_layer"] = region_voxels_df["idx_layer"].astype(int)
    region_voxels_df["idx_noise"] = region_voxels_df["idx_noise"].astype(int)
    region_voxels_df["idx_percent"] = region_voxels_df["idx_percent"].astype(int)

    return region_voxels_df


def normalize_region_summary_columns(region_df: pd.DataFrame):
    required = ["pair_name", "sign", "region_rank"]
    missing = [c for c in required if c not in region_df.columns]

    if missing:
        raise ValueError(
            f"region_csv is missing required columns: {missing}. "
            f"Available columns: {list(region_df.columns)}"
        )

    region_df["region_rank"] = pd.to_numeric(region_df["region_rank"], errors="coerce")
    region_df["sign"] = region_df["sign"].astype(str).str.strip()

    region_df = region_df.dropna(
        subset=["pair_name", "sign", "region_rank"]
    ).copy()

    region_df["region_rank"] = region_df["region_rank"].astype(int)

    return region_df


def normalize_rerun_columns(rerun_df: pd.DataFrame):
    required = [
        "pair_name",
        "sign",
        "region_rank",
        "point_type",
        "Layer",
        "NoiseStd",
        "Percent",
    ]

    missing = [c for c in required if c not in rerun_df.columns]

    if missing:
        raise ValueError(
            f"rerun_csv is missing required columns: {missing}. "
            f"Available columns: {list(rerun_df.columns)}"
        )

    for c in ["region_rank", "Layer", "NoiseStd", "Percent"]:
        rerun_df[c] = pd.to_numeric(rerun_df[c], errors="coerce")

    rerun_df["sign"] = rerun_df["sign"].astype(str).str.strip()

    rerun_df = rerun_df.dropna(
        subset=[
            "pair_name",
            "sign",
            "region_rank",
            "point_type",
            "Layer",
            "NoiseStd",
            "Percent",
        ]
    ).copy()

    rerun_df["region_rank"] = rerun_df["region_rank"].astype(int)

    return rerun_df


# --------------------------------------------------
# Boundary extraction
# --------------------------------------------------

def get_6_neighbors(idx_tuple):
    i, j, k = idx_tuple
    return [
        (i - 1, j, k),
        (i + 1, j, k),
        (i, j - 1, k),
        (i, j + 1, k),
        (i, j, k - 1),
        (i, j, k + 1),
    ]


def extract_boundary_voxels(region_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only boundary voxels for one region.

    A voxel is a boundary voxel if at least one of its 6 axis-aligned neighbors
    is not in the region.
    """
    if region_df.empty:
        return region_df.copy()

    idx_cols = ["idx_layer", "idx_noise", "idx_percent"]
    coord_set = set(
        tuple(x) for x in region_df[idx_cols].drop_duplicates().to_numpy()
    )

    keep_mask = []

    for _, row in region_df.iterrows():
        pt = (
            int(row["idx_layer"]),
            int(row["idx_noise"]),
            int(row["idx_percent"]),
        )
        neighbors = get_6_neighbors(pt)
        is_boundary = any(nb not in coord_set for nb in neighbors)
        keep_mask.append(is_boundary)

    return region_df.loc[keep_mask].copy()


# --------------------------------------------------
# Plot helpers
# --------------------------------------------------

def set_full_grid_axes(
    ax,
    full_layer_min: float,
    full_layer_max: float,
    full_noise_min: float,
    full_noise_max: float,
    full_percent_min: float,
    full_percent_max: float,
):
    ax.set_xlim(full_noise_min, full_noise_max)
    ax.set_ylim(full_percent_min, full_percent_max)
    ax.set_zlim(full_layer_min, full_layer_max)

    noise_ticks = [
        round(x, 1)
        for x in np.arange(full_noise_min, full_noise_max + 0.001, 0.1)
    ]

    percent_ticks = [
        round(x, 1)
        for x in np.arange(full_percent_min, full_percent_max + 0.001, 0.1)
    ]

    layer_ticks = list(range(int(full_layer_min), int(full_layer_max) + 1, 5))

    ax.set_xticks(noise_ticks)
    ax.set_yticks(percent_ticks)
    ax.set_zticks(layer_ticks)


def get_point_marker_style(point_type: str):
    if point_type == "peak":
        return dict(
            marker="*",
            s=240,
            c="white",
            edgecolors="black",
            linewidths=1.5,
            depthshade=False,
        )

    if point_type == "centroid":
        return dict(
            marker="o",
            s=120,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            depthshade=False,
        )

    if point_type == "support":
        return dict(
            marker="x",
            s=90,
            c="black",
            linewidths=1.5,
            depthshade=False,
        )

    return dict(
        marker=".",
        s=60,
        c="black",
        depthshade=False,
    )


def prepare_region_points_for_plot(
    region_voxels_pair_sign: pd.DataFrame,
    selected_regions,
    boundary_only: bool,
):
    """
    Returns:
        region_rank -> DataFrame of voxels to plot
    """
    out = {}

    for region_rank in selected_regions:
        reg_vox = region_voxels_pair_sign[
            region_voxels_pair_sign["region_rank"] == region_rank
        ].copy()

        if reg_vox.empty:
            out[region_rank] = reg_vox
            continue

        n_full = len(reg_vox)

        if boundary_only:
            reg_plot = extract_boundary_voxels(reg_vox)
            n_boundary = len(reg_plot)
            print(f"  Region {region_rank}: full={n_full}, boundary={n_boundary}")
        else:
            reg_plot = reg_vox
            print(f"  Region {region_rank}: full={n_full}")

        out[region_rank] = reg_plot

    return out


def plot_pair_sign_3d(
    region_voxels_pair_sign: pd.DataFrame,
    region_summary_pair_sign: pd.DataFrame,
    rerun_pair_sign: pd.DataFrame,
    pair_name: str,
    sign: str,
    selected_regions,
    outpath: Path,
    dpi: int = 200,
    full_grid: bool = False,
    full_layer_min: float = 0,
    full_layer_max: float = 39,
    full_noise_min: float = 1.1,
    full_noise_max: float = 2.0,
    full_percent_min: float = 0.1,
    full_percent_max: float = 1.0,
    elev: float = 24,
    azim: float = -58,
    voxel_alpha: float = 0.45,
    voxel_size: float = 10,
    candidate_size_scale: float = 1.5,
    boundary_only: bool = False,
):
    if region_voxels_pair_sign.empty:
        print(f"[SKIP] No region voxels for {pair_name} {sign}")
        return

    if not selected_regions:
        print(f"[SKIP] No selected regions for {pair_name} {sign}")
        return

    colors = region_color_list(len(selected_regions))

    print(f"[PLOT] {pair_name} | {sign}")

    plot_points_by_region = prepare_region_points_for_plot(
        region_voxels_pair_sign=region_voxels_pair_sign,
        selected_regions=selected_regions,
        boundary_only=boundary_only,
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot region voxels
    for region_rank, color in zip(selected_regions, colors):
        reg_plot = plot_points_by_region.get(region_rank, pd.DataFrame())

        if reg_plot.empty:
            continue

        ax.scatter(
            reg_plot["NoiseStd"].to_numpy(dtype=float),
            reg_plot["Percent"].to_numpy(dtype=float),
            reg_plot["Layer"].to_numpy(dtype=float),
            s=voxel_size,
            c=[color],
            alpha=voxel_alpha,
            marker="s",
            depthshade=False,
        )

    # Overlay rerun candidate points
    if rerun_pair_sign is not None and not rerun_pair_sign.empty:
        rerun_selected = rerun_pair_sign[
            rerun_pair_sign["region_rank"].isin(selected_regions)
        ].copy()

        point_order = {
            "support": 0,
            "centroid": 1,
            "peak": 2,
        }

        rerun_selected["plot_order"] = (
            rerun_selected["point_type"].map(point_order).fillna(0).astype(int)
        )

        rerun_selected = rerun_selected.sort_values(
            ["plot_order", "region_rank"],
            ascending=[True, True],
        )

        for _, p in rerun_selected.iterrows():
            x = float(p["NoiseStd"])
            y = float(p["Percent"])
            z = float(p["Layer"])
            point_type = str(p["point_type"])
            region_rank = int(p["region_rank"])

            style = get_point_marker_style(point_type)

            if "s" in style:
                style["s"] = float(style["s"]) * float(candidate_size_scale)

            ax.scatter([x], [y], [z], **style)

            if point_type == "peak":
                ax.text(
                    x,
                    y,
                    z,
                    f"R{region_rank}",
                    fontsize=9,
                    color="black",
                )

    title = pretty_contrast_title(pair_name, sign)
    title_suffix = "full grid" if full_grid else "zoomed"
    mode_suffix = "boundary only" if boundary_only else "full voxels"

    if len(selected_regions) == 1:
        region_text = f"Region {selected_regions[0]}"
    else:
        region_text = f"top {len(selected_regions)} regions"

    ax.set_title(
        f"{title} | {sign} | {region_text} | {mode_suffix} | {title_suffix}",
        pad=18,
        fontsize=15,
    )

    ax.set_xlabel("NoiseStd", labelpad=10)
    ax.set_ylabel("Percent", labelpad=10)
    ax.set_zlabel("Layer", labelpad=10)

    if full_grid:
        set_full_grid_axes(
            ax=ax,
            full_layer_min=full_layer_min,
            full_layer_max=full_layer_max,
            full_noise_min=full_noise_min,
            full_noise_max=full_noise_max,
            full_percent_min=full_percent_min,
            full_percent_max=full_percent_max,
        )

    ax.view_init(elev=elev, azim=azim)

    region_handles = make_region_legend(
        region_ranks=selected_regions,
        region_summary_pair_sign=region_summary_pair_sign,
        colors=colors,
    )

    point_handles = make_point_legend()

    leg1 = ax.legend(
        handles=region_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        title="Regions",
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=point_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.02),
        title="Candidates",
    )

    plt.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {outpath}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--region_voxels_csv",
        required=True,
        help="Path to subtraction *_region_voxels.csv",
    )
    ap.add_argument(
        "--region_csv",
        required=True,
        help="Path to subtraction *_region_summary.csv",
    )
    ap.add_argument(
        "--rerun_csv",
        required=True,
        help="Path to subtraction *_rerun_candidates.csv",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Output directory",
    )

    ap.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help=(
            "Optional pair_name values to plot, e.g. "
            "Phonemic_minus_Neologism Semantic_minus_Neologism"
        ),
    )

    ap.add_argument(
        "--signs",
        nargs="+",
        default=["positive", "negative"],
        help="Signs to plot. Default: positive negative",
    )

    ap.add_argument(
        "--top_regions",
        type=int,
        default=3,
        help="Top N regions per pair/sign. Use 0 or negative to plot all available regions.",
    )

    ap.add_argument(
        "--boundary_only",
        action="store_true",
        help="Plot only surface/boundary voxels of each region",
    )

    ap.add_argument(
        "--also_full_grid",
        action="store_true",
        help="Also save an unzoomed version showing the full grid",
    )

    ap.add_argument(
        "--no_individual_regions",
        action="store_true",
        help="If set, do not generate separate per-region figures",
    )

    ap.add_argument("--full_layer_min", type=float, default=0)
    ap.add_argument("--full_layer_max", type=float, default=39)
    ap.add_argument("--full_noise_min", type=float, default=1.1)
    ap.add_argument("--full_noise_max", type=float, default=2.0)
    ap.add_argument("--full_percent_min", type=float, default=0.1)
    ap.add_argument("--full_percent_max", type=float, default=1.0)

    ap.add_argument(
        "--voxel_alpha",
        type=float,
        default=0.45,
        help="Transparency for plotted voxels",
    )
    ap.add_argument(
        "--voxel_size",
        type=float,
        default=10,
        help="Marker size for region voxels",
    )
    ap.add_argument(
        "--candidate_size_scale",
        type=float,
        default=1.5,
        help="Scale factor for peak/centroid/support marker sizes",
    )

    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--elev", type=float, default=24.0)
    ap.add_argument("--azim", type=float, default=-58.0)

    args = ap.parse_args()

    region_voxels_csv = Path(args.region_voxels_csv).expanduser().resolve()
    region_csv = Path(args.region_csv).expanduser().resolve()
    rerun_csv = Path(args.rerun_csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    if not region_voxels_csv.exists():
        raise SystemExit(f"region_voxels_csv not found: {region_voxels_csv}")
    if not region_csv.exists():
        raise SystemExit(f"region_csv not found: {region_csv}")
    if not rerun_csv.exists():
        raise SystemExit(f"rerun_csv not found: {rerun_csv}")

    region_voxels_df = pd.read_csv(region_voxels_csv)
    region_df = pd.read_csv(region_csv)
    rerun_df = pd.read_csv(rerun_csv)

    region_voxels_df = normalize_region_voxels_columns(region_voxels_df)
    region_df = normalize_region_summary_columns(region_df)
    rerun_df = normalize_rerun_columns(rerun_df)

    if args.pairs:
        region_voxels_df = region_voxels_df[
            region_voxels_df["pair_name"].isin(args.pairs)
        ].copy()

        region_df = region_df[
            region_df["pair_name"].isin(args.pairs)
        ].copy()

        rerun_df = rerun_df[
            rerun_df["pair_name"].isin(args.pairs)
        ].copy()

    selected_signs = [str(s).strip() for s in args.signs]

    region_voxels_df = region_voxels_df[
        region_voxels_df["sign"].isin(selected_signs)
    ].copy()

    region_df = region_df[
        region_df["sign"].isin(selected_signs)
    ].copy()

    rerun_df = rerun_df[
        rerun_df["sign"].isin(selected_signs)
    ].copy()

    if region_voxels_df.empty:
        raise SystemExit("No region voxel rows remain after filtering.")

    ensure_dir(outdir)

    pair_names = sorted(region_voxels_df["pair_name"].dropna().unique().tolist())

    print(f"Pairs to plot: {pair_names}")
    print(f"Signs to plot: {selected_signs}")
    print(f"Output directory: {outdir}")
    print(f"Region voxel rows: {len(region_voxels_df)}")
    print(f"Region summary rows: {len(region_df)}")
    print(f"Rerun candidate rows: {len(rerun_df)}")
    print(f"boundary_only={args.boundary_only}")
    print(f"top_regions={args.top_regions}")
    print(f"also_full_grid={args.also_full_grid}")
    print(f"generate_individual_regions={not args.no_individual_regions}")
    print(f"voxel_size={args.voxel_size}")
    print(f"voxel_alpha={args.voxel_alpha}")
    print(f"candidate_size_scale={args.candidate_size_scale}")

    for pair_name in pair_names:
        print()
        print("=" * 80)
        print(f"Processing pair: {pair_name}")

        for sign in selected_signs:
            print("-" * 80)
            print(f"Processing sign: {sign}")

            region_voxels_pair_sign = region_voxels_df[
                (region_voxels_df["pair_name"] == pair_name)
                & (region_voxels_df["sign"] == sign)
            ].copy()

            region_summary_pair_sign = region_df[
                (region_df["pair_name"] == pair_name)
                & (region_df["sign"] == sign)
            ].copy()

            rerun_pair_sign = rerun_df[
                (rerun_df["pair_name"] == pair_name)
                & (rerun_df["sign"] == sign)
            ].copy()

            if region_voxels_pair_sign.empty:
                print(f"[SKIP] No region voxels for {pair_name} {sign}")
                continue

            # Preserve original ranking order if summary exists.
            if not region_summary_pair_sign.empty:
                rank_order = (
                    region_summary_pair_sign
                    .sort_values("region_rank")["region_rank"]
                    .astype(int)
                    .tolist()
                )
            else:
                rank_order = sorted(
                    region_voxels_pair_sign["region_rank"].dropna().unique().tolist()
                )
                rank_order = [int(x) for x in rank_order]

            if args.top_regions is None or args.top_regions <= 0:
                selected_regions = rank_order
            else:
                selected_regions = rank_order[:args.top_regions]

            if not selected_regions:
                print(f"[SKIP] No selected regions for {pair_name} {sign}")
                continue

            print(f"Selected regions: {selected_regions}")

            suffix = "_boundary" if args.boundary_only else ""

            # --------------------------------------------------
            # Combined plot
            # --------------------------------------------------
            combined_zoom_name = sanitize_filename(
                f"{pair_name}_{sign}_regions3d_v2{suffix}.png"
            )
            combined_zoom_path = outdir / combined_zoom_name

            plot_pair_sign_3d(
                region_voxels_pair_sign=region_voxels_pair_sign,
                region_summary_pair_sign=region_summary_pair_sign,
                rerun_pair_sign=rerun_pair_sign,
                pair_name=pair_name,
                sign=sign,
                selected_regions=selected_regions,
                outpath=combined_zoom_path,
                dpi=args.dpi,
                full_grid=False,
                full_layer_min=args.full_layer_min,
                full_layer_max=args.full_layer_max,
                full_noise_min=args.full_noise_min,
                full_noise_max=args.full_noise_max,
                full_percent_min=args.full_percent_min,
                full_percent_max=args.full_percent_max,
                elev=args.elev,
                azim=args.azim,
                voxel_alpha=args.voxel_alpha,
                voxel_size=args.voxel_size,
                candidate_size_scale=args.candidate_size_scale,
                boundary_only=args.boundary_only,
            )

            if args.also_full_grid:
                combined_full_name = sanitize_filename(
                    f"{pair_name}_{sign}_regions3d_v2{suffix}_fullgrid.png"
                )
                combined_full_path = outdir / combined_full_name

                plot_pair_sign_3d(
                    region_voxels_pair_sign=region_voxels_pair_sign,
                    region_summary_pair_sign=region_summary_pair_sign,
                    rerun_pair_sign=rerun_pair_sign,
                    pair_name=pair_name,
                    sign=sign,
                    selected_regions=selected_regions,
                    outpath=combined_full_path,
                    dpi=args.dpi,
                    full_grid=True,
                    full_layer_min=args.full_layer_min,
                    full_layer_max=args.full_layer_max,
                    full_noise_min=args.full_noise_min,
                    full_noise_max=args.full_noise_max,
                    full_percent_min=args.full_percent_min,
                    full_percent_max=args.full_percent_max,
                    elev=args.elev,
                    azim=args.azim,
                    voxel_alpha=args.voxel_alpha,
                    voxel_size=args.voxel_size,
                    candidate_size_scale=args.candidate_size_scale,
                    boundary_only=args.boundary_only,
                )

            # --------------------------------------------------
            # Separate plot for each region
            # --------------------------------------------------
            if not args.no_individual_regions:
                for rr in selected_regions:
                    one_region = [rr]

                    one_zoom_name = sanitize_filename(
                        f"{pair_name}_{sign}_region{rr}_3d_v2{suffix}.png"
                    )
                    one_zoom_path = outdir / one_zoom_name

                    plot_pair_sign_3d(
                        region_voxels_pair_sign=region_voxels_pair_sign,
                        region_summary_pair_sign=region_summary_pair_sign,
                        rerun_pair_sign=rerun_pair_sign,
                        pair_name=pair_name,
                        sign=sign,
                        selected_regions=one_region,
                        outpath=one_zoom_path,
                        dpi=args.dpi,
                        full_grid=False,
                        full_layer_min=args.full_layer_min,
                        full_layer_max=args.full_layer_max,
                        full_noise_min=args.full_noise_min,
                        full_noise_max=args.full_noise_max,
                        full_percent_min=args.full_percent_min,
                        full_percent_max=args.full_percent_max,
                        elev=args.elev,
                        azim=args.azim,
                        voxel_alpha=args.voxel_alpha,
                        voxel_size=args.voxel_size,
                        candidate_size_scale=args.candidate_size_scale,
                        boundary_only=args.boundary_only,
                    )

                    if args.also_full_grid:
                        one_full_name = sanitize_filename(
                            f"{pair_name}_{sign}_region{rr}_3d_v2{suffix}_fullgrid.png"
                        )
                        one_full_path = outdir / one_full_name

                        plot_pair_sign_3d(
                            region_voxels_pair_sign=region_voxels_pair_sign,
                            region_summary_pair_sign=region_summary_pair_sign,
                            rerun_pair_sign=rerun_pair_sign,
                            pair_name=pair_name,
                            sign=sign,
                            selected_regions=one_region,
                            outpath=one_full_path,
                            dpi=args.dpi,
                            full_grid=True,
                            full_layer_min=args.full_layer_min,
                            full_layer_max=args.full_layer_max,
                            full_noise_min=args.full_noise_min,
                            full_noise_max=args.full_noise_max,
                            full_percent_min=args.full_percent_min,
                            full_percent_max=args.full_percent_max,
                            elev=args.elev,
                            azim=args.azim,
                            voxel_alpha=args.voxel_alpha,
                            voxel_size=args.voxel_size,
                            candidate_size_scale=args.candidate_size_scale,
                            boundary_only=args.boundary_only,
                        )

    print()
    print(f"All 3D plots saved under: {outdir}")


if __name__ == "__main__":
    main()


# python plot_subtraction_3d_regions_v2.py \
#   --region_voxels_csv ./tfce_results/sub3d_targetpairs_region_voxels.csv \
#   --region_csv ./tfce_results/sub3d_targetpairs_region_summary.csv \
#   --rerun_csv ./tfce_results/sub3d_targetpairs_rerun_candidates.csv \
#   --outdir ./tfce_results/sub3d_targetpairs_3dplots_v2 \
#   --pairs Phonemic_minus_Neologism Semantic_minus_Neologism \
#   --top_regions 0 \
#   --boundary_only \
#   --also_full_grid