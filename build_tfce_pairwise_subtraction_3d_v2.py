#!/usr/bin/env python3
"""
build_tfce_pairwise_subtraction_3d_v2.py

Build pairwise 3D subtraction TFCE maps from a seed-level summary CSV.

For each selected error-category pair A and B:
    diff_s = p_s(A) - p_s(B)

Across seeds, compute a full 3D volume over:
    Layer x NoiseStd x Percent

For each pair:
    mean_map     = mean(diff_s across seeds, NaN-safe)
    t_map        = one-sample t-statistic vs 0 across seeds, NaN-safe
    tfce_map     = signed 3D TFCE applied to t_map
    valid_n_map  = number of seeds contributing to each voxel

Then:
- save voxel-level long-format results for plotting
- extract positive and negative 3D TFCE regions separately
- summarize those regions
- choose representative rerun points from top regions
- save full region voxel membership for true 3D region plotting

Outputs
-------
Given --out_prefix /path/to/sub3d , the script writes:

1) /path/to/sub3d_voxel_long.csv
2) /path/to/sub3d_region_summary.csv
3) /path/to/sub3d_rerun_candidates.csv
4) /path/to/sub3d_region_voxels.csv
5) /path/to/sub3d_voxel_wide.csv

region_voxels.csv contains the full voxel membership of each selected connected
region. This is used by plot_subtraction_3d_regions_v2.py to plot true 3D
region clouds instead of fake blobs around rerun candidates.

Notes
-----
- Uses proportions, not raw counts
- Merges Formal + Nonword -> Phonemic by default
- Uses signed TFCE for subtraction
- Uses 6-neighbor 3D connectivity by default
- Handles missing seed cells safely
- Filters tiny connected components using --min_region_size
"""

import argparse
import itertools
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORIES_8 = [
    "Correct", "Semantic", "Unrelated", "Formal",
    "Nonword", "Mixed", "Neologism", "NoResponse",
]

CATEGORIES_7 = [
    "Correct", "Semantic", "Unrelated", "Phonemic",
    "Mixed", "Neologism", "NoResponse",
]


def merge_formal_nonword_into_phonemic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Formal" not in out.columns or "Nonword" not in out.columns:
        raise SystemExit("Need both 'Formal' and 'Nonword' columns to merge into Phonemic.")

    out["Phonemic"] = (
        pd.to_numeric(out["Formal"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["Nonword"], errors="coerce").fillna(0.0)
    )
    out = out.drop(columns=["Formal", "Nonword"])
    return out


def build_pairs(categories: list[str], exclude_categories: set[str]) -> list[tuple[str, str]]:
    usable = [c for c in categories if c not in exclude_categories]
    return list(itertools.combinations(usable, 2))


def safe_nanmean_map(stack: np.ndarray) -> np.ndarray:
    """
    stack shape: (n_seed, ...)
    Returns NaN-aware mean without RuntimeWarning on all-NaN slices.
    """
    stack = np.asarray(stack, dtype=float)
    valid = ~np.isnan(stack)
    count = valid.sum(axis=0)

    summed = np.nansum(stack, axis=0)
    out = np.full(stack.shape[1:], np.nan, dtype=float)

    good = count > 0
    out[good] = summed[good] / count[good]
    return out


def safe_t_map_centered(stack: np.ndarray, eps: float = 1e-10, t_cap: float = 50.0) -> np.ndarray:
    """
    stack shape: (n_seed, n_layer, n_noise, n_percent)

    Computes one-sample t-statistic against 0 without RuntimeWarnings.

    Behavior:
    - if n < 2 -> 0
    - if sd <= eps and mean ~= 0 -> 0
    - if sd <= eps and mean != 0 -> signed t_cap
    - otherwise t = mean / se, clipped to [-t_cap, t_cap]
    """
    stack = np.asarray(stack, dtype=float)

    valid = ~np.isnan(stack)
    n = valid.sum(axis=0).astype(float)

    summed = np.nansum(stack, axis=0)
    mean = np.zeros(stack.shape[1:], dtype=float)

    good_mean = n > 0
    mean[good_mean] = summed[good_mean] / n[good_mean]

    var = np.zeros_like(mean, dtype=float)
    good_var = n >= 2

    if np.any(good_var):
        centered = np.where(valid, stack - mean[None, ...], 0.0)
        ss = np.sum(centered ** 2, axis=0)
        var[good_var] = ss[good_var] / (n[good_var] - 1.0)

    sd = np.sqrt(var)

    t = np.zeros_like(mean, dtype=float)

    se = np.full_like(mean, np.nan, dtype=float)
    se[good_var] = sd[good_var] / np.sqrt(n[good_var])

    nonzero_se = good_var & (se > eps)
    nearzero_se = good_var & (se <= eps)

    t[nonzero_se] = mean[nonzero_se] / se[nonzero_se]
    t[nearzero_se] = np.where(
        np.abs(mean[nearzero_se]) <= eps,
        0.0,
        np.sign(mean[nearzero_se]) * t_cap
    )

    t = np.clip(t, -t_cap, t_cap)
    t = np.nan_to_num(t, nan=0.0, posinf=t_cap, neginf=-t_cap)
    return t


def get_neighbors_3d(
    i: int,
    j: int,
    k: int,
    n1: int,
    n2: int,
    n3: int,
    connectivity: int = 6,
):
    if connectivity == 6:
        offsets = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ]
    elif connectivity == 18:
        offsets = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    if abs(di) + abs(dj) + abs(dk) <= 2:
                        offsets.append((di, dj, dk))
    elif connectivity == 26:
        offsets = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    offsets.append((di, dj, dk))
    else:
        raise ValueError(f"Unsupported 3D connectivity: {connectivity}")

    for di, dj, dk in offsets:
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < n1 and 0 <= nj < n2 and 0 <= nk < n3:
            yield ni, nj, nk


def connected_components_extent_3d(mask: np.ndarray, connectivity: int = 6) -> np.ndarray:
    n1, n2, n3 = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    extent = np.zeros_like(mask, dtype=int)

    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                if not mask[i, j, k] or visited[i, j, k]:
                    continue

                q = deque([(i, j, k)])
                visited[i, j, k] = True
                comp = []

                while q:
                    ci, cj, ck = q.popleft()
                    comp.append((ci, cj, ck))

                    for ni, nj, nk in get_neighbors_3d(
                        ci, cj, ck, n1, n2, n3, connectivity=connectivity
                    ):
                        if mask[ni, nj, nk] and not visited[ni, nj, nk]:
                            visited[ni, nj, nk] = True
                            q.append((ni, nj, nk))

                comp_size = len(comp)
                for ci, cj, ck in comp:
                    extent[ci, cj, ck] = comp_size

    return extent


def tfce_3d_nonnegative(
    z: np.ndarray,
    E: float = 0.5,
    H: float = 2.0,
    connectivity: int = 6,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z, dtype=float)

    positive_vals = np.unique(z[z > 0])
    if positive_vals.size == 0:
        return out

    positive_vals = np.sort(positive_vals)
    prev_h = 0.0

    for h in positive_vals:
        dh = float(h - prev_h)
        if dh <= 0:
            continue

        mask = z >= h
        ext = connected_components_extent_3d(mask, connectivity=connectivity)
        valid = ext > 0

        if np.any(valid):
            out[valid] += (ext[valid].astype(float) ** E) * (float(h) ** H) * dh

        prev_h = float(h)

    return out


def tfce_3d_signed(
    z: np.ndarray,
    E: float = 0.5,
    H: float = 2.0,
    connectivity: int = 6,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)

    pos = np.maximum(z, 0.0)
    neg = np.maximum(-z, 0.0)

    out = np.zeros_like(z, dtype=float)

    if np.any(pos > 0):
        out += tfce_3d_nonnegative(pos, E=E, H=H, connectivity=connectivity)

    if np.any(neg > 0):
        out -= tfce_3d_nonnegative(neg, E=E, H=H, connectivity=connectivity)

    return out


def pivot_seed_grid_3d(
    df_seed: pd.DataFrame,
    value_col: str,
    layer_vals: np.ndarray,
    noise_vals: np.ndarray,
    percent_vals: np.ndarray,
) -> np.ndarray:
    arr = np.full((len(layer_vals), len(noise_vals), len(percent_vals)), np.nan, dtype=float)

    layer_to_i = {v: i for i, v in enumerate(layer_vals)}
    noise_to_j = {v: j for j, v in enumerate(noise_vals)}
    pct_to_k = {v: k for k, v in enumerate(percent_vals)}

    tmp = (
        df_seed.groupby(["Layer", "NoiseStd", "Percent"], as_index=False)[value_col]
        .mean()
    )

    for _, row in tmp.iterrows():
        i = layer_to_i[row["Layer"]]
        j = noise_to_j[row["NoiseStd"]]
        k = pct_to_k[row["Percent"]]
        arr[i, j, k] = float(row[value_col])

    return arr


def build_pairwise_maps_3d(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    tfce_E: float,
    tfce_H: float,
    connectivity: int,
    t_eps: float,
    t_cap: float,
) -> tuple[pd.DataFrame, dict]:
    seeds = np.array(sorted(df["Seed"].dropna().unique()))
    layer_vals = np.array(sorted(df["Layer"].dropna().unique()), dtype=float)
    noise_vals = np.array(sorted(df["NoiseStd"].dropna().unique()), dtype=float)
    percent_vals = np.array(sorted(df["Percent"].dropna().unique()), dtype=float)

    rows = []
    for layer in layer_vals:
        for noise in noise_vals:
            for pct in percent_vals:
                rows.append({
                    "Layer": float(layer),
                    "NoiseStd": float(noise),
                    "Percent": float(pct),
                })

    out_df = pd.DataFrame(rows)
    out_df["NumSeeds"] = int(len(seeds))

    pair_data = {}

    for a, b in pairs:
        diff_stack = []

        for seed in seeds:
            df_seed = df[df["Seed"] == seed].copy()

            a_vol = pivot_seed_grid_3d(
                df_seed,
                f"{a}_prop",
                layer_vals,
                noise_vals,
                percent_vals,
            )

            b_vol = pivot_seed_grid_3d(
                df_seed,
                f"{b}_prop",
                layer_vals,
                noise_vals,
                percent_vals,
            )

            diff_stack.append(a_vol - b_vol)

        diff_stack = np.asarray(diff_stack, dtype=float)

        mean_map = safe_nanmean_map(diff_stack)
        t_map = safe_t_map_centered(diff_stack, eps=t_eps, t_cap=t_cap)
        tfce_map = tfce_3d_signed(t_map, E=tfce_E, H=tfce_H, connectivity=connectivity)
        valid_n_map = np.sum(~np.isnan(diff_stack), axis=0).astype(int)

        base = f"{a}_minus_{b}"

        out_df[f"{base}_mean"] = mean_map.reshape(-1)
        out_df[f"{base}_t"] = t_map.reshape(-1)
        out_df[f"{base}_tfce"] = tfce_map.reshape(-1)
        out_df[f"{base}_n_valid"] = valid_n_map.reshape(-1)

        pair_data[base] = {
            "pair_a": a,
            "pair_b": b,
            "mean_map": mean_map,
            "t_map": t_map,
            "tfce_map": tfce_map,
            "valid_n_map": valid_n_map,
            "layer_vals": layer_vals,
            "noise_vals": noise_vals,
            "percent_vals": percent_vals,
            "num_seeds": int(len(seeds)),
        }

    return out_df, pair_data


def label_components_3d(mask: np.ndarray, connectivity: int = 6):
    n1, n2, n3 = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    labels = np.zeros_like(mask, dtype=int)
    regions = []
    label_id = 0

    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                if not mask[i, j, k] or visited[i, j, k]:
                    continue

                label_id += 1
                q = deque([(i, j, k)])
                visited[i, j, k] = True
                comp = []

                while q:
                    ci, cj, ck = q.popleft()
                    comp.append((ci, cj, ck))
                    labels[ci, cj, ck] = label_id

                    for ni, nj, nk in get_neighbors_3d(
                        ci, cj, ck, n1, n2, n3, connectivity=connectivity
                    ):
                        if mask[ni, nj, nk] and not visited[ni, nj, nk]:
                            visited[ni, nj, nk] = True
                            q.append((ni, nj, nk))

                regions.append(comp)

    return labels, regions


def choose_threshold_from_percentile(tfce_abs: np.ndarray, percentile: float) -> float:
    vals = np.asarray(tfce_abs, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]

    if vals.size == 0:
        return np.inf

    return float(np.percentile(vals, percentile))


def choose_representative_points(
    comp: list[tuple[int, int, int]],
    score_map: np.ndarray,
    layer_vals: np.ndarray,
    noise_vals: np.ndarray,
    percent_vals: np.ndarray,
    max_support_points: int = 3,
    min_manhattan_dist: int = 2,
):
    ranked = sorted(
        comp,
        key=lambda xyz: score_map[xyz[0], xyz[1], xyz[2]],
        reverse=True,
    )

    peak = ranked[0]

    coords = np.array(comp, dtype=int)
    centroid_idx = np.round(coords.mean(axis=0)).astype(int)

    centroid = min(
        comp,
        key=lambda xyz: (
            abs(xyz[0] - centroid_idx[0]) +
            abs(xyz[1] - centroid_idx[1]) +
            abs(xyz[2] - centroid_idx[2])
        )
    )

    out = []

    def pack_point(pt, point_type):
        i, j, k = pt
        return {
            "point_type": point_type,
            "Layer": float(layer_vals[i]),
            "NoiseStd": float(noise_vals[j]),
            "Percent": float(percent_vals[k]),
            "score_value": float(score_map[i, j, k]),
            "idx_layer": int(i),
            "idx_noise": int(j),
            "idx_percent": int(k),
        }

    out.append(pack_point(peak, "peak"))

    if centroid != peak:
        out.append(pack_point(centroid, "centroid"))

    chosen = [peak]

    if centroid != peak:
        chosen.append(centroid)

    support_added = 0

    for pt in ranked:
        if pt in chosen:
            continue

        ok = True

        for c in chosen:
            dist = abs(pt[0] - c[0]) + abs(pt[1] - c[1]) + abs(pt[2] - c[2])

            if dist < min_manhattan_dist:
                ok = False
                break

        if ok:
            out.append(pack_point(pt, "support"))
            chosen.append(pt)
            support_added += 1

        if support_added >= max_support_points:
            break

    return out


def extract_regions_for_pair(
    pair_name: str,
    pair_info: dict,
    region_percentile: float,
    connectivity: int,
    max_regions_per_sign: int,
    max_support_points: int,
    min_manhattan_dist: int,
    min_region_size: int,
):
    tfce_map = np.asarray(pair_info["tfce_map"], dtype=float)
    mean_map = np.asarray(pair_info["mean_map"], dtype=float)
    t_map = np.asarray(pair_info["t_map"], dtype=float)
    valid_n_map = np.asarray(pair_info["valid_n_map"], dtype=int)

    layer_vals = pair_info["layer_vals"]
    noise_vals = pair_info["noise_vals"]
    percent_vals = pair_info["percent_vals"]
    num_seeds = pair_info["num_seeds"]
    pair_a = pair_info["pair_a"]
    pair_b = pair_info["pair_b"]

    region_rows = []
    rerun_rows = []
    region_voxel_rows = []

    for sign_name, sign_map in [
        ("positive", np.maximum(tfce_map, 0.0)),
        ("negative", np.maximum(-tfce_map, 0.0)),
    ]:
        thresh = choose_threshold_from_percentile(sign_map, region_percentile)

        if not np.isfinite(thresh):
            continue

        mask = sign_map >= thresh

        if not np.any(mask):
            continue

        _, regions = label_components_3d(mask, connectivity=connectivity)

        regions = [comp for comp in regions if len(comp) >= int(min_region_size)]

        if not regions:
            continue

        reg_rows = []

        for reg_idx, comp in enumerate(regions, start=1):
            coords = np.array(comp, dtype=int)

            tfce_vals = np.array([sign_map[i, j, k] for i, j, k in comp], dtype=float)
            signed_tfce_vals = np.array([tfce_map[i, j, k] for i, j, k in comp], dtype=float)
            t_vals = np.array([t_map[i, j, k] for i, j, k in comp], dtype=float)
            mean_vals = np.array([mean_map[i, j, k] for i, j, k in comp], dtype=float)
            n_valid_vals = np.array([valid_n_map[i, j, k] for i, j, k in comp], dtype=float)

            peak_local_idx = int(np.argmax(tfce_vals))
            pi, pj, pk = comp[peak_local_idx]

            centroid_idx = np.round(coords.mean(axis=0)).astype(int)

            centroid_pt = min(
                comp,
                key=lambda xyz: (
                    abs(xyz[0] - centroid_idx[0]) +
                    abs(xyz[1] - centroid_idx[1]) +
                    abs(xyz[2] - centroid_idx[2])
                )
            )

            ci, cj, ck = centroid_pt

            reg_rows.append({
                "pair_name": pair_name,
                "pair_a": pair_a,
                "pair_b": pair_b,
                "sign": sign_name,
                "region_id_raw": reg_idx,
                "num_seeds_total": int(num_seeds),
                "size_voxels": int(len(comp)),
                "tfce_threshold_used": float(thresh),
                "tfce_sum": float(tfce_vals.sum()),
                "tfce_mean": float(tfce_vals.mean()),
                "tfce_max": float(tfce_vals.max()),
                "tfce_signed_mean": float(np.mean(signed_tfce_vals)),
                "t_mean": float(np.mean(t_vals)),
                "t_max_abs": float(np.max(np.abs(t_vals))),
                "mean_subtraction_mean": float(np.mean(mean_vals)),
                "mean_subtraction_max_abs": float(np.max(np.abs(mean_vals))),
                "n_valid_mean": float(np.mean(n_valid_vals)),
                "n_valid_min": int(np.min(n_valid_vals)),
                "n_valid_max": int(np.max(n_valid_vals)),
                "peak_n_valid": int(valid_n_map[pi, pj, pk]),
                "peak_layer": float(layer_vals[pi]),
                "peak_noise": float(noise_vals[pj]),
                "peak_percent": float(percent_vals[pk]),
                "peak_tfce_signed": float(tfce_map[pi, pj, pk]),
                "peak_tfce_abs_for_sign": float(sign_map[pi, pj, pk]),
                "peak_t_signed": float(t_map[pi, pj, pk]),
                "peak_mean_signed": float(mean_map[pi, pj, pk]),
                "centroid_layer": float(layer_vals[ci]),
                "centroid_noise": float(noise_vals[cj]),
                "centroid_percent": float(percent_vals[ck]),
                "component_points": comp,
            })

        if not reg_rows:
            continue

        reg_df = pd.DataFrame(reg_rows)

        reg_df = reg_df.sort_values(
            ["tfce_sum", "tfce_max", "size_voxels"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        if max_regions_per_sign is not None and max_regions_per_sign > 0:
            reg_df = reg_df.head(max_regions_per_sign).copy()

        reg_df["region_rank"] = np.arange(1, len(reg_df) + 1)

        for _, row in reg_df.iterrows():
            comp = row["component_points"]
            region_rank = int(row["region_rank"])

            # Save every voxel in the selected connected component.
            # This is the new output needed for true full-region 3D plotting.
            for i, j, k in comp:
                region_voxel_rows.append({
                    "pair_name": pair_name,
                    "pair_a": pair_a,
                    "pair_b": pair_b,
                    "sign": sign_name,
                    "region_rank": region_rank,
                    "region_id_raw": int(row["region_id_raw"]),
                    "size_voxels": int(row["size_voxels"]),
                    "tfce_threshold_used": float(row["tfce_threshold_used"]),
                    "tfce_sum": float(row["tfce_sum"]),
                    "tfce_max": float(row["tfce_max"]),

                    "Layer": float(layer_vals[i]),
                    "NoiseStd": float(noise_vals[j]),
                    "Percent": float(percent_vals[k]),

                    "idx_layer": int(i),
                    "idx_noise": int(j),
                    "idx_percent": int(k),

                    # Signed and sign-specific values
                    "tfce_value": float(tfce_map[i, j, k]),
                    "tfce_abs_for_sign": float(sign_map[i, j, k]),
                    "t_value": float(t_map[i, j, k]),
                    "mean_subtraction": float(mean_map[i, j, k]),
                    "n_valid": int(valid_n_map[i, j, k]),
                })

            pts = choose_representative_points(
                comp=comp,
                score_map=sign_map,
                layer_vals=layer_vals,
                noise_vals=noise_vals,
                percent_vals=percent_vals,
                max_support_points=max_support_points,
                min_manhattan_dist=min_manhattan_dist,
            )

            for p in pts:
                rerun_rows.append({
                    "pair_name": pair_name,
                    "pair_a": pair_a,
                    "pair_b": pair_b,
                    "sign": sign_name,
                    "region_rank": region_rank,
                    "size_voxels": int(row["size_voxels"]),
                    "tfce_sum": float(row["tfce_sum"]),
                    "tfce_max": float(row["tfce_max"]),
                    "n_valid_mean": float(row["n_valid_mean"]),
                    "n_valid_min": int(row["n_valid_min"]),
                    "point_type": p["point_type"],
                    "Layer": p["Layer"],
                    "NoiseStd": p["NoiseStd"],
                    "Percent": p["Percent"],
                    "score_value": p["score_value"],
                    "idx_layer": p["idx_layer"],
                    "idx_noise": p["idx_noise"],
                    "idx_percent": p["idx_percent"],
                })

        reg_df = reg_df.drop(columns=["component_points"])
        region_rows.extend(reg_df.to_dict(orient="records"))

    region_df = pd.DataFrame(region_rows)
    rerun_df = pd.DataFrame(rerun_rows)
    region_voxels_df = pd.DataFrame(region_voxel_rows)

    if not region_df.empty:
        region_df = region_df.sort_values(
            ["pair_name", "sign", "region_rank"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    if not rerun_df.empty:
        point_order = {
            "peak": 0,
            "centroid": 1,
            "support": 2,
        }

        rerun_df["point_order"] = (
            rerun_df["point_type"]
            .map(point_order)
            .fillna(99)
            .astype(int)
        )

        rerun_df = (
            rerun_df
            .sort_values(
                ["pair_name", "sign", "region_rank", "point_order", "score_value"],
                ascending=[True, True, True, True, False],
            )
            .drop(columns=["point_order"])
            .reset_index(drop=True)
        )

    if not region_voxels_df.empty:
        region_voxels_df = region_voxels_df.sort_values(
            ["pair_name", "sign", "region_rank", "Layer", "NoiseStd", "Percent"],
            ascending=[True, True, True, True, True, True],
        ).reset_index(drop=True)

    return region_df, rerun_df, region_voxels_df


def build_voxel_long_from_wide(
    voxel_wide_df: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    long_parts = []

    base_cols = ["Layer", "NoiseStd", "Percent", "NumSeeds"]

    for a, b in pairs:
        base = f"{a}_minus_{b}"

        cols = base_cols + [
            f"{base}_mean",
            f"{base}_t",
            f"{base}_tfce",
            f"{base}_n_valid",
        ]

        tmp = voxel_wide_df[cols].copy()

        tmp = tmp.rename(columns={
            f"{base}_mean": "mean_subtraction",
            f"{base}_t": "t_value",
            f"{base}_tfce": "tfce_value",
            f"{base}_n_valid": "n_valid",
        })

        tmp["pair_name"] = base
        tmp["pair_a"] = a
        tmp["pair_b"] = b

        long_parts.append(tmp)

    voxel_long_df = pd.concat(long_parts, axis=0, ignore_index=True)

    voxel_long_df = voxel_long_df[
        [
            "pair_name",
            "pair_a",
            "pair_b",
            "Layer",
            "NoiseStd",
            "Percent",
            "NumSeeds",
            "n_valid",
            "mean_subtraction",
            "t_value",
            "tfce_value",
        ]
    ].copy()

    voxel_long_df["tfce_abs"] = np.abs(voxel_long_df["tfce_value"])
    voxel_long_df["t_abs"] = np.abs(voxel_long_df["t_value"])
    voxel_long_df["direction"] = np.where(
        voxel_long_df["tfce_value"] > 0,
        "positive",
        np.where(voxel_long_df["tfce_value"] < 0, "negative", "zero")
    )

    return voxel_long_df


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("input_csv", help="Seed-level input CSV")
    ap.add_argument("--out_prefix", required=True, help="Output prefix, e.g. ./tfce_results/sub3d")

    ap.add_argument(
        "--merge_7cat",
        action="store_true",
        default=True,
        help="Merge Formal+Nonword into Phonemic (default: on)",
    )
    ap.add_argument(
        "--no_merge_7cat",
        action="store_true",
        help="Disable merge and keep 8 categories",
    )
    ap.add_argument(
        "--exclude_categories",
        nargs="+",
        default=["Correct", "NoResponse"],
        help="Categories to exclude from pair generation (default: Correct NoResponse)",
    )

    ap.add_argument(
        "--connectivity",
        type=int,
        choices=[6, 18, 26],
        default=6,
        help="3D TFCE neighborhood connectivity (default: 6)",
    )
    ap.add_argument("--tfce_E", type=float, default=0.5)
    ap.add_argument("--tfce_H", type=float, default=2.0)

    ap.add_argument("--round_noise", type=int, default=6)
    ap.add_argument("--round_percent", type=int, default=6)
    ap.add_argument("--t_eps", type=float, default=1e-10)
    ap.add_argument("--t_cap", type=float, default=50.0)

    ap.add_argument(
        "--region_percentile",
        type=float,
        default=90.0,
        help="Threshold positive TFCE magnitude at this percentile to define candidate regions",
    )
    ap.add_argument("--max_regions_per_sign", type=int, default=5)
    ap.add_argument("--max_support_points", type=int, default=3)
    ap.add_argument("--min_manhattan_dist", type=int, default=2)
    ap.add_argument(
        "--min_region_size",
        type=int,
        default=3,
        help="Minimum connected-component size in voxels to keep as a region (default: 3)",
    )

    ap.add_argument("--min_noise", type=float, default=None)
    ap.add_argument("--max_noise", type=float, default=None)
    ap.add_argument("--min_percent", type=float, default=None)
    ap.add_argument("--max_percent", type=float, default=None)

    ap.add_argument(
        "--only_pairs",
        nargs="+",
        default=None,
        help="Explicit subtraction pairs formatted as A:B . Example: Semantic:Neologism Phonemic:Neologism",
    )

    args = ap.parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    out_prefix = Path(args.out_prefix).expanduser().resolve()

    if not input_csv.exists():
        raise SystemExit(f"Input CSV does not exist: {input_csv}")

    df = pd.read_csv(input_csv)

    required_base = {"Layer", "NoiseStd", "Percent", "Seed"}
    missing_base = required_base - set(df.columns)

    if missing_base:
        raise SystemExit(f"Missing required columns: {sorted(missing_base)}")

    required_cats = CATEGORIES_8
    missing_cats = [c for c in required_cats if c not in df.columns]

    if missing_cats:
        raise SystemExit(f"Missing category columns: {missing_cats}")

    merge_7cat = args.merge_7cat and (not args.no_merge_7cat)

    df["Layer"] = pd.to_numeric(df["Layer"], errors="coerce")
    df["NoiseStd"] = pd.to_numeric(df["NoiseStd"], errors="coerce").round(args.round_noise)
    df["Percent"] = pd.to_numeric(df["Percent"], errors="coerce").round(args.round_percent)
    df["Seed"] = pd.to_numeric(df["Seed"], errors="coerce")

    for c in required_cats:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Layer", "NoiseStd", "Percent", "Seed"]).copy()
    df["Layer"] = df["Layer"].astype(int)
    df["Seed"] = df["Seed"].astype(int)

    if args.min_noise is not None:
        df = df[df["NoiseStd"] >= float(args.min_noise)].copy()
    if args.max_noise is not None:
        df = df[df["NoiseStd"] <= float(args.max_noise)].copy()
    if args.min_percent is not None:
        df = df[df["Percent"] >= float(args.min_percent)].copy()
    if args.max_percent is not None:
        df = df[df["Percent"] <= float(args.max_percent)].copy()

    if df.empty:
        raise SystemExit("No rows remain after filtering.")

    group_cols = ["Percent", "NoiseStd", "Layer", "Seed"]
    df = df.groupby(group_cols, as_index=False)[CATEGORIES_8].mean()

    df["TotalCount"] = df[CATEGORIES_8].sum(axis=1)

    bad_total = (df["TotalCount"] <= 0).sum()

    if bad_total > 0:
        print(f"[WARN] {bad_total} rows have TotalCount <= 0. Their proportions will be NaN.")

    for c in CATEGORIES_8:
        df[f"{c}_prop"] = np.where(
            df["TotalCount"] > 0,
            df[c] / df["TotalCount"],
            np.nan,
        )

    if merge_7cat:
        df = merge_formal_nonword_into_phonemic(df)

        if "Formal_prop" not in df.columns or "Nonword_prop" not in df.columns:
            raise SystemExit("Missing Formal_prop / Nonword_prop during merge.")

        df["Phonemic_prop"] = df["Formal_prop"] + df["Nonword_prop"]
        df = df.drop(columns=["Formal_prop", "Nonword_prop"])
        categories = CATEGORIES_7
    else:
        categories = CATEGORIES_8

    exclude_categories = set(args.exclude_categories)

    if args.only_pairs:
        pairs = []

        for item in args.only_pairs:
            a, b = item.split(":")
            a = a.strip()
            b = b.strip()

            if a not in categories or b not in categories:
                raise SystemExit(f"Invalid pair in --only_pairs: {item}")

            pairs.append((a, b))

        excluded_msg = "not used because --only_pairs was provided"
    else:
        pairs = build_pairs(categories, exclude_categories)
        excluded_msg = sorted(exclude_categories)

    if not pairs:
        raise SystemExit("No pairs left after applying pair selection.")

    print(f"Input : {input_csv}")
    print(f"Rows after cleanup: {len(df)}")
    print(f"Categories used: {categories}")
    print(f"Excluded from pairs: {excluded_msg}")
    print(f"Pairs ({len(pairs)}):")

    for a, b in pairs:
        print(f"  - {a} minus {b}")

    full_combos = df[["Layer", "NoiseStd", "Percent"]].drop_duplicates().shape[0]

    seed_combo_counts = (
        df.groupby("Seed")[["Layer", "NoiseStd", "Percent"]]
        .apply(lambda x: x.drop_duplicates().shape[0])
    )

    print(f"Unique grid cells overall: {full_combos}")
    print("Per-seed unique grid-cell counts summary:")
    print(seed_combo_counts.describe())

    print(f"TFCE params: E={args.tfce_E}, H={args.tfce_H}, connectivity={args.connectivity}")
    print(f"t_eps={args.t_eps}, t_cap={args.t_cap}")
    print(f"region_percentile={args.region_percentile}")
    print(f"max_regions_per_sign={args.max_regions_per_sign}")
    print(f"max_support_points={args.max_support_points}")
    print(f"min_manhattan_dist={args.min_manhattan_dist}")
    print(f"min_region_size={args.min_region_size}")

    voxel_wide_df, pair_data = build_pairwise_maps_3d(
        df=df,
        pairs=pairs,
        tfce_E=args.tfce_E,
        tfce_H=args.tfce_H,
        connectivity=args.connectivity,
        t_eps=args.t_eps,
        t_cap=args.t_cap,
    )

    voxel_long_df = build_voxel_long_from_wide(voxel_wide_df, pairs)

    region_parts = []
    rerun_parts = []
    region_voxel_parts = []

    for pair_name, info in pair_data.items():
        region_df, rerun_df, region_voxels_df = extract_regions_for_pair(
            pair_name=pair_name,
            pair_info=info,
            region_percentile=args.region_percentile,
            connectivity=args.connectivity,
            max_regions_per_sign=args.max_regions_per_sign,
            max_support_points=args.max_support_points,
            min_manhattan_dist=args.min_manhattan_dist,
            min_region_size=args.min_region_size,
        )

        if not region_df.empty:
            region_parts.append(region_df)

        if not rerun_df.empty:
            rerun_parts.append(rerun_df)

        if not region_voxels_df.empty:
            region_voxel_parts.append(region_voxels_df)

        print(f"[DONE] {pair_name}")

    region_summary_df = (
        pd.concat(region_parts, axis=0, ignore_index=True)
        if region_parts else pd.DataFrame()
    )

    rerun_candidates_df = (
        pd.concat(rerun_parts, axis=0, ignore_index=True)
        if rerun_parts else pd.DataFrame()
    )

    region_voxels_df = (
        pd.concat(region_voxel_parts, axis=0, ignore_index=True)
        if region_voxel_parts else pd.DataFrame()
    )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    voxel_wide_path = out_prefix.with_name(out_prefix.name + "_voxel_wide.csv")
    voxel_long_path = out_prefix.with_name(out_prefix.name + "_voxel_long.csv")
    region_path = out_prefix.with_name(out_prefix.name + "_region_summary.csv")
    rerun_path = out_prefix.with_name(out_prefix.name + "_rerun_candidates.csv")
    region_voxels_path = out_prefix.with_name(out_prefix.name + "_region_voxels.csv")

    voxel_wide_df.to_csv(voxel_wide_path, index=False)
    voxel_long_df.to_csv(voxel_long_path, index=False)
    region_summary_df.to_csv(region_path, index=False)
    rerun_candidates_df.to_csv(rerun_path, index=False)
    region_voxels_df.to_csv(region_voxels_path, index=False)

    print(f"Saved voxel_wide      : {voxel_wide_path}")
    print(f"Saved voxel_long      : {voxel_long_path}")
    print(f"Saved region_summary  : {region_path}")
    print(f"Saved rerun_candidates: {rerun_path}")
    print(f"Saved region_voxels   : {region_voxels_path}")

    print(f"voxel_wide rows      : {len(voxel_wide_df)}")
    print(f"voxel_long rows      : {len(voxel_long_df)}")
    print(f"region_summary rows  : {len(region_summary_df)}")
    print(f"rerun_candidates rows: {len(rerun_candidates_df)}")
    print(f"region_voxels rows   : {len(region_voxels_df)}")


if __name__ == "__main__":
    main()


# Example:
# python build_tfce_pairwise_subtraction_3d_v2.py \
#   ./data/PNT_80_seeds_discovery_40_summary.csv \
#   --out_prefix ./tfce_results/sub3d_targetpairs \
#   --merge_7cat \
#   --connectivity 6 \
#   --region_percentile 90 \
#   --min_region_size 3 \
#   --max_regions_per_sign 10 \
#   --max_support_points 20 \
#   --min_manhattan_dist 2 \
#   --only_pairs Phonemic:Neologism Semantic:Neologism