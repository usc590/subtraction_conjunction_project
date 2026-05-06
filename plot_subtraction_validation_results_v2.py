#!/usr/bin/env python3
"""
plot_subtraction_validation_results_v2.py

Paper-ready subtraction validation plots and tables.

Inputs
------
1) validation_region_summary.csv
   Example:
   ./tfce_results/sub3d_targetpairs_validation_eval_validation_region_summary.csv

2) validation_point_lookup.csv
   Example:
   ./tfce_results/sub3d_targetpairs_validation_eval_validation_point_lookup.csv

3) validation_point_seed_lookup.csv
   Example:
   ./tfce_results/sub3d_targetpairs_validation_eval_validation_point_seed_lookup.csv

Main added feature over v1
--------------------------
This version computes seed-level validation success counts:

For subtraction:
    positive discovery region:
        correct seed = validation_seed_subtraction > 0

    negative discovery region:
        correct seed = validation_seed_subtraction < 0

So each rerun candidate condition gets:
    n_correct / n_total

This is the number for paper tables:
    e.g., 40/40, 37/40, 18/40

Outputs
-------
Tables:
  paper_region_summary_table.csv
  paper_candidate_seed_success_table.csv
  paper_region_seed_success_table.csv
  paper_peak_candidate_table.csv

Plots:
  paper_region_validation_mean_subtraction.png
  paper_region_mean_seed_match_rate.png
  paper_candidate_seed_success_count.png
  paper_candidate_success_rate.png
  paper_discovery_vs_validation_subtraction.png
  paper_validation_subtraction_by_region.png

Example
-------
python plot_subtraction_validation_results_v2.py \
  --region_summary_csv ./tfce_results/sub3d_targetpairs_validation_eval_validation_region_summary.csv \
  --point_lookup_csv ./tfce_results/sub3d_targetpairs_validation_eval_validation_point_lookup.csv \
  --point_seed_lookup_csv ./tfce_results/sub3d_targetpairs_validation_eval_validation_point_seed_lookup.csv \
  --outdir ./tfce_results/sub3d_targetpairs_validation_plots_v2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def pretty_pair_label(pair_name: str) -> str:
    if "_minus_" in str(pair_name):
        a, b = str(pair_name).split("_minus_", 1)
        return f"{a} - {b}"
    return str(pair_name)


def pretty_sign_label(pair_name: str, sign: str) -> str:
    if "_minus_" not in str(pair_name):
        return f"{pair_name} ({sign})"

    a, b = str(pair_name).split("_minus_", 1)
    sign = str(sign).lower()

    if sign == "positive":
        return f"{a} > {b}"
    if sign == "negative":
        return f"{b} > {a}"

    return f"{pair_name} ({sign})"


def short_point_label(row) -> str:
    label = pretty_sign_label(row["pair_name"], row["sign"])
    region = f"R{int(row['region_rank'])}"
    point = str(row["point_type"])
    layer = int(float(row["Layer"]))
    noise = float(row["NoiseStd"])
    percent = float(row["Percent"])
    return f"{label}\n{region}-{point}\nL{layer}, N{noise:.1f}, P{percent:.1f}"


def region_label(row) -> str:
    return f"{pretty_sign_label(row['pair_name'], row['sign'])}\nR{int(row['region_rank'])}"


def add_zero_line(ax, axis="y"):
    if axis == "y":
        ax.axhline(0, linestyle="--", linewidth=1)
    elif axis == "x":
        ax.axvline(0, linestyle="--", linewidth=1)


def add_threshold_line(ax, y, label=None):
    ax.axhline(y, linestyle="--", linewidth=1)
    if label:
        ax.text(
            0.01,
            y + 0.015,
            label,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=9,
        )


def require_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}. Available: {list(df.columns)}")


def point_type_order_value(x):
    order = {"peak": 0, "centroid": 1, "support": 2}
    return order.get(str(x), 99)


def find_seed_subtraction_column(point_seed_df: pd.DataFrame) -> str:
    """
    Try common column names used by subtraction validation scripts.
    """
    candidates = [
        "validation_seed_subtraction",
        "validation_seed_difference",
        "validation_seed_diff",
        "validation_seed_mean_subtraction",
        "validation_seed_value",
    ]

    for c in candidates:
        if c in point_seed_df.columns:
            return c

    raise ValueError(
        "Could not find seed-level subtraction value column. "
        f"Tried: {candidates}. "
        f"Available columns: {list(point_seed_df.columns)}"
    )


# --------------------------------------------------
# Table builders
# --------------------------------------------------

def build_candidate_seed_success_table(
    point_seed_df: pd.DataFrame,
    seed_zero_eps: float = 0.0,
) -> pd.DataFrame:
    """
    One row per selected rerun candidate point.

    For subtraction:
        positive sign:
            correct = validation_seed_subtraction > seed_zero_eps

        negative sign:
            correct = validation_seed_subtraction < -seed_zero_eps
    """
    value_col = find_seed_subtraction_column(point_seed_df)

    require_columns(
        point_seed_df,
        [
            "pair_name",
            "sign",
            "region_rank",
            "point_type",
            "Layer",
            "NoiseStd",
            "Percent",
            "Seed",
            value_col,
        ],
        "point_seed_lookup",
    )

    work = point_seed_df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work["sign"] = work["sign"].astype(str).str.lower().str.strip()

    group_cols = [
        "pair_name",
        "pair_a",
        "pair_b",
        "sign",
        "region_rank",
        "point_type",
        "Layer",
        "NoiseStd",
        "Percent",
        "idx_layer",
        "idx_noise",
        "idx_percent",
    ]
    group_cols = [c for c in group_cols if c in work.columns]

    rows = []

    for keys, sub in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_dict = dict(zip(group_cols, keys))

        valid = sub.dropna(subset=[value_col]).copy()

        if "Seed" in valid.columns:
            n_total = int(valid["Seed"].nunique())
        else:
            n_total = int(len(valid))

        sign = str(key_dict.get("sign", "")).lower().strip()

        if sign == "positive":
            correct_mask = valid[value_col] > float(seed_zero_eps)
            expected_pattern = "validation_subtraction_positive"
        elif sign == "negative":
            correct_mask = valid[value_col] < -float(seed_zero_eps)
            expected_pattern = "validation_subtraction_negative"
        else:
            correct_mask = pd.Series([False] * len(valid), index=valid.index)
            expected_pattern = "unknown_sign"

        n_correct = int(correct_mask.sum())
        seed_values = valid[value_col].astype(float)

        success_rate = n_correct / n_total if n_total > 0 else np.nan

        row = {
            **key_dict,
            "expected_validation_pattern": expected_pattern,
            "seed_value_column": value_col,
            "n_correct": n_correct,
            "n_total": n_total,
            "success_rate": success_rate,
            "success_percent": success_rate * 100 if np.isfinite(success_rate) else np.nan,
            "success_text": f"{n_correct}/{n_total}",
            "validation_seed_mean": float(seed_values.mean()) if len(seed_values) else np.nan,
            "validation_seed_median": float(seed_values.median()) if len(seed_values) else np.nan,
            "validation_seed_min": float(seed_values.min()) if len(seed_values) else np.nan,
            "validation_seed_max": float(seed_values.max()) if len(seed_values) else np.nan,
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["point_type_order"] = out["point_type"].apply(point_type_order_value)
    out = out.sort_values(
        ["pair_name", "sign", "region_rank", "point_type_order", "Layer", "NoiseStd", "Percent"]
    ).drop(columns=["point_type_order"]).reset_index(drop=True)

    return out


def build_region_seed_success_table(candidate_success_df: pd.DataFrame) -> pd.DataFrame:
    """
    Region-level summary of seed success counts across candidate points.

    This does not redefine validation success. It summarizes the candidate-level
    n_correct / n_total values inside each discovered region.
    """
    if candidate_success_df.empty:
        return pd.DataFrame()

    group_cols = ["pair_name", "pair_a", "pair_b", "sign", "region_rank"]
    group_cols = [c for c in group_cols if c in candidate_success_df.columns]

    rows = []

    for keys, sub in candidate_success_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_dict = dict(zip(group_cols, keys))

        peak = sub[sub["point_type"] == "peak"].copy()
        centroid = sub[sub["point_type"] == "centroid"].copy()
        support = sub[sub["point_type"] == "support"].copy()

        row = {
            **key_dict,
            "n_candidate_points": int(len(sub)),
            "mean_n_correct": float(sub["n_correct"].mean()),
            "median_n_correct": float(sub["n_correct"].median()),
            "min_n_correct": int(sub["n_correct"].min()),
            "max_n_correct": int(sub["n_correct"].max()),
            "mean_success_rate": float(sub["success_rate"].mean()),
            "median_success_rate": float(sub["success_rate"].median()),
            "min_success_rate": float(sub["success_rate"].min()),
            "max_success_rate": float(sub["success_rate"].max()),
            "n_total": int(sub["n_total"].max()),
        }

        if not peak.empty:
            row["peak_n_correct"] = int(peak["n_correct"].iloc[0])
            row["peak_success_text"] = str(peak["success_text"].iloc[0])
            row["peak_success_rate"] = float(peak["success_rate"].iloc[0])
        else:
            row["peak_n_correct"] = np.nan
            row["peak_success_text"] = ""
            row["peak_success_rate"] = np.nan

        if not centroid.empty:
            row["centroid_n_correct"] = int(centroid["n_correct"].iloc[0])
            row["centroid_success_text"] = str(centroid["success_text"].iloc[0])
            row["centroid_success_rate"] = float(centroid["success_rate"].iloc[0])
        else:
            row["centroid_n_correct"] = np.nan
            row["centroid_success_text"] = ""
            row["centroid_success_rate"] = np.nan

        if not support.empty:
            row["support_mean_n_correct"] = float(support["n_correct"].mean())
            row["support_min_n_correct"] = int(support["n_correct"].min())
            row["support_max_n_correct"] = int(support["n_correct"].max())
        else:
            row["support_mean_n_correct"] = np.nan
            row["support_min_n_correct"] = np.nan
            row["support_max_n_correct"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["pair_name", "sign", "region_rank"]).reset_index(drop=True)
    return out


def build_paper_region_summary_table(
    region_df: pd.DataFrame,
    region_seed_success_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact region-level table for paper/report.
    """
    region_cols = [
        "pair_name",
        "pair_a",
        "pair_b",
        "sign",
        "region_rank",
        "size_voxels",
        "tfce_sum",
        "tfce_max",
        "peak_layer",
        "peak_noise",
        "peak_percent",
        "validation_mean_subtraction",
        "validation_median_subtraction",
        "validation_min_subtraction",
        "validation_max_subtraction",
        "sign_match_rate_meanmap",
        "mean_seed_match_rate",
        "min_seed_match_rate",
        "max_seed_match_rate",
        "region_success_primary",
        "n_points",
    ]

    keep = [c for c in region_cols if c in region_df.columns]
    out = region_df[keep].copy()

    merge_cols = [
        "pair_name",
        "sign",
        "region_rank",
        "peak_success_text",
        "peak_n_correct",
        "peak_success_rate",
        "centroid_success_text",
        "centroid_n_correct",
        "centroid_success_rate",
        "mean_n_correct",
        "min_n_correct",
        "max_n_correct",
        "mean_success_rate",
        "min_success_rate",
        "max_success_rate",
        "n_total",
    ]

    if region_seed_success_df is not None and not region_seed_success_df.empty:
        merge_keep = [c for c in merge_cols if c in region_seed_success_df.columns]
        out = out.merge(
            region_seed_success_df[merge_keep],
            on=["pair_name", "sign", "region_rank"],
            how="left",
        )

    if "pair_name" in out.columns and "sign" in out.columns and "region_rank" in out.columns:
        out = out.sort_values(["pair_name", "sign", "region_rank"]).reset_index(drop=True)

    return out


def build_peak_candidate_table(candidate_success_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per region peak candidate. Useful for a simple paper table.
    """
    if candidate_success_df.empty:
        return pd.DataFrame()

    out = candidate_success_df[candidate_success_df["point_type"] == "peak"].copy()

    keep_cols = [
        "pair_name",
        "pair_a",
        "pair_b",
        "sign",
        "region_rank",
        "point_type",
        "Layer",
        "NoiseStd",
        "Percent",
        "expected_validation_pattern",
        "n_correct",
        "n_total",
        "success_text",
        "success_rate",
        "validation_seed_mean",
        "validation_seed_min",
        "validation_seed_max",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]

    out = out[keep_cols].copy()
    out = out.sort_values(["pair_name", "sign", "region_rank"]).reset_index(drop=True)
    return out


# --------------------------------------------------
# Plots
# --------------------------------------------------

def plot_region_validation_mean_subtraction(region_df: pd.DataFrame, outpath: Path, dpi: int = 220):
    col = "validation_mean_subtraction"
    if col not in region_df.columns:
        print(f"[SKIP] {outpath.name}: missing {col}")
        return

    work = region_df.copy()
    work["label"] = work.apply(region_label, axis=1)
    work = work.sort_values(["pair_name", "sign", "region_rank"]).reset_index(drop=True)

    x = np.arange(len(work))
    y = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(9, len(work) * 0.75), 5))
    ax.bar(x, y)
    add_zero_line(ax, "y")

    ax.set_ylabel("Validation mean subtraction")
    ax.set_title("Validation subtraction magnitude by discovered region")
    ax.set_xticks(x)
    ax.set_xticklabels(work["label"], rotation=45, ha="right")

    for xi, yi in zip(x, y):
        if np.isfinite(yi):
            va = "bottom" if yi >= 0 else "top"
            offset = 0.003 if yi >= 0 else -0.003
            ax.text(xi, yi + offset, f"{yi:.3f}", ha="center", va=va, fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {outpath}")


def plot_region_mean_seed_match_rate(
    region_df: pd.DataFrame,
    outpath: Path,
    seed_match_rate_threshold: float = 0.70,
    dpi: int = 220,
):
    col = "mean_seed_match_rate"
    if col not in region_df.columns:
        print(f"[SKIP] {outpath.name}: missing {col}")
        return

    work = region_df.copy()
    work["label"] = work.apply(region_label, axis=1)
    work = work.sort_values(["pair_name", "sign", "region_rank"]).reset_index(drop=True)

    x = np.arange(len(work))
    y = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(9, len(work) * 0.75), 5))
    ax.bar(x, y)
    ax.set_ylim(0, 1.05)
    add_threshold_line(ax, seed_match_rate_threshold, f"threshold = {seed_match_rate_threshold:.2f}")

    ax.set_ylabel("Mean validation seed sign-match rate")
    ax.set_title("Validation seed sign consistency by discovered region")
    ax.set_xticks(x)
    ax.set_xticklabels(work["label"], rotation=45, ha="right")

    for xi, yi in zip(x, y):
        if np.isfinite(yi):
            ax.text(xi, yi + 0.015, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {outpath}")


def plot_candidate_seed_success_count(candidate_df: pd.DataFrame, outpath: Path, dpi: int = 220):
    if candidate_df.empty:
        print(f"[SKIP] {outpath.name}: empty candidate table")
        return

    work = candidate_df.copy()
    work["label"] = work.apply(short_point_label, axis=1)
    work["point_type_order"] = work["point_type"].apply(point_type_order_value)
    work = work.sort_values(
        ["pair_name", "sign", "region_rank", "point_type_order", "Layer", "NoiseStd", "Percent"]
    ).reset_index(drop=True)

    x = np.arange(len(work))
    y = pd.to_numeric(work["n_correct"], errors="coerce").to_numpy(dtype=float)
    n_total = int(pd.to_numeric(work["n_total"], errors="coerce").max())

    fig, ax = plt.subplots(figsize=(max(12, len(work) * 0.55), 6))
    ax.bar(x, y)
    ax.set_ylim(0, max(n_total + 2, 42))
    ax.set_ylabel(f"Correct validation seeds out of {n_total}")
    ax.set_title("Seed-level validation sign consistency by selected condition")
    ax.set_xticks(x)
    ax.set_xticklabels(work["label"], rotation=75, ha="right", fontsize=8)

    for xi, yi, txt in zip(x, y, work["success_text"]):
        if np.isfinite(yi):
            ax.text(xi, yi + 0.6, txt, ha="center", va="bottom", fontsize=7, rotation=90)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {outpath}")


def plot_candidate_success_rate(candidate_df: pd.DataFrame, outpath: Path, dpi: int = 220):
    if candidate_df.empty:
        print(f"[SKIP] {outpath.name}: empty candidate table")
        return

    work = candidate_df.copy()
    work["label"] = work.apply(short_point_label, axis=1)
    work["point_type_order"] = work["point_type"].apply(point_type_order_value)
    work = work.sort_values(
        ["pair_name", "sign", "region_rank", "point_type_order", "Layer", "NoiseStd", "Percent"]
    ).reset_index(drop=True)

    x = np.arange(len(work))
    y = pd.to_numeric(work["success_rate"], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(12, len(work) * 0.55), 6))
    ax.bar(x, y)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Seed-level success rate")
    ax.set_title("Seed-level validation sign-consistency rate by selected condition")
    ax.set_xticks(x)
    ax.set_xticklabels(work["label"], rotation=75, ha="right", fontsize=8)

    for xi, yi in zip(x, y):
        if np.isfinite(yi):
            ax.text(xi, yi + 0.015, f"{yi:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {outpath}")


def plot_discovery_vs_validation_subtraction(point_df: pd.DataFrame, outpath: Path, dpi: int = 220):
    required = ["discovery_mean_subtraction", "validation_mean_subtraction"]
    if any(c not in point_df.columns for c in required):
        print(f"[SKIP] {outpath.name}: missing discovery/validation subtraction columns")
        return

    work = point_df.copy()
    work = work.dropna(subset=required)

    if work.empty:
        print(f"[SKIP] {outpath.name}: no usable rows")
        return

    marker_map = {"peak": "*", "centroid": "o", "support": "x"}

    fig, ax = plt.subplots(figsize=(6.5, 6))

    for point_type, sub in work.groupby("point_type"):
        ax.scatter(
            sub["discovery_mean_subtraction"].to_numpy(dtype=float),
            sub["validation_mean_subtraction"].to_numpy(dtype=float),
            label=point_type,
            marker=marker_map.get(point_type, "o"),
            s=120 if point_type == "peak" else 70,
            alpha=0.85,
        )

    all_vals = np.concatenate([
        work["discovery_mean_subtraction"].to_numpy(dtype=float),
        work["validation_mean_subtraction"].to_numpy(dtype=float),
    ])

    lo = float(np.nanmin(all_vals))
    hi = float(np.nanmax(all_vals))
    pad = max(0.01, 0.05 * (hi - lo if hi > lo else 1.0))
    lo -= pad
    hi += pad

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, label="y = x")
    add_zero_line(ax, "x")
    add_zero_line(ax, "y")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Discovery mean subtraction")
    ax.set_ylabel("Validation mean subtraction")
    ax.set_title("Discovery vs. validation at selected conditions")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {outpath}")


def plot_validation_subtraction_by_region(point_df: pd.DataFrame, outpath: Path, dpi: int = 220):
    col = "validation_mean_subtraction"
    if col not in point_df.columns:
        print(f"[SKIP] {outpath.name}: missing {col}")
        return

    work = point_df.copy()
    work = work.dropna(subset=[col])

    if work.empty:
        print(f"[SKIP] {outpath.name}: no usable rows")
        return

    grouped = []
    labels = []

    for keys, sub in work.groupby(["pair_name", "sign", "region_rank"], dropna=False):
        pair_name, sign, region_rank = keys
        grouped.append(sub[col].to_numpy(dtype=float))
        labels.append(f"{pretty_sign_label(pair_name, sign)}\nR{int(region_rank)}")

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.8), 5.5))
    ax.boxplot(grouped, tick_labels=labels, showfliers=True)
    add_zero_line(ax, "y")

    ax.set_ylabel("Validation mean subtraction")
    ax.set_title("Distribution of selected-point validation subtraction by region")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {outpath}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--region_summary_csv",
        required=True,
        help="Path to *_validation_region_summary.csv",
    )
    ap.add_argument(
        "--point_lookup_csv",
        required=True,
        help="Path to *_validation_point_lookup.csv",
    )
    ap.add_argument(
        "--point_seed_lookup_csv",
        required=True,
        help="Path to *_validation_point_seed_lookup.csv",
    )
    ap.add_argument(
        "--outdir",
        required=True,
        help="Output directory",
    )
    ap.add_argument(
        "--seed_zero_eps",
        type=float,
        default=0.0,
        help=(
            "For sign consistency, positive is value > eps and negative is value < -eps. "
            "Default 0.0."
        ),
    )
    ap.add_argument(
        "--seed_match_rate_threshold",
        type=float,
        default=0.70,
        help="Threshold line for mean seed match-rate plot.",
    )
    ap.add_argument("--dpi", type=int, default=220)

    args = ap.parse_args()

    region_summary_csv = Path(args.region_summary_csv).expanduser().resolve()
    point_lookup_csv = Path(args.point_lookup_csv).expanduser().resolve()
    point_seed_lookup_csv = Path(args.point_seed_lookup_csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    if not region_summary_csv.exists():
        raise SystemExit(f"region_summary_csv not found: {region_summary_csv}")
    if not point_lookup_csv.exists():
        raise SystemExit(f"point_lookup_csv not found: {point_lookup_csv}")
    if not point_seed_lookup_csv.exists():
        raise SystemExit(f"point_seed_lookup_csv not found: {point_seed_lookup_csv}")

    ensure_dir(outdir)

    region_df = pd.read_csv(region_summary_csv)
    point_df = pd.read_csv(point_lookup_csv)
    point_seed_df = pd.read_csv(point_seed_lookup_csv)

    print(f"region_summary rows: {len(region_df)}")
    print(f"point_lookup rows: {len(point_df)}")
    print(f"point_seed_lookup rows: {len(point_seed_df)}")

    # -------------------------
    # Build tables
    # -------------------------
    candidate_success_df = build_candidate_seed_success_table(
        point_seed_df=point_seed_df,
        seed_zero_eps=args.seed_zero_eps,
    )

    region_seed_success_df = build_region_seed_success_table(candidate_success_df)

    paper_region_summary_df = build_paper_region_summary_table(
        region_df=region_df,
        region_seed_success_df=region_seed_success_df,
    )

    peak_candidate_df = build_peak_candidate_table(candidate_success_df)

    paper_region_summary_path = outdir / "paper_region_summary_table.csv"
    candidate_success_path = outdir / "paper_candidate_seed_success_table.csv"
    region_seed_success_path = outdir / "paper_region_seed_success_table.csv"
    peak_candidate_path = outdir / "paper_peak_candidate_table.csv"

    paper_region_summary_df.to_csv(paper_region_summary_path, index=False)
    candidate_success_df.to_csv(candidate_success_path, index=False)
    region_seed_success_df.to_csv(region_seed_success_path, index=False)
    peak_candidate_df.to_csv(peak_candidate_path, index=False)

    print(f"[SAVED] {paper_region_summary_path}")
    print(f"[SAVED] {candidate_success_path}")
    print(f"[SAVED] {region_seed_success_path}")
    print(f"[SAVED] {peak_candidate_path}")

    # -------------------------
    # Plots
    # -------------------------
    plot_region_validation_mean_subtraction(
        region_df,
        outdir / "paper_region_validation_mean_subtraction.png",
        dpi=args.dpi,
    )

    plot_region_mean_seed_match_rate(
        region_df,
        outdir / "paper_region_mean_seed_match_rate.png",
        seed_match_rate_threshold=args.seed_match_rate_threshold,
        dpi=args.dpi,
    )

    plot_candidate_seed_success_count(
        candidate_success_df,
        outdir / "paper_candidate_seed_success_count.png",
        dpi=args.dpi,
    )

    plot_candidate_success_rate(
        candidate_success_df,
        outdir / "paper_candidate_success_rate.png",
        dpi=args.dpi,
    )

    plot_discovery_vs_validation_subtraction(
        point_df,
        outdir / "paper_discovery_vs_validation_subtraction.png",
        dpi=args.dpi,
    )

    plot_validation_subtraction_by_region(
        point_df,
        outdir / "paper_validation_subtraction_by_region.png",
        dpi=args.dpi,
    )

    print(f"All paper-ready subtraction validation outputs saved under: {outdir}")


if __name__ == "__main__":
    main()


# python plot_subtraction_validation_results_v2.py \
#   --region_summary_csv ./tfce_results/sub3d_targetpairs_validation_eval_validation_region_summary.csv \
#   --point_lookup_csv ./tfce_results/sub3d_targetpairs_validation_eval_validation_point_lookup.csv \
#   --point_seed_lookup_csv ./tfce_results/sub3d_targetpairs_validation_eval_validation_point_seed_lookup.csv \
#   --outdir ./tfce_results/sub3d_targetpairs_validation_plots_v2