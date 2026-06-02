#!/usr/bin/env python3
"""
Regression Analysis
===================
Fit a linear regression between the batch processor's predicted counts
and manually collected ground truth counts, then plot and report stats.

Usage:
    python regression_analysis.py

Ground truth CSV format (Representative Sample Counts):
    Video Title, Colony ID, Weather, Video Date,
    NW Enter, NW Exit, NE Enter, NE Exit,
    SW Enter, SW Exit, SE Enter, SE Exit

Predicted output format (output/<stem>_summary.csv):
    quadrant, enters, exits
    NE, 5, 3
    SE, 0, 0
    SW, 0, 0
    NW, 3, 1
    TOTAL, 8, 4
"""

from __future__ import annotations
import os
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ── Configuration ─────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the ground truth CSV
GROUND_TRUTH_CSV = os.path.join(_SCRIPT_DIR, "Representative Sample Counts - Sheet1.csv")

# Folder(s) to search for output/*_summary.csv files
# Searches recursively inside each directory listed here
SEARCH_DIRS = [_SCRIPT_DIR]

# What to regress: "enters", "exits", or "net" (enters - exits)
COUNT_TYPE = "enters"

# True  → one data point per quadrant per video (4× more points)
# False → one data point per video (total across all quadrants)
PER_QUADRANT = False

# Save merged predictions + ground truth to a CSV for further inspection
SAVE_RESULTS_CSV = True
RESULTS_CSV_PATH = os.path.join(_SCRIPT_DIR, "regression_results.csv")

_QUADRANTS = ("NE", "SE", "SW", "NW")

# ── Load predictions ──────────────────────────────────────────────────────────

def load_predictions(search_dirs: list[str], per_quadrant: bool) -> pd.DataFrame:
    """
    Find all output/*_summary.csv files under search_dirs and load them.
    Returns a DataFrame with columns: stem, [quadrant,] pred_enters, pred_exits, pred_net.
    """
    rows = []
    for base in search_dirs:
        for summary_path in sorted(glob.glob(
                os.path.join(base, "**", "output", "*_summary.csv"), recursive=True)):
            stem = os.path.basename(summary_path).replace("_summary.csv", "")
            enters_by_q: dict[str, int] = {}
            exits_by_q:  dict[str, int] = {}

            with open(summary_path, newline="") as f:
                for row in csv.DictReader(f):
                    q = row["quadrant"].strip()
                    if q == "TOTAL":
                        continue
                    enters_by_q[q] = int(row["enters"])
                    exits_by_q[q]  = int(row["exits"])

            if per_quadrant:
                for q in _QUADRANTS:
                    e = enters_by_q.get(q, 0)
                    x = exits_by_q.get(q, 0)
                    rows.append({"stem": stem, "quadrant": q,
                                 "pred_enters": e, "pred_exits": x,
                                 "pred_net": e - x})
            else:
                total_e = sum(enters_by_q.values())
                total_x = sum(exits_by_q.values())
                rows.append({"stem": stem,
                             "pred_enters": total_e, "pred_exits": total_x,
                             "pred_net": total_e - total_x})

    if not rows:
        raise FileNotFoundError(
            "No output/*_summary.csv files found.\n"
            f"Searched under: {SEARCH_DIRS}\n"
            "Run the batch processor first.")
    return pd.DataFrame(rows)


# ── Load ground truth ─────────────────────────────────────────────────────────

def load_ground_truth(csv_path: str, per_quadrant: bool) -> pd.DataFrame:
    """
    Load 'Representative Sample Counts' CSV.

    Expected columns:
        Video Title, Colony ID, Weather, Video Date,
        NW Enter, NW Exit, NE Enter, NE Exit,
        SW Enter, SW Exit, SE Enter, SE Exit

    Returns a DataFrame with: stem, [quadrant,] gt_enters, gt_exits, gt_net.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Ground truth file not found:\n  {csv_path}")

    df = pd.read_csv(csv_path)
    # Normalise column names: strip whitespace, lowercase
    df.columns = [c.strip() for c in df.columns]

    if "Video Title" not in df.columns:
        raise ValueError(
            f"Expected a 'Video Title' column. Found: {list(df.columns)}")

    df["stem"] = df["Video Title"].astype(str).str.strip()

    # Parse per-quadrant counts from "NE Enter", "NE Exit", … columns
    rows = []
    for _, row in df.iterrows():
        for q in _QUADRANTS:
            e_col = f"{q} Enter"
            x_col = f"{q} Exit"
            e = int(pd.to_numeric(row.get(e_col, 0), errors="coerce") or 0)
            x = int(pd.to_numeric(row.get(x_col, 0), errors="coerce") or 0)
            rows.append({"stem": row["stem"], "quadrant": q,
                         "gt_enters": e, "gt_exits": x, "gt_net": e - x})

    long_df = pd.DataFrame(rows)

    if per_quadrant:
        return long_df.reset_index(drop=True)
    else:
        total_df = (
            long_df.groupby("stem", as_index=False)
            .agg(gt_enters=("gt_enters", "sum"),
                 gt_exits=("gt_exits",  "sum"))
        )
        total_df["gt_net"] = total_df["gt_enters"] - total_df["gt_exits"]
        return total_df


# ── Regression ────────────────────────────────────────────────────────────────

def run_regression(merged: pd.DataFrame, count_type: str) -> dict:
    """
    Fit OLS linear regression: predicted ~ ground_truth.
    Returns a dict of statistics plus the raw arrays for plotting.
    """
    pred_col = f"pred_{count_type}"
    gt_col   = f"gt_{count_type}"

    x = merged[gt_col].values.astype(float)   # ground truth  (independent)
    y = merged[pred_col].values.astype(float)  # predicted     (dependent)

    slope, intercept, r, p_value, std_err = stats.linregress(x, y)
    r2    = r ** 2
    y_hat = slope * x + intercept
    rmse  = np.sqrt(np.mean((y - y_hat) ** 2))
    mae   = np.mean(np.abs(y - y_hat))

    return dict(slope=slope, intercept=intercept, r2=r2, r=r,
                p_value=p_value, std_err=std_err, rmse=rmse, mae=mae,
                n=len(x), x=x, y=y, y_hat=y_hat,
                pred_col=pred_col, gt_col=gt_col)


# ── Plot ──────────────────────────────────────────────────────────────────────

_QUAD_COLORS = {
    "NE": "gold", "SE": "limegreen",
    "SW": "cornflowerblue", "NW": "violet",
}


def plot_regression(merged: pd.DataFrame, res: dict,
                    count_type: str, per_quadrant: bool):
    x, y, y_hat   = res["x"], res["y"], res["y_hat"]
    slope, intercept = res["slope"], res["intercept"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: scatter + fit line ──────────────────────────────────────────────
    ax = axes[0]

    if per_quadrant:
        for q in _QUADRANTS:
            sub = merged[merged["quadrant"] == q]
            ax.scatter(sub[res["gt_col"]], sub[res["pred_col"]],
                       color=_QUAD_COLORS[q], label=q, s=70, zorder=3)
    else:
        ax.scatter(x, y, color="steelblue", s=70, zorder=3)
        for _, row in merged.iterrows():
            ax.annotate(row["stem"],
                        (row[res["gt_col"]], row[res["pred_col"]]),
                        fontsize=7, xytext=(5, 4), textcoords="offset points")

    # Regression line
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="tomato", linewidth=2,
            label=f"fit:  y = {slope:.2f}x + {intercept:.2f}")

    # Perfect prediction (y = x)
    all_vals = np.concatenate([x, y])
    pad = max((all_vals.max() - all_vals.min()) * 0.1, 0.5)
    lim = (all_vals.min() - pad, all_vals.max() + pad)
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.4, label="perfect (y = x)")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"Ground truth  ({count_type})")
    ax.set_ylabel(f"Predicted  ({count_type})")
    ax.set_title(f"Predicted vs Ground Truth — {count_type}")
    ax.legend(fontsize=8)
    ax.text(0.05, 0.95,
            f"R² = {res['r2']:.3f}\n"
            f"RMSE = {res['rmse']:.2f}\n"
            f"MAE  = {res['mae']:.2f}\n"
            f"n = {res['n']}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    # ── Right: residuals ──────────────────────────────────────────────────────
    ax2 = axes[1]
    residuals = y - y_hat
    ax2.axhline(0, color="tomato", linewidth=1.5, linestyle="--")

    if per_quadrant:
        for q in _QUADRANTS:
            sub = merged[merged["quadrant"] == q]
            sub_x   = sub[res["gt_col"]].values.astype(float)
            sub_res = sub[res["pred_col"]].values.astype(float) - (slope * sub_x + intercept)
            ax2.scatter(sub_x, sub_res, color=_QUAD_COLORS[q], label=q, s=70)
        ax2.legend(fontsize=8)
    else:
        ax2.scatter(x, residuals, color="steelblue", s=70)
        for _, row in merged.iterrows():
            gt  = row[res["gt_col"]]
            res_ = row[res["pred_col"]] - (slope * gt + intercept)
            ax2.annotate(row["stem"], (gt, res_),
                         fontsize=7, xytext=(5, 4), textcoords="offset points")

    ax2.set_xlabel(f"Ground truth  ({count_type})")
    ax2.set_ylabel("Residual  (predicted − fitted)")
    ax2.set_title("Residual plot")

    plt.suptitle(
        f"Linear regression — {count_type} — "
        f"{'per quadrant' if per_quadrant else 'per video total'}",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading predictions…")
    pred_df = load_predictions(SEARCH_DIRS, PER_QUADRANT)
    print(f"  {pred_df['stem'].nunique()} video(s), "
          f"{len(pred_df)} prediction row(s) loaded.")

    print("Loading ground truth…")
    gt_df = load_ground_truth(GROUND_TRUTH_CSV, PER_QUADRANT)
    print(f"  {gt_df['stem'].nunique()} video(s), "
          f"{len(gt_df)} ground truth row(s) loaded.")

    # Merge on stem (+ quadrant when per_quadrant=True)
    join_keys = ["stem", "quadrant"] if PER_QUADRANT else ["stem"]
    merged = pd.merge(pred_df, gt_df, on=join_keys, how="inner")

    if merged.empty:
        raise ValueError(
            "No rows matched after merge.\n"
            f"  Prediction stems : {sorted(pred_df['stem'].unique())}\n"
            f"  Ground truth stems: {sorted(gt_df['stem'].unique())}\n"
            "Check that 'Video Title' values match the output file name stems.")

    print(f"  {len(merged)} row(s) matched.")

    unmatched_pred = set(pred_df["stem"]) - set(merged["stem"])
    unmatched_gt   = set(gt_df["stem"])   - set(merged["stem"])
    if unmatched_pred:
        print(f"  WARNING — predictions with no ground truth : {unmatched_pred}")
    if unmatched_gt:
        print(f"  WARNING — ground truth with no predictions : {unmatched_gt}")

    print(f"\nFitting linear regression on '{COUNT_TYPE}'…")
    result = run_regression(merged, COUNT_TYPE)

    print("\n── Results ───────────────────────────────────────────────────")
    print(f"  n          : {result['n']}")
    print(f"  Slope      : {result['slope']:.4f}   (ideal = 1.0)")
    print(f"  Intercept  : {result['intercept']:.4f}   (ideal = 0.0)")
    print(f"  R²         : {result['r2']:.4f}")
    print(f"  Pearson r  : {result['r']:.4f}")
    print(f"  p-value    : {result['p_value']:.4g}")
    print(f"  RMSE       : {result['rmse']:.2f}")
    print(f"  MAE        : {result['mae']:.2f}")

    if result["slope"] < 0.8:
        print("\n  ⚠  Slope < 0.8 — model is undercounting relative to ground truth.")
    elif result["slope"] > 1.2:
        print("\n  ⚠  Slope > 1.2 — model is overcounting relative to ground truth.")
    else:
        print("\n  ✓  Slope within 20% of 1.0 — counts are roughly proportional.")

    if SAVE_RESULTS_CSV:
        merged.to_csv(RESULTS_CSV_PATH, index=False)
        print(f"\n  Saved merged results to:\n  {RESULTS_CSV_PATH}")

    plot_regression(merged, result, COUNT_TYPE, PER_QUADRANT)


if __name__ == "__main__":
    main()
