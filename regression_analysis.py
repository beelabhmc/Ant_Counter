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
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ── Configuration ─────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the ground truth CSV
# GROUND_TRUTH_CSV = os.path.join(_SCRIPT_DIR, "Natalie Thesis Data - Count Data Representative Sample.csv")

# Folder(s) to search for output/*_summary.csv files
# Searches recursively inside each directory listed here
SEARCH_DIRS = [os.path.join(_SCRIPT_DIR, "outputs")]

# Save merged predictions + ground truth to a CSV for further inspection
SAVE_RESULTS_CSV = True
RESULTS_CSV_PATH = os.path.join(_SCRIPT_DIR, "regression_results.csv")

_QUADRANTS = ("NE", "SE", "SW", "NW")

# ── Load predictions ──────────────────────────────────────────────────────────

def load_predictions(search_dirs: list[str]) -> pd.DataFrame:
    """
    Find all output/*_summary.csv files under search_dirs and load them.
    Returns a long DataFrame with one row per (stem, quadrant, direction),
    where direction is 'enters' or 'exits'.
    Columns: stem, quadrant, direction, predicted.
    """
    rows = []
    for base in search_dirs:
        for summary_path in sorted(glob.glob(
                os.path.join(base, "**", "*_summary.csv"), recursive=True)):
            stem = os.path.basename(summary_path).replace("_summary.csv", "")
            with open(summary_path, newline="") as f:
                for row in csv.DictReader(f):
                    q = row["quadrant"].strip()
                    if q == "TOTAL":
                        continue
                    rows.append({"stem": stem, "quadrant": q,
                                 "direction": "enters",
                                 "predicted": int(row["enters"]),
                                 "mean_grayscale": float(row.get("mean_grayscale", 0))})
                    rows.append({"stem": stem, "quadrant": q,
                                 "direction": "exits",
                                 "predicted": int(row["exits"]),
                                 "mean_grayscale": float(row.get("mean_grayscale", 0))})

    if not rows:
        raise FileNotFoundError(
            "No *_summary.csv files found.\n"
            f"Searched under: {search_dirs}\n"
            "Run the batch processor first.")
    return pd.DataFrame(rows)


# ── Load ground truth ─────────────────────────────────────────────────────────

def load_ground_truth(csv_path: str) -> pd.DataFrame:
    """
    Load 'Representative Sample Counts' CSV.

    Expected columns:
        Video Title, Colony ID, Weather, Video Date,
        NW Enter, NW Exit, NE Enter, NE Exit,
        SW Enter, SW Exit, SE Enter, SE Exit

    Returns a long DataFrame with one row per (stem, quadrant, direction).
    Columns: stem, quadrant, direction, ground_truth.
    """
    if not csv_path:
        raise ValueError("Ground truth CSV path not provided. Please use the --ground-truth-csv argument to specify the path.")
    
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Ground truth file not found:\n  {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "Video Title" not in df.columns:
        raise ValueError(
            f"Expected a 'Video Title' column. Found: {list(df.columns)}")

    rows = []
    for _, row in df.iterrows():
        stem = str(row["Video Title"]).strip()
        for q in _QUADRANTS:
            try:
                e = int(pd.to_numeric(row.get(f"{q} Enter", 0), errors="coerce") or 0)
                x = int(pd.to_numeric(row.get(f"{q} Exit",  0), errors="coerce") or 0)
            except (ValueError, TypeError):
                warnings.warn(
                    f"Invalid count value for video '{stem}', quadrant '{q}'.\n"
                    f"Expected integer counts in columns '{q} Enter' and '{q} Exit'.\n"
                    f"Found: '{row.get(f'{q} Enter')}' and '{row.get(f'{q} Exit')}'")
            rows.append({"stem": stem, "quadrant": q,
                         "direction": "enters", "ground_truth": e})
            rows.append({"stem": stem, "quadrant": q,
                         "direction": "exits",  "ground_truth": x})

    return pd.DataFrame(rows)


# ── Regression ────────────────────────────────────────────────────────────────

def run_regression(merged: pd.DataFrame) -> dict:
    """
    Fit OLS linear regression: predicted ~ ground_truth across all
    (stem, quadrant, direction) data points.
    Returns a dict of statistics plus the raw arrays for plotting.
    """
    x = merged["ground_truth"].values.astype(float)
    y = merged["predicted"].values.astype(float)

    slope, intercept, r, p_value, std_err = stats.linregress(x, y)
    r2    = r ** 2
    y_hat = slope * x + intercept
    sq_err = (y - x) ** 2          # squared error vs perfect prediction
    rmse  = np.sqrt(np.mean((y - y_hat) ** 2))
    mae   = np.mean(np.abs(y - x))

    return dict(slope=slope, intercept=intercept, r2=r2, r=r,
                p_value=p_value, std_err=std_err, rmse=rmse, mae=mae,
                n=len(x), x=x, y=y, y_hat=y_hat, sq_err=sq_err)


# ── Plot ──────────────────────────────────────────────────────────────────────

_QUAD_COLORS = {
    "NE": "#e6b800",        # amber
    "SE": "#2ca02c",        # green
    "SW": "#1f77b4",        # blue
    "NW": "#9467bd",        # purple
}
_DIR_MARKERS = {"enters": "o", "exits": "^"}   # circle = enters, triangle = exits


def plot_regression(merged: pd.DataFrame, res: dict):
    """
    Two-panel figure:
      Left  — ground truth vs predicted, coloured by quadrant,
               shape by direction (enters ○ / exits △),
               with the regression line and perfect y=x line.
      Right — ground truth vs squared error (predicted − ground truth)²,
               same colour/shape coding.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    slope     = res["slope"]
    intercept = res["intercept"]

    for ax_idx, ax in enumerate(axes):
        for q in _QUADRANTS:
            for direction, marker in _DIR_MARKERS.items():
                sub = merged[
                    (merged["quadrant"] == q) &
                    (merged["direction"] == direction)
                ]
                if sub.empty:
                    continue

                gt  = sub["ground_truth"].values.astype(float)
                pr  = sub["predicted"].values.astype(float)
                sq  = (pr - gt) ** 2

                label = f"{q} {direction}"
                y_vals = pr if ax_idx == 0 else sq

                sc = ax.scatter(gt, y_vals,
                                color=_QUAD_COLORS[q],
                                marker=marker,
                                s=70, zorder=3,
                                label=label)

                # Label each point with its video stem
                for _, row in sub.iterrows():
                    y_val = row["predicted"] if ax_idx == 0 else (row["predicted"] - row["ground_truth"]) ** 2
                    ax.annotate(row["stem"], (row["ground_truth"], y_val),
                                fontsize=6, xytext=(4, 3),
                                textcoords="offset points", color=_QUAD_COLORS[q])

        if ax_idx == 0:
            # Regression line
            all_gt = merged["ground_truth"].values.astype(float)
            all_pr = merged["predicted"].values.astype(float)
            pad = max((all_gt.max() - all_gt.min()) * 0.12, 0.5)
            xlim = (all_gt.min() - pad, all_gt.max() + pad)
            ylim = (min(all_pr.min(), all_gt.min()) - pad,
                    max(all_pr.max(), all_gt.max()) + pad)

            x_line = np.linspace(xlim[0], xlim[1], 200)
            ax.plot(x_line, slope * x_line + intercept,
                    color="tomato", linewidth=2,
                    label=f"fit: y={slope:.2f}x+{intercept:.2f}")
            ax.plot(xlim, xlim, "k--", linewidth=1, alpha=0.4, label="perfect (y=x)")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_ylabel("Predicted count")
            ax.set_title("Predicted vs Ground Truth")
            ax.text(0.05, 0.95,
                    f"R²   = {res['r2']:.3f}\n"
                    f"RMSE = {res['rmse']:.2f}\n"
                    f"MAE  = {res['mae']:.2f}\n"
                    f"n    = {res['n']}",
                    transform=ax.transAxes, fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
        else:
            all_gt = merged["ground_truth"].values.astype(float)
            all_pr = merged["predicted"].values.astype(float)
            all_sq = (all_pr - all_gt) ** 2
            pad_x = max((all_gt.max() - all_gt.min()) * 0.12, 0.5)
            pad_y = max(all_sq.max() * 0.08, 0.1)
            ax.set_xlim(all_gt.min() - pad_x, all_gt.max() + pad_x)
            ax.set_ylim(-pad_y, all_sq.max() + pad_y)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_ylabel("Squared error  (predicted − ground truth)²")
            ax.set_title("Squared Error vs Ground Truth")

        ax.set_xlabel("Ground truth count")
        ax.legend(fontsize=7, ncol=2, loc="upper left",
                  bbox_to_anchor=(0.0, 0.85) if ax_idx == 0 else (0.0, 1.0))

    plt.suptitle(
        "Regression analysis — enters ○  exits △  |  colour = quadrant",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(search_dirs: list[str] | None = None, ground_truth_csv: str | None = None):
    if search_dirs is None:
        search_dirs = SEARCH_DIRS

    print("Loading predictions…")
    pred_df = load_predictions(search_dirs)
    print(f"  {pred_df['stem'].nunique()} video(s), "
          f"{len(pred_df)} prediction row(s) loaded.")

    print("Loading ground truth…")
    gt_df = load_ground_truth(ground_truth_csv)
    print(f"  {gt_df['stem'].nunique()} video(s), "
          f"{len(gt_df)} ground truth row(s) loaded.")

    # Merge on stem + quadrant + direction
    merged = pd.merge(pred_df, gt_df, on=["stem", "quadrant", "direction"], how="inner")

    if merged.empty:
        raise ValueError(
            "No rows matched after merge.\n"
            f"  Prediction stems : {sorted(pred_df['stem'].unique())}\n"
            f"  Ground truth stems: {sorted(gt_df['stem'].unique())}\n"
            "Check that 'Video Title' values match the output file name stems.")

    print(f"  {len(merged)} row(s) matched "
          f"({merged['stem'].nunique()} videos × 4 quadrants × 2 directions).")

    unmatched_pred = set(pred_df["stem"]) - set(merged["stem"])
    unmatched_gt   = set(gt_df["stem"])   - set(merged["stem"])
    if unmatched_pred:
        print(f"  WARNING — predictions with no ground truth : {unmatched_pred}")
    if unmatched_gt:
        print(f"  WARNING — ground truth with no predictions : {unmatched_gt}")

    print("\nFitting linear regression (all quadrants + directions combined)…")
    result = run_regression(merged)

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
        merged["squared_error"] = (merged["predicted"] - merged["ground_truth"]) ** 2
        merged.to_csv(RESULTS_CSV_PATH, index=False)
        print(f"\n  Saved merged results to:\n  {RESULTS_CSV_PATH}")
    
    plot_regression(merged, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit linear regression between predicted and ground truth ant counts.")
    parser.add_argument(
        "--search-dirs",
        nargs="+",
        default=None,
        help="Directories to search for output/*_summary.csv files (default: current script directory)")
    parser.add_argument(
        "--ground-truth-csv",
        default=None,
        help="Path to ground truth CSV")
    args = parser.parse_args()
    
    main(search_dirs=[os.path.abspath(dir) for dir in args.search_dirs], ground_truth_csv=os.path.abspath(args.ground_truth_csv))
