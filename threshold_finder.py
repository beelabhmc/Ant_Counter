#!/usr/bin/env python3
"""
Threshold Finder
================
Collect LD1 diff values from two videos — one quiet (no ants), one active
(ants present) — and plot their distributions to determine the best DIFF_THRESH.

Usage:
    python threshold_finder.py

Steps:
    1. Set QUIET_VIDEO_PATH to a video with no ant movement.
    2. Set ACTIVE_VIDEO_PATH to a video with ant movement.
    3. Run the script — prints recommended values and shows a histogram.
"""

from __future__ import annotations
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from ant_counter_circle_shadow import compute_ld1_diff, PROCESS_SCALE, FRAME_STRIDE

# ── Configuration ─────────────────────────────────────────────────────────────

QUIET_VIDEO_PATH  = "path/to/quiet_video.MP4"    # no ants
ACTIVE_VIDEO_PATH = "path/to/active_video.MP4"   # ants present

# ── Collect diff values ───────────────────────────────────────────────────────

def collect_diff_values(video_path: str) -> np.ndarray:
    """
    Collect LD1 diff values across all frames of a video using the same
    pipeline as ant_counter_circle_shadow:
      - resize to PROCESS_SCALE
      - compare frame t to frame t-FRAME_STRIDE (ring buffer)
      - apply Gaussian blur (3x3)
      - collect all pixel values across the whole frame

    Returns a flat float32 array of diff values (same scale as DIFF_THRESH).
    """
    cap    = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    proc_w = int(orig_w * PROCESS_SCALE)
    proc_h = int(orig_h * PROCESS_SCALE)

    diff_values: list[float] = []
    frame_buf: list = []

    for fid in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (proc_w, proc_h))

        # Identical ring-buffer logic to ant_counter_circle_shadow._process_frames
        frame_buf.append(small.copy())
        if len(frame_buf) > FRAME_STRIDE + 1:
            frame_buf.pop(0)

        if len(frame_buf) > FRAME_STRIDE:
            diff = compute_ld1_diff(frame_buf[0], small)
            diff = cv2.GaussianBlur(diff, (3, 3), 0)
            diff_values.extend(diff.flatten().tolist())

        if fid % 50 == 0:
            print(f"  frame {fid}/{total}  ({len(diff_values):,} pixels collected)",
                  end="\r")

    cap.release()
    print()
    return np.array(diff_values, dtype=np.float32)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Collecting quiet video:  {QUIET_VIDEO_PATH}")
    quiet = collect_diff_values(QUIET_VIDEO_PATH)

    print(f"Collecting active video: {ACTIVE_VIDEO_PATH}")
    active = collect_diff_values(ACTIVE_VIDEO_PATH)

    # ── Statistics ────────────────────────────────────────────────────────────
    bg_99   = np.percentile(quiet,  99)
    bg_995  = np.percentile(quiet,  99.5)
    ant_95  = np.percentile(active, 95)
    ant_999 = np.percentile(active, 99.9)

    recommended_scalar = 255 / ant_999 if ant_999 > 0 else 255 / 30
    recommended_thresh = bg_99 * recommended_scalar

    print("\n── LD1 diff statistics ───────────────────────────────────────")
    print(f"  Quiet   99th pct  : {bg_99:.3f}")
    print(f"  Quiet   99.5th pct: {bg_995:.3f}")
    print(f"  Active  95th pct  : {ant_95:.3f}")
    print(f"  Active  99.9th pct: {ant_999:.3f}")
    print()
    print("── Recommended values ────────────────────────────────────────")
    print(f"  Scalar  (255 / 99.9th pct) : 255 / {ant_999:.3f} = {recommended_scalar:.2f}")
    print(f"  DIFF_THRESH (scaled)        : {recommended_thresh:.1f}")
    print()
    print("  In ant_counter_circle_shadow.py set:")
    print(f"    return (diff * {recommended_scalar:.2f}).clip(0, 255).astype(np.uint8)")
    print(f"    DIFF_THRESH = {recommended_thresh:.0f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    x_max = float(np.percentile(active, 99.9)) * 1.1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw LD1 diff distributions
    ax = axes[0]
    ax.hist(quiet[quiet <= x_max],   bins=150, alpha=0.6, density=True,
            color="steelblue", label="quiet (no ants)")
    ax.hist(active[active <= x_max], bins=150, alpha=0.6, density=True,
            color="tomato",    label="active (ants present)")
    ax.axvline(bg_99,   color="steelblue", linestyle="--", linewidth=1.5,
               label=f"quiet 99th pct = {bg_99:.3f}")
    ax.axvline(ant_999, color="tomato",    linestyle="--", linewidth=1.5,
               label=f"active 99.9th pct = {ant_999:.3f}")
    ax.set_xlabel("LD1 diff value (0–255 scale)")
    ax.set_ylabel("Density")
    ax.set_title("LD1 diff distributions")
    ax.set_xlim(0, x_max)
    ax.legend(fontsize=8)

    # Right: full 0–255 range with recommended threshold marked
    ax2 = axes[1]
    ax2.hist(quiet,  bins=150, alpha=0.6, density=True,
             color="steelblue", label="quiet (no ants)")
    ax2.hist(active, bins=150, alpha=0.6, density=True,
             color="tomato",    label="active (ants present)")
    ax2.axvline(recommended_thresh, color="black", linestyle="--", linewidth=1.5,
                label=f"recommended DIFF_THRESH = {recommended_thresh:.0f}")
    ax2.set_xlabel("LD1 diff value (0–255 scale)")
    ax2.set_ylabel("Density")
    ax2.set_title("Full range with recommended threshold")
    ax2.set_xlim(0, 255)
    ax2.legend(fontsize=8)

    plt.suptitle("LD1 diff threshold calibration", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
