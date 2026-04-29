#!/usr/bin/env python3
"""
debug_gen.py  –  Ant detection debug video generator
=====================================================
Generates two 5-minute debug videos for GX010314 saved to
  GX010314/debug_outputs/

Video 1: debug_motion.mp4
  Three panels side-by-side at every frame:
    Left  – original frame (960x540)
    Mid   – raw frame diff ×5 (what the sensor sees before filtering)
    Right – cleaned diff after Gaussian pre-blur + 7×7 morph + threshold,
            showing only the blobs that make it through to the tracker.
            Cyan circles = detected blobs.  Red border = vibration frame (skipped).

Video 2: debug_labeled.mp4
  Original frame at 960x540 with:
    – Final filtered blobs drawn as cyan circles (size ∝ sqrt(area))
    – Blob area label printed next to each
    – Live count of detected blobs + vibration-skip counter

Run:
    python debug_gen.py

Tune the parameters at the top if needed.
"""

import cv2
import numpy as np
import os
from datetime import timedelta

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "GX010314", "GX010314.MP4")
OUT_DIR    = os.path.join(SCRIPT_DIR, "GX010314", "debug_outputs")

# ── parameters ───────────────────────────────────────────────────────────────
PROCESS_SCALE   = 0.25    # 4K → 960x540
MINUTES         = 5       # how many minutes of video to process
BG_SAMPLES      = 300     # frames sampled for the median background

DIFF_THRESH     = 15#20      # threshold on the Gaussian-blurred diff (was 12)
FRAME_STRIDE    = 2       # compare frame t to frame t-STRIDE

# Morphological kernels (larger = fewer, bigger blobs)
K_CLOSE_SIZE    = 7       # closing kernel – merges nearby motion pixels
K_OPEN_SIZE     = 5       # opening kernel – removes speckle below ant size

# Blob size limits at PROCESS_SCALE resolution
MIN_BLOB_AREA   = 10
MAX_BLOB_AREA   = 1200

# Vibration guard: skip detection when this % of pixels are "moving"
VIBRATION_PCT   = 8.0

# Amplification for the raw-diff visualisation panel (doesn't affect detection)
VIS_AMP         = 5.0

# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

cap    = cv2.VideoCapture(VIDEO_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS)
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

proc_w   = int(orig_w * PROCESS_SCALE)
proc_h   = int(orig_h * PROCESS_SCALE)
n_frames = min(total, int(MINUTES * 60 * fps))

print(f"Video : {orig_w}×{orig_h}  {fps:.0f} fps")
print(f"Output: {proc_w}×{proc_h}  {n_frames:,} frames  ({MINUTES} min)")
print(f"Saving to: {OUT_DIR}")
print()

# ── Build median background (for reference / BG-sub panel) ───────────────────
print("Step 1/3  Building median background …")
idxs = np.linspace(0, total - 1, BG_SAMPLES).astype(int)
cap  = cv2.VideoCapture(VIDEO_PATH)
bg_samples = []
for i, idx in enumerate(idxs):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ret, frame = cap.read()
    if ret:
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                       (proc_w, proc_h)).astype(np.float32)
        bg_samples.append(g)
    if (i + 1) % 60 == 0:
        print(f"  {i+1}/{BG_SAMPLES}")
cap.release()

bg_gray = np.median(np.stack(bg_samples), axis=0).astype(np.float32)
bg_blur = cv2.GaussianBlur(bg_gray, (5, 5), 0)
print("  done.\n")

# ── Video writers ─────────────────────────────────────────────────────────────
# Each panel is proc_w × proc_h; three panels side by side → 2880×540
panel_w = proc_w
panel_h = proc_h

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
w_motion  = cv2.VideoWriter(os.path.join(OUT_DIR, "debug_motion.mp4"),
                            fourcc, fps, (panel_w * 3, panel_h), True)
w_labeled = cv2.VideoWriter(os.path.join(OUT_DIR, "debug_labeled.mp4"),
                            fourcc, fps, (proc_w, proc_h), True)

# ── Morphological kernels (match ant_counter.py exactly) ─────────────────────
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_CLOSE_SIZE, K_CLOSE_SIZE))
k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K_OPEN_SIZE,  K_OPEN_SIZE))
frame_pixels  = proc_w * proc_h
vibration_lim = int(frame_pixels * VIBRATION_PCT / 100)


def colorize(diff_u8, amp=VIS_AMP):
    """Amplify and apply COLORMAP_HOT to a grayscale diff image."""
    clipped = np.clip(diff_u8.astype(np.float32) * amp, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(clipped, cv2.COLORMAP_HOT)


def draw_blobs_on(img, blobs, color, draw_labels=True):
    """Draw blob circles + area labels on img in-place."""
    for b in blobs:
        cx, cy = b["cx"], b["cy"]
        r = max(5, int(np.sqrt(b["area"] / np.pi)))
        cv2.circle(img, (cx, cy), r, color, 2)
        cv2.circle(img, (cx, cy), 2, color, -1)
        if draw_labels:
            cv2.putText(img, str(b["area"]), (cx + r + 2, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)


# ── Main processing loop ──────────────────────────────────────────────────────
print("Step 2/3  Processing frames …")

cap           = cv2.VideoCapture(VIDEO_PATH)
frame_buf     = []
n_vib_skips   = 0

for fid in range(n_frames):
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (proc_w, proc_h))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    ts    = str(timedelta(seconds=int(fid / fps)))

    # ── Frame-to-frame diff ───────────────────────────────────────────────────
    frame_buf.append(gray.copy())
    if len(frame_buf) > FRAME_STRIDE + 1:
        frame_buf.pop(0)

    if len(frame_buf) > FRAME_STRIDE:
        raw_diff = cv2.absdiff(gray, frame_buf[0])
        smooth   = cv2.GaussianBlur(raw_diff, (3, 3), 0)   # pre-blur: kills speckle
    else:
        raw_diff = np.zeros((proc_h, proc_w), dtype=np.uint8)
        smooth   = raw_diff.copy()

    # ── Threshold + morphological cleanup ────────────────────────────────────
    _, mask_raw = cv2.threshold(smooth, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask,     cv2.MORPH_OPEN,  k_open)

    # ── Vibration guard ───────────────────────────────────────────────────────
    n_fg        = int(np.count_nonzero(mask))
    is_vibration = n_fg > vibration_lim
    if is_vibration:
        n_vib_skips += 1

    # ── Blob extraction (skipped on vibration frames) ─────────────────────────
    blobs = []
    if not is_vibration:
        n_lbl, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n_lbl):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
                blobs.append({
                    "cx": int(centroids[i][0]),
                    "cy": int(centroids[i][1]),
                    "area": area,
                })

    # ── Panel A: original ─────────────────────────────────────────────────────
    pa = small.copy()
    cv2.rectangle(pa, (0, 0), (270, 28), (0, 0, 0), -1)
    cv2.putText(pa, f"Original  {ts}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Panel B: raw diff (no filtering) ─────────────────────────────────────
    # Shows what the sensor sees before any processing — lets you spot the flash.
    pb = colorize(raw_diff)
    cv2.rectangle(pb, (0, 0), (300, 28), (0, 0, 0), -1)
    cv2.putText(pb, f"Raw diff ×{VIS_AMP:.0f}  stride={FRAME_STRIDE}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1, cv2.LINE_AA)

    # ── Panel C: cleaned diff + final blobs ───────────────────────────────────
    pc = colorize(smooth)   # show Gaussian-blurred diff
    # overlay the cleaned mask boundary in white
    contours_c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pc, contours_c, -1, (255, 255, 255), 1)
    draw_blobs_on(pc, blobs, (0, 220, 255))
    # red border on vibration frames
    if is_vibration:
        cv2.rectangle(pc, (0, 0), (proc_w - 1, proc_h - 1), (0, 0, 255), 6)
    cv2.rectangle(pc, (0, 0), (420, 28), (0, 0, 0), -1)
    label_c = (f"VIBRATION SKIP ({n_fg/frame_pixels*100:.0f}% fg)"
               if is_vibration else
               f"Filtered  thr={DIFF_THRESH}  blobs={len(blobs)}  shakes={n_vib_skips}")
    col_c = (0, 80, 255) if is_vibration else (0, 220, 255)
    cv2.putText(pc, label_c, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, col_c, 1, cv2.LINE_AA)

    w_motion.write(np.hstack([pa, pb, pc]))

    # ── Labeled video ─────────────────────────────────────────────────────────
    labeled = small.copy()
    draw_blobs_on(labeled, blobs, (0, 220, 255), draw_labels=True)
    if is_vibration:
        cv2.rectangle(labeled, (0, 0), (proc_w - 1, proc_h - 1), (0, 0, 200), 4)
    cv2.rectangle(labeled, (6, 6), (400, 50), (0, 0, 0), -1)
    cv2.putText(labeled,
                f"Blobs: {len(blobs):3d}   thr={DIFF_THRESH}   shakes={n_vib_skips}",
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(labeled, ts, (proc_w - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1, cv2.LINE_AA)
    w_labeled.write(labeled)

    if fid % 300 == 0:
        pct = 100 * fid / n_frames
        print(f"  {fid:,}/{n_frames:,}  ({pct:.0f}%)  "
              f"blobs={len(blobs):3d}  vib_skips={n_vib_skips}")

cap.release()
w_motion.release()
w_labeled.release()

print(f"\nStep 3/3  Done.  Vibration frames skipped: {n_vib_skips}")
print(f"  {OUT_DIR}/debug_motion.mp4")
print(f"    Panel A: original")
print(f"    Panel B: raw frame diff ×{VIS_AMP:.0f}  (shows the flash source)")
print(f"    Panel C: Gaussian-blurred + {K_CLOSE_SIZE}×{K_CLOSE_SIZE} close + "
      f"{K_OPEN_SIZE}×{K_OPEN_SIZE} open, thr={DIFF_THRESH}  (what tracker sees)")
print(f"             Cyan circles = accepted blobs. Red border = vibration skip.")
print(f"  {OUT_DIR}/debug_labeled.mp4  ← blobs on original frame")
print()
print("What to look for:")
print("  Panel B (raw diff): should be nearly black except at ant locations.")
print("  If the whole panel flashes → camera vibration; VIBRATION_PCT will catch it.")
print("  Panel C: target is 5-15 blobs/frame, all on moving ants.")
print("  Tune DIFF_THRESH up if still too many blobs, down if ants are missed.")
