#!/usr/bin/env python3
# ============================================================================
# ANT HOLE COUNTER — UNIFIED COMMAND LINE (GUI / BATCH / REGRESSION / OPTUNA)
# ============================================================================
# COMMANDS (choose ONE mode; no mode flag opens the GUI):
#   python ant_tracker_modular.py
#       Open the GUI (single video or a whole folder; see GUI below).
#   python ant_tracker_modular.py --gui --videos ./videos --recursive
#       Open the GUI with --run options pre-filled (all except --circle-file;
#       nothing starts until you click Process).
#   python ant_tracker_modular.py --run --videos ./videos [--recursive] [--strict]
#   python ant_tracker_modular.py --run --video ./videos/GX010287.MP4
#   python ant_tracker_modular.py --run --video ./day1/A.MP4 --video ./day2/B.MP4
#       Process video(s) headlessly (no GUI).
#   python ant_tracker_modular.py --regression ./counts.csv --videos ./videos
#       Compare per-quadrant IN/OUT counts with ONE consolidated CSV (Video Title,
#       Colony ID, Video Date, NW Enter/Exit, ...) or a folder of per-video CSVs;
#       fit OLS regression statistics and save a two-panel graph.
#   python ant_tracker_modular.py --gui --regression ./counts.csv --videos ./videos
#       The same in the GUI's Regression tab (pre-filled; click Run). The
#       plain GUI also has the Regression tab.
#   python ant_tracker_modular.py --optimize --ground-truth ./labels --videos ./videos
#   python ant_tracker_modular.py --optimize ./dataset      (dataset/{ground-truth,videos}/)
#       Use Optuna to optimize ant DETECTION F1 from positional annotations.
#
# SHARED OPTIONS:
#   --videos DIR            Video folder for --run, --gui, --regression, --optimize.
#   --video FILE            A video for --run or --gui (instead of --videos);
#                           repeat it for several videos from any folders.
#                           Their outputs go directly in the output folder.
#   --parameters FILE       JSON parameter preset (GUI, --run, regression; base
#                           settings for Optuna). --settings is an alias.
#   --circle-file FILE      One circle for every video (--run, --gui, regression).
#   --output DIR            Output root for batch/regression/Optuna.
#   --render-video          Write annotated MP4s (off by default in headless
#                           modes; --no-render-video disables).
#   --ground-truth PATH     Count CSV(s) for --regression or position CSVs for
#                           --optimize.
#   --no-prompt             Never prompt for missing inputs (see PROMPTS).
#   --code-version TEXT     GitHub version (commit, tag, release) to record in
#                           the run metadata. --run and --regression ask for it
#                           in a terminal (Enter = version detected from git);
#                           the GUI asks before each run.
#
# BATCH OPTIONS (--run and --gui):
#   --recursive             Include every subfolder of --videos. Skips hidden
#                           folders, macOS ._ files, *_counted.mp4 renders and
#                           the output folder. Outputs mirror the folder tree:
#                           videos/a/b/X.MP4 -> <output>/a/b/X_*.
#   --strict                Stop BEFORE processing if any circle is missing or
#                           invalid, and stop at the first video that fails.
#                           Without it those videos are skipped and recorded.
#   At the end, every skipped / failed / not-run video is listed with its
#   reason. Exit code: 0 = done (skipped videos are fine), 1 = at least one
#   video FAILED (the rest were still processed), 2 = stopped early
#   (--strict, bad --circle-file, or nothing processable).
#
# REGRESSION / OPTUNA OPTIONS:
#   --quadrant NE|SE|SW|NW  Restrict regression to one labeled quadrant.
#   --show-plot             Also show the regression figure (PNG is always saved).
#   --skip-missing-circles  Regression: skip and list matched videos that have
#                           no valid circle, instead of stopping the run.
#   --trials N              Additional Optuna trials (default 30).
#   --match-distance PX     Detection-match radius in original pixels (default 25).
#   --seed N                Optuna sampler random seed (default 42).
#
# CIRCLE LOOKUP for --run / --gui folder runs (first match wins):
#   1. --circle-file FILE (used for every video)
#   2. <video's folder>/<stem>_circle.json
#   3. <output folder>/<stem>_circle.json  (mirrored subfolder when recursive;
#      so circles the GUI wrote to its output folder are reused directly)
#   Every circle is checked before processing starts and a summary is printed:
#   "Found 42 video(s) | 40 with circles | 2 missing | 0 invalid".
#   Regression keeps its own lookup: video folder, ground-truth folder,
#   videos/circles/, or --circle-file.
#
# PROMPTS: in an interactive terminal, a missing REQUIRED input (the video
#   folder/file for --run; ground truth + videos for --regression; dataset or
#   ground truth + videos for --optimize) is asked for instead of erroring.
#   Paths are checked as typed (quotes, drag-and-drop escapes and ~ handled),
#   3 tries each, empty answer or Ctrl-C cancels, and the equivalent full
#   command is printed. Scripts/pipes/--no-prompt get the usual error.
#   Contradictory flags are always errors.
#
# OUTPUT DEFAULTS:
#   --run: <folder>/batch_results for --videos, ./outputs for --video.
#   regression: ./videos/regression_results/regression_<date-time>/ (a new
#     folder for every run, so results are never overwritten).
#   Optuna: dataset/ or ./videos/optuna_results; each run's results go in
#     optuna_<date-time>/ inside it (the study database stays at the top so
#     later runs resume the study).
#
# METADATA: every --run, regression and Optuna run writes run_metadata.json
#   (regression: regression_run_metadata.json) with the date and time, the
#   code version entered, git details, the command, the parameter file and
#   the exact parameter values used.
#
# INPUT CONVENTIONS:
#   videos/foo.mp4 pairs with ground-truth/foo.csv (same stem).
#   Optuna position CSV: timestamp,frame,ant_x,ant_y (ant_id optional), zero-based
#     frame indices and source-resolution x/y pixel coordinates.
#
# NOTES:
# * --regression FITS OLS on predicted vs ground-truth count pairs;
#   reports fit quality and counting errors. It does NOT tune parameters.
# * --optimize scores detected positions, NOT entrance counts.
# * The GUI and --run share plan_batch()/execute_batch(), so they behave and
#   write batch_summary.json identically.
# * Detection uses frame differencing, Kalman filtering and Hungarian matching.
# ============================================================================
"""
Ant Hole Counter
================
Counts ants entering and exiting an anthill, per compass quadrant.

Detection method: frame-to-frame difference (NOT background subtraction).
  Comparing adjacent frames detects only things that MOVED, so static shadows
  — even if they look very different from the median background — register as
  zero motion and are ignored.  Moving ants show up clearly.

GUI workflow:
    1. "Videos": add video files from any folders (Add videos…, Add a
       folder's videos…, or drag & drop files/folders onto the window), or
       "Folder": choose one folder (optionally "Include subfolders").
       Drag & drop needs: pip install tkinterdnd2
    2. Optionally load a parameter preset and choose an output folder.
    3. Scrub the frame slider to a clear view of the hole entrance.
    4. Click the entrance center, then click toward north; set the radius.
       Choose where circles come from: "Find <name>_circle.json in the input
       and output folders" (as --run does), or "Click it manually in the GUI"
       (double-click each video in the table and click its circle).
    5. Click Process. "Copy CLI command" gives the equivalent --run command.

Outputs (per video):
    <name>_counts.csv       — timestamp, event (enter/exit), running count
    <name>_summary.csv      — enters/exits per quadrant
    <name>_ant_boxes.csv    — per-frame ant boxes
    <name>_circle.json      — circle parameters (center, radius, north_angle)
    <name>_counted.mp4      — annotated video (when rendering is on)
    batch_summary.json      — one per run, at the output root
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from collections import deque, defaultdict
from pathlib import Path
import cv2
import numpy as np
import os
import json
import csv
import threading
import time
import math
import re
import shlex
import subprocess
import hashlib
import sys
from datetime import datetime, timezone, timedelta
from scipy.optimize import linear_sum_assignment as _hungarian_solve

# GUI-only and Optuna-only dependencies are optional so that headless modes
# (--run, --regression) work on machines without a display stack or Optuna.
# Each mode checks for what it needs and exits with an install hint.
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog, simpledialog
except ImportError:  # pragma: no cover - depends on the Python build
    tk = ttk = messagebox = filedialog = simpledialog = None

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    Image = ImageTk = None

# Optional: drag & drop of files/folders onto the GUI (pip install tkinterdnd2).
# Without it the GUI works the same, minus drag & drop.
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
except Exception:  # pragma: no cover - ImportError, or Tk missing
    TkinterDnD = DND_FILES = None

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


def _require_gui_dependencies():
    """Exit with a clear message if Tkinter or Pillow is unavailable."""
    missing = []
    if tk is None:
        missing.append("tkinter (install a Python build with Tk support)")
    if Image is None or ImageTk is None:
        missing.append("Pillow  (pip install Pillow)")
    if missing:
        raise SystemExit("The GUI needs:\n  " + "\n  ".join(missing))


def _require_optuna():
    """Exit with a clear message if Optuna is unavailable."""
    if optuna is None:
        raise SystemExit("--optimize needs Optuna:  pip install optuna")


# --- Cluster blobs to reduce shadow double-counting ---
# Groups nearby motion components by distance to the initial component in each group.
# Coordinates are processing-scale pixels; area is summed, centroids are averaged.
# Caution: this is a one-hop greedy cluster, not transitive clustering; two ants can merge.
def cluster_blobs(blobs, max_dist=15):
    """
    Merge blobs that are within max_dist of each other.
    Returns a new list of merged blobs (average position, summed area).
    """
    if not blobs:
        return []
    clusters = []
    used = set()
    for i, b in enumerate(blobs):
        if i in used:
            continue
        cluster = [b]
        used.add(i)
        for j, b2 in enumerate(blobs):
            if j in used:
                continue
            d = math.hypot(b2["cx"] - b["cx"], b2["cy"] - b["cy"])
            if d < max_dist:
                cluster.append(b2)
                used.add(j)
        # Merge cluster
        if len(cluster) == 1:
            clusters.append(cluster[0])
        else:
            cx = np.mean([x["cx"] for x in cluster])
            cy = np.mean([x["cy"] for x in cluster])
            area = sum(x["area"] for x in cluster)
            merged = {"cx": cx, "cy": cy, "area": area}
            if all("x" in x and "y" in x and "w" in x and "h" in x for x in cluster):
                x1 = min(x["x"] for x in cluster)
                y1 = min(x["y"] for x in cluster)
                x2 = max(x["x"] + x["w"] for x in cluster)
                y2 = max(x["y"] + x["h"] for x in cluster)
                merged.update(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
            clusters.append(merged)
    return clusters

# Experimental shadow detector (NOT invoked in the active frame-processing loop).
# Converts the image to HSV; compares brightness and hue to local blurred averages.
# Returns 0/255 mask; dark real ants may satisfy the same tests as shadows.
def build_shadow_mask(frame_bgr, kernel_size=61):

    SHADOW_VAL_DROP = 40

    SHADOW_HUE_STABLE = 20


    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # Local neighbourhood average using a large blur — this is our
    # estimate of what the pixel "should" look like without a shadow
    k = kernel_size | 1   # ensure odd
    v_local_avg = cv2.blur(v.astype(np.float32), (k, k))
    h_local_avg = cv2.blur(h.astype(np.float32), (k, k))

    # Shadow condition: value dropped a lot, but hue stayed similar
    val_drop  = (v_local_avg - v.astype(np.float32)) > SHADOW_VAL_DROP
    hue_stable = np.abs(h.astype(np.float32) - h_local_avg) < SHADOW_HUE_STABLE

    shadow_raw = (val_drop & hue_stable).astype(np.uint8) * 255

    # Clean up speckle
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    shadow_clean = cv2.morphologyEx(shadow_raw, cv2.MORPH_OPEN, k_open)

    return shadow_clean

# Experimental hue-only difference metric; NOT used in the active pipeline.
# Hue is circular (0 and 179 are neighbors). Inputs should be floating-point
# or signed before subtraction to avoid uint8 wraparound.
def compute_hsv_diff(hsv1, hsv2):
    """
    Frame difference using V and S channels weighted by ant/shadow separation.
    Returns a single-channel diff image.
    """
    # hsv1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    # hsv2 = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # # V channel: ants ~200+, shadows ~50 — huge separation, weight heavily
    # v_diff = np.abs(hsv2[:,:,2] - hsv1[:,:,2])

    # # S channel: ants ~115-252, shadows ~100 — moderate separation
    # s_diff = np.abs(hsv2[:,:,1] - hsv1[:,:,1])

    h_diff = np.abs(hsv2[:,:,0] - hsv1[:,:,0])

    h_diff = np.minimum(h_diff, 180-h_diff)

    # H channel: nearly identical for both — don't use it for diff
    # Weighted combination: V dominates
    combined = (h_diff * (255/90)).clip(0,255).astype(np.uint8)
    return combined

# EXPLANATION: Experimental HSV projection coefficients for unused LD1 difference path.
LD1_VEC = np.array([0.00469, -0.01605, -0.01250])


# Experimental per-pixel weighted HSV change score; not currently executed.
# Blur frames, unwrap hue differences, project delta HSV onto LD1_VEC, rescale.
def compute_ld1_diff(frame1_bgr, frame2_bgr):

    f1= cv2.GaussianBlur(frame1_bgr, (3,3), 0)
    f2= cv2.GaussianBlur(frame2_bgr, (3,3), 0)
    hsv1 = cv2.cvtColor(f1, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv2 = cv2.cvtColor(f2, cv2.COLOR_BGR2HSV).astype(np.float32)

    h_diff = hsv2[:,:,0] - hsv1[:,:,0]

    hsv2[:,:,0] = np.where(h_diff < -90, hsv2[:,:,0] + 180, hsv2[:,:,0])
    hsv1[:,:,0] = np.where(h_diff >  90, hsv1[:,:,0] + 180, hsv1[:,:,0])

    diff = np.abs(np.dot(hsv2-hsv1, LD1_VEC))

    return (diff * (255/7.7024)).clip(0,255).astype(np.uint8)

# Optional HSV ant-color threshold; NOT active in the current detector.
# Returns 255 where HSV falls inside the specified range, zero elsewhere.
# The original docstring mentions two frames, but this implementation uses only one.
def build_ant_color_gate(frame1_bgr):
    """Gate passes if either frame looks ant-colored — catches leading edges."""
    hsv1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2HSV)
    # hsv2 = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0,  80, 100], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    gate1 = cv2.inRange(hsv1, lower, upper)
    # gate2 = cv2.inRange(hsv2, lower, upper)
    # return cv2.bitwise_or(gate1, gate2)  # pass if ant-colored in either frame
    return gate1

# Optional post-detection shadow classifier, currently unused.
# Tests fraction of shadow pixels in a square near each blob centroid; partitions
# blobs into ant and shadow candidates; this is NOT an actual blob bounding box.
def separate_shadow_blobs(blobs, shadow_mask, min_shadow_overlap=0.4):
    """
    Given blobs and a shadow mask, return (ant_blobs, shadow_blobs).
    A blob is classified as shadow if >min_shadow_overlap fraction of its
    bounding box pixels are in the shadow mask.
    """
    ant_blobs = []
    shadow_blobs = []
    for b in blobs:
        # Sample shadow mask at blob centroid neighborhood
        cx, cy = int(b["cx"]), int(b["cy"])
        r = max(3, int(math.sqrt(b["area"] / math.pi)))
        y1 = max(0, cy - r)
        y2 = min(shadow_mask.shape[0], cy + r)
        x1 = max(0, cx - r)
        x2 = min(shadow_mask.shape[1], cx + r)
        region = shadow_mask[y1:y2, x1:x2]
        if region.size == 0:
            ant_blobs.append(b)
            continue
        shadow_frac = np.count_nonzero(region) / region.size
        if shadow_frac >= min_shadow_overlap:
            shadow_blobs.append(b)
        else:
            ant_blobs.append(b)
    
    return ant_blobs, shadow_blobs


# ── Processing parameters (tune these if detection is noisy) ─────────────────
# EXPLANATION: Every pixel-distance and area threshold below applies to the resized processing image.
PROCESS_SCALE    = 0.25   # resize factor: 4K→960×540 for processing
DIFF_THRESH      = 15   # motion threshold — higher = fewer blobs (was 12, see notes)
FRAME_STRIDE     = 2      # compare frame t to frame t-STRIDE
MIN_BLOB_AREA    = 10    # min blob area in pixels at PROCESS_SCALE
MAX_BLOB_AREA    = 1200   # max blob area in pixels at PROCESS_SCALE
MORPH_CLOSE_SIZE = 7
MORPH_OPEN_SIZE = 5
CLUSTER_DISTANCE = 15
MAX_TRACK_DIST   = 50     # max pixel distance to link blobs between frames

# How long a lost track stays in memory (no draw, no match, no crossings).
# Short coast only — long persistence let shadows re-attach and fake crossings.
MAX_COAST_FRAMES = 4

# Consecutive frames with a real blob match before crossings are allowed.
MIN_MATCH_STREAK = 3

# Crossing hysteresis: must stay on the same side this many matched frames.
# At 60 fps, 8 frames ≈ 0.13 s.
CROSS_HYSTERESIS_FRAMES = 4

# Camera-vibration guard: if >VIBRATION_PCT % of pixels change simultaneously,
# it's a whole-frame shake → skip detection for that frame (don't count blobs)
VIBRATION_PCT    = 8.0    # percent of frame pixels; raise if too many skips

# Buffer zone around the circle boundary (pixels at PROCESS_SCALE).
# An ant must be this far inside OR outside the boundary before a crossing
# is registered — prevents bouncing counts from ants lingering on the edge.
# At PROCESS_SCALE=0.25 on 4K footage, 10 px here ≈ 40 px in the original.
BUFFER_ZONE      = 10

# ── Circular quadrant detection parameters ─────────────────────────────────
DEFAULT_RADIUS   = 410    # Default circle radius in pixels
_QUADRANTS = ("NE", "SE", "SW", "NW")
QUAD_COLORS = {
    "NE": (0, 255, 255),   # yellow
    "SE": (0, 255, 0),     # green
    "SW": (255, 0, 0),     # blue
    "NW": (255, 0, 255),   # magenta
}
THICKNESS = 4
CENTER_DOT_RADIUS = 5

# ─────────────────────────────────────────────────────────────────────────────
# Circle Parameter File Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

# Constructs the JSON filename used to persist circle calibration for a video.
def _stem_circle_path(out_dir: str, stem: str) -> str:
    return os.path.join(out_dir, f"{stem}_circle.json")


# Validates input JSON and returns center, radius, and north_angle in a
# canonical dictionary. Can derive angle from north_point for old file formats.
def _normalize_circle_params(raw: dict) -> dict | None:
    if not isinstance(raw, dict):
        return None

    params = raw.get("circle_params") if "circle_params" in raw else raw
    if not isinstance(params, dict):
        return None

    center = params.get("center")
    radius = params.get("radius")
    if center is None or radius is None:
        return None

    try:
        cx, cy = float(center[0]), float(center[1])
        radius = float(radius)
    except (TypeError, ValueError, IndexError):
        return None

    north_angle = params.get("north_angle")
    if north_angle is None and "north_point" in params:
        np_ = params["north_point"]
        try:
            nx, ny = float(np_[0]), float(np_[1])
            north_angle = angle_deg_from_center_to_point(cx, cy, nx, ny)
        except (TypeError, ValueError, IndexError):
            return None
    if north_angle is None:
        return None

    return {
        "center": (cx, cy),
        "radius": radius,
        "north_angle": float(north_angle),
    }


# Returns validated circle parameters from a file, or None if invalid/missing.
def _load_circle_json(path: str) -> dict | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return _normalize_circle_params(data)
    except (json.JSONDecodeError, OSError, TypeError):
        return None

# Writes the circle calibration to JSON (creating or replacing the file).
# Returns the path written, or None when no path was given.
def _save_circle_json(circle_params, path: str):
    if not path:
        return None
    params = dict(circle_params)
    if "center" in params:
        params["center"] = [float(params["center"][0]), float(params["center"][1])]
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"circle_params": params}, f, indent=2)
    return path

# Packages video-space center/radius and computes north angle from a clicked point.
def _circle_params_dict(center, radius, north_point) -> dict:
    north_angle = angle_deg_from_center_to_point(
        center[0], center[1], north_point[0], north_point[1])
    return {
        "center": center,
        "radius": radius,
        "north_angle": north_angle,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Circular Quadrant Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

# Returns the image-coordinate angle from center to point, modulo 360.
# Since image y increases downward, angles increase clockwise on screen.
def angle_deg_from_center_to_point(cx, cy, px, py):
    """
    Returns angle in degrees for OpenCV's arc system.
    OpenCV ellipse angles are in degrees, measured clockwise from the +x axis.
    """
    dx = px - cx
    dy = py - cy
    ang = math.degrees(math.atan2(dy, dx))  # in [-180, 180]
    # Convert to [0, 360) and keep as clockwise-from-+x convention
    ang = ang % 360
    return ang


# Maps an outside point to NE, SE, SW or NW by subtracting the user-set
# north direction, then dividing the full turn into four 90-degree sectors.
def get_quadrant(cx, cy, px, py, north_angle_deg):
    """
    Determine which quadrant a point is in relative to center and north direction.
    Returns: "NE", "SE", "SW", "NW"
    Note: This function should only be called for points outside the circle.
    """
    # Calculate angle from center to point
    angle = angle_deg_from_center_to_point(cx, cy, px, py)
    
    # Normalize angle relative to north direction
    relative_angle = (angle - north_angle_deg) % 360
    
    # Determine quadrant based on relative angle
    # NE: 315-45, SE: 45-135, SW: 135-225, NW: 225-315
    # if relative_angle >= 315 or relative_angle < 45:
    #     return "NE"
    # elif relative_angle >= 45 and relative_angle < 135:
    #     return "SE"
    # elif relative_angle >= 135 and relative_angle < 225:
    #     return "SW"
    # else:  # relative_angle >= 225 and relative_angle < 315
    #     return "NW"

    if relative_angle >= 0 and relative_angle < 90:
        return "NE"
    elif relative_angle >= 90 and relative_angle < 180:
        return "SE"
    elif relative_angle >= 180 and relative_angle < 270:
        return "SW"
    else:  # relative_angle >= 270 and relative_angle < 360
        return "NW"


# Tests Euclidean distance from center against radius; includes points on edge.
def is_inside_circle(cx, cy, px, py, radius):
    """Check if point is inside the circle."""
    dist = math.hypot(px - cx, py - cy)
    return dist <= radius


# Draws four colored 90-degree circle sectors and N/E/S/W radial guides.
# This is visualization only; counting happens inside Tracker.update().
def draw_quadrant_arcs(frame, center, radius, north_angle_deg):
    """
    Draw 4 quadrant arcs (NE, SE, SW, NW) based on where North points.
    """
    cx, cy = center
    r = int(round(radius))

    # North is a direction. Define East/South/West as +90/+180/+270 degrees clockwise.
    ang_N = north_angle_deg
    ang_E = ang_N + 90
    ang_S = ang_N + 180
    ang_W = ang_N + 270

    # Helper method norm in the module subsystem.
    def norm(a):
        return a % 360

    ang_N = norm(ang_N)
    ang_E = norm(ang_E)
    ang_S = norm(ang_S)
    ang_W = norm(ang_W)

    # Draw an arc from start to start+90 (clockwise)
    # Helper method draw_arc in the module subsystem.
    def draw_arc(start_deg, color):
        end_deg = (start_deg + 90) % 360
        # cv2.ellipse uses start/end angles in degrees. If end < start, split the arc.
        if end_deg > start_deg:
            cv2.ellipse(frame, (cx, cy), (r, r), 0, start_deg, end_deg, color, THICKNESS)
        else:
            cv2.ellipse(frame, (cx, cy), (r, r), 0, start_deg, 360, color, THICKNESS)
            cv2.ellipse(frame, (cx, cy), (r, r), 0, 0, end_deg, color, THICKNESS)

    # Define quadrants starting at North going clockwise:
    # NE: N -> E, SE: E -> S, SW: S -> W, NW: W -> N
    draw_arc(ang_N, QUAD_COLORS["NE"])
    draw_arc(ang_E, QUAD_COLORS["SE"])
    draw_arc(ang_S, QUAD_COLORS["SW"])
    draw_arc(ang_W, QUAD_COLORS["NW"])

    # Draw axis lines for N, E, S, W so the cardinal directions are explicit.
    axis_color = (255, 255, 255)
    axis_thickness = max(2, THICKNESS - 1)
    for ang in (ang_N, ang_E, ang_S, ang_W):
        rad = math.radians(ang)
        x = int(round(cx + 1.2 * r * math.cos(rad)))
        y = int(round(cy + 1.2 * r * math.sin(rad)))
        cv2.line(frame, (cx, cy), (x, y), axis_color, axis_thickness)

    # Mark center
    cv2.circle(frame, (cx, cy), CENTER_DOT_RADIUS, (255, 255, 255), -1)

# ─────────────────────────────────────────────────────────────────────────────
# Circle Preview Mixin to show circle and quadrants on the video preview
# ─────────────────────────────────────────────────────────────────────────────

# Tkinter preview/circle-editing behavior shared by App through inheritance.
# Uses full-video pixel coordinates for calibration and canvas coordinates for clicks.
class CirclePreviewMixin:
    # Seeks to the chosen video frame, loads it, updates timestamp text, and redraws.
    def _show_frame(self, idx: int):
        if not self.cap:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_bgr = frame
        self._frame_lbl.config(
            text=f"Frame {idx:,} / {self.total_frames - 1:,}  "
                 f"({timedelta(seconds=int(idx / self.fps))})")
        self._render()

    # Scales the current video frame to fit Tkinter canvas while preserving aspect
    # ratio, overlays circle guides, converts to a PhotoImage, and displays it.
    def _render(self):
        if self.current_bgr is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 4 or ch < 4:
            return

        scale = min(cw / self.orig_w, ch / self.orig_h)
        self.disp_scale = scale
        dw = int(self.orig_w * scale)
        dh = int(self.orig_h * scale)
        self.disp_off_x = (cw - dw) // 2
        self.disp_off_y = (ch - dh) // 2

        small = cv2.resize(self.current_bgr, (dw, dh))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        if self.circle_center:
            cx_disp, cy_disp = self._v2c([self.circle_center])[0]
            cx_img = int(cx_disp - self.disp_off_x)
            cy_img = int(cy_disp - self.disp_off_y)
            radius_img = int(self.circle_radius * self.disp_scale)

            if self.north_point and self.circle_complete:
                north_angle = angle_deg_from_center_to_point(
                    self.circle_center[0], self.circle_center[1],
                    self.north_point[0], self.north_point[1])
                draw_quadrant_arcs(rgb, (cx_img, cy_img), radius_img, north_angle)
            else:
                cv2.circle(rgb, (cx_img, cy_img), radius_img, (50, 255, 50), 2)
                cv2.circle(rgb, (cx_img, cy_img), 5, (50, 255, 50), -1)
                if self.north_point:
                    nx_disp, ny_disp = self._v2c([self.north_point])[0]
                    nx_img = int(nx_disp - self.disp_off_x)
                    ny_img = int(ny_disp - self.disp_off_y)
                    cv2.line(rgb, (cx_img, cy_img), (nx_img, ny_img), (255, 255, 0), 2)
                    cv2.circle(rgb, (nx_img, ny_img), 5, (255, 255, 0), -1)

        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.disp_off_x, self.disp_off_y,
            anchor=tk.NW, image=self._photo)

    # Converts original video pixel locations to Tkinter canvas locations.
    def _v2c(self, pts):
        return [(x * self.disp_scale + self.disp_off_x,
                 y * self.disp_scale + self.disp_off_y)
                for x, y in pts]

    # Converts a Tkinter canvas mouse location to original video coordinates.
    def _c2v(self, cx, cy):
        return ((cx - self.disp_off_x) / self.disp_scale,
                (cy - self.disp_off_y) / self.disp_scale)

    # First click selects circle center; second click sets north direction.
    def _click(self, event):
        if not self.cap:
            return
        vx, vy = self._c2v(event.x, event.y)

        if self.circle_center is None:
            self.circle_center = (vx, vy)
            self.circle_radius = self._radius_var.get()
            self._render()
            self._circle_lbl.config(
                text="Center set — click for north direction",
                fg="#664400")
        elif self.north_point is None:
            self.north_point = (vx, vy)
            self.circle_complete = True
            self._render()
            self._circle_lbl.config(
                text="Circle ready with quadrants",
                fg="#006600")
            if hasattr(self, "_refresh_proc_btn"):
                self._refresh_proc_btn()
        else:
            return
        self._notify_circle_edited()

    # Hook for subclasses: called whenever the preview circle is edited by hand
    # (click, clear, radius change) so they can mark it as unsaved.
    def _notify_circle_edited(self):
        hook = getattr(self, "_circle_edited", None)
        if hook is not None:
            hook()

    # No action on double-click when editing circle calibration.
    def _dblclick(self, event):
        """Double-click does nothing in circle mode."""
        pass
    
    # Right-click removes current circle definition.
    def _rclick(self, event):
        self.clear_circle()

    # Resets all circle selections and disables processing until reconfigured.
    def clear_circle(self):
        self.circle_center = None
        self.north_point = None
        self.circle_complete = False
        self._circle_lbl.config(text="No circle defined", fg="#b00")
        if self.current_bgr is not None:
            self._render()
        if hasattr(self, "_refresh_proc_btn"):
            self._refresh_proc_btn()
        self._notify_circle_edited()

    # Updates preview radius when user changes radius control.
    def _on_radius_change(self, _event=None):
        try:
            radius = float(self._radius_var.get())
        except (tk.TclError, ValueError):
            return  # half-typed value in the spinbox; wait for a valid number
        if radius <= 0 or radius == self.circle_radius:
            return
        self.circle_radius = radius
        if self.current_bgr is not None:
            self._render()
        if self.circle_center is not None:
            self._notify_circle_edited()

    # Handles user scrub position and renders selected frame.
    def _on_slider(self, val):
        self._show_frame(int(float(val)))

    # Redraws preview to account for a resized window/canvas.
    def _resize(self, _event):
        if self.current_bgr is not None:
            self._render()

    # Shows a save dialog and writes current circle calibration to JSON.
    def _save_circle(self):
        if not self.circle_complete:
            messagebox.showwarning("Circle", "Define center and north first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save circle parameters",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        params = _circle_params_dict(
            self.circle_center, self.circle_radius, self.north_point)
        _save_circle_json(params, path)
        self._status(f"Saved circle to:\n{path}")

    # Shows a file dialog, validates JSON, then applies stored calibration.
    def _load_circle(self):
        path = filedialog.askopenfilename(
            title="Load circle parameters",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            messagebox.showerror("Load failed", "No file selected.")
            return
        params = _load_circle_json(path)
        if not params:
            messagebox.showerror(
                "Load failed",
                f"Could not read a valid circle from:\n{path}\n\n"
                "Expected JSON with center, radius, and north_angle\n"
                "(or north_point).")
            return
        self._apply_circle_params(params)

    # Restores saved center/radius/north angle; reconstructs a point along
    # north vector for UI display and rerenders the circle.
    def _apply_circle_params(self, params: dict):
        normalized = _normalize_circle_params(
            params if "circle_params" in params else {"circle_params": params})
        if not normalized:
            messagebox.showerror(
                "Invalid circle",
                "Circle JSON must include center, radius, and north_angle\n"
                "(or north_point to derive north).")
            return
        self.circle_center = normalized["center"]
        self.circle_radius = normalized["radius"]
        self._radius_var.set(int(round(self.circle_radius)))
        ang = normalized["north_angle"]
        rad = math.radians(ang)
        cx, cy = self.circle_center
        self.north_point = (
            cx + math.cos(rad) * max(self.circle_radius * 0.5, 50),
            cy + math.sin(rad) * max(self.circle_radius * 0.5, 50),
        )
        self.circle_complete = True
        self._circle_lbl.config(text="Circle loaded", fg="#006600")
        if self.current_bgr is not None:
            self._render()
        if hasattr(self, "_refresh_proc_btn"):
            self._refresh_proc_btn()



# ─────────────────────────────────────────────────────────────────────────────
# Tracker (Kalman filter + Hungarian assignment)
# ─────────────────────────────────────────────────────────────────────────────



# Kalman tuning (constant-velocity model in image pixels at PROCESS_SCALE)
# EXPLANATION: Q diagonal entries are VARIANCES for position-model uncertainty (pixel^2).
KF_PROC_POS = 40.0     # position process noise
# EXPLANATION: Q velocity components let predicted motion adapt when ants turn or accelerate.
KF_PROC_VEL = 8.0      # velocity process noise
# EXPLANATION: R entries are centroid measurement VARIANCES: 4 corresponds to 2px standard deviation.
KF_MEAS_POS = 4.0      # measurement noise (centroid observation)
MIN_TRACK_AGE = 8      # frames before a young track is kept when briefly lost

# ── Visualization-only options (do not affect tracking or counting) ─────────
SHOW_KALMAN_VECTORS = True
SHOW_TRACK_TRAILS = True
VECTOR_FUTURE_FRAMES = 6     # project 6 video frames beyond corrected state
VECTOR_SCALE = 1.0           # visualization scale (1 = physically predicted pixels)
TRACK_TRAIL_LENGTH = 16      # number of observed centroids in the trail
# Require this many successive measured points beyond the destination buffer.
# This is only a confirmation step: a path through BOTH buffer boundaries
# and the true circle is still mandatory for a crossing event.
PATH_CONFIRM_FRAMES = CROSS_HYSTERESIS_FRAMES

# Keep a tracked ant through brief false-negative detections (blinking).
# 2 means the ant may be absent for TWO consecutive frames and still rejoin
# its original Kalman track at the next observation (same ant ID).
TEMPORAL_MAX_GAP = 2

# Wider matching radius when re-acquiring a temporarily hidden ant.
# Avoid making it too large: nearby ants could otherwise exchange IDs.
REACQUIRE_DISTANCE_SCALE = 1.25

# Draw a hollow pink circle at the estimated location of a missing ant.
SHOW_TEMPORARY_TRACKS = True

# OpenCV uses BGR, not RGB.
COLOR_MEASUREMENT = (0, 255, 0)       # green: measured blob centroid
COLOR_FILTERED = (255, 255, 255)      # white: Kalman corrected position
COLOR_PRIOR = (255, 0, 255)           # magenta: prediction BEFORE correction
COLOR_VELOCITY = (255, 255, 0)        # cyan: corrected velocity (future projection)
COLOR_TRAIL = (180, 80, 255)          # bright pink in BGR: previous measured ant path



# Constant-velocity Kalman filter for a single blob track.
# State x=[x,y,vx,vy]; pixel-per-frame velocity, in PROCESS_SCALE coordinates.
# F predicts motion, H maps state to measured centroid, Q is process-noise
# covariance, R is centroid measurement-noise covariance, P uncertainty.
class KalmanFilter2D:
    """Constant-velocity Kalman filter for blob centroids (state: x, y, vx, vy)."""

    # Initializes four-dimensional state, uncertainty P, transition F,
    # measurement H, and noise covariance matrices Q/R.
    def __init__(self, x: float, y: float):
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([500.0, 500.0, 1000.0, 1000.0]).astype(np.float64)
        # EXPLANATION: F maps [x,y,vx,vy] to [x+vx,y+vy,vx,vy] per processed frame.
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        # EXPLANATION: Only x,y are observed; the detector does not directly measure velocities.
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        self.Q = np.diag([KF_PROC_POS, KF_PROC_POS, KF_PROC_VEL, KF_PROC_VEL])
        self.R = np.diag([KF_MEAS_POS, KF_MEAS_POS])
        self._I = np.eye(4, dtype=np.float64)

    # Time update: x <- F x and P <- F P F^T + Q.
    # Returns predicted centroid before incorporating this frame's detection.
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    # Measurement correction using observed centroid z=[zx,zy].
    # Innovation = z-Hx; S = HPH^T+R; gain K=PH^T S^-1;
    # correct x and P; velocity is learned implicitly from positions.
    def update(self, zx, zy):
        z = np.array([zx, zy], dtype=np.float64)
        innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (self._I - K @ self.H) @ self.P


# Solves the global minimum-cost track-to-blob assignment.
# Pads the rectangular cost matrix to a square, runs SciPy assignment,
# and discards padded or too-distant matches.
def _hungarian_match(cost, max_cost):
    """
    Minimum-cost assignment.  Returns (track_row, blob_col) pairs under max_cost.
    """
    n_tracks, n_blobs = cost.shape
    if n_tracks == 0 or n_blobs == 0:
        return []

    pad = np.full((max(n_tracks, n_blobs),) * 2, max_cost * 10 + 1e3)
    pad[:n_tracks, :n_blobs] = np.minimum(cost, max_cost * 10 + 1e2)
    row_ind, col_ind = _hungarian_solve(pad)
    pairs = []
    for r, c in zip(row_ind, col_ind):
        if r < n_tracks and c < n_blobs and cost[r, c] <= max_cost:
            pairs.append((int(r), int(c)))
    return pairs


# Owns all ant IDs, per-ant Kalman filters, detection associations and events.
# On every frame: predict -> Hungarian matching -> update/delete/new tracks
# -> hysteresis-confirmed circle crossings. Input blobs contain cx/cy/area.
class Tracker:
    """
    Multi-object tracker: Kalman-filtered centroids + Hungarian assignment,
    with hysteresis crossing detection for circular quadrants.

    Each frame:
      1. Predict all track positions with Kalman filters.
      2. Build a cost matrix (Euclidean distance, gated by max_dist).
      3. Hungarian algorithm assigns blobs to tracks globally.
      4. Matched tracks get a Kalman measurement update; unmatched tracks coast
         on prediction until max_missing, then are removed.

    Crossing events (enter/exit per quadrant) use the same hysteresis logic as before.
    """

    # Configures distance gate, max-missing age and crossing hysteresis;
    # maintains dictionary ID -> track and next never-reused ID.
    def __init__(self, max_dist: float, max_missing_frames: int,
                 hysteresis_frames: int = 6):
        self.max_dist = max_dist
        self.max_missing = max_missing_frames
        self.hyst = hysteresis_frames
        self._ants: dict = {}
        # id → {cx, cy, missing, committed_inside, committed_quadrant, pending_inside, pending_quadrant, pending_count}
        self._next_id: int = 0

    @staticmethod
    def _path_circle_intersections(previous, current, center, radii):
        """Return ordered segment crossings of concentric circle boundaries.

        We connect two *observed centroids* with a straight-line segment. For
        each requested radius, solve ||previous + t*(current-previous) - center||²=r²
        for t in [0, 1]. A crossing is outward if the segment's radial derivative
        at the intersection is positive, inward if negative.

        A segment may intersect the same circle twice; process both intersections
        in time order. The Kalman-predicted position is NEVER used here.
        """
        p = np.asarray(previous, dtype=np.float64)
        q = np.asarray(current, dtype=np.float64)
        c = np.asarray(center, dtype=np.float64)
        d = q - p
        a = float(np.dot(d, d))
        if a < 1e-12:
            return []
        f = p - c
        b = 2.0 * float(np.dot(f, d))
        intersections = []
        for name, r in radii:
            discriminant = b * b - 4.0 * a * (float(np.dot(f, f)) - r*r)
            if discriminant <= 1e-9:  # Tangential contact is NOT a crossing.
                continue
            root = math.sqrt(discriminant)
            for t in ((-b-root)/(2*a), (-b+root)/(2*a)):
                if not (0.0 < t <= 1.0):
                    continue
                location = p + t*d
                derivative = float(np.dot(location-c, d))
                if abs(derivative) < 1e-9:
                    continue
                intersections.append((t, name, 1 if derivative > 0 else -1,
                                      (float(location[0]), float(location[1]))))
        return sorted(intersections, key=lambda event: event[0])

    @staticmethod
    def _safe_side(point, center, radius):
        """True = inside inner buffer; False = outside outer; None = buffer."""
        distance = math.hypot(point[0]-center[0], point[1]-center[1])
        if distance < radius - BUFFER_ZONE:
            return True
        if distance > radius + BUFFER_ZONE:
            return False
        return None

    def _check_path_crossing(self, ant, center, radius, north_angle):
        """Process actual measured motion and optionally return one crossing event.

        Exit requires an observed continuous path crossing, in order:
            inner radius OUTWARD -> circle OUTWARD -> outer radius OUTWARD.
        Entry requires the reverse inward sequence:
            outer radius INWARD -> circle INWARD -> inner radius INWARD.

        Crossing stages persist across frames, so a slow ant need not travel
        all the way through the annulus within a single frame. Going back to
        its departure-side safe zone cancels the incomplete crossing. A brief
        gap in detections also resets the sequence rather than treating an
        unobserved jump as evidence of travel.
        """
        current = (float(ant["meas_cx"]), float(ant["meas_cy"]))
        previous = ant.get("path_previous")
        ant["path_previous"] = current
        side = self._safe_side(current, center, radius)
        origin = ant.get("path_origin")

        # Initial detection inside the buffer alone cannot establish which
        # direction this ant came from: wait until it is observed on a safe side.
        if origin is None:
            if side is not None:
                ant["path_origin"] = side
                ant["committed_inside"] = side
                ant["committed_quadrant"] = (
                    "CENTER" if side else get_quadrant(*center, *current, north_angle)
                )
            return None

        if previous is None:
            return None

        # Monitor true segment/circle intersections, not merely start/end zones.
        boundaries = (("inner", radius-BUFFER_ZONE),
                      ("circle", radius),
                      ("outer", radius+BUFFER_ZONE))
        if boundaries[0][1] <= 0:
            return None
        expected = (("inner", "circle", "outer") if origin
                    else ("outer", "circle", "inner"))
        direction = 1 if origin else -1

        for _, boundary, motion, position in self._path_circle_intersections(
                previous, current, center, boundaries):
            if motion != direction:
                # Turning around can invalidate an unfinished traversal.
                continue
            stage = ant.get("path_stage", 0)
            if stage < 3 and boundary == expected[stage]:
                ant["path_stage"] = stage + 1
                if boundary == "circle":
                    ant["path_crossing_point"] = position

        # If ant returns to the side where it began, cancel any progress.
        if side is origin:
            ant["path_stage"] = 0
            ant["path_confirm"] = 0
            ant["path_crossing_point"] = None
            return None

        # Reaching the destination zone only counts if the actual observed
        # trajectory crossed all THREE geometric thresholds in order.
        if side is not None and side != origin and ant.get("path_stage", 0) == 3:
            ant["path_confirm"] = ant.get("path_confirm", 0) + 1
        else:
            ant["path_confirm"] = 0

        if (ant.get("path_stage", 0) != 3
                or ant["path_confirm"] < PATH_CONFIRM_FRAMES
                or ant.get("match_streak", 0) < MIN_MATCH_STREAK):
            return None

        # Label the direction using the ACTUAL circle intersection point,
        # not the later position at which hysteresis confirmation completed.
        crossing = ant.get("path_crossing_point") or current
        quadrant = get_quadrant(*center, *crossing, north_angle)
        new_inside = not origin
        event = ("enter_" if new_inside else "exit_") + quadrant
        ant["committed_inside"] = new_inside
        ant["committed_quadrant"] = "CENTER" if new_inside else quadrant
        ant["pending_inside"] = new_inside
        ant["pending_quadrant"] = ant["committed_quadrant"]
        ant["pending_count"] = 0
        ant["path_origin"] = new_inside
        ant["path_stage"] = 0
        ant["path_confirm"] = 0
        ant["path_crossing_point"] = None
        return {"event": event, "cx": crossing[0], "cy": crossing[1],
                "quadrant": quadrant}

    # Mirrors Kalman state into handy scalar fields in the track dictionary.
    def _sync_track_state(self, ant: dict) -> None:
        kf = ant["kf"]
        ant["cx"], ant["cy"] = float(kf.x[0]), float(kf.x[1])
        ant["vx"], ant["vy"] = float(kf.x[2]), float(kf.x[3])
        

    # Initializes a tentative track for an unmatched blob; its initial inside/
    # outside classification does NOT generate a crossing event.
    def _new_track(self, blob: dict, inside: bool, quadrant: str) -> dict:
        kf = KalmanFilter2D(blob["cx"], blob["cy"])
        return {
            "kf": kf,
            "cx": blob["cx"], "cy": blob["cy"],
            "last_cx": blob["cx"], "last_cy": blob["cy"],
            "vx": 0.0, "vy": 0.0,
            "missing": 0,
            "age": 0,
            "committed_inside": inside,
            "committed_quadrant": quadrant,
            "pending_inside": inside,
            "pending_quadrant": quadrant,
            "pending_count": 0,
            "matched_this_frame": True,
            "match_streak": 1,
            "meas_cx": blob["cx"],
            "meas_cy": blob["cy"],
            # Most recent detected box dimensions, used to place a box around
            # the Kalman-predicted center on every frame.
            "box_w": blob.get("w", max(1.0, 2.0 * math.sqrt(blob["area"] / math.pi))),
            "box_h": blob.get("h", max(1.0, 2.0 * math.sqrt(blob["area"] / math.pi))),
            # Diagnostic state only; values below never feed the counting logic.
            "prior_cx": blob["cx"],
            "prior_cy": blob["cy"],
            "trail": [(float(blob["cx"]), float(blob["cy"]))],
            # Path-crossing state; initialized only if starting beyond the buffer.
            "path_origin": None, "path_previous": None,
            "path_stage": 0, "path_confirm": 0,
            "path_crossing_point": None,
            "was_reacquired": False,
        }

    def _coast_missing_track(self, ant: dict) -> None:
        """Leave the Kalman prediction alive when a detection blinks out.

        The Kalman filter was already predicted exactly once in this frame.
        We intentionally DO NOT correct it using a fabricated observation.
        Retain the ant's ID, last REAL centroid, observed trail, and path
        crossing progress. No crossing is confirmed during this missing frame.
        """
        ant["matched_this_frame"] = False
        ant["was_reacquired"] = False
        # Freeze confirmation streaks while the ant is invisible. Predictions
        # cannot satisfy the requirement for observed destination-side frames.
        # The estimate in ant['cx'], ant['cy'] is still the KF prediction.

    # Clears the matched_this_frame flags before assigning current detections.
    def _mark_all_unmatched(self) -> None:
        for ant in self._ants.values():
            ant["matched_this_frame"] = False

    # ------------------------------------------------------------------
    # Main multi-ant matching and crossing method. It returns newly confirmed
    # enter/exit events rather than a count of every visible object.
    # Briefly missing tracks remain eligible for Kalman/Hungarian re-matching.
    # Kalman-only positions do not produce crossing events.
    def update(self, blobs: list, circle_params) -> list:
        """
        Update with blobs from the current frame.

        circle_params: dict with keys 'center', 'radius', 'north_angle'

        Returns list of committed crossing events:
            {"ant_id", "event": "enter_NE"|"exit_NE"|etc, "cx", "cy", "quadrant"}
        """
        if not circle_params:
            return []

        cx, cy = circle_params["center"]
        radius = circle_params["radius"]
        north_angle = circle_params["north_angle"]

        events = []
        all_track_ids = list(self._ants.keys())
        self._mark_all_unmatched()

        # Tracks with 0..TEMPORAL_MAX_GAP missing frames can all be matched.
        # On frame N+1 after disappearance, the KF will predict forward again.
        active_ids = [
            tid for tid in all_track_ids
            if self._ants[tid]["missing"] <= TEMPORAL_MAX_GAP
        ]

        # ── 1. Kalman predict (active tracks only) ────────────────────
        predictions = {}
        for ant_id in active_ids:
            ant = self._ants[ant_id]
            px, py = ant["kf"].predict()
            predictions[ant_id] = (px, py)
            # Save the a-priori Kalman estimate before the measured centroid
            # is incorporated by kf.update(). This is the actual position
            # predicted for the CURRENT frame from the previous frame.
            ant["prior_cx"], ant["prior_cy"] = px, py
            self._sync_track_state(ant)

        # ── 2. Cost matrix + Hungarian match (active tracks only) ─────
        n_tracks = len(active_ids)
        n_blobs = len(blobs)
        matched_track_rows: set[int] = set()
        matched_blob_cols: set[int] = set()

        if n_tracks and n_blobs:
            # EXPLANATION: Rows are tracks, columns are detected blobs; large costs represent disallowed pairs.
            cost = np.full((n_tracks, n_blobs), self.max_dist * 100.0, dtype=np.float64)
            for ri, ant_id in enumerate(active_ids):
                ant = self._ants[ant_id]
                px, py = predictions[ant_id]
                for bi, blob in enumerate(blobs):
                    # Allow a slightly larger search radius for re-acquisition.
                    # Penalize a missed track lightly in the assignment cost,
                    # favoring a continuously visible ant when equally close.
                    allowable = self.max_dist * (
                        REACQUIRE_DISTANCE_SCALE if ant["missing"] > 0 else 1.0
                    )
                    d = math.hypot(blob["cx"] - px, blob["cy"] - py)
                    if d <= allowable:
                        # The Hungarian global gate below is fixed at max_dist,
                        # so scale wider-range costs back into the allowed band.
                        cost[ri, bi] = d / (allowable / self.max_dist) + ant["missing"] * 0.1

            # EXPLANATION: Matched blob updates existing ID and its Kalman filter, rather than spawning a new ant.
            for ri, bi in _hungarian_match(cost, self.max_dist):
                matched_track_rows.add(ri)
                matched_blob_cols.add(bi)
                ant_id = active_ids[ri]
                ant = self._ants[ant_id]
                blob = blobs[bi]
                gap = ant["missing"]
                ant["kf"].update(blob["cx"], blob["cy"])
                ant["missing"] = 0
                ant["was_reacquired"] = gap > 0
                if gap > 0:
                    # Do not create a straight-line crossing across a gap:
                    # motion in those missing frames was not observed.
                    # Keep the partial crossing sequence (path_stage/origin),
                    # but restart segment intersection at the new observation.
                    ant["path_previous"] = None
                ant["age"] += 1
                self._sync_track_state(ant)
                ant["last_cx"] = ant["cx"]
                ant["last_cy"] = ant["cy"]
                ant["meas_cx"] = blob["cx"]
                ant["meas_cy"] = blob["cy"]
                ant["box_w"] = blob.get("w", ant.get("box_w", 1.0))
                ant["box_h"] = blob.get("h", ant.get("box_h", 1.0))
                # Keep a short history of REAL detections, not KF forecasts.
                # Limit size so long videos do not accumulate unbounded memory.
                ant.setdefault("trail", []).append((float(blob["cx"]), float(blob["cy"])))
                if len(ant["trail"]) > TRACK_TRAIL_LENGTH:
                    ant["trail"].pop(0)
                ant["matched_this_frame"] = True
                ant["match_streak"] = ant.get("match_streak", 0) + 1

        matched_ant_ids = {active_ids[ri] for ri in matched_track_rows}

        # ── 3. Unmatched tracks: freeze briefly, then delete ──────────
        for ant_id in all_track_ids:
            if ant_id in matched_ant_ids:
                continue
            ant = self._ants[ant_id]
            ant["missing"] += 1
            # Give both new and established ants a short recovery window.
            # Delete when the permitted gap expires. Resetting the track later
            # will prevent a spurious path continuation across a long absence.
            if ant["missing"] > min(self.max_missing, TEMPORAL_MAX_GAP):
                del self._ants[ant_id]
            else:
                self._coast_missing_track(ant)

        # ── 4. New tracks for unmatched detections ────────────────────
        for bi, blob in enumerate(blobs):
            if bi in matched_blob_cols:
                continue
            inside = is_inside_circle(cx, cy, blob["cx"], blob["cy"], radius)
            quadrant = (
                get_quadrant(cx, cy, blob["cx"], blob["cy"], north_angle)
                if not inside else "CENTER"
            )
            self._ants[self._next_id] = self._new_track(blob, inside, quadrant)
            self._next_id += 1

        # ── Measured-PATH-based boundary crossing ────────────────────────
        # Do not decide IN or OUT from an isolated centroid. For every matched
        # ant, analyze its observed trajectory segment against all 3 radii.
        # A new event requires a complete ordered crossing and confirmation
        # on the far side; Kalman-only predictions never trigger events.
        for ant_id, ant in self._ants.items():
            if not ant.get("matched_this_frame", False) or ant["missing"] > 0:
                continue
            ev = self._check_path_crossing(ant, (cx, cy), radius, north_angle)
            if ev is not None:
                ev["ant_id"] = ant_id
                events.append(ev)

        return events

    # Returns the dictionary of retained tracks, used by drawing code.
    def ants(self) -> dict:
        return self._ants

    # Reads the last hysteresis-confirmed inside/outside state for one ID.
    def committed_inside(self, ant_id) -> bool:
        """Return the committed (hysteresis-filtered) inside state."""
        ant = self._ants.get(ant_id)
        return ant["committed_inside"] if ant else False

    # Reads the last confirmed quadrant label for one ID.
    def get_committed_quadrant(self, ant_id) -> str:
        """Return the committed quadrant."""
        ant = self._ants.get(ant_id)
        return ant["committed_quadrant"] if ant else "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Video Processor (runs in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

def draw_kalman_diagnostics(frame, ant_id, ant):
    """Draw measured position, previous-frame prediction, and future motion.

    Green dot: measured blob center z_t.
    White ring: Kalman posterior (corrected) position x_{t|t}.
    Magenta line: previous-frame prior x_{t|t-1} -> observation z_t,
                  visualizing the prediction residual (innovation).
    Magenta cross: where Kalman predicted the ant BEFORE seeing this frame.
    Cyan arrow: posterior position -> estimated position N frames ahead,
                using x_next = x + N*vx, y_next = y + N*vy.
    Pink polyline: most recent REAL centroid observations.

    Note: future arrow is a constant-velocity extrapolation, NOT a guarantee
    that the ant follows that direction. This function never changes track state.
    """
    if ant["missing"] > 0:
        if SHOW_TEMPORARY_TRACKS:
            # Pink hollow circle: KF estimate of ant location in a short gap.
            predicted = (int(round(ant["cx"])), int(round(ant["cy"])))
            cv2.circle(frame, predicted, 7, COLOR_TRAIL, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{ant_id}?", (predicted[0]+8, predicted[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TRAIL, 1, cv2.LINE_AA)
        return
    if not ant.get("matched_this_frame"):
        return

    measured = (int(round(ant["meas_cx"])), int(round(ant["meas_cy"])))
    posterior = (int(round(ant["cx"])), int(round(ant["cy"])))
    prior = (int(round(ant["prior_cx"])), int(round(ant["prior_cy"])))

    if SHOW_TRACK_TRAILS:
        points = [(int(round(x)), int(round(y))) for x, y in ant.get("trail", [])]
        for start, end in zip(points, points[1:]):
            cv2.line(frame, start, end, COLOR_TRAIL, 2, cv2.LINE_AA)

    if SHOW_KALMAN_VECTORS:
        # Prediction error: prior -> observation, with a cross at prior.
        cv2.line(frame, prior, measured, COLOR_PRIOR, 1, cv2.LINE_AA)
        cv2.drawMarker(frame, prior, COLOR_PRIOR, cv2.MARKER_CROSS, 9, 1)

        # Estimated velocity vector. The KF state uses pixels/frame, so N
        # times that velocity estimates N frames of displacement.
        vx, vy = float(ant["vx"]), float(ant["vy"])
        future = (
            int(round(ant["cx"] + VECTOR_SCALE * VECTOR_FUTURE_FRAMES * vx)),
            int(round(ant["cy"] + VECTOR_SCALE * VECTOR_FUTURE_FRAMES * vy)),
        )
        if future != posterior:
            cv2.arrowedLine(frame, posterior, future, COLOR_VELOCITY,
                            2, cv2.LINE_AA, tipLength=0.22)

    # Draw after the vectors so the measured and corrected points stay visible.
    cv2.circle(frame, measured, 3, COLOR_MEASUREMENT, -1, cv2.LINE_AA)
    cv2.circle(frame, posterior, 5, COLOR_FILTERED, 1, cv2.LINE_AA)
    cv2.putText(frame, str(ant_id), (posterior[0] + 7, posterior[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_FILTERED, 1, cv2.LINE_AA)


# Runs video frame processing, foreground extraction, tracking and output writing.
# The active detector uses grayscale frame DIFFERENCING, not MOG2 subtraction.
class VideoProcessor:
    # Stores input/output paths, circle parameters, callbacks and cancel flag.
    def __init__(self, video_path, 
                 circle_params,
                 progress_cb, status_cb,
                 out_dir: str | None = None,
                 stem: str | None = None,
                 render_video: bool = True,
                 cancel_event: threading.Event | None = None):
        self.render_video = render_video
        # Optional shared Event (batch runs): setting it cancels this video too.
        self._cancel_event = cancel_event
        self.video_path = video_path # full path to video for processing
        if not out_dir:
            self.out_dir = os.path.join(os.path.abspath(os.path.curdir),"outputs")
        else:
            self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        if not stem:
            self.stem = os.path.splitext(os.path.basename(self.video_path))[0] # name of video without extension
        else:
            self.stem = stem
        self.circle_params = circle_params  # dict with center, radius, north_angle
        self.first_frame = None
        self.progress_cb = progress_cb
        self.status_cb = status_cb
        self._cancel = False

    # Requests cooperative cancellation; next iteration of frame loop will stop.
    def cancel(self):
        self._cancel = True

    def _cancelled(self) -> bool:
        return self._cancel or bool(self._cancel_event and self._cancel_event.is_set())
    
    # ------------------------------------------------------------------
    # Reads video metadata, scales circle coordinates, writes calibration JSON,
    # processes video once, then aggregates entry/exit counts by quadrant.
    def run(self):
        """Main pipeline.  
        Returns (events, enter_totals, exit_totals, paths_dict) 
        or (None, None, None, None)."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Cannot open video: {self.video_path}")
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, self.first_frame = cap.read()
        if not ret:
            self.first_frame = None
        cap.release()

        proc_w = int(orig_w * PROCESS_SCALE)
        proc_h = int(orig_h * PROCESS_SCALE)

        # Scale circle parameters to process resolution
        circle_proc = {
            'center': (self.circle_params['center'][0] * PROCESS_SCALE,
                      self.circle_params['center'][1] * PROCESS_SCALE),
            'radius': self.circle_params['radius'] * PROCESS_SCALE,
            'north_angle': self.circle_params['north_angle']
        }

        # Save circle parameters
        circle_path = _stem_circle_path(self.out_dir, self.stem)
        with open(circle_path, "w") as f:
            json.dump({
                "video": self.video_path,
                "circle_params": self.circle_params,
                "orig_width": orig_w, "orig_height": orig_h,
                "process_scale": PROCESS_SCALE,
                "detection": "frame_diff_quadrants",
                "frame_stride": FRAME_STRIDE,
                "diff_thresh": DIFF_THRESH,
            }, f, indent=2)

        # Single-pass: frame-to-frame diff (no background estimation needed)
        csv_path = os.path.join(self.out_dir, f"{self.stem}_counts.csv")
        bbox_csv_path = os.path.join(self.out_dir, f"{self.stem}_ant_boxes.csv")
        vid_path = os.path.join(self.out_dir, f"{self.stem}_counted.mp4")
        summary_path = os.path.join(self.out_dir, f"{self.stem}_summary.csv")
        self.progress_cb(0, "Processing frames…")
        events = self._process_frames(
            circle_proc, proc_w, proc_h, fps, total, 
            csv_path, bbox_csv_path, vid_path, summary_path)
        
        # Calculate summary counts per quadrant
        enter_totals = {q: 0 for q in _QUADRANTS}
        exit_totals = {q: 0 for q in _QUADRANTS}
        for e in events:
            quad = e["quadrant"]
            if e["event"].startswith("enter_"):
                enter_totals[quad] += 1
            elif e["event"].startswith("exit_"):
                exit_totals[quad] += 1

        if self._cancelled():
            return None, None, None, None

        # Write the previously missing per-video summary. This is the exact
        # quadrant,enters,exits format expected by regression_analysis.py.
        # The TOTAL row is for human review; regression intentionally skips it.
        with open(summary_path, 'w', newline='', encoding='utf-8') as handle:
            writer_csv = csv.writer(handle)
            writer_csv.writerow(['quadrant', 'enters', 'exits'])
            for quad in _QUADRANTS:
                writer_csv.writerow([quad, enter_totals[quad], exit_totals[quad]])
            writer_csv.writerow(['TOTAL', sum(enter_totals.values()),
                                 sum(exit_totals.values())])

        return events, enter_totals, exit_totals, {
            "csv": csv_path, 
            "ant_boxes_csv": bbox_csv_path,
            "video": vid_path if self.render_video else None,
            "circle": circle_path, 
            "summary": summary_path,
            "output_dir": self.out_dir,
        }

    # ------------------------------------------------------------------
    # Main per-frame pipeline. Resize -> grayscale -> frame differencing ->
    # binary threshold -> morphological cleanup -> connected components ->
    # blob clustering -> multi-ant tracking -> event CSV and rendered MP4.
    # Coordinates here are reduced to PROCESS_SCALE of original video.
    def _process_frames(self, circle_proc, proc_w, proc_h,
                        fps, total, csv_path, bbox_csv_path, vid_path, summary_path):
        tracker = Tracker(
            max_dist=MAX_TRACK_DIST,
            max_missing_frames=MAX_COAST_FRAMES,
            hysteresis_frames=CROSS_HYSTERESIS_FRAMES,
        )
        #total = min(total, int(fps * 60))   # first 60 seconds only
        
        # Extract circle parameters
        cx, cy = circle_proc['center']
        radius = circle_proc['radius']
        north_angle = circle_proc['north_angle']
        circle_int = (int(cx), int(cy), int(radius))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = (cv2.VideoWriter(vid_path, fourcc, fps, (proc_w, proc_h), True)
                  if self.render_video else None)

        # 7×7 close merges nearby motion pixels into one blob;
        # 5×5 open removes isolated speckles smaller than an ant.
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_SIZE, MORPH_OPEN_SIZE))
        frame_pixels = proc_w * proc_h
        vibration_limit = int(frame_pixels * VIBRATION_PCT / 100)

        all_events = []
        # Track entries and exits per quadrant to include in the CSV and summary.
        enter_counts = {q: 0 for q in _QUADRANTS}
        exit_counts = {q: 0 for q in _QUADRANTS}
        n_vibration_skips = 0

        # Ring buffer: diff frame t against frame t-FRAME_STRIDE.
        frame_buf: list = []

        cap = cv2.VideoCapture(self.video_path)
        csv_rows = [["timestamp_s", "time", "frame", "event", "quadrant",
                     "enter_count", "exit_count", "ant_id", "x_orig", "y_orig"]]
        bbox_rows = [["frame", "timestamp_s", "ant_id", "x_min", "y_min",
                      "x_max", "y_max", "status"]]

        for fid in range(total):
            if self._cancelled():
                break
            ret, frame = cap.read()
            if not ret:
                break

            # ── resize + grayscale ────────────────────────────────────
            small = cv2.resize(frame, (proc_w, proc_h))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            # gray[gray < 100] = 0
            # hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            # gray = hsv[:,:,2]

            # ── create shadow mask ───────────────────────────────────────
            shadow_mask = gray < 110

            # ── Frame-to-frame diff ───────────────────────────────────────
            # EXPLANATION: Buffers grayscale frames. Existing inline comment saying BGR is outdated.
            frame_buf.append(gray.copy())   # store full BGR, not gray
            if len(frame_buf) > FRAME_STRIDE + 1:
                frame_buf.pop(0)

            if len(frame_buf) > FRAME_STRIDE:
                # diff = compute_ld1_diff(frame_buf[0], small)
                # diff = compute_hsv_diff(frame_buf[0], small)
                # EXPLANATION: Difference is absolute intensity change vs a prior frame, not learned-background subtraction.
                diff = cv2.absdiff(gray, frame_buf[0])
                diff = cv2.GaussianBlur(diff, (3, 3), 0)
            else:
                diff = np.zeros((proc_h, proc_w), dtype=np.uint8)

            # ── Color gate: only keep motion where pixels look like ants ─────
            # ant_gate = build_ant_color_gate(frame_buf[0],small)

            # Apply gate to diff — shadow motion gets zeroed out
            # EXPLANATION: Threshold turns continuous motion magnitude into binary foreground pixels.
            _, mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
            # EXPLANATION: NOTE: mask was computed ABOVE. This modifies diff visualization, NOT the mask used for detections. Existing behavior preserved.
            diff[shadow_mask] = 0
            # mask = cv2.bitwise_and(mask, ant_gate)
            # EXPLANATION: Closing dilates then erodes to fill small gaps and join nearby motion pixels.
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
            # EXPLANATION: Opening erodes then dilates, removing small isolated foreground speckles.
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)

            # ── vibration guard ───────────────────────────────────────
            # If the whole frame moved (wind shake), skip blob detection
            # this frame so we don't inject hundreds of false blobs.
            n_fg = int(np.count_nonzero(mask))
            is_vibration = n_fg > vibration_limit
            if is_vibration:
                n_vibration_skips += 1
                blobs = []
            else:
                # ── blob extraction ───────────────────────────────────
                # EXPLANATION: Label 0 is background; stats gives bounding boxes/pixel area; centroids gives region centers.
                n_lbl, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
                blobs = []
                for i in range(1, n_lbl):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
                        blobs.append({"cx": centroids[i][0],
                                      "cy": centroids[i][1],
                                      "area": int(area),
                                      "x": int(stats[i, cv2.CC_STAT_LEFT]),
                                      "y": int(stats[i, cv2.CC_STAT_TOP]),
                                      "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                                      "h": int(stats[i, cv2.CC_STAT_HEIGHT])})
                # ant_blobs, shadow_blobs = separate_shadow_blobs(blobs, shadow_mask)
                # EXPLANATION: Merge close components to reduce split detections; might accidentally merge distinct nearby ants.
                blobs = cluster_blobs(blobs, max_dist=CLUSTER_DISTANCE)

            # ── tracking & crossing detection ─────────────────────────
            # EXPLANATION: Tracker maintains IDs across frames and emits only confirmed boundary transitions.
            events = tracker.update(blobs, circle_proc)
            ts  = fid / fps
            ts_str = str(timedelta(seconds=int(ts)))

            # One row per retained ant and frame. Matched rows use the current
            # detection center; missing rows use the Kalman predicted center.
            for ant_id, ant in tracker.ants().items():
                box_w = float(ant.get("box_w", 1.0))
                box_h = float(ant.get("box_h", 1.0))
                x_min = (ant["cx"] - box_w / 2) / PROCESS_SCALE
                y_min = (ant["cy"] - box_h / 2) / PROCESS_SCALE
                x_max = (ant["cx"] + box_w / 2) / PROCESS_SCALE
                y_max = (ant["cy"] + box_h / 2) / PROCESS_SCALE
                bbox_rows.append([
                    fid, f"{ts:.3f}", ant_id,
                    f"{x_min:.1f}", f"{y_min:.1f}",
                    f"{x_max:.1f}", f"{y_max:.1f}",
                    "detected" if ant["matched_this_frame"] else "predicted",
                ])

            for ev in events:
                quadrant = ev["quadrant"]
                if ev["event"].startswith("enter_"):
                    enter_counts[quadrant] += 1
                elif ev["event"].startswith("exit_"):
                    exit_counts[quadrant] += 1
                    
                rec = {
                    "timestamp_s": ts, "time": ts_str, "frame": fid,
                    "event": ev["event"], "quadrant": quadrant,
                    "enter_count": enter_counts[quadrant],
                    "exit_count": exit_counts[quadrant],
                    "ant_id": ev["ant_id"],
                    "x_orig": ev["cx"] / PROCESS_SCALE,
                    "y_orig": ev["cy"] / PROCESS_SCALE,
                }
                all_events.append(rec)
                csv_rows.append([
                    f"{ts:.3f}", ts_str, fid, ev["event"], quadrant,
                    enter_counts[quadrant], exit_counts[quadrant], ev["ant_id"],
                    f"{rec['x_orig']:.1f}", f"{rec['y_orig']:.1f}",
                ])

            if self.render_video:
                # ── render annotated frame ────────────────────────────────
                out = small.copy()

                # motion diff inset (top-right corner, 1/4 size) – helps verify
                # what the detector sees while watching the output video
                inset_w = proc_w // 4
                inset_h = proc_h // 4
                diff_amp = np.clip(diff.astype(np.float32) * 5, 0, 255).astype(np.uint8)
                diff_color = cv2.applyColorMap(diff_amp, cv2.COLORMAP_HOT)
                inset = cv2.resize(diff_color, (inset_w, inset_h))
                ix = proc_w - inset_w - 4
                iy = 4
                out[iy:iy+inset_h, ix:ix+inset_w] = inset
                cv2.rectangle(out, (ix-1, iy-1), (ix+inset_w, iy+inset_h), (180,180,180), 1)

                # Circle overlay with quadrants plus BOTH safe-zone boundaries.
                # The highlighted ring is the ambiguity/buffer zone. A verified
                # crossing must travel through inner -> circle -> outer (exit),
                # or outer -> circle -> inner (entry) using observed ant positions.
                draw_quadrant_arcs(out, (int(cx), int(cy)), int(radius), north_angle)
                if radius - BUFFER_ZONE > 0:
                    cv2.circle(out, (int(cx), int(cy)),
                               int(round(radius - BUFFER_ZONE)),
                               (150, 150, 150), 1, cv2.LINE_AA)
                cv2.circle(out, (int(cx), int(cy)),
                           int(round(radius + BUFFER_ZONE)),
                           (150, 150, 150), 1, cv2.LINE_AA)

                # ant blobs — only draw actively detected tracks (not coasting ghosts)
                for ant_id, ant in tracker.ants().items():
                    if ant["missing"] > 0:
                        # Still render the last observed path and the predicted
                        # location without treating the ant as actually detected.
                        if SHOW_TRACK_TRAILS:
                            pts = [(int(x), int(y)) for x, y in ant.get("trail", [])]
                            for a, b in zip(pts, pts[1:]):
                                cv2.line(out, a, b, COLOR_TRAIL, 2, cv2.LINE_AA)
                        draw_kalman_diagnostics(out, ant_id, ant)
                        continue
                    ax, ay = int(ant["cx"]), int(ant["cy"])
                    committed_inside = ant["committed_inside"]
                    committed_quad = ant["committed_quadrant"]
                    pending_inside = ant["pending_inside"]
                    pending_quad = ant["pending_quadrant"]
                    in_hyst = (committed_inside != pending_inside) or (committed_quad != pending_quad)
                
                    if in_hyst:
                        color = (0, 220, 255)   # yellow-ish: pending crossing
                    elif committed_inside:
                        color = (255, 255, 255)   # white: inside center
                    else:
                        # Orange for all outside ants regardless of quadrant
                        color = (0, 165, 255)    # orange: outside ants
                    
                    cv2.circle(out, (ax, ay), 6, color, 2)
                    cv2.circle(out, (ax, ay), 2, color, -1)

                    # Overlay the Kalman forecast, velocity direction, raw blob
                    # centroid and previous detected positions for this ant.
                    draw_kalman_diagnostics(out, ant_id, ant)

              
                


                # quadrant count box (top-left)
                cv2.rectangle(out, (8, 8), (340, 120), (0, 0, 0), -1)
                y_offset = 25
                for i, quad in enumerate(_QUADRANTS):
                    color = QUAD_COLORS[quad]
                    label = (f"{quad}  in:{enter_counts[quad]}  "
                             f"out:{exit_counts[quad]}")
                    cv2.putText(out, label, (12, y_offset + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

                # recent event flash (bottom-left, fades after 1 s)
                if all_events and fid - all_events[-1]["frame"] < fps:
                    last = all_events[-1]
                    event_name = last["event"]
                    quadrant = last["quadrant"]
                    if event_name.startswith("enter_"):
                        ev_txt = f"ENTER {quadrant} (#{last['enter_count']})"
                        ev_col = (0, 255, 90)
                    elif event_name.startswith("exit_"):
                        ev_txt = f"EXIT {quadrant} (#{last['exit_count']})"
                        ev_col = (50, 50, 255)
                    else:
                        ev_txt = event_name
                        ev_col = (255, 255, 255)
                    cv2.rectangle(out, (6, proc_h - 45), (300, proc_h - 8), (0, 0, 0), -1)
                    cv2.putText(out, ev_txt, (10, proc_h - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, ev_col, 2, cv2.LINE_AA)

                # timestamp (below count box)
                cv2.putText(out, ts_str, (12, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1, cv2.LINE_AA)

                # EXPLANATION: Annotated frame is written to MP4 at processing resolution.
                # Short legend clarifies the colored visualizations.
                if SHOW_KALMAN_VECTORS:
                    legend = "green:obs  pink:prior  cyan:future  white:KF"
                    cv2.putText(out, legend, (8, proc_h - 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                                (220, 220, 220), 1, cv2.LINE_AA)

                writer.write(out)

            if fid % 200 == 0:
                pct = 100 * fid / total
                self.progress_cb(
                    pct,
                    f"Frame {fid:,}/{total:,}  blobs:{len(blobs):3d}  "
                    f"events:{len(all_events)}  shakes:{n_vibration_skips}")

        cap.release()
        if writer is not None:
            writer.release()

        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerows(csv_rows)

        with open(bbox_csv_path, "w", newline="") as f:
            csv.writer(f).writerows(bbox_rows)

        return all_events

    # Optional diagnostic that measures mean grayscale brightness along each
    # quadrant arc of first frame; not currently part of detector.
    def calculate_quadrant_arc_pixel_values(self):
        """
        Calculate the average pixel values for each quadrant arc of the circle.
        Returns a dict: { "NE": mean_grayscale, ... }
        """
        if not self.circle_params or self.first_frame is None:
            return {}

        cx, cy = self.circle_params["center"]
        radius = self.circle_params["radius"]
        north_angle = self.circle_params["north_angle"]

        # Create a template for the masks with the same size as the first frame
        mask = np.zeros(self.first_frame.shape[:2], dtype=np.uint8)

        # Create masks for each quadrant arc
        quadrant_masks = {}
        start_angle = 0
        for quad in _QUADRANTS:
            quad_mask = mask.copy()
            cv2.ellipse(quad_mask, (int(cx), int(cy)), (int(radius), int(radius)), north_angle, start_angle, start_angle + 90, 255, 6)
            quadrant_masks[quad] = quad_mask
            start_angle += 90

        # Calculate mean pixel values for each quadrant arc
        quadrant_values = {}
        for quad, q_mask in quadrant_masks.items():
            gray_frame = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2GRAY)
            mean_val = cv2.mean(gray_frame, mask=q_mask)[0]  # mean returns a tuple, take the greyscale element
            quadrant_values[quad] = mean_val

        return quadrant_values


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────

# Tkinter GUI with everything --run has, plus a video list built from any
# folders and drag & drop: add video files (from as many folders as you like)
# or a whole folder (optionally including subfolders), parameter presets,
# circles (clicked in the GUI, or found as <name>_circle.json files),
# output folder, render/strict options and a
# copyable equivalent command. Processing goes through plan_batch() and
# execute_batch(), exactly the functions the CLI uses.
class App(CirclePreviewMixin):
    # Table colours per row state.
    ROW_COLORS = {
        "ready": "#1b5e20", "missing": "#b00020", "invalid": "#b00020",
        "running": "#0d47a1", "processed": "#0d47a1", "failed": "#b00020",
        "skipped": "#757575", "cancelled": "#757575",
    }
    # Circle sources shown in the table for circles clicked in the GUI.
    CLICKED = "clicked in GUI"
    CLICKED_UNSAVED = "clicked in GUI (not saved)"

    def __init__(self, prefill: dict | None = None):
        _require_gui_dependencies()
        # Drag & drop needs a TkinterDnD root; fall back to plain Tk if the
        # tkdnd library cannot load on this machine.
        self._dnd = False
        root = None
        if TkinterDnD is not None:
            try:
                root = TkinterDnD.Tk()
                self._dnd = True
            except Exception:
                root = None
        self.root = root or tk.Tk()
        self.root.title("Ant Hole Counter")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 640)

        # ── preview state (used by CirclePreviewMixin), VIDEO pixel coords ──
        self.video_path: str | None = None      # video shown in the preview
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.orig_w = self.orig_h = 0
        self.current_bgr = None
        self.circle_center: tuple = None
        self.circle_radius: float = float(DEFAULT_RADIUS)
        self.north_point: tuple = None
        self.circle_complete: bool = False
        self.disp_off_x = self.disp_off_y = 0
        self.disp_scale = 1.0
        self._photo = None

        # ── run inputs (mirror the --run options) ──
        self.video_files: list[str] = []        # --video (repeatable), any folders
        self.folder: str | None = None          # --videos
        self.out_dir: str | None = None         # --output
        self.parameters_file: str | None = None  # --parameters
        self.parameter_values: dict = dict(DEFAULT_PARAMETERS)
        # Circles clicked for individual videos in "Click it manually" mode:
        # resolved video path -> (circle, saved file path or None=unsaved).
        self.video_circles: dict[str, tuple[dict, str | None]] = {}
        # File the preview circle came from; None = drawn/edited, not saved.
        self.circle_source_path: str | None = None
        self._suspend_circle_hook = False
        self._render_touched = False
        # GitHub version recorded in run metadata; asked before each run.
        self.code_version: str | None = None

        # ── batch state ──
        self.plan: BatchPlan | None = None
        self._plan_error = ""
        self._iid_by_item: dict[int, str] = {}
        self._run_order: dict[int, tuple[int, int]] = {}
        self._busy = False
        self._cancel_event: threading.Event | None = None

        self._build_ui()
        self.regression = RegressionTab(self._reg_tab, self)
        self._register_drop_targets()
        self.regression.register_drop_targets(self._dnd)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        prefill = prefill or {}
        self.code_version = prefill.get("code_version") or None
        self._apply_prefill(prefill)
        self._on_input_mode()
        if prefill.get("regression"):
            self.regression.apply_prefill(prefill["regression"])
            self.notebook.select(self._reg_tab)

    # ══════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        self.input_mode = tk.StringVar(value="files")         # files | folder
        self.recursive_var = tk.BooleanVar(value=False)
        self.strict_var = tk.BooleanVar(value=False)
        self.render_var = tk.BooleanVar(value=False)
        # files: find <name>_circle.json in the input/output folders (as --run)
        # manual: every video's circle is clicked in the GUI
        self.circle_mode = tk.StringVar(value="files")

        # Two tabs: counting (this class) and regression (RegressionTab).
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        count_tab = tk.Frame(self.notebook)
        self._reg_tab = tk.Frame(self.notebook)
        self.notebook.add(count_tab, text="  Count ants  ")
        self.notebook.add(self._reg_tab, text="  Regression  ")
        pane = tk.PanedWindow(count_tab, orient=tk.HORIZONTAL,
                              sashwidth=6, sashrelief=tk.RAISED, bg="#333")
        pane.pack(fill=tk.BOTH, expand=True)
        side = self._scrollable_sidebar(pane)

        tk.Label(side, text="🐜  Ant Hole Counter",
                 font=("Arial", 15, "bold"), bg="#f4f4f4").pack(pady=(2, 8))

        # ── Input ──────────────────────────────────────────────────────
        box = self._panel(side, "Input")
        row = tk.Frame(box)
        row.pack(fill=tk.X, padx=4, pady=2)
        ttk.Radiobutton(row, text="Videos (any folders)", value="files",
                        variable=self.input_mode,
                        command=self._on_input_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(row, text="Folder", value="folder",
                        variable=self.input_mode,
                        command=self._on_input_mode).pack(side=tk.LEFT, padx=10)
        holder = tk.Frame(box)
        holder.pack(fill=tk.X)
        self._files_box = tk.Frame(holder)
        row = tk.Frame(self._files_box)
        row.pack(fill=tk.X, padx=4, pady=1)
        tk.Button(row, text="Add videos…", command=self._add_videos_dialog).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(row, text="Add a folder's videos…",
                  command=self._add_folder_dialog).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        row = tk.Frame(self._files_box)
        row.pack(fill=tk.X, padx=4, pady=1)
        tk.Button(row, text="Remove selected", command=self._remove_selected).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(row, text="Clear list", command=self._clear_list).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        self._folder_box = tk.Frame(holder)
        tk.Button(self._folder_box, text="Choose folder…", relief=tk.GROOVE,
                  command=self._choose_folder_dialog).pack(fill=tk.X, padx=4, pady=2)
        self._recursive_chk = ttk.Checkbutton(
            box, text="Include subfolders (--recursive)",
            variable=self.recursive_var, command=self._on_recursive)
        self._recursive_chk.pack(anchor=tk.W, padx=4, pady=(2, 2))
        self._drop_lbl = tk.Label(box, text="", relief=tk.GROOVE, bd=2, height=2,
                                  font=("Arial", 9), fg="#555", bg="#fafafa",
                                  wraplength=260)
        self._drop_lbl.pack(fill=tk.X, padx=4, pady=3)
        self._input_lbl = self._note(box, "")
        self._info_lbl = tk.Label(box, text="", font=("Courier", 8),
                                  justify=tk.LEFT, fg="#444")
        self._info_lbl.pack(anchor=tk.W, padx=4)

        # ── Parameters ─────────────────────────────────────────────────
        box = self._panel(side, "Parameters")
        self._params_lbl = self._note(box, "Defaults (built-in values)")
        row = tk.Frame(box)
        row.pack(fill=tk.X, padx=4, pady=(0, 4))
        tk.Button(row, text="Load…", command=self._load_parameters).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(row, text="Reset", command=self._reset_parameters).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(row, text="View values", command=self._view_parameters).pack(
            side=tk.LEFT, expand=True, fill=tk.X)

        # ── Frame scrub ────────────────────────────────────────────────
        box = self._panel(side, "Preview frame")
        self._frame_var = tk.IntVar()
        self._slider = ttk.Scale(box, from_=0, to=100, variable=self._frame_var,
                                 orient=tk.HORIZONTAL, command=self._on_slider)
        self._slider.pack(fill=tk.X, padx=4, pady=2)
        self._frame_lbl = tk.Label(box, text="—", font=("Arial", 8))
        self._frame_lbl.pack()

        # ── Circle ─────────────────────────────────────────────────────
        box = self._panel(side, "Circle")
        tk.Label(box, text=("1st click: center   2nd click: north\n"
                            "Right-click: clear"),
                 font=("Arial", 8), justify=tk.LEFT).pack(anchor=tk.W, padx=4)
        self._circle_lbl = tk.Label(box, text="No circle defined", fg="#b00",
                                    font=("Arial", 9, "bold"), wraplength=270,
                                    justify=tk.LEFT)
        self._circle_lbl.pack(anchor=tk.W, padx=4, pady=2)
        row = tk.Frame(box)
        row.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(row, text="Radius (px):", font=("Arial", 8)).pack(side=tk.LEFT)
        self._radius_var = tk.IntVar(value=DEFAULT_RADIUS)
        spin = tk.Spinbox(row, from_=10, to=5000, textvariable=self._radius_var,
                          width=7, command=self._on_radius_change)
        spin.pack(side=tk.LEFT, padx=(5, 0))
        spin.bind("<Return>", self._on_radius_change)
        spin.bind("<FocusOut>", self._on_radius_change)
        row = tk.Frame(box)
        row.pack(fill=tk.X, padx=4, pady=2)
        tk.Button(row, text="Save circle…", command=self._save_circle).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(row, text="Clear", command=self.clear_circle).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        ttk.Separator(box).pack(fill=tk.X, padx=4, pady=4)
        tk.Label(box, text="Where each video's circle comes from:",
                 font=("Arial", 8, "bold")).pack(anchor=tk.W, padx=4)
        ttk.Radiobutton(box, text="Find <name>_circle.json in the input and\n"
                                  "output folders (same as the CLI)",
                        value="files", variable=self.circle_mode,
                        command=self._on_circle_mode).pack(anchor=tk.W, padx=4)
        ttk.Radiobutton(box, text="Click it manually in the GUI\n(each video separately)",
                        value="manual", variable=self.circle_mode,
                        command=self._on_circle_mode).pack(anchor=tk.W, padx=4)
        self._circle_mode_lbl = self._note(
            box, "Looks next to each video first, then in the output folder.")

        # ── Output ─────────────────────────────────────────────────────
        box = self._panel(side, "Output")
        self._out_dir_lbl = self._note(box, "")
        row = tk.Frame(box)
        row.pack(fill=tk.X, padx=4, pady=(0, 4))
        tk.Button(row, text="Select output folder…",
                  command=self._select_out_dir).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(row, text="Use default",
                  command=self._clear_out_dir).pack(side=tk.LEFT, padx=(2, 0))

        # ── Options ────────────────────────────────────────────────────
        box = self._panel(side, "Options")
        ttk.Checkbutton(box, text="Render annotated MP4 (--render-video)",
                        variable=self.render_var,
                        command=self._on_render_toggle).pack(anchor=tk.W, padx=4)
        ttk.Checkbutton(box, text="Strict (--strict): no skipping; stop on errors",
                        variable=self.strict_var,
                        command=self._refresh_proc_btn).pack(anchor=tk.W, padx=4)

        # ── Process ────────────────────────────────────────────────────
        box = self._panel(side, "Process")
        self._proc_btn = tk.Button(
            box, text="▶  Process", font=("Arial", 11, "bold"),
            bg="#2e7d32", fg="white", state=tk.DISABLED,
            activebackground="#1b5e20", activeforeground="white",
            command=self.start_processing)
        self._proc_btn.pack(fill=tk.X, padx=4, pady=(4, 0))
        self._proc_hint = tk.Label(box, text="", font=("Arial", 8), fg="#8a5a00",
                                   wraplength=270, justify=tk.LEFT)
        self._proc_hint.pack(anchor=tk.W, padx=4)
        self._cancel_btn = tk.Button(box, text="Cancel", state=tk.DISABLED,
                                     command=self.cancel_processing)
        self._cancel_btn.pack(fill=tk.X, padx=4, pady=(2, 4))
        self._overall_lbl = tk.Label(box, text="", font=("Arial", 9, "bold"),
                                     wraplength=270, justify=tk.LEFT)
        self._overall_lbl.pack(anchor=tk.W, padx=4)
        self._prog_var = tk.DoubleVar()
        ttk.Progressbar(box, variable=self._prog_var, maximum=100).pack(
            fill=tk.X, padx=4, pady=2)
        self._prog_lbl = tk.Label(box, text="", font=("Arial", 8),
                                  wraplength=270, justify=tk.LEFT)
        self._prog_lbl.pack(anchor=tk.W, padx=4)
        tk.Button(box, text="Copy CLI command",
                  command=self._copy_cli_command).pack(fill=tk.X, padx=4, pady=4)

        self._status_lbl = tk.Label(side, text="Add videos or choose a folder to get started.",
                                    wraplength=280, justify=tk.LEFT,
                                    font=("Arial", 9), fg="#333", bg="#f4f4f4")
        self._status_lbl.pack(pady=6, anchor=tk.W)

        # ── right side: preview canvas over the video table ────────────
        right = tk.PanedWindow(pane, orient=tk.VERTICAL, sashwidth=6,
                               sashrelief=tk.RAISED, bg="#333")
        pane.add(right, minsize=500)
        canvas_frame = tk.Frame(right, bg="#111")
        right.add(canvas_frame, minsize=250)
        self.canvas = tk.Canvas(canvas_frame, bg="#111", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Double-Button-1>", self._dblclick)
        self.canvas.bind("<Button-3>", self._rclick)
        self.canvas.bind("<Button-2>", self._rclick)   # macOS right-click
        self.canvas.bind("<Configure>", self._resize)

        table_frame = tk.Frame(right)
        right.add(table_frame, minsize=140, height=260)
        top = tk.Frame(table_frame)
        top.pack(fill=tk.X, padx=4, pady=2)
        self._plan_lbl = tk.Label(top, text="", font=("Arial", 9, "bold"), anchor=tk.W)
        self._plan_lbl.pack(side=tk.LEFT)
        tk.Button(top, text="Refresh", command=self._refresh_plan).pack(side=tk.RIGHT)
        tk.Label(top, text="Double-click a video to preview it or fix its circle",
                 font=("Arial", 8), fg="#666").pack(side=tk.RIGHT, padx=8)
        body = tk.Frame(table_frame)
        body.pack(fill=tk.BOTH, expand=True)
        self.table = ttk.Treeview(body, columns=("status", "details"),
                                  show="tree headings", selectmode="extended")
        self.table.heading("#0", text="Video")
        self.table.heading("status", text="Status")
        self.table.heading("details", text="Circle / details")
        self.table.column("#0", width=340, stretch=False)
        self.table.column("status", width=100, stretch=False)
        self.table.column("details", width=560)
        for tag, colour in self.ROW_COLORS.items():
            self.table.tag_configure(tag, foreground=colour)
        bar = ttk.Scrollbar(body, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=bar.set)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.table.bind("<Double-1>", self._on_table_double)
        self.table.bind("<Delete>", lambda e: self._remove_selected())
        self.table.bind("<BackSpace>", lambda e: self._remove_selected())

    def _scrollable_sidebar(self, pane):
        """A vertically scrollable sidebar so all panels fit on small screens."""
        outer = tk.Frame(pane, bg="#f4f4f4")
        pane.add(outer, minsize=300, width=330)
        canvas = tk.Canvas(outer, bg="#f4f4f4", highlightthickness=0)
        bar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas, bg="#f4f4f4", padx=8, pady=6)
        window = canvas.create_window((0, 0), window=inner, anchor=tk.NW)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        prefix = str(canvas)

        def on_wheel(event):
            # Only scroll when the pointer is over the sidebar.
            if not str(event.widget).startswith(prefix):
                return
            if getattr(event, "num", None) == 4:
                step = -1
            elif getattr(event, "num", None) == 5:
                step = 1
            else:
                delta = event.delta
                step = -int(delta / 120) if abs(delta) >= 120 else -int(delta)
            if step:
                canvas.yview_scroll(step, "units")

        self.root.bind_all("<MouseWheel>", on_wheel, add="+")
        self.root.bind_all("<Button-4>", on_wheel, add="+")
        self.root.bind_all("<Button-5>", on_wheel, add="+")
        return inner

    @staticmethod
    def _panel(parent, title):
        frame = ttk.LabelFrame(parent, text=f"  {title}  ")
        frame.pack(fill=tk.X, pady=4)
        return frame

    @staticmethod
    def _note(parent, text):
        label = tk.Label(parent, text=text, font=("Arial", 8), fg="#444",
                         justify=tk.LEFT, anchor=tk.W, wraplength=270)
        label.pack(fill=tk.X, padx=4, pady=(2, 2))
        return label

    @staticmethod
    def _set_enabled(widgets, enabled: bool):
        for widget in widgets:
            if isinstance(widget, ttk.Widget):
                widget.state(["!disabled"] if enabled else ["disabled"])
            else:
                widget.config(state=tk.NORMAL if enabled else tk.DISABLED)

    # ══════════════════════════════════════════════════════════════════════
    # Drag & drop
    # ══════════════════════════════════════════════════════════════════════
    def _register_drop_targets(self):
        """Accept dropped files/folders on the preview, table and drop box."""
        if self._dnd:
            try:
                for widget in (self.canvas, self.table, self._drop_lbl):
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind("<<Drop>>", self._on_drop)
                self._drop_lbl.dnd_bind("<<DropEnter>>",
                                        lambda e: self._drop_highlight(e, True))
                self._drop_lbl.dnd_bind("<<DropLeave>>",
                                        lambda e: self._drop_highlight(e, False))
            except Exception:
                self._dnd = False
        if self._dnd:
            self._drop_lbl.config(
                text="⬇  Drop videos or folders here\n(or onto the preview / table)")
        else:
            self._drop_lbl.config(
                text="Drag & drop is off. To turn it on:\npip install tkinterdnd2")

    def _drop_highlight(self, event, on: bool):
        self._drop_lbl.config(bg="#e3f2fd" if on else "#fafafa")
        return getattr(event, "action", None)

    def _on_drop(self, event):
        self._drop_lbl.config(bg="#fafafa")
        try:
            paths = list(self.root.tk.splitlist(event.data))
        except Exception:
            paths = [event.data]
        self.handle_dropped_paths(paths)
        return getattr(event, "action", None)

    def handle_dropped_paths(self, paths: list[str]):
        """Dropped files are added to the video list. A single folder dropped in
        Folder mode becomes the folder; otherwise a folder's videos are added
        to the list (including subfolders when that box is ticked)."""
        if self._busy:
            self._status("Busy processing; drop videos again when it finishes.")
            return
        folders = [p for p in paths if os.path.isdir(p)]
        files = [p for p in paths if os.path.isfile(p)]
        if self.input_mode.get() == "folder" and len(folders) == 1 and not files:
            self._set_folder(folders[0])
            return
        if self.input_mode.get() != "files":
            self.input_mode.set("files")
            self._on_input_mode_quiet()
        added, rejected = self._add_paths(files, folders)
        self._after_list_change(added)
        self._report_added(added, rejected)

    # ══════════════════════════════════════════════════════════════════════
    # Thread-safe UI helpers
    # ══════════════════════════════════════════════════════════════════════
    def _ui(self, fn):
        """Run fn on the Tk main loop (safe to call from the worker thread)."""
        self.root.after(0, fn)

    def _status(self, msg: str):
        self._ui(lambda m=msg: self._status_lbl.config(text=m))

    # ══════════════════════════════════════════════════════════════════════
    # Input: video list (any folders) or one folder
    # ══════════════════════════════════════════════════════════════════════
    def _apply_prefill(self, prefill: dict):
        params = prefill.get("parameters")
        if params:
            self._set_parameters_from_file(params, show_errors=True)
        if prefill.get("output"):
            self.out_dir = str(Path(prefill["output"]).expanduser().resolve())
        self.strict_var.set(bool(prefill.get("strict")))
        files = _as_file_list(prefill.get("video"))
        if files:
            self.input_mode.set("files")
            self._add_paths(files, [])
        elif prefill.get("videos"):
            self.input_mode.set("folder")
            self.folder = str(Path(prefill["videos"]).expanduser().resolve())
            self.recursive_var.set(bool(prefill.get("recursive")))
        if prefill.get("render_video"):
            self.render_var.set(True)
            self._render_touched = True

    def _on_input_mode(self):
        files_mode = self.input_mode.get() == "files"
        if files_mode:
            self._folder_box.pack_forget()
            self._files_box.pack(fill=tk.X)
        else:
            self._files_box.pack_forget()
            self._folder_box.pack(fill=tk.X)
        self._apply_render_default()
        self._update_input_label()
        self._update_out_dir_label()
        self._refresh_plan()
        self._sync_preview()

    def _on_recursive(self):
        # Folder mode: changes which videos are found. Video-list mode: applies
        # to the next "Add a folder's videos" or dropped folder.
        if self.input_mode.get() == "folder":
            self._refresh_plan()
            self._sync_preview()

    def _on_render_toggle(self):
        self._render_touched = True

    def _apply_render_default(self):
        """Until the user touches it: render only when one video is selected
        (annotated MP4s are slow; the CLI never renders unless asked)."""
        if not self._render_touched:
            self.render_var.set(self.input_mode.get() == "files"
                                and len(self.video_files) == 1)

    def _update_input_label(self):
        if self.input_mode.get() == "folder":
            self._input_lbl.config(text=f"Folder: {self.folder}" if self.folder
                                   else "No folder chosen")
        else:
            n = len(self.video_files)
            folders = len({str(Path(v).parent) for v in self.video_files})
            self._input_lbl.config(
                text=(f"{n} video(s) from {folders} folder(s)" if n
                      else "No videos yet: add them or drop them here"))

    def _add_paths(self, files: list[str], folders: list[str]) -> tuple[int, list[str]]:
        """Add video files, and the videos inside folders, to the list."""
        existing = set(self.video_files)
        added, rejected = [], []
        candidates = [Path(f).expanduser().resolve() for f in files]
        for folder in folders:
            try:
                candidates.extend(_videos_from_inputs(folder, None, self.recursive_var.get()))
            except ValueError:
                rejected.append(f"{Path(folder).name}/ (no videos)")
        for path in candidates:
            if path.is_file() and _is_video_file(path):
                if str(path) not in existing:
                    existing.add(str(path))
                    self.video_files.append(str(path))
                    added.append(str(path))
            else:
                rejected.append(path.name)
        return len(added), rejected

    def _report_added(self, added: int, rejected: list[str]):
        msg = f"Added {added} video(s)." if added else "No new videos added."
        if rejected:
            shown = ", ".join(rejected[:5]) + (" …" if len(rejected) > 5 else "")
            msg += f"\nIgnored (not a supported video): {shown}"
        self._status(msg)

    def _after_list_change(self, added: int = 0):
        self._apply_render_default()
        self._update_input_label()
        self._update_out_dir_label()
        self._refresh_plan()
        if added and self.video_files and (
                not self.video_path or not self._in_plan(self.video_path)):
            self._open_video(self.video_files[-added])
        else:
            self._sync_preview()

    def _add_videos_dialog(self):
        if self._busy:
            return
        exts = " ".join(f"*{e} *{e.upper()}" for e in sorted(OPTIMIZE_VIDEO_EXTENSIONS))
        paths = filedialog.askopenfilenames(title="Add videos (select several)",
                                            filetypes=[("Video", exts), ("All", "*.*")])
        paths = list(self.root.tk.splitlist(paths)) if isinstance(paths, str) else list(paths or [])
        if not paths:
            return
        self.input_mode.set("files")
        added, rejected = self._add_paths(paths, [])
        self._on_input_mode_quiet()
        self._after_list_change(added)
        self._report_added(added, rejected)

    def _add_folder_dialog(self):
        if self._busy:
            return
        folder = filedialog.askdirectory(title="Add all videos in a folder")
        if not folder:
            return
        self.input_mode.set("files")
        added, rejected = self._add_paths([], [folder])
        self._on_input_mode_quiet()
        self._after_list_change(added)
        self._report_added(added, rejected)

    def _on_input_mode_quiet(self):
        """Show the right input buttons without rebuilding the plan."""
        if self.input_mode.get() == "files":
            self._folder_box.pack_forget()
            self._files_box.pack(fill=tk.X)

    def _remove_selected(self):
        if self._busy or self.input_mode.get() != "files" or self.plan is None:
            return
        selected = self.table.selection()
        if not selected:
            self._status("Select videos in the table first (Shift/Cmd-click for several).")
            return
        remove = {str(self.plan.items[int(iid)].video) for iid in selected}
        self.video_files = [v for v in self.video_files if v not in remove]
        self._status(f"Removed {len(remove)} video(s) from the list.")
        self._after_list_change()

    def _clear_list(self):
        if self._busy:
            return
        self.video_files = []
        self._status("Video list cleared.")
        self._after_list_change()

    def _choose_folder_dialog(self):
        if self._busy:
            return
        folder = filedialog.askdirectory(title="Select video folder")
        if folder:
            self._set_folder(folder)

    def _set_folder(self, folder: str):
        self.input_mode.set("folder")
        self.folder = str(Path(folder).expanduser().resolve())
        self._on_input_mode()
        if self.plan:
            self._status(f"Folder: {self.plan.summary_line()}")

    def _sync_preview(self):
        """Keep the preview on a video that belongs to the current input."""
        if self.plan and self.plan.items:
            if not self.video_path or not self._in_plan(self.video_path):
                self._open_video(str(self.plan.items[0].video))
        elif self.video_path:
            self._close_preview()

    # ══════════════════════════════════════════════════════════════════════
    # Preview video and circle handling
    # ══════════════════════════════════════════════════════════════════════
    def load_video(self, path: str):
        if not os.path.exists(path):
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return
        self.handle_dropped_paths([path])

    def _close_preview(self):
        if self.cap:
            self.cap.release()
        self.cap = None
        self.video_path = None
        self.current_bgr = None
        self._suspend_circle_hook = True
        try:
            self.clear_circle()
        finally:
            self._suspend_circle_hook = False
        self.canvas.delete("all")
        self._info_lbl.config(text="")
        self._frame_lbl.config(text="—")

    def _open_video(self, path: str) -> bool:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open:\n{path}")
            return False
        if self.cap:
            self.cap.release()
        self.cap = cap
        self.video_path = path
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._slider.config(to=max(0, self.total_frames - 1))
        self._frame_var.set(0)

        # Show this video's own circle (clicked, or found as a file).
        self._suspend_circle_hook = True
        try:
            self.clear_circle()
        finally:
            self._suspend_circle_hook = False
        self._autoload_circle(Path(path))
        self._show_frame(0)
        dur = self.total_frames / self.fps
        self._info_lbl.config(
            text=f"Preview: {os.path.basename(path)}\n"
                 f"{self.orig_w}×{self.orig_h}  {self.fps:.0f} fps\n"
                 f"{self.total_frames:,} frames  ({dur:.0f} s)")
        self._refresh_proc_btn()
        return True

    def _manual(self) -> bool:
        return self.circle_mode.get() == "manual"

    def _autoload_circle(self, video: Path):
        """Show the circle this video will use in the current circle mode."""
        if self._manual():
            entry = self.video_circles.get(str(video.resolve()))
            if entry is None:
                self._circle_lbl.config(text="Not clicked yet: click the center, then north",
                                        fg="#b00")
                return
            circle, path = entry
            self._apply_circle_params(circle)
            self.circle_source_path = path
            self._circle_lbl.config(
                text=(f"Clicked circle (saved as {Path(path).name})" if path
                      else "Clicked circle (used when you process)"),
                fg="#006600")
            return
        lookup = _batch_circle(video, self._output_dir_for(video))
        if lookup.status == "ok":
            self._apply_circle_params(lookup.circle)
            self.circle_source_path = str(lookup.path)
            self._circle_lbl.config(text=f"Found in {lookup.source}:\n{lookup.path.name}",
                                    fg="#006600")
        elif lookup.status == "invalid":
            self._circle_lbl.config(text=f"Circle file invalid: {lookup.reason}", fg="#b00")
        else:
            self._circle_lbl.config(
                text=f"No {video.stem}_circle.json in the input or output folder",
                fg="#b00")

    def _show_circle_file(self, path: str, label: str):
        circle, reason = _read_circle_file(Path(path))
        if circle is None:
            self._circle_lbl.config(text=f"Circle file invalid: {reason}", fg="#b00")
            return
        self._apply_circle_params(circle)
        self.circle_source_path = str(path)
        self._circle_lbl.config(text=f"Circle loaded ({label}):\n{Path(path).name}",
                                fg="#006600")

    def _remember_preview_circle(self, path: str | None):
        """Record the preview circle as this video's own circle (per-video mode)."""
        if not self._manual() or not self.video_path:
            return
        key = str(Path(self.video_path).resolve())
        if self.circle_complete:
            self.video_circles[key] = (self._preview_circle(), path)
        else:
            self.video_circles.pop(key, None)

    def _circle_edited(self):
        """Called by CirclePreviewMixin after a click, clear or radius change."""
        if self._suspend_circle_hook:
            return
        self.circle_source_path = None
        self._remember_preview_circle(None)
        if not self._manual() and self.circle_complete:
            # Find-files mode only uses circles saved as files.
            self._circle_lbl.config(
                text="Not used until saved: click Save circle… (keep the suggested "
                     "<name>_circle.json next to the video)", fg="#8a5a00")
        self._refresh_plan()

    def _preview_circle(self) -> dict | None:
        if not self.circle_complete:
            return None
        return _circle_params_dict(self.circle_center, float(self.circle_radius),
                                   self.north_point)

    def _save_circle(self):
        if not self.circle_complete:
            messagebox.showwarning("Circle", "Define the center and north first.")
            return
        video = Path(self.video_path) if self.video_path else None
        path = filedialog.asksaveasfilename(
            title="Save circle parameters",
            initialdir=str(video.parent) if video else None,
            initialfile=f"{video.stem}_circle.json" if video else "circle.json",
            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            _save_circle_json(self._preview_circle(), path)
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.circle_source_path = path
        self._remember_preview_circle(path)
        self._circle_lbl.config(text=f"Circle saved:\n{Path(path).name}", fg="#006600")
        self._refresh_plan()
        message = f"Saved circle to:\n{path}"
        if not self._manual() and video is not None:
            name = f"{video.stem}_circle.json"
            spots = {(video.parent / name).resolve()}
            out = self._output_dir_for(video)
            if out:
                spots.add((out / name).resolve())
            if Path(path).resolve() not in spots:
                message += (f"\n\nNote: it is not named {name} in the video or output "
                            "folder, so it won't be found automatically.")
        self._status(message)

    def _on_circle_mode(self):
        if self._manual():
            self._circle_mode_lbl.config(
                text="Double-click each video in the table and click its circle. "
                     "Circle files on disk are ignored.")
        else:
            self._circle_mode_lbl.config(
                text="Looks next to each video first, then in the output folder.")
        self._refresh_plan()
        if self.video_path:
            # Show the circle this video uses in the new mode.
            self._suspend_circle_hook = True
            try:
                self.clear_circle()
            finally:
                self._suspend_circle_hook = False
            self._autoload_circle(Path(self.video_path))
            self._refresh_proc_btn()

    # ══════════════════════════════════════════════════════════════════════
    # Parameters
    # ══════════════════════════════════════════════════════════════════════
    def _set_parameters_from_file(self, path: str, show_errors: bool = True) -> bool:
        try:
            values = load_tune_values(path)
            if not values:
                raise ValueError("No recognized numeric parameters in the file.")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if show_errors:
                messagebox.showerror("Parameters", f"Could not load {path}:\n\n{exc}")
            return False
        self.parameter_values = {**DEFAULT_PARAMETERS, **values}
        self.parameters_file = str(Path(path).expanduser().resolve())
        self._params_lbl.config(text=f"{Path(path).name}  ({len(values)} values)\n{self.parameters_file}")
        return True

    def _load_parameters(self):
        path = filedialog.askopenfilename(title="Load parameter preset",
                                          filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path and self._set_parameters_from_file(path):
            self._status("Parameters loaded; they apply when you click Process.")

    def _reset_parameters(self):
        self.parameter_values = dict(DEFAULT_PARAMETERS)
        self.parameters_file = None
        self._params_lbl.config(text="Defaults (built-in values)")
        self._status("Parameters reset to the built-in defaults.")

    def _view_parameters(self):
        win = tk.Toplevel(self.root)
        win.title("Parameter values")
        source = self.parameters_file or "built-in defaults"
        text = tk.Text(win, width=64, height=min(32, len(self.parameter_values) + 4),
                       font=("Courier", 10))
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, f"Source: {source}\n* = differs from the default\n\n")
        for key in sorted(self.parameter_values):
            value = self.parameter_values[key]
            default = DEFAULT_PARAMETERS.get(key)
            mark = "*" if value != default else " "
            suffix = f"   (default {default})" if value != default else ""
            text.insert(tk.END, f"{mark} {key:<26} {value}{suffix}\n")
        text.config(state=tk.DISABLED)

    # ══════════════════════════════════════════════════════════════════════
    # Output folder
    # ══════════════════════════════════════════════════════════════════════
    def _default_output(self) -> Path | None:
        if self.input_mode.get() == "folder":
            return default_output_root(self.folder, None) if self.folder else None
        return default_output_root(None, ["x"])

    def _output_root(self) -> Path | None:
        return Path(self.out_dir).resolve() if self.out_dir else self._default_output()

    def _output_dir_for(self, video: Path) -> Path | None:
        """This video's own output folder (mirrored subfolder for folder runs)."""
        if self.plan is not None:
            for item in self.plan.items:
                if item.video == video.resolve():
                    return item.output_dir
        return self._output_root()

    def _select_out_dir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if not path:
            return
        self.out_dir = path
        self._update_out_dir_label()
        self._refresh_plan()

    def _clear_out_dir(self):
        self.out_dir = None
        self._update_out_dir_label()
        self._refresh_plan()

    def _update_out_dir_label(self):
        if self.out_dir:
            self._out_dir_lbl.config(text=f"Output folder: {self.out_dir}")
            return
        default = self._default_output()
        where = str(default) if default else "<folder>/batch_results"
        self._out_dir_lbl.config(text=f"Output folder (default): {where}")

    # ══════════════════════════════════════════════════════════════════════
    # Batch plan (pre-run check) and the video table
    # ══════════════════════════════════════════════════════════════════════
    def _build_plan(self) -> BatchPlan | None:
        """Build the same plan the CLI would, then apply the GUI circle mode."""
        if self.input_mode.get() == "files":
            if not self.video_files:
                return None
            plan = plan_batch(video_file=list(self.video_files), output_dir=self.out_dir)
        else:
            if not self.folder:
                return None
            plan = plan_batch(videos_dir=self.folder, recursive=self.recursive_var.get(),
                              output_dir=self.out_dir)
        if self._manual():
            # Only circles clicked in the GUI count; files on disk are ignored.
            for item in plan.items:
                entry = self.video_circles.get(str(item.video))
                if entry is None:
                    item.circle = CircleLookup(
                        "missing", reason="not clicked yet (double-click the row, then "
                                          "click the center and north)")
                else:
                    circle, path = entry
                    item.circle = CircleLookup(
                        "ok", circle, Path(path) if path else None,
                        self.CLICKED if path else self.CLICKED_UNSAVED)
        return plan

    def _refresh_plan(self):
        if self._busy:
            return
        try:
            self.plan = self._build_plan()
            self._plan_error = ""
        except ValueError as exc:
            self.plan = None
            self._plan_error = str(exc)
        self._fill_table()
        self._refresh_proc_btn()

    def _in_plan(self, path: str) -> bool:
        return bool(self.plan) and any(item.video == Path(path).resolve()
                                       for item in self.plan.items)

    def _row_values(self, item: BatchItem) -> tuple[tuple, str]:
        if item.status == "pending":
            tag = item.state
            if tag == "ready":
                where = item.circle.path if item.circle.path else ""
                details = f"{item.circle.source}: {where}".rstrip(": ")
            else:
                details = item.reason
        else:
            tag = item.status
            details = item.message
        return (tag, details), tag

    def _fill_table(self):
        self.table.delete(*self.table.get_children())
        self._iid_by_item.clear()
        if self.plan is None:
            empty = ("Add videos or drop them here" if self.input_mode.get() == "files"
                     else "Choose a folder of videos")
            self._plan_lbl.config(text=self._plan_error or empty)
            return
        for index, item in enumerate(self.plan.items):
            iid = str(index)
            values, tag = self._row_values(item)
            # Folder mode: path inside the folder. Video list: path relative to
            # the folders' common parent (just the name when all share one).
            self.table.insert("", tk.END, iid=iid, text=item.relative,
                              values=values, tags=(tag,))
            self._iid_by_item[id(item)] = iid
        c = self.plan.counts()
        self._plan_lbl.config(
            text=f"{c['total']} videos · {c['ready']} ready · {c['missing']} missing · "
                 f"{c['invalid']} invalid   →   output: {self.plan.output_root}")

    def _update_row(self, item: BatchItem):
        iid = self._iid_by_item.get(id(item))
        if iid is None or not self.table.exists(iid):
            return
        values, tag = self._row_values(item)
        self.table.item(iid, values=values, tags=(tag,))
        if item.status == "running":
            self.table.see(iid)

    def _on_table_double(self, event):
        if self._busy:
            return
        iid = self.table.identify_row(event.y) or self.table.focus()
        if not iid or self.plan is None:
            return
        item = self.plan.items[int(iid)]
        if self._open_video(str(item.video)):
            if item.state == "ready":
                self._status(f"Previewing {item.relative}.")
            else:
                if self._manual():
                    hint = "Click the center, then north. It's used when you process."
                else:
                    hint = ("Click the center, then north, then Save circle… (keep the "
                            "suggested <name>_circle.json) so it is found.")
                self._status(f"Previewing {item.relative} ({item.reason}).\n{hint}")

    # ══════════════════════════════════════════════════════════════════════
    # Processing
    # ══════════════════════════════════════════════════════════════════════
    def _can_process(self) -> tuple[bool, str]:
        if self.plan is None:
            if self._plan_error:
                return False, self._plan_error
            return False, ("Add videos or drop them here." if self.input_mode.get() == "files"
                           else "Choose a folder.")
        c = self.plan.counts()
        problems = c["missing"] + c["invalid"]
        if c["ready"] == 0:
            if c["total"] == 1:
                return False, "Draw or load a circle (center, then north)."
            return False, "No video has a valid circle yet."
        if problems and self.strict_var.get():
            return False, f"Strict: fix {problems} missing/invalid circle(s) first."
        if problems:
            return True, f"{problems} video(s) without a valid circle will be skipped."
        return True, ""

    def _refresh_proc_btn(self):
        if not hasattr(self, "_proc_btn"):
            return
        regression = getattr(self, "regression", None)
        if self._busy or (regression is not None and regression._busy):
            self._proc_btn.config(state=tk.DISABLED)
            return
        ok, hint = self._can_process()
        self._proc_btn.config(state=tk.NORMAL if ok else tk.DISABLED)
        self._proc_hint.config(text=hint)

    def start_processing(self):
        if self._busy:
            return
        if self.regression._busy:
            messagebox.showwarning("Busy", "The Regression tab is running; wait for it to finish.")
            return
        self._refresh_plan()
        ok, hint = self._can_process()
        if not ok:
            messagebox.showwarning("Cannot process yet", hint)
            return
        plan = self.plan
        strict = self.strict_var.get()
        render = self.render_var.get()
        code_version = self.ask_code_version()
        if code_version is None:
            self._status("Run cancelled (no code version given).")
            return
        # Parameters apply at the START of a run: defaults first, then the preset.
        apply_tune_values(DEFAULT_PARAMETERS)
        apply_tune_values(self.parameter_values)

        runnable = [item for item in plan.items if item.state == "ready"]
        self._run_order = {id(item): (k, len(runnable)) for k, item in enumerate(runnable, 1)}
        self._busy = True
        self._cancel_event = threading.Event()
        self._proc_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        self._prog_var.set(0)
        self._prog_lbl.config(text="Starting…")
        self._overall_lbl.config(text=f"0/{len(runnable)} videos")
        self._status("Processing…")
        threading.Thread(target=self._thread_body,
                         args=(plan, strict, render, self.parameters_file, code_version),
                         daemon=True).start()

    def _thread_body(self, plan, strict, render, parameters_file, code_version=None):
        try:
            summary = execute_batch(
                plan, strict=strict, render_video=render, parameters_file=parameters_file,
                progress_cb=self._on_batch_progress, video_cb=self._on_item_changed,
                log_cb=lambda m: print(m, flush=True), cancel_event=self._cancel_event,
                code_version=code_version, interface='gui')
            self._ui(lambda s=summary: self._finish_run(s, None))
        except BatchAbort as exc:
            message, summary = str(exc), exc.summary
            self._ui(lambda s=summary, m=message: self._finish_run(s, m))
        except Exception as exc:  # show unexpected errors instead of dying silently
            import traceback
            message = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()[-600:]}"
            self._ui(lambda m=message: self._finish_run(None, m))

    def _on_batch_progress(self, index, total, pct, msg):
        def update():
            self._prog_var.set(pct)
            if msg and msg != "done":
                self._prog_lbl.config(text=msg)
        self._ui(update)

    def _on_item_changed(self, item):
        def update(it=item):
            self._update_row(it)
            if it.status == "running":
                k, n = self._run_order.get(id(it), (0, 0))
                self._overall_lbl.config(text=f"Video {k}/{n}: {it.relative}")
                self._prog_var.set(0)
        self._ui(update)

    def _finish_run(self, summary: dict | None, error: str | None):
        self._busy = False
        self._cancel_event = None
        self._cancel_btn.config(state=tk.DISABLED)
        lines = []
        if summary:
            state = "Cancelled" if summary["cancelled"] else ("Stopped" if error else "Done")
            lines.append(f"{state}: {summary['videos_processed']} processed · "
                         f"{summary['videos_skipped']} skipped · "
                         f"{summary['videos_failed']} failed")
            entries, exits = summary["totals"]["entries"], summary["totals"]["exits"]
            lines.append("  ".join(f"{q} in:{entries[q]} out:{exits[q]}" for q in _QUADRANTS))
            lines.append(f"Saved to:\n{summary['output']}")
            lines.append(f"Metadata: run_metadata.json (code version {summary['code_version']})")
            if not error and not summary["cancelled"]:
                self._prog_var.set(100)
            self._overall_lbl.config(text=lines[0])
        self._prog_lbl.config(text="Complete!" if summary and not error else "")
        if error:
            lines.append(error.splitlines()[0])
            messagebox.showerror("Processing stopped", error)
        self._status("\n".join(lines))
        self._refresh_proc_btn()
        self.regression._refresh_btn()

    def cancel_processing(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._status("Cancelling… (finishing the current frame)")

    # ══════════════════════════════════════════════════════════════════════
    # Equivalent CLI command
    # ══════════════════════════════════════════════════════════════════════
    def _copy_cli_command(self):
        files_mode = self.input_mode.get() == "files"
        if self.plan is None:
            messagebox.showinfo("Copy CLI command",
                                "Add videos first." if files_mode else "Choose a folder first.")
            return
        opts = {"parameters": self.parameters_file, "output": self.out_dir,
                "render_video": self.render_var.get(), "strict": self.strict_var.get(),
                "code_version": self.code_version}
        if files_mode:
            opts["video"] = list(self.video_files)
        else:
            opts.update(videos=self.folder, recursive=self.recursive_var.get())
        if self._manual():
            self._offer_to_save_gui_circles()
        command = build_command_line("run", **opts)
        self.root.clipboard_clear()
        self.root.clipboard_append(command)
        print(command, flush=True)
        self._status(f"Copied to clipboard:\n{command}")

    def _cli_findable(self, item: BatchItem, path: Path | None) -> bool:
        """Would the CLI's circle lookup find this exact file for this video?"""
        if path is None:
            return False
        name = f"{item.video.stem}_circle.json"
        spots = {(item.video.parent / name).resolve(), (item.output_dir / name).resolve()}
        return path.resolve() in spots

    def _offer_to_save_gui_circles(self):
        """Circles clicked in the GUI are invisible to the CLI (which finds
        <name>_circle.json files); offer to save them next to each video."""
        pending = [item for item in self.plan.items
                   if item.circle.source in (self.CLICKED, self.CLICKED_UNSAVED)
                   and not self._cli_findable(item, item.circle.path)]
        if not pending:
            return
        names = ", ".join(item.video.name for item in pending[:4]) + \
            (" …" if len(pending) > 4 else "")
        if not messagebox.askyesno(
                "Copy CLI command",
                f"{len(pending)} circle(s) clicked in the GUI aren't saved where the "
                f"command looks ({names}).\n\nSave them as <name>_circle.json next to "
                "each video now? (An existing file with that name is replaced.)\n\n"
                "If you choose No, the command will skip those videos."):
            return
        saved = 0
        for item in pending:
            target = item.video.parent / f"{item.video.stem}_circle.json"
            try:
                _save_circle_json(item.circle.circle, str(target))
            except OSError as exc:
                messagebox.showerror("Save failed", f"{target}\n\n{exc}")
                continue
            self.video_circles[str(item.video)] = (item.circle.circle, str(target))
            saved += 1
        if self.video_path and str(Path(self.video_path).resolve()) in self.video_circles:
            self.circle_source_path = self.video_circles[str(Path(self.video_path).resolve())][1]
        self._refresh_plan()
        self._status(f"Saved {saved} circle file(s) next to their videos.")

    def ask_code_version(self) -> str | None:
        """Ask which GitHub version of the code this run uses (for the run
        metadata). Pre-filled with the last answer or the version detected
        from git. Returns None if the user cancels (the run is not started)."""
        detected = detect_code_version()
        answer = simpledialog.askstring(
            "Code version",
            "GitHub version of the code for this run\n(commit, tag or release).\n"
            "It is saved in the run's metadata file."
            + (f"\n\nDetected from git: {detected}" if detected else ""),
            initialvalue=self.code_version or detected, parent=self.root)
        if answer is None:
            return None
        self.code_version = answer.strip() or detected or "not provided"
        return self.code_version

    # ══════════════════════════════════════════════════════════════════════
    def _on_close(self):
        busy = self._busy or self.regression._busy
        if busy and not messagebox.askyesno(
                "Quit", "Processing is running. Cancel it and quit?"):
            return
        for event in (self._cancel_event, self.regression._cancel_event):
            if event is not None:
                event.set()
        self.root.destroy()

    # Starts Tkinter event loop and releases the preview capture on exit.
    def run(self):
        self.root.mainloop()
        if self.cap:
            self.cap.release()



# GUI tab for --regression: choose hand counts + videos, check matches and
# circles, run, then see the fit statistics, plot and per-video errors.
# Uses plan_regression()/run_regression_mode(), exactly what the CLI runs.
class RegressionTab:
    ROW_COLORS = App.ROW_COLORS

    def __init__(self, parent, app: "App"):
        self.app = app
        self.root = app.root
        self.ground_truth: str | None = None
        self.videos_dir: str | None = None
        self.parameters_file: str | None = None
        self.parameter_values: dict = dict(DEFAULT_PARAMETERS)
        self.circle_file: str | None = None
        self.out_dir: str | None = None
        self.plan: dict | None = None
        self._plan_error = ""
        self._runnable: list[dict] = []
        self._busy = False
        self._cancel_event: threading.Event | None = None
        self._plot_photo = None
        self._last_summary: dict | None = None
        self._build(parent)
        self._refresh_plan()

    # ── layout ───────────────────────────────────────────────────────────
    def _build(self, parent):
        self.quadrant_var = tk.StringVar(value="All")
        self.render_var = tk.BooleanVar(value=False)
        self.skip_var = tk.BooleanVar(value=False)
        note = App._note
        panel = App._panel

        pane = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=6,
                              sashrelief=tk.RAISED, bg="#333")
        pane.pack(fill=tk.BOTH, expand=True)
        side = tk.Frame(pane, bg="#f4f4f4", padx=8, pady=6)
        pane.add(side, minsize=300, width=330)

        tk.Label(side, text="📈  Regression", font=("Arial", 15, "bold"),
                 bg="#f4f4f4").pack(pady=(2, 6))
        tk.Label(side, text="Compare the tracker's counts with hand counts.",
                 font=("Arial", 8), bg="#f4f4f4", fg="#555").pack(anchor=tk.W)

        box = panel(side, "Hand counts (ground truth)")
        self._gt_lbl = note(box, "Not chosen")
        row = tk.Frame(box)
        row.pack(fill=tk.X, padx=4, pady=(0, 4))
        tk.Button(row, text="Choose CSV…", command=self._choose_gt_csv).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(row, text="Choose folder…", command=self._choose_gt_folder).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        box = panel(side, "Videos")
        self._vid_lbl = note(box, "Not chosen")
        tk.Button(box, text="Choose video folder…", command=self._choose_videos).pack(
            fill=tk.X, padx=4, pady=(0, 4))
        self._drop_lbl = tk.Label(box, text="", relief=tk.GROOVE, bd=2, height=2,
                                  font=("Arial", 8), fg="#555", bg="#fafafa", wraplength=260)
        self._drop_lbl.pack(fill=tk.X, padx=4, pady=(0, 4))

        box = panel(side, "Parameters")
        self._params_lbl = note(box, "Defaults (built-in values)")
        row = tk.Frame(box)
        row.pack(fill=tk.X, padx=4, pady=(0, 4))
        tk.Button(row, text="Load…", command=self._load_parameters).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(row, text="Reset", command=self._reset_parameters).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        box = panel(side, "Circles")
        tk.Label(box, text=("Each video uses <name>_circle.json next to the video,\n"
                            "in the hand-counts folder, or in <videos>/circles/."),
                 font=("Arial", 8), justify=tk.LEFT, fg="#444").pack(anchor=tk.W, padx=4)
        self._circle_lbl = note(box, "Optional: one circle file for every video")
        row = tk.Frame(box)
        row.pack(fill=tk.X, padx=4, pady=(0, 2))
        tk.Button(row, text="Circle file for all…", command=self._choose_circle).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(row, text="Clear", command=self._clear_circle).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Checkbutton(box, text="Skip videos without a circle\n(--skip-missing-circles)",
                        variable=self.skip_var,
                        command=self._refresh_btn).pack(anchor=tk.W, padx=4, pady=(0, 4))

        box = panel(side, "Options")
        row = tk.Frame(box)
        row.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(row, text="Quadrant:", font=("Arial", 8)).pack(side=tk.LEFT)
        combo = ttk.Combobox(row, textvariable=self.quadrant_var, width=6, state="readonly",
                             values=["All", *_QUADRANTS])
        combo.pack(side=tk.LEFT, padx=4)
        combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_plan())
        ttk.Checkbutton(box, text="Render annotated MP4 (--render-video)",
                        variable=self.render_var).pack(anchor=tk.W, padx=4)
        self._out_lbl = note(box, "")
        row = tk.Frame(box)
        row.pack(fill=tk.X, padx=4, pady=(0, 4))
        tk.Button(row, text="Select output folder…", command=self._select_out).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(row, text="Use default", command=self._clear_out).pack(side=tk.LEFT, padx=(2, 0))

        box = panel(side, "Run")
        self._run_btn = tk.Button(box, text="▶  Run regression", font=("Arial", 11, "bold"),
                                  bg="#1565c0", fg="white", state=tk.DISABLED,
                                  activebackground="#0d47a1", activeforeground="white",
                                  command=self.start)
        self._run_btn.pack(fill=tk.X, padx=4, pady=(4, 0))
        self._hint = tk.Label(box, text="", font=("Arial", 8), fg="#8a5a00",
                              wraplength=270, justify=tk.LEFT)
        self._hint.pack(anchor=tk.W, padx=4)
        self._cancel_btn = tk.Button(box, text="Cancel", state=tk.DISABLED, command=self.cancel)
        self._cancel_btn.pack(fill=tk.X, padx=4, pady=2)
        self._overall_lbl = tk.Label(box, text="", font=("Arial", 9, "bold"),
                                     wraplength=270, justify=tk.LEFT)
        self._overall_lbl.pack(anchor=tk.W, padx=4)
        self._prog_var = tk.DoubleVar()
        ttk.Progressbar(box, variable=self._prog_var, maximum=100).pack(fill=tk.X, padx=4, pady=2)
        self._prog_lbl = tk.Label(box, text="", font=("Arial", 8), wraplength=270, justify=tk.LEFT)
        self._prog_lbl.pack(anchor=tk.W, padx=4)
        row = tk.Frame(box)
        row.pack(fill=tk.X, padx=4, pady=4)
        tk.Button(row, text="Copy CLI command", command=self._copy_command).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(row, text="Open output folder", command=self._open_output).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))
        self._status_lbl = tk.Label(side, text="Choose the hand counts and the video folder.",
                                    wraplength=290, justify=tk.LEFT, font=("Arial", 9),
                                    bg="#f4f4f4", fg="#333")
        self._status_lbl.pack(anchor=tk.W, pady=6)

        right = tk.PanedWindow(pane, orient=tk.VERTICAL, sashwidth=6,
                               sashrelief=tk.RAISED, bg="#333")
        pane.add(right, minsize=500)
        top = tk.Frame(right)
        right.add(top, minsize=180, height=300)
        head = tk.Frame(top)
        head.pack(fill=tk.X, padx=4, pady=2)
        self._plan_lbl = tk.Label(head, text="", font=("Arial", 9, "bold"), anchor=tk.W,
                                  justify=tk.LEFT, wraplength=900)
        self._plan_lbl.pack(side=tk.LEFT)
        tk.Button(head, text="Refresh", command=self._refresh_plan).pack(side=tk.RIGHT)
        body = tk.Frame(top)
        body.pack(fill=tk.BOTH, expand=True)
        self.table = ttk.Treeview(body, columns=("title", "status", "details"),
                                  show="tree headings", selectmode="browse")
        for col, text, width in (("#0", "Video", 230), ("title", "Hand-count title", 150),
                                 ("status", "Status", 90), ("details", "Circle / result", 560)):
            self.table.heading(col, text=text)
            self.table.column(col, width=width, stretch=(col == "details"))
        for tag, colour in self.ROW_COLORS.items():
            self.table.tag_configure(tag, foreground=colour)
        self.table.tag_configure("unmatched", foreground="#757575")
        bar = ttk.Scrollbar(body, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=bar.set)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        results = tk.Frame(right, bg="white")
        right.add(results, minsize=200)
        self._stats = tk.Text(results, width=36, font=("Courier", 10), relief=tk.FLAT,
                              bg="white", wrap=tk.WORD)
        self._stats.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        self._stats.insert(tk.END, "Results appear here after a run.")
        self._stats.config(state=tk.DISABLED)
        self._plot_lbl = tk.Label(results, bg="white")
        self._plot_lbl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._plot_lbl.bind("<Configure>", lambda e: self._show_plot())
        self._update_out_label()

    def register_drop_targets(self, enabled: bool):
        if enabled:
            try:
                for widget in (self._drop_lbl, self.table):
                    widget.drop_target_register(DND_FILES)
                    widget.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                enabled = False
        self._drop_lbl.config(
            text=("⬇  Drop the hand-count CSV and/or the video folder here" if enabled
                  else "Drag & drop is off (pip install tkinterdnd2)"))

    def _on_drop(self, event):
        try:
            paths = list(self.root.tk.splitlist(event.data))
        except Exception:
            paths = [event.data]
        self.handle_dropped_paths(paths)
        return getattr(event, "action", None)

    def handle_dropped_paths(self, paths: list[str]):
        """A .csv becomes the hand counts; a folder with videos becomes the
        video folder; a folder with only CSVs becomes the hand-count folder."""
        if self._busy:
            return
        notes = []
        for raw in paths:
            path = Path(raw).expanduser().resolve()
            if path.is_file() and path.suffix.lower() == ".csv":
                self.ground_truth = str(path)
                notes.append(f"hand counts: {path.name}")
            elif path.is_dir():
                has_videos = any(p.is_file() and _is_video_file(p) for p in path.iterdir())
                has_csv = any(p.is_file() and p.suffix.lower() == ".csv" for p in path.iterdir())
                if has_videos:
                    self.videos_dir = str(path)
                    notes.append(f"videos: {path.name}/")
                elif has_csv:
                    self.ground_truth = str(path)
                    notes.append(f"hand-count folder: {path.name}/")
                else:
                    notes.append(f"ignored {path.name}/ (no videos or CSVs)")
            else:
                notes.append(f"ignored {path.name}")
        self._update_labels()
        self._refresh_plan()
        self._status("Dropped → " + "; ".join(notes))

    # ── helpers ──────────────────────────────────────────────────────────
    def _ui(self, fn):
        self.root.after(0, fn)

    def _status(self, msg: str):
        self._ui(lambda m=msg: self._status_lbl.config(text=m))

    def _quadrant(self) -> str | None:
        value = self.quadrant_var.get()
        return None if value in ("", "All") else value

    def _default_output(self) -> Path | None:
        return Path(self.videos_dir) / "regression_results" if self.videos_dir else None

    def _output(self) -> Path | None:
        return Path(self.out_dir).resolve() if self.out_dir else self._default_output()

    def _update_labels(self):
        self._gt_lbl.config(text=self.ground_truth or "Not chosen")
        self._vid_lbl.config(text=self.videos_dir or "Not chosen")
        self._circle_lbl.config(text=(f"For all videos: {self.circle_file}" if self.circle_file
                                      else "Optional: one circle file for every video"))
        self._update_out_label()

    def _update_out_label(self):
        if self.out_dir:
            text = f"Output folder: {self.out_dir}"
        else:
            text = (f"Output folder (default): "
                    f"{self._default_output() or '<videos>/regression_results'}")
        self._out_lbl.config(text=text + "\nEach run gets its own regression_<date-time> "
                                         "folder inside it.")

    # ── inputs ───────────────────────────────────────────────────────────
    def apply_prefill(self, prefill: dict):
        if prefill.get("ground_truth"):
            self.ground_truth = str(Path(prefill["ground_truth"]).expanduser().resolve())
        if prefill.get("videos"):
            self.videos_dir = str(Path(prefill["videos"]).expanduser().resolve())
        if prefill.get("parameters"):
            self._set_parameters(prefill["parameters"])
        if prefill.get("circle_file"):
            self.circle_file = str(Path(prefill["circle_file"]).expanduser().resolve())
        if prefill.get("output"):
            self.out_dir = str(Path(prefill["output"]).expanduser().resolve())
        if prefill.get("quadrant"):
            self.quadrant_var.set(prefill["quadrant"])
        self.render_var.set(bool(prefill.get("render_video")))
        self.skip_var.set(bool(prefill.get("skip_missing_circles")))
        self._update_labels()
        self._refresh_plan()

    def _choose_gt_csv(self):
        path = filedialog.askopenfilename(title="Hand-count CSV",
                                          filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            self.ground_truth = path
            self._update_labels()
            self._refresh_plan()

    def _choose_gt_folder(self):
        path = filedialog.askdirectory(title="Folder of per-video hand-count CSVs")
        if path:
            self.ground_truth = path
            self._update_labels()
            self._refresh_plan()

    def _choose_videos(self):
        path = filedialog.askdirectory(title="Video folder")
        if path:
            self.videos_dir = path
            self._update_labels()
            self._refresh_plan()

    def _set_parameters(self, path: str) -> bool:
        try:
            values = load_tune_values(path)
            if not values:
                raise ValueError("No recognized numeric parameters in the file.")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            messagebox.showerror("Parameters", f"Could not load {path}:\n\n{exc}")
            return False
        self.parameter_values = {**DEFAULT_PARAMETERS, **values}
        self.parameters_file = str(Path(path).expanduser().resolve())
        self._params_lbl.config(text=f"{Path(path).name}  ({len(values)} values)\n"
                                     f"{self.parameters_file}")
        return True

    def _load_parameters(self):
        path = filedialog.askopenfilename(title="Load parameter preset",
                                          filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self._set_parameters(path)

    def _reset_parameters(self):
        self.parameter_values = dict(DEFAULT_PARAMETERS)
        self.parameters_file = None
        self._params_lbl.config(text="Defaults (built-in values)")

    def _choose_circle(self):
        path = filedialog.askopenfilename(title="One circle file for every video",
                                          filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        circle, reason = _read_circle_file(Path(path))
        if circle is None:
            messagebox.showerror("Invalid circle file", f"{path}\n\n{reason}")
            return
        self.circle_file = path
        self._update_labels()
        self._refresh_plan()

    def _clear_circle(self):
        self.circle_file = None
        self._update_labels()
        self._refresh_plan()

    def _select_out(self):
        path = filedialog.askdirectory(title="Regression output folder")
        if path:
            self.out_dir = path
            self._update_out_label()

    def _clear_out(self):
        self.out_dir = None
        self._update_out_label()

    # ── pre-run check ────────────────────────────────────────────────────
    def _refresh_plan(self):
        if self._busy:
            return
        self.plan, self._plan_error = None, ""
        if self.ground_truth and self.videos_dir:
            try:
                self.plan = plan_regression(self.ground_truth, self.videos_dir,
                                            self._quadrant(), self.circle_file)
            except (ValueError, OSError) as exc:
                self._plan_error = str(exc)
        self._fill_table()
        self._refresh_btn()

    def _fill_table(self):
        self.table.delete(*self.table.get_children())
        if self.plan is None:
            self._plan_lbl.config(text=self._plan_error or
                                  "Choose the hand counts and the video folder")
            return
        for index, pair in enumerate(self.plan["pairs"]):
            circle = pair["circle"]
            if circle.status == "ok":
                tag, details = "ready", f"{circle.source}: {circle.path}"
            else:
                tag, details = circle.status, circle.reason
            self.table.insert("", tk.END, iid=f"p{index}", text=pair["video"].name,
                              values=(pair["title"], tag, details), tags=(tag,))
        for name in self.plan["unmatched_videos"]:
            self.table.insert("", tk.END, text=name,
                              values=("—", "no counts", "not in the hand counts; not scored"),
                              tags=("unmatched",))
        c = self._counts()
        text = (f"{c['matched']} matched · {c['ready']} with circles · "
                f"{c['missing']} missing/invalid circle · "
                f"{len(self.plan['unmatched_videos'])} video(s) without hand counts")
        if self.plan["unmatched_truth"]:
            text += f" · {len(self.plan['unmatched_truth'])} hand-count row(s) without a video"
        self._plan_lbl.config(text=text)

    def _counts(self) -> dict:
        pairs = self.plan["pairs"] if self.plan else []
        ready = sum(p["circle"].status == "ok" for p in pairs)
        return {"matched": len(pairs), "ready": ready, "missing": len(pairs) - ready}

    def _can_run(self) -> tuple[bool, str]:
        if not self.ground_truth or not self.videos_dir:
            return False, "Choose the hand counts and the video folder."
        if self.plan is None:
            return False, self._plan_error or "Nothing to check."
        c = self._counts()
        if c["matched"] == 0:
            return False, "No video names match the hand-count titles."
        if c["ready"] == 0:
            return False, "No matched video has a valid circle."
        if c["missing"] and not self.skip_var.get():
            return False, (f"{c['missing']} matched video(s) have no valid circle. Make them in "
                           "the Count ants tab, choose a circle file for all, or tick "
                           "'Skip videos without a circle'.")
        if c["missing"]:
            return True, f"{c['missing']} video(s) without a circle will be skipped."
        return True, ""

    def _refresh_btn(self):
        if self._busy or self.app._busy:
            self._run_btn.config(state=tk.DISABLED)
            return
        ok, hint = self._can_run()
        self._run_btn.config(state=tk.NORMAL if ok else tk.DISABLED)
        self._hint.config(text=hint)

    # ── run ──────────────────────────────────────────────────────────────
    def start(self):
        if self._busy:
            return
        if self.app._busy:
            messagebox.showwarning("Busy", "The Count ants tab is processing; wait for it to finish.")
            return
        self._refresh_plan()
        ok, hint = self._can_run()
        if not ok:
            messagebox.showwarning("Cannot run yet", hint)
            return
        code_version = self.app.ask_code_version()
        if code_version is None:
            self._status("Run cancelled (no code version given).")
            return
        apply_tune_values(DEFAULT_PARAMETERS)
        apply_tune_values(self.parameter_values)
        self._runnable = [p for p in self.plan["pairs"] if p["circle"].status == "ok"]
        self._busy = True
        self._cancel_event = threading.Event()
        self._run_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        self._prog_var.set(0)
        self._overall_lbl.config(text=f"0/{len(self._runnable)} videos")
        self._prog_lbl.config(text="Starting…")
        self._status("Running regression…")
        args = dict(ground_truth_dir=self.ground_truth, videos_dir=self.videos_dir,
                    parameters_file=self.parameters_file, quadrant=self._quadrant(),
                    circle_file=self.circle_file, output_dir=self.out_dir,
                    render_video=self.render_var.get(), show_plot=False,
                    skip_missing_circles=self.skip_var.get(), code_version=code_version)
        threading.Thread(target=self._thread_body, args=(args,), daemon=True).start()

    def _thread_body(self, args):
        try:
            summary = run_regression_mode(
                **args, progress_cb=self._on_progress,
                log_cb=lambda m: print(m, flush=True), cancel_event=self._cancel_event)
            self._ui(lambda s=summary: self._finish(s, None))
        except Exception as exc:
            cancelled = self._cancel_event is not None and self._cancel_event.is_set()
            message = "Cancelled." if cancelled else f"{type(exc).__name__}: {exc}"
            self._ui(lambda m=message, c=cancelled: self._finish(None, m, c))

    def _on_progress(self, index, total, pct, msg):
        def update():
            pair = self._runnable[index - 1] if 0 < index <= len(self._runnable) else None
            if pair is not None:
                self._overall_lbl.config(text=f"Video {index}/{total}: {pair['video'].name}")
                iid = f"p{self.plan['pairs'].index(pair)}"
                if self.table.exists(iid):
                    self.table.item(iid, values=(pair["title"], "running",
                                                 self.table.item(iid, "values")[2]),
                                    tags=("running",))
            self._prog_var.set(pct)
            if msg:
                self._prog_lbl.config(text=msg)
        self._ui(update)

    def _finish(self, summary: dict | None, error: str | None, cancelled: bool = False):
        self._busy = False
        self._cancel_event = None
        self._cancel_btn.config(state=tk.DISABLED)
        if summary is None:
            self._overall_lbl.config(text="Cancelled" if cancelled else "Stopped")
            self._prog_lbl.config(text="")
            self._status(error or "")
            if not cancelled:
                messagebox.showerror("Regression stopped", error)
            self._refresh_plan()
            self.app._refresh_proc_btn()
            return
        self._last_summary = summary
        self._prog_var.set(100)
        self._overall_lbl.config(text=f"Done: {summary['videos_evaluated']} video(s) scored")
        self._prog_lbl.config(text="Complete!")
        self._show_results(summary)
        self._status(f"Saved to:\n{summary['output_dir']}")
        self._refresh_btn()
        self.app._refresh_proc_btn()

    def cancel(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._status("Cancelling… (finishing the current frame)")

    # ── results ──────────────────────────────────────────────────────────
    def _show_results(self, summary: dict):
        fit = summary["linear_regression"]

        def fmt(key, digits=3):
            value = fit.get(key)
            return f"{value:.{digits}f}" if isinstance(value, float) else ("N/A" if value is None else str(value))

        lines = [
            "Predicted vs hand counts",
            "",
            f"Points (n)         {fit['n']}",
            f"Slope   (ideal 1)  {fmt('slope')}",
            f"Intercept (ideal 0){fmt('intercept'):>6}",
            f"R²      (ideal 1)  {fmt('r2')}",
            f"Pearson r          {fmt('r')}",
            f"p-value            {fmt('p_value', 4)}",
            f"MAE  (ideal 0)     {fmt('mae', 2)}",
            f"RMSE (ideal 0)     {fmt('count_rmse', 2)}",
            f"RMSE (fit line)    {fmt('rmse', 2)}",
            "",
            f"Videos scored      {summary['videos_evaluated']}",
            f"Code version       {summary.get('code_version', '')}",
            f"Total abs. error   {summary['total_absolute_error']}",
            f"Exact rows         {summary['exact_match_rows']}",
        ]
        if fit.get("fit_note"):
            lines += ["", fit["fit_note"]]
        if summary.get("skipped_missing_circle"):
            lines += ["", f"Skipped (no circle): {len(summary['skipped_missing_circle'])}"]
        self._stats.config(state=tk.NORMAL)
        self._stats.delete("1.0", tk.END)
        self._stats.insert(tk.END, "\n".join(lines))
        self._stats.config(state=tk.DISABLED)

        # Per-video result in the table.
        by_name = {v["video"]: v for v in summary["per_video"]}
        skipped = {s["video"]: s["reason"] for s in summary.get("skipped_missing_circle", [])}
        for index, pair in enumerate(self.plan["pairs"]):
            iid = f"p{index}"
            if not self.table.exists(iid):
                continue
            name = pair["video"].name
            if name in by_name:
                parts = []
                for row in by_name[name]["comparison"]:
                    parts.append(f"{row['quadrant']} in {row['predicted_in']}/{row['ground_truth_in']}"
                                 f" out {row['predicted_out']}/{row['ground_truth_out']}")
                total = sum(row["total_abs_error"] for row in by_name[name]["comparison"])
                self.table.item(iid, values=(pair["title"], "processed",
                                             f"abs error {total} · " + " · ".join(parts)),
                                tags=("processed",))
            elif name in skipped:
                self.table.item(iid, values=(pair["title"], "skipped", skipped[name]),
                                tags=("skipped",))
        self._show_plot()

    def _show_plot(self):
        summary = self._last_summary
        if not summary:
            return
        path = summary["files"].get("plot")
        if not path or not os.path.isfile(path):
            return
        try:
            width = max(300, int(self._plot_lbl.winfo_width()) - 10)
            height = max(200, int(self._plot_lbl.winfo_height()) - 10)
            image = Image.open(path)
            image.thumbnail((width, height))
            self._plot_photo = ImageTk.PhotoImage(image)
            self._plot_lbl.config(image=self._plot_photo)
        except Exception as exc:  # display only; the PNG is saved regardless
            self._plot_lbl.config(text=f"Plot saved to {path}\n({exc})")

    # ── extras ───────────────────────────────────────────────────────────
    def _copy_command(self):
        if not (self.ground_truth and self.videos_dir):
            messagebox.showinfo("Copy CLI command", "Choose the hand counts and the video folder first.")
            return
        command = build_command_line(
            "regression", regression=self.ground_truth, videos=self.videos_dir,
            quadrant=self._quadrant(), parameters=self.parameters_file,
            circle_file=self.circle_file, output=self.out_dir,
            render_video=self.render_var.get(), skip_missing_circles=self.skip_var.get(),
            code_version=self.app.code_version)
        self.root.clipboard_clear()
        self.root.clipboard_append(command)
        print(command, flush=True)
        self._status(f"Copied to clipboard:\n{command}")

    def _open_output(self):
        # After a run, open that run's own folder (regression_<date-time>/).
        folder = (Path(self._last_summary["output_dir"]) if self._last_summary
                  else self._output())
        if not folder or not folder.is_dir():
            messagebox.showinfo("Output folder", "The output folder doesn't exist yet; run first.")
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            elif os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as exc:
            messagebox.showerror("Output folder", f"{folder}\n\n{exc}")

# ─────────────────────────────────────────────────────────────────────────────


# ============================================================================
# INTERACTIVE TUNING MODE
# ============================================================================
# This mode runs separately from Tkinter's normal batch-processing interface.
# It uses OpenCV trackbars and updates the SAME Tracker and KalmanFilter2D classes.
# You can test parameter changes frame by frame before saving a JSON preset.
# See --help for command-line usage.

# Map: key -> (slider label, min, max, factor).  Every slider value maps to the
# actual runtime parameter as slider/factor.  We use factors for decimal noise
# variances such as KF_MEAS_POS and REACQUIRE_DISTANCE_SCALE.
TUNE_SPECS = {
    # Detection: foreground extraction and connected component filtering.
    'DIFF_THRESH': ('Diff threshold', 1, 100, 1),
    'FRAME_STRIDE': ('Frame stride', 1, 8, 1),
    'MIN_BLOB_AREA': ('Min blob area', 1, 250, 1),
    'MAX_BLOB_AREA': ('Max area x10', 10, 500, 0.1),
    'MORPH_CLOSE_SIZE': ('Closing kernel', 1, 15, 1),
    'MORPH_OPEN_SIZE': ('Opening kernel', 1, 15, 1),
    'CLUSTER_DISTANCE': ('Blob merge dist', 0, 50, 1),
    'VIBRATION_PCT': ('Vibration % x10', 1, 400, 10),
    # KF process/measurement uncertainties and track-association gate.
    'KF_PROC_POS': ('KF position Q x10', 0, 2000, 10),
    'KF_PROC_VEL': ('KF velocity Q x10', 0, 1000, 10),
    'KF_MEAS_POS': ('KF measurement R x10', 1, 1000, 10),
    'MAX_TRACK_DIST': ('Max track distance', 1, 200, 1),
    # Gaps, reassociation, tentative ant lifespan and crossing eligibility.
    'TEMPORAL_MAX_GAP': ('Allowed missing frames', 0, 15, 1),
    'REACQUIRE_DISTANCE_SCALE': ('Reacquire scale x100', 100, 400, 100),
    'MAX_COAST_FRAMES': ('Coast max frames', 0, 15, 1),
    'MIN_TRACK_AGE': ('Min track age', 0, 30, 1),
    'MIN_MATCH_STREAK': ('Min match streak', 1, 15, 1),
}

def _odd_kernel(value):
    """OpenCV morphological kernels should have odd positive side lengths."""
    value = max(1, int(value))
    return value if value % 2 else value + 1


def apply_tune_values(settings, tracker=None):
    """Apply recognized tuning keys (including optimizer-only values).

    Parameter presets are plain flat JSON objects. Live sliders remain the
    authority over their own parameters when adjusted interactively.
    """
    recognized = set(TUNE_SPECS) | {'PROCESS_SCALE', 'BUFFER_ZONE',
                'CROSS_HYSTERESIS_FRAMES', 'PATH_CONFIRM_FRAMES'}
    for name, value in settings.items():
        if name not in recognized or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value):
            continue
        if name in ('MORPH_OPEN_SIZE', 'MORPH_CLOSE_SIZE'):
            value = _odd_kernel(value)
        globals()[name] = value
    if tracker is not None:
        tracker.max_dist = float(MAX_TRACK_DIST)
        tracker.max_missing = int(MAX_COAST_FRAMES)
        for ant in tracker.ants().values():
            kf = ant['kf']
            kf.Q = np.diag([KF_PROC_POS, KF_PROC_POS, KF_PROC_VEL, KF_PROC_VEL])
            kf.R = np.diag([KF_MEAS_POS, KF_MEAS_POS])


def save_tune_values(settings, filename):
    """Persist numeric slider values; use --settings FILE for batch processing."""
    with open(filename, 'w', encoding='utf-8') as fp:
        json.dump(settings, fp, indent=2)
    print('Saved tuning parameters to:', filename)


def load_tune_values(filename):
    """Load numerical tuning preset from JSON including optimization-only keys."""
    with open(filename, encoding='utf-8') as fp:
        saved = json.load(fp)
    if not isinstance(saved, dict):
        raise ValueError('Settings JSON must be an object of parameter names and numeric values')
    extras = {'PROCESS_SCALE': (0.1, 1.0), 'BUFFER_ZONE': (0, 100),
              'CROSS_HYSTERESIS_FRAMES': (1, 30), 'PATH_CONFIRM_FRAMES': (1, 30)}
    result = {}
    for key, value in saved.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        if key in TUNE_SPECS:
            _, low, high, factor = TUNE_SPECS[key]
            result[key] = max(low / factor, min(high / factor, value))
        elif key in extras:
            low, high = extras[key]
            result[key] = max(low, min(high, value))
    for key in ('MORPH_OPEN_SIZE', 'MORPH_CLOSE_SIZE'):
        if key in result:
            result[key] = _odd_kernel(result[key])
    return result


# Optimization terminal progress utilities. Flush ensures live output even when
# standard output is redirected to a log file or displayed in an IDE terminal.
def _opt_progress(message: str) -> None:
    print(f"[OPTUNA {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _opt_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes:d}m {secs:02d}s"


# ==========================================================================
# OPTUNA OPTIMIZATION: annotated ant positions, not entrance-crossing counts
# ==========================================================================
# Usage: python ant_tracker_optuna.py --optimize /path/to/folder --trials 30
#        python ant_tracker_optuna.py --settings /path/to/folder/ant_tuning_parameters.json
# Input folder/{ground-truth, videos}; CSV header:
# timestamp,frame,ant_x,ant_y (ant_id optional). Frames are ZERO-BASED and coordinates
# correspond to the source video's original (unscaled) pixel resolution.
# A separate ground-truth row is required for EACH visible ant at each labeled
# frame. Omitted ants count as unannotated (predictions will be false positives).
# Videos must share filenames (stem) with their ground-truth CSV files.
#
# Crossing-only parameters are retained in output presets but are not tuned:
# position-only annotations do not establish true entry/exit events.
# Scoring: one-to-one spatial Hungarian matching within --match-distance
# original-image pixels; maximize 90% detection F1 + 10% ID consistency.
# Tracks in a brief Kalman-only gap are included as predictions; this lets
# the optimization reward successful temporal gap handling. The regular
# tracker receives ALL intervening frames between 2-second annotations.
# A holdout dataset is NOT implied by these reported training-set scores.
# ==========================================================================

OPTIMIZE_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}
# ant_id is optional (older annotator CSVs have it; scoring never uses it).
GROUND_TRUTH_COLUMNS = {'timestamp', 'frame', 'ant_x', 'ant_y'}


@dataclass(frozen=True)
class OptimizationVideoPair:
    """One video and corresponding annotation CSV."""
    stem: str
    video_path: Path
    ground_truth_path: Path
    annotation_count: int


@dataclass(frozen=True)
class OptimizationDataset:
    """Validated directory of video/CSV pairs."""
    root: Path
    pairs: tuple[OptimizationVideoPair, ...]


def load_optimization_dataset(folder: str | Path, ground_truth_dir: str | Path | None = None,
                              videos_dir: str | Path | None = None,
                              output_dir: str | Path | None = None) -> OptimizationDataset:
    """Pair basename-matched video/CSV files and validate required headers."""
    root = Path(output_dir or folder).expanduser().resolve()
    gt_dir = Path(ground_truth_dir).expanduser().resolve() if ground_truth_dir else Path(folder).expanduser().resolve() / 'ground-truth'
    video_dir = Path(videos_dir).expanduser().resolve() if videos_dir else Path(folder).expanduser().resolve() / 'videos'
    if not gt_dir.is_dir() or not video_dir.is_dir():
        raise ValueError(f'Optuna requires existing ground-truth and videos folders: {gt_dir} and {video_dir}')
    root.mkdir(parents=True, exist_ok=True)
    gt_files = sorted(p for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() == '.csv')
    videos = sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in OPTIMIZE_VIDEO_EXTENSIONS)
    if not gt_files or not videos:
        raise ValueError('Optimization requires CSV files in ground-truth/ and videos in videos/')
    by_stem = {}
    for video in videos:
        key = video.stem.casefold()
        if key in by_stem:
            raise ValueError(f'Duplicate videos named {video.stem!r} with different extensions')
        by_stem[key] = video
    pairs = []
    matched = set()
    for csv_path in gt_files:
        key = csv_path.stem.casefold()
        if key in matched:
            raise ValueError(f'Duplicate ground-truth stem: {csv_path.stem}')
        video = by_stem.get(key)
        if video is None:
            raise ValueError(f'No corresponding video found for {csv_path.name}')
        with csv_path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            actual = {c.strip().lower() for c in reader.fieldnames or []}
            missing = GROUND_TRUTH_COLUMNS - actual
            if missing:
                raise ValueError(f'{csv_path.name} missing CSV headers: {sorted(missing)}')
            count = sum(bool(any(str(v or '').strip() for v in row.values())) for row in reader)
        if not count:
            raise ValueError(f'{csv_path.name} has no labeled ants')
        matched.add(key)
        pairs.append(OptimizationVideoPair(csv_path.stem, video, csv_path, count))
    unused = [p.name for k, p in by_stem.items() if k not in matched]
    if unused:
        print('Warning: ignoring videos without annotation CSVs:', ', '.join(unused))
    return OptimizationDataset(root, tuple(pairs))


@dataclass(frozen=True)
class GroundTruthObservation:
    """Human-labeled ant at one zero-based frame in original video pixels."""
    timestamp_seconds: float
    frame_index: int
    ant_id: str
    x: float
    y: float


@dataclass(frozen=True)
class PredictedAntObservation:
    """Tracked ant at an annotation frame. is_observed=False means Kalman coast."""
    frame_index: int
    timestamp_seconds: float
    predicted_id: int
    x: float
    y: float
    is_observed: bool


@dataclass(frozen=True)
class VideoTrialPredictions:
    """Ant locations sampled ONLY at annotated frames, after full tracking."""
    stem: str
    observations: tuple[PredictedAntObservation, ...]


@dataclass(frozen=True)
class MatchedAntPair:
    """A spatially valid one-to-one ground-truth/prediction correspondence."""
    truth: GroundTruthObservation
    prediction: PredictedAntObservation
    distance_pixels: float


@dataclass(frozen=True)
class FrameMatchResult:
    """Matched and unmatched ants for one labeled frame."""
    frame_index: int
    matches: tuple[MatchedAntPair, ...]
    unmatched_truth: tuple[GroundTruthObservation, ...]
    unmatched_predictions: tuple[PredictedAntObservation, ...]


@dataclass(frozen=True)
class VideoAccuracyMetrics:
    """Ant-detection metrics at labeled frames; ground-truth IDs are ignored."""
    stem: str
    true_positives: int
    false_positives: int
    false_negatives: int
    detection_f1: float
    mean_position_error_px: float | None


class GroundTruthReader:
    """Parse CSV rows and validate frame positions against actual video FPS.

    Treat `frame` as zero-based. Timestamp is checked against frame/fps,
    but minor rounding of floating video timestamps is tolerated. Text
    timestamps HH:MM:SS.sss or numeric seconds are both supported.
    """

    @staticmethod
    def _time_seconds(value: str) -> float:
        value = value.strip()
        if ':' in value:
            parts = [float(p) for p in value.split(':')]
            if len(parts) > 3:
                raise ValueError('Too many colon-separated timestamp fields')
            return sum(part * (60 ** index) for index, part in enumerate(reversed(parts)))
        return float(value)

    def load(self, pair: OptimizationVideoPair) -> tuple[GroundTruthObservation, ...]:
        cap = cv2.VideoCapture(str(pair.video_path))
        try:
            if not cap.isOpened():
                raise ValueError(f'Cannot decode video: {pair.video_path}')
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        if fps <= 0 or total <= 0 or width <= 0 or height <= 0:
            raise ValueError(f'Invalid video metadata: {pair.video_path.name}')
        observations = []
        seen = set()
        with pair.ground_truth_path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            for line, raw in enumerate(reader, start=2):
                if not any(str(v or '').strip() for v in raw.values()):
                    continue
                row = {k.strip().lower(): v for k, v in raw.items() if k is not None}
                try:
                    timestamp = self._time_seconds(str(row['timestamp']))
                    frame_as_float = float(row['frame'])
                    frame = int(frame_as_float)
                    ant_id = str(row.get('ant_id') or '').strip() or f'row{line}'
                    x, y = float(row['ant_x']), float(row['ant_y'])
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f'{pair.ground_truth_path.name}, row {line}: invalid annotation: {exc}') from exc
                if not all(math.isfinite(v) for v in (timestamp, frame_as_float, x, y)):
                    raise ValueError(f'{pair.stem}, row {line}: non-finite field')
                if frame < 0 or frame >= total or frame_as_float != frame:
                    raise ValueError(f'{pair.stem}, row {line}: frame {frame_as_float} not in [0, {total-1}]')
                if not ant_id or not (0 <= x < width and 0 <= y < height):
                    raise ValueError(f'{pair.stem}, row {line}: invalid ID or coordinate outside {width}x{height}')
                if timestamp < 0 or abs(timestamp - frame / fps) > max(0.15, 2.0 / fps):
                    raise ValueError(f'{pair.stem}, row {line}: timestamp {timestamp} disagrees with frame {frame} at {fps:.3f} FPS')
                # The same point twice on one frame is dropped with a warning.
                key = (frame, round(x, 3), round(y, 3))
                if key in seen:
                    print(f'Warning: {pair.ground_truth_path.name}, row {line}: duplicate point '
                          f'({x:g}, {y:g}) on frame {frame}; ignored', flush=True)
                    continue
                seen.add(key)
                observations.append(GroundTruthObservation(timestamp, frame, ant_id, x, y))
        if not observations:
            raise ValueError(f'{pair.stem}: no annotations found')
        return tuple(sorted(observations, key=lambda v: (v.frame_index, v.ant_id)))


# Source defaults that apply to both live GUI and optimizer presets.
# The live sliders expose the entries from TUNE_SPECS; the extra keys are
# applied through --settings and through trial-scoped global settings.
OPTIMIZER_EXTRA_KEYS = ('PROCESS_SCALE', 'BUFFER_ZONE', 'CROSS_HYSTERESIS_FRAMES',
                        'PATH_CONFIRM_FRAMES')
OPTIMIZER_SETTINGS_KEYS = tuple(dict.fromkeys((*TUNE_SPECS, *OPTIMIZER_EXTRA_KEYS)))


class OptunaParameterSampler:
    """Sample a concrete candidate, with valid odd morphology kernels.

    Search bounds are intentionally limited to plausible starting ranges.
    Increase trials/bounds once a baseline is established. Some parameters
    such as MIN_TRACK_AGE do not affect the current tracker; they are omitted.
    Crossing-only controls (buffer and hysteresis) also cannot be learned from
    position-only ground truth: the output preset retains baseline values.
    """
    def sample(self, trial) -> dict:
        params = {
            'PROCESS_SCALE': trial.suggest_categorical('PROCESS_SCALE', [0.25, 0.35, 0.5]),
            'DIFF_THRESH': trial.suggest_int('DIFF_THRESH', 5, 55),
            'FRAME_STRIDE': trial.suggest_int('FRAME_STRIDE', 1, 5),
            'MIN_BLOB_AREA': trial.suggest_int('MIN_BLOB_AREA', 2, 65),
            'MAX_BLOB_AREA': trial.suggest_int('MAX_BLOB_AREA', 200, 2200, step=100),
            'MORPH_CLOSE_SIZE': trial.suggest_categorical('MORPH_CLOSE_SIZE', [1, 3, 5, 7, 9]),
            'MORPH_OPEN_SIZE': trial.suggest_categorical('MORPH_OPEN_SIZE', [1, 3, 5, 7]),
            'CLUSTER_DISTANCE': trial.suggest_int('CLUSTER_DISTANCE', 0, 30),
            'VIBRATION_PCT': trial.suggest_float('VIBRATION_PCT', 4, 30, step=1),
            'KF_PROC_POS': trial.suggest_float('KF_PROC_POS', 1, 90, log=True),
            'KF_PROC_VEL': trial.suggest_float('KF_PROC_VEL', 0.5, 30, log=True),
            'KF_MEAS_POS': trial.suggest_float('KF_MEAS_POS', 0.5, 35, log=True),
            'MAX_TRACK_DIST': trial.suggest_int('MAX_TRACK_DIST', 12, 100, step=4),
            'TEMPORAL_MAX_GAP': trial.suggest_int('TEMPORAL_MAX_GAP', 0, 8),
            'REACQUIRE_DISTANCE_SCALE': trial.suggest_float('REACQUIRE_DISTANCE_SCALE', 1.0, 2.0, step=0.1),
            'MAX_COAST_FRAMES': trial.suggest_int('MAX_COAST_FRAMES', 0, 10),
        }
        # Do not search visualization-only settings, which have no effect on accuracy.
        # MAX_COAST_FRAMES and TEMPORAL_MAX_GAP jointly limit actual gap retention.
        return params

    @contextmanager
    def apply_for_trial(self, params: dict):
        """Apply optimizer parameters and restore all module globals on exit.

        The existing pipeline reads globals from KalmanFilter2D, Tracker and
        morphology/detection code; serial trials isolate changes via this
        context manager. Do not parallelize these trials within one process.
        """
        previous = {k: globals()[k] for k in params}
        try:
            apply_tune_values(params)
            yield
        finally:
            globals().update(previous)


class TrackingTrialRunner:
    """Run the same frame-difference/morphology/blob/tracker pipeline headlessly.

    Every intermediate frame is processed. Only requested frames are stored.
    Tracker.update() currently requires circle geometry even for location
    tracking; supply a dummy, valid circle to preserve identical association
    and Kalman behavior without requiring manual circle calibration.
    """

    def run_video(self, pair: OptimizationVideoPair, params: dict,
                  annotation_frames: tuple[int, ...], progress_callback=None) -> VideoTrialPredictions:
        # progress_callback(current_frame, total_frames_to_process) is optional.
        # Called every 10% and on completion to avoid per-frame terminal spam.
        requested = set(annotation_frames)
        if not requested:
            return VideoTrialPredictions(pair.stem, ())
        scale = float(params['PROCESS_SCALE'])
        if not (0 < scale <= 1):
            raise ValueError('PROCESS_SCALE must be in (0,1]')
        cap = cv2.VideoCapture(str(pair.video_path))
        if not cap.isOpened():
            raise ValueError(f'Cannot open video {pair.video_path}')
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0 or width <= 0 or height <= 0:
                raise ValueError(f'Invalid video metadata: {pair.stem}')
            pw, ph = max(1, int(width * scale)), max(1, int(height * scale))
            frame_pixels = pw * ph
            vibration_limit = frame_pixels * float(VIBRATION_PCT) / 100.0
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (int(MORPH_CLOSE_SIZE), int(MORPH_CLOSE_SIZE)))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                     (int(MORPH_OPEN_SIZE), int(MORPH_OPEN_SIZE)))
            tracker = Tracker(float(MAX_TRACK_DIST), int(MAX_COAST_FRAMES),
                              int(CROSS_HYSTERESIS_FRAMES))
            # Path events are irrelevant to location scoring; use a circle
            # fully within the image so crossing functions can still run.
            circle = {'center': (pw / 2, ph / 2),
                      'radius': max(float(BUFFER_ZONE) + 2.0, min(pw, ph) * 0.30),
                      'north_angle': 270.0}
            frame_history = deque(maxlen=int(FRAME_STRIDE) + 1)
            results = []
            last_frame = max(requested)
            total_to_process = last_frame + 1
            progress_step = max(1, total_to_process // 10)
            for fid in range(last_frame + 1):
                okay, frame = cap.read()
                if not okay:
                    raise ValueError(f'{pair.stem}: video ended at frame {fid}, before annotation frame {last_frame}')
                resized = cv2.resize(frame, (pw, ph))
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                frame_history.append(gray)
                if len(frame_history) > FRAME_STRIDE:
                    diff = cv2.absdiff(gray, frame_history[0])
                    diff = cv2.GaussianBlur(diff, (3, 3), 0)
                else:
                    diff = np.zeros_like(gray)
                _, mask = cv2.threshold(diff, int(DIFF_THRESH), 255, cv2.THRESH_BINARY)
                # Important: keep the legacy processing behavior, which applies
                # the shadow suppression to diff AFTER the binary mask is built.
                diff[gray < 110] = 0
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
                blobs = []
                if np.count_nonzero(mask) <= vibration_limit:
                    n, _, stats, centers = cv2.connectedComponentsWithStats(mask)
                    for i in range(1, n):
                        area = int(stats[i, cv2.CC_STAT_AREA])
                        if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
                            blobs.append({'cx': float(centers[i, 0]), 'cy': float(centers[i, 1]),
                                          'area': area,
                                          'x': int(stats[i, cv2.CC_STAT_LEFT]),
                                          'y': int(stats[i, cv2.CC_STAT_TOP]),
                                          'w': int(stats[i, cv2.CC_STAT_WIDTH]),
                                          'h': int(stats[i, cv2.CC_STAT_HEIGHT])})
                    blobs = cluster_blobs(blobs, max_dist=float(CLUSTER_DISTANCE))
                tracker.update(blobs, circle)
                if fid in requested:
                    for ant_id, ant in tracker.ants().items():
                        results.append(PredictedAntObservation(
                            fid, fid / fps, int(ant_id), float(ant['cx']) / scale,
                            float(ant['cy']) / scale,
                            bool(ant.get('matched_this_frame', False))))
                if progress_callback is not None and (
                        (fid + 1) % progress_step == 0 or fid == last_frame):
                    progress_callback(fid + 1, total_to_process)
            return VideoTrialPredictions(pair.stem, tuple(results))
        finally:
            cap.release()


class PredictionAligner:
    """Align annotated frames; include observed detections only, not Kalman ghosts."""
    def align(self, ground_truth: tuple[GroundTruthObservation, ...],
              predictions: VideoTrialPredictions):
        truth_by_frame = defaultdict(list)
        predictions_by_frame = defaultdict(list)
        for truth in ground_truth:
            truth_by_frame[truth.frame_index].append(truth)
        for predicted in predictions.observations:
            if predicted.is_observed:
                predictions_by_frame[predicted.frame_index].append(predicted)
        return tuple((fid, tuple(truth_by_frame[fid]), tuple(predictions_by_frame[fid]))
                     for fid in sorted(truth_by_frame))


class DetectionMatcher:
    """One-to-one assignment of GT/predicted centroids at each annotated frame."""
    def match(self, truth_at_frame, predictions_at_frame,
              distance_threshold_px: float, frame_index: int = 0) -> FrameMatchResult:
        truths, predictions = tuple(truth_at_frame), tuple(predictions_at_frame)
        if not truths or not predictions:
            return FrameMatchResult(frame_index, (), truths, predictions)
        positions_t = np.array([(t.x, t.y) for t in truths])
        positions_p = np.array([(p.x, p.y) for p in predictions])
        dist = np.linalg.norm(positions_t[:, None, :] - positions_p[None, :, :], axis=2)
        # Pad with dummy rows/columns so unmatched nodes cost slightly more
        # than an allowed match; impossible matches have much larger cost.
        n, m = len(truths), len(predictions)
        gate = float(distance_threshold_px)
        size = n + m
        cost = np.full((size, size), gate + 1.0, dtype=np.float64)
        cost[:n, :m] = np.where(dist <= gate, dist, 4 * (gate + 1))
        cost[n:, m:] = 0.0
        row_idx, col_idx = _hungarian_solve(cost)
        matched = []
        mt, mp = set(), set()
        for i, j in zip(row_idx, col_idx):
            if i < n and j < m and dist[i, j] <= gate:
                matched.append(MatchedAntPair(truths[i], predictions[j], float(dist[i, j])))
                mt.add(int(i))
                mp.add(int(j))
        return FrameMatchResult(frame_index, tuple(matched),
                                tuple(t for i, t in enumerate(truths) if i not in mt),
                                tuple(p for i, p in enumerate(predictions) if i not in mp))


class AccuracyScorer:
    """Score ant DETECTION only, independent of object tracking identity.

    A true positive is one one-to-one spatial match between a predicted ant
    center and a labeled ant center within --match-distance source pixels.
    Any unmatched prediction is a false positive; any unmatched label is a
    false negative. Ground-truth ant_id and the track's predicted_id NEVER
    enter the objective, so ID switches, reassignments, and differing numbers
    do not directly affect optimization. Each annotated frame is evaluated
    separately; micro-averaging gives every labeled ant the same weight.

    Only actual detected/matched centroids are scored. Kalman-only estimates
    of an invisible ant are excluded upstream by PredictionAligner, since the
    goal is DETECTION rather than maintaining a track without a measurement.
    """
    def score_video(self, stem: str, frame_matches) -> VideoAccuracyMetrics:
        tp = fp = fn = 0
        distances = []
        for result in frame_matches:
            tp += len(result.matches)
            fp += len(result.unmatched_predictions)
            fn += len(result.unmatched_truth)
            distances.extend(match.distance_pixels for match in result.matches)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
        return VideoAccuracyMetrics(
            stem=stem, true_positives=tp, false_positives=fp,
            false_negatives=fn, detection_f1=float(f1),
            mean_position_error_px=(float(np.mean(distances)) if distances else None),
        )

    def aggregate(self, per_video_metrics: tuple[VideoAccuracyMetrics, ...]) -> float:
        """Micro-average F1 across ALL annotated frames in ALL training videos.

        Optuna maximizes the detection F1 exactly, with no ID consistency
        bonus or penalty. Returning 0 when no detections/labels occur avoids
        falsely optimizing an entirely empty annotation dataset.
        """
        tp = sum(v.true_positives for v in per_video_metrics)
        fp = sum(v.false_positives for v in per_video_metrics)
        fn = sum(v.false_negatives for v in per_video_metrics)
        return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0


class OptimizationResultWriter:
    """Write a plain numeric JSON preset compatible with --settings and sliders.

    Other files (metrics, trial history, SQLite) are kept in the dataset root.
    Atomic JSON replacement avoids leaving a partially written settings file.
    """
    @staticmethod
    def _atomic_json(path: Path, payload):
        temporary = path.with_suffix(path.suffix + '.tmp')
        try:
            with temporary.open('w', encoding='utf-8') as fp:
                json.dump(payload, fp, indent=2, allow_nan=False)
                fp.write('\n')
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def save(self, study, output_directory: Path, optimizer) -> None:
        if not study.best_trials:
            raise ValueError('No completed trials; optimized settings cannot be written')
        best = study.best_trial
        # Save FULL set of current configurable values, not just sampled keys.
        settings = {k: optimizer.base_settings[k] for k in OPTIMIZER_SETTINGS_KEYS}
        settings.update(best.params)
        self._atomic_json(output_directory / 'ant_tuning_parameters.json', settings)
        # A human-readable summary and complete CSV history are separate from
        # the plain preset so --settings can parse it without special handling.
        self._atomic_json(output_directory / 'optuna_results.json', {
            'best_trial': best.number,
            'best_objective_score': best.value,
            'best_trial_params': best.params,
            'settings_path': 'ant_tuning_parameters.json',
            'objective': 'Detection-only micro-averaged F1 (spatial matching, no identity score)',
            'match_distance_px': optimizer.match_distance_px,
            'ground_truth_coordinate_system': 'original-resolution pixels',
            'frame_numbering': 'zero-based',
            'n_complete_trials': sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
            'seed': optimizer.seed,
            'training_videos': [pair.stem for pair in optimizer.dataset.pairs],
            'per_video_metrics': best.user_attrs.get('per_video_metrics', []),
        })
        history = output_directory / 'optuna_trials.csv'
        with history.open('w', newline='', encoding='utf-8') as fp:
            fieldnames = ['trial', 'state', 'value', *sorted(OPTIMIZER_SETTINGS_KEYS)]
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for trial in study.trials:
                writer.writerow({'trial': trial.number, 'state': trial.state.name,
                                 'value': trial.value if trial.state == optuna.trial.TrialState.COMPLETE else '',
                                 **trial.params})


class AntOptunaOptimizer:
    """Coordinate reproducible sequential trials and save best runtime preset.

    Persistent SQLite allows resuming the same study. Ground truths are cached
    because annotations do not change between parameter trials. All source
    settings are restored after each evaluation, preventing trial leakage.
    """
    def __init__(self, dataset: OptimizationDataset, n_trials: int = 30,
                 match_distance_px: float = 25.0, seed: int = 42,
                 code_version: str | None = None, parameters_file: str | None = None):
        self.dataset = dataset
        self.code_version = code_version
        self.parameters_file = parameters_file
        self.n_trials = int(n_trials)
        self.match_distance_px = float(match_distance_px)
        self.seed = int(seed)
        self.ground_truth_reader = GroundTruthReader()
        self.parameter_sampler = OptunaParameterSampler()
        self.trial_runner = TrackingTrialRunner()
        self.prediction_aligner = PredictionAligner()
        self.matcher = DetectionMatcher()
        self.accuracy_scorer = AccuracyScorer()
        self.result_writer = OptimizationResultWriter()
        self.base_settings = {k: globals()[k] for k in OPTIMIZER_SETTINGS_KEYS}
        # Fail before launching a long study on a broken CSV/video pair.
        _opt_progress(f'Loading annotations for {len(dataset.pairs)} video(s)...')
        self.ground_truth = {}
        for pair in dataset.pairs:
            loaded = self.ground_truth_reader.load(pair)
            self.ground_truth[pair.stem] = loaded
            _opt_progress(f'  Loaded {pair.stem}: {len(loaded)} labeled ants, '
                          f'{len({g.frame_index for g in loaded})} annotated frames')

    def create_study(self):
        """Resume or create a seeded maximize-study persisted in folder root."""
        storage_path = self.dataset.root / 'ant_optuna_study.sqlite3'
        storage = optuna.storages.RDBStorage(url=f'sqlite:///{storage_path.as_posix()}')
        study = optuna.create_study(study_name='ant_detection_accuracy_v2',
                                    storage=storage, direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=self.seed),
                                    load_if_exists=True)
        # Avoid resuming studies with a different dataset or scoring radius.
        signature = json.dumps({
            'pairs': [(p.video_path.name, p.ground_truth_path.name,
                       p.video_path.stat().st_size, p.ground_truth_path.stat().st_size,
                       p.video_path.stat().st_mtime_ns, p.ground_truth_path.stat().st_mtime_ns)
                      for p in self.dataset.pairs],
            'match_distance_px': self.match_distance_px,
            'objective_version': 2,
            'scoring': 'observed_ant_detection_f1_no_ids',
        }, sort_keys=True)
        previous_signature = study.user_attrs.get('dataset_signature')
        if previous_signature is not None and previous_signature != signature:
            raise ValueError('Existing Optuna database was created for different data/scoring. '
                             'Move ant_optuna_study.sqlite3 out of the folder to start a fresh study.')
        study.set_user_attr('dataset_signature', signature)
        return study

    def objective(self, trial) -> float:
        """Run every video with sampled settings and score annotated frames."""
        trial_start = time.monotonic()
        parameters = self.parameter_sampler.sample(trial)
        metrics = []
        _opt_progress(f'Trial {trial.number + 1}: START ({len(self.dataset.pairs)} videos)')
        with self.parameter_sampler.apply_for_trial(parameters):
            for video_index, pair in enumerate(self.dataset.pairs, start=1):
                observations = self.ground_truth[pair.stem]
                frames = tuple(sorted({g.frame_index for g in observations}))
                video_start = time.monotonic()
                _opt_progress(f'  Trial {trial.number + 1} | Video {video_index}/'
                              f'{len(self.dataset.pairs)}: {pair.video_path.name} | '
                              f'{len(frames)} annotated frames')
                last_reported_percent = -1

                def video_progress(current, total):
                    nonlocal last_reported_percent
                    # Display percent milestones (roughly every 10%), not every frame.
                    percent = min(100, int(current * 100 / total))
                    bucket = percent // 10
                    if bucket > last_reported_percent:
                        last_reported_percent = bucket
                        _opt_progress(f'    {pair.stem}: {percent:3d}% '
                                      f'({current:,}/{total:,} frames) | '
                                      f'elapsed {_opt_elapsed(time.monotonic() - video_start)}')

                predicted = self.trial_runner.run_video(
                    pair, parameters, frames, progress_callback=video_progress)
                aligned = self.prediction_aligner.align(observations, predicted)
                frame_results = tuple(self.matcher.match(truth, prediction,
                                          self.match_distance_px, frame)
                                      for frame, truth, prediction in aligned)
                metric = self.accuracy_scorer.score_video(pair.stem, frame_results)
                metrics.append(metric)
                _opt_progress(f'  Completed {pair.stem} | F1={metric.detection_f1:.4f} | '
                              f'elapsed {_opt_elapsed(time.monotonic() - video_start)}')
        trial.set_user_attr('per_video_metrics', [asdict(v) for v in metrics])
        score = self.accuracy_scorer.aggregate(tuple(metrics))
        _opt_progress(f'Trial {trial.number + 1}: DONE | detection F1={score:.5f} | '
                      f'elapsed {_opt_elapsed(time.monotonic() - trial_start)}')
        return score

    def execute_study(self) -> None:
        """Optimize n additional trials and export best preset/metrics/history."""
        study = self.create_study()
        completed_before = sum(t.state == optuna.trial.TrialState.COMPLETE
                               for t in study.trials)
        _opt_progress(f'Study ready | {len(self.dataset.pairs)} video(s), '
                      f'{self.n_trials} new trials, {completed_before} previous completed trials')
        _opt_progress('Objective: ANT DETECTION F1 ONLY (ground-truth IDs ignored)')
        _opt_progress(f'Matching radius: {self.match_distance_px} source pixels')
        _opt_progress(f'Study database: {self.dataset.root / "ant_optuna_study.sqlite3"}')
        started = time.monotonic()
        finished_this_run = 0

        def on_trial_complete(study, trial):
            nonlocal finished_this_run
            finished_this_run += 1
            duration = time.monotonic() - started
            remaining = max(0, self.n_trials - finished_this_run)
            eta = remaining * (duration / finished_this_run)
            score = f'{trial.value:.5f}' if trial.value is not None else trial.state.name
            _opt_progress(f'PROGRESS {finished_this_run}/{self.n_trials} '
                          f'({100 * finished_this_run / self.n_trials:.1f}%) | '
                          f'last={score} | best={study.best_value:.5f} '
                          f'(trial {study.best_trial.number + 1}) | '
                          f'elapsed {_opt_elapsed(duration)} | ETA {_opt_elapsed(eta)}')

        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1,
                       gc_after_trial=True, callbacks=[on_trial_complete])
        _opt_progress('Saving optimized parameters and trial history...')
        run_dir = _unique_run_folder(self.dataset.root, 'optuna')
        self.result_writer.save(study, run_dir, self)
        write_run_metadata(run_dir, build_run_metadata(
            'optimize', self.code_version, self.parameters_file, {
                'study_database': str(self.dataset.root / 'ant_optuna_study.sqlite3'),
                'trials_this_run': self.n_trials, 'match_distance_px': self.match_distance_px,
                'seed': self.seed,
                'videos': [str(p.video_path) for p in self.dataset.pairs],
                'ground_truth': [str(p.ground_truth_path) for p in self.dataset.pairs],
                'note': 'parameters.values_used are the BASE settings; the tuned '
                        'values are in ant_tuning_parameters.json'}))
        _opt_progress(f'FINISHED | best objective={study.best_value:.5f} | '
                      f'total elapsed {_opt_elapsed(time.monotonic() - started)}')
        _opt_progress(f'Results folder: {run_dir}')
        _opt_progress(f'Best settings: {run_dir / "ant_tuning_parameters.json"}')
        _opt_progress(f'Detailed metrics: {run_dir / "optuna_results.json"}')
        _opt_progress(f'Trial history: {run_dir / "optuna_trials.csv"}')
        _opt_progress(f'Run metadata: {run_dir / "run_metadata.json"}')

    def run(self) -> None:
        _opt_progress(f'Optimization dataset: {self.dataset.root}')
        for pair in self.dataset.pairs:
            _opt_progress(f'  {pair.video_path.name} <-> {pair.ground_truth_path.name} '
                          f'({pair.annotation_count} labeled ants)')
        self.execute_study()


def run_optimization_mode(folder: str | Path, n_trials: int = 30,
                          match_distance_px: float = 25.0, seed: int = 42,
                          ground_truth_dir: str | Path | None = None,
                          videos_dir: str | Path | None = None,
                          output_dir: str | Path | None = None,
                          code_version: str | None = None,
                          parameters_file: str | None = None) -> None:
    """CLI optimization entry point. No Tkinter GUI or preview is opened."""
    _require_optuna()
    if n_trials < 1:
        raise ValueError('--trials must be >= 1')
    if not math.isfinite(match_distance_px) or match_distance_px <= 0:
        raise ValueError('--match-distance must be a positive finite pixel distance')
    dataset = load_optimization_dataset(folder, ground_truth_dir, videos_dir, output_dir)
    AntOptunaOptimizer(dataset, n_trials, match_distance_px, seed,
                       code_version, parameters_file).run()




# -----------------------------------------------------------------------------
# BATCH COUNT REGRESSION: compare entry/exit totals to labeled truth per quadrant.
# This evaluates counts; it does NOT fit a statistical regression model or
# optimize any settings. Optuna remains a separate mode.
# -----------------------------------------------------------------------------

def _regression_read_truth(csv_path: Path, selected_quadrant: str | None):
    """Accept per-quadrant counts (long/wide) or individual crossing events.

    Supported long format: quadrant,ins,outs  (or enter,exit)
    Supported wide format: NE_in,NE_out,SE_in,SE_out,...
    Event format: event,quadrant, where event is enter/exit or enter_NE/exit_NE.
    A one-quadrant file with ins,outs needs --quadrant.
    Omitted quadrants are intentionally NOT scored as zero.
    """
    with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f'{csv_path}: CSV has no header')
        rows = [{str(k).strip().lower(): str(v or '').strip() for k, v in raw.items()
                 if k is not None} for raw in reader]
    if not rows:
        raise ValueError(f'{csv_path}: ground-truth CSV is empty')
    columns = set(rows[0]); truth = {}

    def count(raw, label):
        try:
            num = float(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{csv_path}: invalid count {label}={raw!r}') from exc
        if not math.isfinite(num) or num < 0 or not num.is_integer():
            raise ValueError(f'{csv_path}: {label} must be a nonnegative integer')
        return int(num)

    def value(row, aliases):
        for alias in aliases:
            if alias in row and row[alias] != '':
                return row[alias]
        return None

    in_names = ('ins', 'in', 'enter', 'enters', 'entries', 'enter_count', 'in_count')
    out_names = ('outs', 'out', 'exit', 'exits', 'departures', 'exit_count', 'out_count')
    if 'event' in columns:
        for row in rows:
            raw_event = row.get('event', '').lower().replace('-', '_').strip()
            quad = row.get('quadrant', '').upper()
            if raw_event.startswith(('enter_', 'exit_')):
                direction, embedded_quad = raw_event.split('_', 1)
                if quad and quad != embedded_quad.upper():
                    raise ValueError(f'{csv_path}: event/quadrant mismatch: {row}')
                quad = embedded_quad.upper()
            elif raw_event in ('enter', 'entry', 'in'):
                direction = 'enter'
            elif raw_event in ('exit', 'out'):
                direction = 'exit'
            else:
                raise ValueError(f'{csv_path}: unsupported event {raw_event!r}')
            if quad not in _QUADRANTS:
                raise ValueError(f'{csv_path}: invalid/missing quadrant {quad!r}')
            counts = truth.setdefault(quad, {'enter': 0, 'exit': 0})
            counts[direction] += 1
    elif 'quadrant' in columns or (columns & set(in_names) and columns & set(out_names)):
        for row in rows:
            quad = (row.get('quadrant') or selected_quadrant or '').upper()
            if quad not in _QUADRANTS:
                raise ValueError(f'{csv_path}: include quadrant column or specify --quadrant NE/SE/SW/NW')
            entered = value(row, in_names); exited = value(row, out_names)
            if entered is None or exited is None:
                raise ValueError(f'{csv_path}: each count row needs both ins and outs columns')
            if quad in truth:
                raise ValueError(f'{csv_path}: duplicate quadrant {quad} in count table')
            truth[quad] = {'enter': count(entered, 'ins'), 'exit': count(exited, 'outs')}
    else:
        row = rows[0]
        if len(rows) != 1:
            raise ValueError(f'{csv_path}: wide-format CSV must have one row')
        for quad in _QUADRANTS:
            q = quad.lower()
            entered = value(row, (f'{q}_in', f'{q}_ins', f'{q}_enter', f'{q}_entries', f'{q}_enter_count'))
            exited = value(row, (f'{q}_out', f'{q}_outs', f'{q}_exit', f'{q}_exits', f'{q}_exit_count'))
            if entered is not None or exited is not None:
                if entered is None or exited is None:
                    raise ValueError(f'{csv_path}: both {quad} in/out columns are required')
                truth[quad] = {'enter': count(entered, f'{quad}_in'),
                               'exit': count(exited, f'{quad}_out')}
    if selected_quadrant:
        if selected_quadrant not in truth:
            raise ValueError(f'{csv_path}: missing ground truth for {selected_quadrant}')
        truth = {selected_quadrant: truth[selected_quadrant]}
    if not truth:
        raise ValueError(f'{csv_path}: no recognized quadrant counts or events')
    return truth


def _regression_circle_lookup(video: Path, gt_directory: Path,
                              circle_file: str | None) -> 'CircleLookup':
    """Find a regression video's circle without raising.

    Order: --circle-file (every video) -> next to the video -> ground-truth
    folder -> <videos>/circles/. The first file found decides.
    """
    if circle_file:
        candidates = [('shared', Path(circle_file).expanduser().resolve())]
    else:
        name = f'{video.stem}_circle.json'
        candidates = [('video folder', video.with_name(name)),
                      ('ground-truth folder', gt_directory / name),
                      ('circles/ folder', video.parent / 'circles' / name)]
    for source, candidate in candidates:
        if candidate.is_file():
            circle, reason = _read_circle_file(candidate)
            return CircleLookup('ok' if circle else 'invalid', circle, candidate, source, reason)
    if circle_file:
        return CircleLookup('invalid', None, candidates[0][1], 'shared',
                            f'--circle-file not found: {candidates[0][1]}')
    return CircleLookup('missing', reason=(f'no {video.stem}_circle.json next to the video, '
                                           'in the ground-truth folder or in circles/'))


def _regression_circle(video: Path, gt_directory: Path, circle_file: str | None):
    """Raising wrapper: (circle, path) or ValueError."""
    lookup = _regression_circle_lookup(video, gt_directory, circle_file)
    if lookup.status == 'ok':
        return lookup.circle, lookup.path
    raise ValueError(f'{video.name}: {lookup.reason}')


def plan_regression(ground_truth: str, videos_dir: str, quadrant: str | None = None,
                    circle_file: str | None = None) -> dict:
    """Pre-run check for regression (used by the CLI and the GUI).

    Matches videos (top level of videos_dir) to ground-truth titles and finds
    each matched video's circle. Nothing is processed. Raises ValueError for
    unusable inputs (missing folder, no videos, unreadable ground truth).
    """
    gt_path = Path(ground_truth).expanduser().resolve()
    vid_dir = Path(videos_dir).expanduser().resolve()
    if not vid_dir.is_dir():
        raise ValueError(f'--videos is not a directory: {vid_dir}')
    if quadrant is not None:
        quadrant = quadrant.upper()
        if quadrant not in _QUADRANTS:
            raise ValueError('--quadrant must be NE, SE, SW or NW')
    videos = sorted(p for p in vid_dir.iterdir() if p.is_file() and _is_video_file(p))
    if not videos:
        raise ValueError(f'No supported videos found in {vid_dir}')
    labels = _regression_truth_catalog(str(gt_path), quadrant)
    video_catalog = {v.stem.casefold(): v for v in videos}
    gt_directory = gt_path.parent if gt_path.is_file() else gt_path
    pairs = []
    for video in videos:
        key = video.stem.casefold()
        if key not in labels:
            continue
        title, truth, csv_path, colony_id, video_date = labels[key]
        pairs.append({'video': video, 'title': title, 'truth': truth, 'csv_path': csv_path,
                      'colony_id': colony_id, 'video_date': video_date,
                      'circle': _regression_circle_lookup(video, gt_directory, circle_file)})
    return {
        'ground_truth': gt_path, 'videos_dir': vid_dir, 'quadrant': quadrant,
        'videos': videos, 'pairs': pairs,
        'unmatched_videos': sorted(v.name for v in videos if v.stem.casefold() not in labels),
        'unmatched_truth': sorted(item[0] for stem, item in labels.items()
                                  if stem not in video_catalog),
    }


def _regression_truth_catalog(source: str, quadrant: str | None):
    """Read EITHER a consolidated Representative Sample Counts CSV OR per-video CSVs.

    A consolidated CSV has one row per video and columns:
    Video Title, Colony ID, Video Date, then each quadrant's Enter and Exit.
    Colony ID and Video Date are retained as metadata in the regression outputs.
    Video Title may include its extension; names are matched case-insensitively.
    Per-video CSV mode retains prior long, wide, or crossing-event formats.
    Missing quadrant columns are not silently treated as zero.
    """
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise ValueError(f'Ground-truth file/directory does not exist: {source_path}')
    if source_path.is_file():
        files = [source_path]
        if source_path.suffix.lower() != '.csv':
            raise ValueError(f'Ground-truth file must be a CSV: {source_path}')
    else:
        files = sorted(f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() == '.csv')
        if not files:
            raise ValueError(f'No ground-truth CSV files in {source_path}')

    catalog = {}
    for csv_file in files:
        with csv_file.open('r', newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            columns = {c.strip().lower().replace('_', ' ') for c in (reader.fieldnames or [])}
            if 'video title' in columns:
                # Consolidated annotated sample count table.
                required_metadata = {'video title', 'colony id', 'video date'}
                missing_metadata = required_metadata - columns
                if missing_metadata:
                    raise ValueError(
                        f'{csv_file}: consolidated regression CSV is missing required '
                        f'metadata columns: {", ".join(sorted(missing_metadata))}. '
                        'Expected Video Title, Colony ID, Video Date, followed by quadrant Enter/Exit columns.')
                for raw in reader:
                    row = {str(k).strip().lower().replace('_', ' '): (v or '').strip()
                           for k, v in raw.items() if k is not None}
                    video_title = row.get('video title', '')
                    if not video_title:
                        raise ValueError(f'{csv_file}: blank Video Title')
                    colony_id = row.get('colony id', '')
                    video_date = row.get('video date', '')
                    stem = Path(video_title).stem.casefold()
                    if stem in catalog:
                        raise ValueError(f'Duplicate ground-truth Video Title: {video_title!r}')
                    truth = {}
                    for quad in _QUADRANTS:
                        e_key, x_key = f'{quad.lower()} enter', f'{quad.lower()} exit'
                        entered, exited = row.get(e_key), row.get(x_key)
                        if entered in (None, '') and exited in (None, ''):
                            continue  # No annotation for this quadrant.
                        if entered in (None, '') or exited in (None, ''):
                            raise ValueError(f'{csv_file}: {video_title} requires both {quad} Enter/Exit')
                        def parse_count(v):
                            n = float(v)
                            if not math.isfinite(n) or n < 0 or not n.is_integer():
                                raise ValueError(f'Invalid ground-truth count {v!r} for {video_title}')
                            return int(n)
                        truth[quad] = {'enter': parse_count(entered), 'exit': parse_count(exited)}
                    if quadrant:
                        truth = {quadrant: truth[quadrant]} if quadrant in truth else {}
                    if truth:
                        catalog[stem] = (video_title, truth, csv_file, colony_id, video_date)
            else:
                # Existing per-video CSV format.
                stem = csv_file.stem.casefold()
                if stem in catalog:
                    raise ValueError(f'Duplicate ground-truth CSV stem: {csv_file.stem}')
                catalog[stem] = (csv_file.stem, _regression_read_truth(csv_file, quadrant), csv_file, '', '')
    if not catalog:
        raise ValueError(f'No usable labeled quadrant counts in {source_path}')
    return catalog


def _regression_statistics(comparison: list[dict]) -> dict:
    """Ordinary-least-squares y=predicted count vs x=ground-truth count.

    As in the provided regression_analysis.py reference, fitted RMSE is against
    the regression line, while MAE and squared error compare to ideal y=x.
    Count RMSE against y=x is additionally included to avoid ambiguity.
    Constant x or fewer than two points => linear regression undefined.
    """
    from scipy import stats
    x = np.array([r['ground_truth'] for r in comparison], dtype=float)
    y = np.array([r['predicted'] for r in comparison], dtype=float)
    delta = y - x
    valid = len(x) >= 2 and np.ptp(x) > 0
    result = {
        'n': len(x), 'mae': float(np.mean(np.abs(delta))),
        'count_rmse': float(np.sqrt(np.mean(delta ** 2))),
        'total_squared_error': float(np.sum(delta ** 2)),
        'slope': None, 'intercept': None, 'r': None, 'r2': None,
        'p_value': None, 'std_err': None, 'rmse': None,
        'fit_note': None,
    }
    if not valid:
        result['fit_note'] = 'OLS undefined: need at least two observations and varying ground-truth counts.'
        return result
    # SciPy raises if all x are identical; handled above.
    fit = stats.linregress(x, y)
    slope, intercept = float(fit.slope), float(fit.intercept)
    y_hat = slope * x + intercept
    result.update({
        'slope': slope, 'intercept': intercept,
        'r': float(fit.rvalue) if math.isfinite(fit.rvalue) else None,
        'r2': float(fit.rvalue ** 2) if math.isfinite(fit.rvalue) else None,
        'p_value': float(fit.pvalue) if math.isfinite(fit.pvalue) else None,
        'std_err': float(fit.stderr) if math.isfinite(fit.stderr) else None,
        'rmse': float(np.sqrt(np.mean((y - y_hat) ** 2))),
    })
    return result


def _regression_plot(comparison: list[dict], result: dict, destination: Path, show: bool):
    """Reference-style two-panel regression/squared-error figure, saved as PNG.

    Use a noninteractive backend for batch/server execution unless the user
    explicitly requests --show-plot (which needs an active display).
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'NE': '#e6b800', 'SE': '#2ca02c', 'SW': '#1f77b4', 'NW': '#9467bd'}
    directions = {'enters': 'o', 'exits': '^'}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.array([r['ground_truth'] for r in comparison], dtype=float)
    y = np.array([r['predicted'] for r in comparison], dtype=float)
    sq = (y - x) ** 2
    for ax_idx, ax in enumerate(axes):
        for quad in _QUADRANTS:
            for direction, marker in directions.items():
                subset = [r for r in comparison if r['quadrant'] == quad and r['direction'] == direction]
                if not subset:
                    continue
                xx = [r['ground_truth'] for r in subset]
                yy = [r['predicted'] if ax_idx == 0 else r['squared_error'] for r in subset]
                ax.scatter(xx, yy, color=colors[quad], marker=marker, s=70,
                           zorder=3, label=f'{quad} {direction}')
                for r in subset:
                    v = r['predicted'] if ax_idx == 0 else r['squared_error']
                    ax.annotate(r['stem'], (r['ground_truth'], v), fontsize=6,
                                xytext=(4, 3), textcoords='offset points', color=colors[quad])
        padding = max(float(np.ptp(x)) * .12, .5)
        ax.set_xlim(float(x.min()) - padding, float(x.max()) + padding)
        if ax_idx == 0:
            ymin = float(min(x.min(), y.min())) - padding
            ymax = float(max(x.max(), y.max())) + padding
            ax.set_ylim(ymin, ymax)
            lim = np.array([min(ymin, float(x.min()) - padding),
                            max(ymax, float(x.max()) + padding)])
            ax.plot(lim, lim, 'k--', lw=1, alpha=.4, label='perfect (y=x)')
            if result['slope'] is not None:
                ax.plot(lim, result['slope'] * lim + result['intercept'],
                        color='tomato', lw=2,
                        label=f"fit: y={result['slope']:.2f}x+{result['intercept']:.2f}")
            ax.set_ylabel('Predicted count')
            ax.set_title('Predicted vs Ground Truth')
            annotation = (f"R² = {result['r2']:.3f}" if result['r2'] is not None else 'R² = N/A')
            annotation += (f"\nRMSE(fit) = {result['rmse']:.2f}" if result['rmse'] is not None else '\nRMSE(fit) = N/A')
            annotation += f"\nMAE = {result['mae']:.2f}\nn = {result['n']}"
            ax.text(.05, .95, annotation, transform=ax.transAxes, fontsize=9,
                    va='top', bbox=dict(boxstyle='round,pad=.4', facecolor='white', alpha=.8))
        else:
            padding_y = max(float(sq.max()) * .08, .1)
            ax.set_ylim(-padding_y, float(sq.max()) + padding_y)
            ax.axhline(0, color='gray', lw=.8, ls='--')
            ax.set_ylabel('Squared error (predicted − ground truth)²')
            ax.set_title('Squared Error vs Ground Truth')
        ax.set_xlabel('Ground truth count')
        ax.legend(fontsize=7, ncol=2, loc='upper left',
                  bbox_to_anchor=(0, .85) if ax_idx == 0 else (0, 1))
    fig.suptitle('Regression analysis — enters ○  exits △  |  colour = quadrant',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def _git_metadata(source_file: Path) -> dict:
    """Capture the source revision used for this run; gracefully handle non-git copies."""
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(source_file.resolve().parent), *args],
            check=True, capture_output=True, text=True, timeout=5
        ).stdout.strip()

    metadata = {"repository_detected": False, "commit_hash": None,
                "commit_short_hash": None, "commit_date": None,
                "branch": None, "working_tree_dirty": None}
    try:
        metadata.update({
            "repository_detected": True,
            "commit_hash": git("rev-parse", "HEAD"),
            "commit_date": git("show", "-s", "--format=%cI", "HEAD"),
            "branch": git("branch", "--show-current") or None,
            "working_tree_dirty": bool(git("status", "--porcelain")),
        })
        metadata["commit_short_hash"] = metadata["commit_hash"][:12]
    except (subprocess.SubprocessError, OSError):
        metadata["git_note"] = "Git metadata unavailable (not a Git checkout or Git command failed)."
    try:
        metadata["source_file_sha256"] = hashlib.sha256(source_file.read_bytes()).hexdigest()
    except OSError:
        metadata["source_file_sha256"] = None
    return metadata


def _runtime_parameter_snapshot() -> dict:
    """Return the effective module-level parameter values, including defaults."""
    keys = list(TUNE_SPECS) + ["PROCESS_SCALE", "BUFFER_ZONE",
                               "CROSS_HYSTERESIS_FRAMES", "PATH_CONFIRM_FRAMES"]
    return {key: globals()[key] for key in keys if key in globals()}


def _timestamp_slug(moment: datetime | None = None) -> str:
    """Local date-time for folder names, e.g. 2026-11-25_14-53-07."""
    return (moment or datetime.now()).strftime('%Y-%m-%d_%H-%M-%S')


def _unique_run_folder(root: str | Path, prefix: str) -> Path:
    """Create <root>/<prefix>_<date-time>/ (with _2, _3… if it exists) so a run
    never overwrites an earlier run's files."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = root / f'{prefix}_{_timestamp_slug()}'
    folder, n = base, 2
    while folder.exists():
        folder, n = Path(f'{base}_{n}'), n + 1
    folder.mkdir(parents=True)
    return folder


def detect_code_version() -> str:
    """Best guess of the code version from git (tag or commit, '-dirty' when
    there are uncommitted changes); '' if this is not a git checkout."""
    try:
        return subprocess.run(
            ['git', '-C', str(Path(__file__).resolve().parent), 'describe',
             '--tags', '--always', '--dirty'],
            check=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ''


def build_run_metadata(mode: str, code_version: str | None,
                       parameters_file: str | None, extra: dict | None = None) -> dict:
    """Provenance for one run: date, the GitHub version the user entered, git
    details, and the exact parameter values in effect for this run."""
    now = datetime.now().astimezone()
    script_path = Path(__file__).resolve()
    settings_path = Path(parameters_file).expanduser().resolve() if parameters_file else None
    settings_sha256 = None
    if settings_path and settings_path.is_file():
        settings_sha256 = hashlib.sha256(settings_path.read_bytes()).hexdigest()
    payload = {
        'metadata_schema_version': 2,
        'mode': mode,
        'date': now.strftime('%Y-%m-%d'),
        'time': now.strftime('%H:%M:%S'),
        'timezone': now.tzname(),
        'run_started_at_local': now.isoformat(),
        'run_started_at_utc': now.astimezone(timezone.utc).isoformat(),
        'code_version': code_version or 'not provided',
        'git_detected': _git_metadata(script_path),
        'script': {'file_name': script_path.name, 'absolute_path': str(script_path),
                   'source_file_sha256': hashlib.sha256(script_path.read_bytes()).hexdigest()},
        'command_line': sys.argv,
        'parameters': {'file': str(settings_path) if settings_path else None,
                       'file_sha256': settings_sha256,
                       'values_used': _runtime_parameter_snapshot()},
    }
    payload.update(extra or {})
    return payload


def write_run_metadata(folder: Path, payload: dict, name: str = 'run_metadata.json') -> Path:
    destination = Path(folder) / name
    destination.write_text(json.dumps(payload, indent=2, default=str) + '\n', encoding='utf-8')
    return destination


def _write_regression_run_metadata(output: Path, parameters_file: str | None,
                                   videos: list[Path], pairs: list[tuple],
                                   circle_file: str | None, render_video: bool,
                                   code_version: str | None = None) -> Path:
    """Write provenance beside regression metrics for reproducible comparisons."""
    payload = build_run_metadata('regression', code_version, parameters_file, {
        'execution': {'videos_directory': str(Path(videos[0]).parent),
                      'video_count_found': len(videos), 'video_count_evaluated': len(pairs),
                      'render_video': bool(render_video),
                      'shared_circle_file': (str(Path(circle_file).expanduser().resolve())
                                             if circle_file else None)},
        'videos': [{'video': str(item[0]), 'ground_truth_title': item[1],
                    'colony_id': item[6], 'video_date': item[7],
                    'ground_truth_csv': str(item[5]), 'circle_file': str(item[4])}
                   for item in pairs],
    })
    return write_run_metadata(output, payload, 'regression_run_metadata.json')


def run_regression_mode(ground_truth_dir: str, videos_dir: str,
                        parameters_file: str | None = None,
                        quadrant: str | None = None, circle_file: str | None = None,
                        output_dir: str | None = None, render_video: bool = False,
                        show_plot: bool = False, skip_missing_circles: bool = False,
                        progress_cb=None, log_cb=None,
                        cancel_event: threading.Event | None = None,
                        code_version: str | None = None):
    """Run tracker on videos, then calculate the exact regression from the reference.

    Ground truth: one consolidated CSV with Video Title and quadrant Enter/Exit
    columns, OR a directory of per-video CSV count files.
    Outputs: regression_results.csv (long form), regression_comparison.csv,
    regression_summary.json (OLS stats), regression_plot.png and event CSVs.
    Only video titles matching actual video stems are scored. Warnings list any
    unmatched videos/labels rather than assuming missing labels mean zero.
    Matched videos without a valid circle stop the run (all are listed), unless
    skip_missing_circles, which skips and reports them instead.
    progress_cb(video_index, video_total, percent, message), log_cb(message)
    and cancel_event are used by the GUI; the CLI prints to the terminal.
    """
    log = log_cb or (lambda message: print(message, flush=True))
    plan = plan_regression(ground_truth_dir, videos_dir, quadrant, circle_file)
    gt_path, vid_dir, quadrant = plan['ground_truth'], plan['videos_dir'], plan['quadrant']
    videos = plan['videos']
    params = load_tune_values(parameters_file) if parameters_file else {}
    # Every run gets its own folder: <output>/regression_<date-time>/
    output_root = (Path(output_dir).expanduser().resolve() if output_dir
                   else vid_dir / 'regression_results')
    unmatched_videos, unmatched_truth = plan['unmatched_videos'], plan['unmatched_truth']
    for name in unmatched_videos:
        log(f'WARNING: no ground truth for video {name}; skipping')
    for name in unmatched_truth:
        log(f'WARNING: ground truth has no matching video {name}; skipping')
    bad = [p for p in plan['pairs'] if p['circle'].status != 'ok']
    if bad and not skip_missing_circles:
        raise ValueError('These matched videos have no valid circle (create them in the GUI, '
                         'pass --circle-file, or use --skip-missing-circles):\n'
                         + '\n'.join(f"  {p['video'].name}: {p['circle'].reason}" for p in bad))
    skipped_no_circle = [{'video': p['video'].name, 'reason': p['circle'].reason} for p in bad]
    for entry in skipped_no_circle:
        log(f"SKIP {entry['video']}: {entry['reason']}")
    pairs = [(p['video'], p['title'], p['truth'], p['circle'].circle, p['circle'].path,
              p['csv_path'], p['colony_id'], p['video_date'])
             for p in plan['pairs'] if p['circle'].status == 'ok']
    if not pairs:
        raise ValueError('No videos to score: none matched the ground truth'
                         + (' with a valid circle' if bad else ''))
    output = _unique_run_folder(output_root, 'regression')
    log(f'Output folder: {output}')
    metadata_path = _write_regression_run_metadata(output, parameters_file, videos, pairs,
                                                   circle_file, render_video, code_version)
    comparison = []
    long_rows = []
    per_video = []
    totals = {q: {'gt_in': 0, 'gt_out': 0, 'pred_in': 0, 'pred_out': 0} for q in _QUADRANTS}
    for index, (video, title, truth, circle, circle_path, truth_path, colony_id, video_date) in enumerate(pairs, 1):
        log(f'[{index}/{len(pairs)}] Processing {video.name} | circle={circle_path.name}')
        def progress(pct, msg, _index=index):
            if progress_cb is not None:
                progress_cb(_index, len(pairs), pct, msg)
            elif pct > 0 and 'Frame ' in msg:
                log(f'  {pct:.1f}% {msg}')
        processor = VideoProcessor(str(video), circle, progress, lambda message: None,
                                   out_dir=str(output / 'events'), stem=video.stem,
                                   render_video=render_video, cancel_event=cancel_event)
        events, entered, exited, paths = processor.run()
        if events is None:
            raise RuntimeError(f'Processing cancelled: {video.name}')
        video_rows = []
        for q, expected in truth.items():
            gt_in, gt_out = expected['enter'], expected['exit']
            pred_in, pred_out = int(entered[q]), int(exited[q])
            in_error, out_error = pred_in - gt_in, pred_out - gt_out
            row = {'video': video.name, 'video_title': title,
                   'colony_id': colony_id, 'video_date': video_date, 'quadrant': q,
                   'ground_truth_in': gt_in, 'predicted_in': pred_in, 'in_error': in_error,
                   'in_abs_error': abs(in_error),
                   'ground_truth_out': gt_out, 'predicted_out': pred_out, 'out_error': out_error,
                   'out_abs_error': abs(out_error),
                   'total_abs_error': abs(in_error) + abs(out_error),
                   'exact_match': int(in_error == 0 and out_error == 0)}
            comparison.append(row); video_rows.append(row)
            for direction, gt, predicted in (('enters', gt_in, pred_in), ('exits', gt_out, pred_out)):
                long_rows.append({'stem': video.stem, 'video_title': title,
                                  'colony_id': colony_id, 'video_date': video_date, 'quadrant': q,
                                  'direction': direction, 'predicted': predicted,
                                  'ground_truth': gt, 'squared_error': (predicted - gt)**2,
                                  'absolute_error': abs(predicted - gt)})
            totals[q]['gt_in'] += gt_in; totals[q]['gt_out'] += gt_out
            totals[q]['pred_in'] += pred_in; totals[q]['pred_out'] += pred_out
            log(f'  {q}: IN {pred_in} / GT {gt_in} (error {in_error:+d}); '
                  f'OUT {pred_out} / GT {gt_out} (error {out_error:+d})')
        per_video.append({'video': video.name, 'video_title': title,
                          'colony_id': colony_id, 'video_date': video_date,
                          'circle_file': str(circle_path), 'ground_truth_csv': str(truth_path),
                          'ground_truth': truth, 'comparison': video_rows,
                          'event_csv': paths['csv'], 'annotated_video': paths['video']})
    comparison_path = output / 'regression_comparison.csv'
    with comparison_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0].keys()))
        writer.writeheader(); writer.writerows(comparison)
    csv_path = output / 'regression_results.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0].keys()))
        writer.writeheader(); writer.writerows(long_rows)
    fit = _regression_statistics(long_rows)
    plot_path = output / 'regression_plot.png'
    _regression_plot(long_rows, fit, plot_path, show_plot)
    summary = {'videos_evaluated': len(pairs), 'observations': len(long_rows),
               'linear_regression': fit,
               'mean_absolute_error_per_direction': fit['mae'],
               'total_absolute_error': sum(row['total_abs_error'] for row in comparison),
               'exact_match_rows': sum(row['exact_match'] for row in comparison),
               'unmatched_videos': unmatched_videos, 'unmatched_ground_truth': unmatched_truth,
               'skipped_missing_circle': skipped_no_circle,
               'output_dir': str(output),
               'code_version': code_version or 'not provided',
               'quadrant_totals': {q: v for q, v in totals.items()
                                   if any(row['quadrant'] == q for row in comparison)},
               'settings_file': str(parameters_file) if parameters_file else None,
               'settings_applied': params, 'per_video': per_video,
               'files': {'comparison': str(comparison_path), 'results': str(csv_path),
                         'plot': str(plot_path), 'run_metadata': str(metadata_path),
                         'summary': str(output / 'regression_summary.json')}}
    summary_path = output / 'regression_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    log('\n── Linear regression: predicted ~ ground truth ──')
    for label, key in [('n', 'n'), ('Slope (ideal 1)', 'slope'),
                       ('Intercept (ideal 0)', 'intercept'), ('R²', 'r2'),
                       ('Pearson r', 'r'), ('p-value', 'p_value'),
                       ('RMSE (fitted line)', 'rmse'), ('MAE (vs ideal)', 'mae'),
                       ('RMSE (vs ideal)', 'count_rmse')]:
        v = fit[key]
        log(f'  {label:<22}: {v:.4f}' if isinstance(v, float) else
              f'  {label:<22}: {v if v is not None else "N/A"}')
    if fit['fit_note']:
        log(f"  NOTE: {fit['fit_note']}")
    log(f'\nSaved: {csv_path}\n       {comparison_path}\n       {plot_path}\n       {summary_path}\n       {metadata_path}')
    return summary


# ============================================================================
# SHARED RUN-MODE CORE: video discovery, circle lookup, batch plan + execution.
# Used by BOTH the CLI (--run) and the GUI, so the two cannot drift apart.
# ============================================================================

# Effective parameter values at import time (the source-code defaults).
# Taken before any --parameters file is applied; GUI "Reset to defaults" and
# every GUI run start from this snapshot.
DEFAULT_PARAMETERS = _runtime_parameter_snapshot()


def _is_video_file(path: Path) -> bool:
    """Supported video, excluding hidden/AppleDouble files and our own renders."""
    name = path.name
    return (path.suffix.lower() in OPTIMIZE_VIDEO_EXTENSIONS
            and not name.startswith('.')            # hidden and macOS ._ files
            and not path.stem.lower().endswith('_counted'))


def _as_file_list(video_file) -> list[str]:
    """Normalize --video input: None, one path, or a list of paths."""
    if not video_file:
        return []
    if isinstance(video_file, (str, os.PathLike)):
        return [str(video_file)]
    return [str(v) for v in video_file if v]


def _videos_from_inputs(videos_dir: str | None, video_file,
                        recursive: bool = False,
                        exclude_dir: str | Path | None = None) -> list[Path]:
    """Resolve video file(s) or a folder (tree) into a video list.

    video_file may be one path or a list of paths from any folders (repeated
    --video, or the GUI's video list); duplicates are dropped, order is kept.
    Recursive search skips hidden folders and the output folder (exclude_dir)
    so earlier results are never picked up again. Never opens a dialog.
    """
    files = _as_file_list(video_file)
    if bool(videos_dir) == bool(files):
        raise ValueError('Specify exactly one of --videos DIRECTORY or --video FILE')
    if files:
        if recursive:
            raise ValueError('--recursive applies to a folder (--videos), not to --video files')
        videos, seen = [], set()
        for name in files:
            path = Path(name).expanduser().resolve()
            if not path.is_file() or path.suffix.lower() not in OPTIMIZE_VIDEO_EXTENSIONS:
                raise ValueError(f'Invalid or unsupported video: {path}')
            if path not in seen:
                seen.add(path)
                videos.append(path)
        return videos
    directory = Path(videos_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f'Video directory not found: {directory}')
    exclude = Path(exclude_dir).expanduser().resolve() if exclude_dir else None
    if recursive:
        found = []
        for root, dirs, files in os.walk(directory):
            root_path = Path(root)
            dirs[:] = sorted(
                d for d in dirs
                if not d.startswith('.')
                and not (exclude is not None and (root_path / d).resolve() == exclude))
            found.extend(root_path / f for f in sorted(files) if _is_video_file(root_path / f))
    else:
        found = [p for p in sorted(directory.iterdir()) if p.is_file() and _is_video_file(p)]
    if not found:
        where = 'in or below' if recursive else 'in'
        raise ValueError(f'No supported videos found {where}: {directory}')
    return sorted(found, key=lambda p: str(p.relative_to(directory)).casefold())


@dataclass
class CircleLookup:
    """Result of finding one video's circle: ok, missing, or invalid."""
    status: str                      # 'ok' | 'missing' | 'invalid'
    circle: dict | None = None
    path: Path | None = None
    source: str = ''                 # 'shared', 'video folder', 'output folder', 'preview'
    reason: str = ''


def _read_circle_file(path: Path) -> tuple[dict | None, str]:
    """Parse and validate a circle JSON; return (circle, '') or (None, reason)."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        return None, f'unreadable JSON ({exc.msg}, line {exc.lineno})'
    except (OSError, UnicodeDecodeError) as exc:
        return None, f'cannot read file ({exc})'
    circle = _normalize_circle_params(data)
    if circle is None:
        return None, 'missing center, radius, or north_angle/north_point'
    if not (math.isfinite(circle['radius']) and circle['radius'] > 0):
        return None, f"radius must be positive (got {circle['radius']})"
    return circle, ''


def _batch_circle(video: Path, output_dir: str | Path | None,
                  circle_file: str | None = None) -> CircleLookup:
    """Find a video's circle without raising.

    Order: shared circle_file (if given, used for every video) ->
    <video folder>/<stem>_circle.json -> <output_dir>/<stem>_circle.json.
    output_dir is this video's own output folder (mirrored in recursive runs).
    The first file found decides: an invalid file is reported, not skipped over.
    """
    if circle_file:
        path = Path(circle_file).expanduser().resolve()
        if not path.is_file():
            return CircleLookup('invalid', None, path, 'shared',
                                f'shared circle file not found: {path}')
        circle, reason = _read_circle_file(path)
        return CircleLookup('ok' if circle else 'invalid', circle, path, 'shared', reason)
    name = f'{video.stem}_circle.json'
    candidates = [('video folder', video.parent)]
    if output_dir:
        candidates.append(('output folder', Path(output_dir)))
    for source, folder in candidates:
        path = folder / name
        if path.is_file():
            circle, reason = _read_circle_file(path)
            return CircleLookup('ok' if circle else 'invalid', circle, path, source, reason)
    return CircleLookup('missing', reason=f'no {name} in the video folder or output folder')


@dataclass
class BatchItem:
    """One video in a batch plan, with its circle and output folder."""
    video: Path
    relative: str
    output_dir: Path
    circle: CircleLookup
    conflict: str = ''               # e.g. duplicate output name
    status: str = 'pending'          # pending/running/processed/failed/skipped/cancelled
    message: str = ''
    result: dict | None = None

    @property
    def state(self) -> str:
        """Pre-run readiness: 'ready', 'missing' or 'invalid'."""
        if self.conflict:
            return 'invalid'
        return 'ready' if self.circle.status == 'ok' else self.circle.status

    @property
    def reason(self) -> str:
        return self.conflict or self.circle.reason


@dataclass
class BatchPlan:
    """Everything decided before processing starts (the pre-run check)."""
    input_root: Path
    output_root: Path
    items: list
    recursive: bool = False
    single_video: bool = False
    circle_file: str | None = None

    def counts(self) -> dict:
        states = [item.state for item in self.items]
        return {'total': len(states), 'ready': states.count('ready'),
                'missing': states.count('missing'), 'invalid': states.count('invalid')}

    def summary_line(self) -> str:
        c = self.counts()
        return (f"Found {c['total']} video(s) | {c['ready']} with circles | "
                f"{c['missing']} missing | {c['invalid']} invalid")

    def problem_lines(self) -> list[str]:
        return [f'  {item.state.upper():8s} {item.relative}: {item.reason}'
                for item in self.items if item.state != 'ready']


class BatchAbort(RuntimeError):
    """Batch stopped (strict mode problem, or nothing processable)."""
    def __init__(self, message: str, summary: dict | None = None):
        super().__init__(message)
        self.summary = summary


def default_output_root(videos_dir: str | None, video_file) -> Path:
    """Video file(s) -> ./outputs ; folder -> <folder>/batch_results."""
    if _as_file_list(video_file):
        return Path.cwd() / 'outputs'
    return Path(videos_dir).expanduser().resolve() / 'batch_results'


def plan_batch(videos_dir: str | None = None, video_file=None,
               recursive: bool = False, circle_file: str | None = None,
               output_dir: str | None = None) -> BatchPlan:
    """Discover videos and locate every circle; nothing is processed here.

    Folder input: outputs mirror the folder tree under the output root.
    File input (one or more files from any folders): outputs go directly in
    the output root; two files with the same name are flagged as a conflict.
    """
    output_root = (Path(output_dir).expanduser().resolve() if output_dir
                   else default_output_root(videos_dir, video_file))
    videos = _videos_from_inputs(videos_dir, video_file, recursive, exclude_dir=output_root)
    from_files = bool(_as_file_list(video_file))
    if from_files:
        # Label files relative to their deepest common folder (just the file
        # name when they all share one folder).
        common = Path(os.path.commonpath([str(v.parent) for v in videos]))
        root = common
    else:
        root = Path(videos_dir).expanduser().resolve()
    items = []
    claimed = {}
    for video in videos:
        relative = video.relative_to(root)
        out = output_root if from_files else output_root / relative.parent
        item = BatchItem(video, str(relative), out, _batch_circle(video, out, circle_file))
        # Two videos with the same stem in one output folder would overwrite
        # each other's results (e.g. GX01.MP4 and GX01.mov).
        key = (str(out).casefold(), video.stem.casefold())
        if key in claimed:
            item.conflict = (f'same name as {claimed[key]}; outputs would overwrite it '
                             '(rename one of the videos)')
        else:
            claimed[key] = str(relative)
        items.append(item)
    return BatchPlan(root, output_root, items, recursive, len(videos) == 1 and from_files,
                     circle_file)


def execute_batch(plan: BatchPlan, strict: bool = False, render_video: bool = False,
                  parameters_file: str | None = None, progress_cb=None, video_cb=None,
                  log_cb=print, cancel_event: threading.Event | None = None,
                  code_version: str | None = None, interface: str = 'cli') -> dict:
    """Process every ready video in a plan and write batch_summary.json.

    progress_cb(video_index, video_total, percent, message) - per-video progress.
    video_cb(item) - called whenever an item's status changes.
    strict: any missing/invalid circle stops BEFORE processing; any processing
    failure stops the run. Otherwise those videos are skipped/recorded.
    Raises BatchAbort when the run stops early (summary is still written).
    """
    log = log_cb or (lambda message: None)
    notify = video_cb or (lambda item: None)
    counts = plan.counts()
    if strict and (counts['missing'] or counts['invalid']):
        raise BatchAbort('Strict mode: fix these circles before processing '
                         '(nothing was processed):\n' + '\n'.join(plan.problem_lines()))
    plan.output_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    # Written before processing, so it exists even if the run stops early.
    metadata_path = write_run_metadata(plan.output_root, build_run_metadata(
        'run', code_version, parameters_file, {
            'interface': interface,
            'inputs': {'input': str(plan.input_root), 'recursive': plan.recursive,
                       'shared_circle_file': plan.circle_file, 'strict': strict,
                       'render_video': bool(render_video)},
            'output': str(plan.output_root),
            'videos': [{'video': str(item.video), 'relative_path': item.relative,
                        'circle_status': item.state,
                        'circle_file': str(item.circle.path) if item.circle.path else None,
                        'circle_source': item.circle.source,
                        'circle': item.circle.circle}
                       for item in plan.items],
        }))

    for item in plan.items:
        if item.state != 'ready':
            item.status, item.message = 'skipped', item.reason
            log(f'SKIP {item.relative}: {item.reason}')
            notify(item)
    runnable = [item for item in plan.items if item.state == 'ready']
    abort_message = None

    for index, item in enumerate(runnable, 1):
        if cancel_event is not None and cancel_event.is_set():
            item.status, item.message = 'cancelled', 'batch cancelled before this video'
            notify(item)
            continue
        item.status = 'running'
        notify(item)
        log(f'[{index}/{len(runnable)}] Processing {item.relative} | '
            f'circle={item.circle.path or "drawn in GUI"} ({item.circle.source})')

        def per_video(pct, msg, _index=index):
            if progress_cb is not None:
                progress_cb(_index, len(runnable), pct, msg)

        processor = VideoProcessor(str(item.video), item.circle.circle, per_video,
                                   lambda message: None, out_dir=str(item.output_dir),
                                   stem=item.video.stem, render_video=render_video,
                                   cancel_event=cancel_event)
        try:
            events, entered, exited, paths = processor.run()
        except Exception as exc:  # one bad video should not lose the whole batch
            item.status, item.message = 'failed', f'{type(exc).__name__}: {exc}'
            log(f'  FAILED: {item.message}')
            notify(item)
            if strict:
                abort_message = f'Strict mode: stopped after {item.relative} failed: {item.message}'
                break
            continue
        if events is None:
            item.status, item.message = 'cancelled', 'cancelled while processing'
            notify(item)
            continue
        item.status = 'processed'
        item.message = f'{len(events)} events | IN={sum(entered.values())} | OUT={sum(exited.values())}'
        item.result = {'entries': entered, 'exits': exited,
                       'total_events': len(events), 'outputs': paths}
        log(f'  {item.message}')
        if progress_cb is not None:
            progress_cb(index, len(runnable), 100.0, 'done')
        notify(item)

    if abort_message:
        for item in runnable:
            if item.status == 'pending':
                item.status, item.message = 'cancelled', 'strict mode stopped the batch first'
                notify(item)

    summary = _write_batch_summary(plan, strict, render_video, parameters_file,
                                   started, cancel_event, code_version, metadata_path)
    if abort_message:
        raise BatchAbort(abort_message, summary)
    if not runnable:
        raise BatchAbort('No videos could be processed (every video is missing a '
                         'valid circle). Details are in '
                         f"{summary['files']['summary']}", summary)
    return summary


def _write_batch_summary(plan: BatchPlan, strict: bool, render_video: bool,
                         parameters_file: str | None, started, cancel_event,
                         code_version: str | None = None,
                         metadata_path: Path | None = None) -> dict:
    """Write batch_summary.json at the output root and return its content."""
    def entry(item):
        record = {'video': str(item.video), 'relative_path': item.relative,
                  'output_dir': str(item.output_dir),
                  'circle_source': item.circle.source,
                  'circle_file': str(item.circle.path) if item.circle.path else None}
        if item.status == 'processed':
            record.update(item.result)
        else:
            record['reason'] = item.message
        return record

    by_status = defaultdict(list)
    for item in plan.items:
        by_status[item.status].append(entry(item))
    totals_in = {q: 0 for q in _QUADRANTS}
    totals_out = {q: 0 for q in _QUADRANTS}
    for item in plan.items:
        if item.status == 'processed':
            for q in _QUADRANTS:
                totals_in[q] += int(item.result['entries'].get(q, 0))
                totals_out[q] += int(item.result['exits'].get(q, 0))
    summary_path = plan.output_root / 'batch_summary.json'
    summary = {
        'run_started_at_utc': started.isoformat(),
        'run_finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'input': str(plan.input_root), 'output': str(plan.output_root),
        'recursive': plan.recursive, 'strict': strict,
        'render_video': bool(render_video),
        'code_version': code_version or 'not provided',
        'parameters_file': str(parameters_file) if parameters_file else None,
        'effective_parameters': _runtime_parameter_snapshot(),
        'circle_file': plan.circle_file,
        'cancelled': bool(cancel_event is not None and cancel_event.is_set()),
        'videos_found': len(plan.items),
        'videos_processed': len(by_status['processed']),
        'videos_skipped': len(by_status['skipped']),
        'videos_failed': len(by_status['failed']),
        'videos_cancelled': len(by_status['cancelled']),
        'totals': {'entries': totals_in, 'exits': totals_out},
        'per_video': by_status['processed'],
        'skipped': by_status['skipped'],
        'failed': by_status['failed'],
        'cancelled_videos': by_status['cancelled'],
        'files': {'summary': str(summary_path),
                  'metadata': str(metadata_path) if metadata_path else None},
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + '\n', encoding='utf-8')
    return summary


def _apply_cli_parameters(filename: str | None) -> dict:
    """Load one preset consistently before running any selected mode."""
    if not filename:
        return {}
    values = load_tune_values(filename)
    if not values:
        raise ValueError(f'No recognized numeric tuning parameters in {filename}')
    apply_tune_values(values)
    print(f'Loaded {len(values)} settings from {filename}', flush=True)
    return values


def run_batch_mode(videos_dir: str | None = None, video_file: str | None = None,
                   parameters_file: str | None = None, circle_file: str | None = None,
                   output_dir: str | None = None, render_video: bool = False,
                   recursive: bool = False, strict: bool = False,
                   code_version: str | None = None) -> dict:
    """CLI wrapper: plan (pre-run circle check) -> print -> execute.

    Missing/invalid circles are skipped unless strict. An invalid shared
    --circle-file, or a single --video without a circle, always stops.
    """
    if circle_file:
        shared = Path(circle_file).expanduser().resolve()
        if not shared.is_file():
            raise ValueError(f'--circle-file not found: {shared}')
        if _read_circle_file(shared)[0] is None:
            raise ValueError(f'Invalid --circle-file {shared}: {_read_circle_file(shared)[1]}')
    plan = plan_batch(videos_dir, video_file, recursive, circle_file, output_dir)
    print(plan.summary_line(), flush=True)
    for line in plan.problem_lines():
        print(line, flush=True)
    if plan.single_video and plan.items[0].state != 'ready':
        raise ValueError(f'{plan.items[0].relative}: {plan.items[0].reason}. '
                         'Create one in the GUI or pass --circle-file FILE')

    last_print = {}

    def progress(index, total, pct, msg):
        if 'Frame ' in msg and last_print.get(index) != msg:
            last_print[index] = msg
            print(f'  {pct:5.1f}% {msg}', flush=True)

    summary = execute_batch(plan, strict, render_video, parameters_file,
                            progress_cb=progress, log_cb=lambda m: print(m, flush=True),
                            code_version=code_version, interface='cli')
    _print_batch_report(summary, 'Batch complete')
    return summary


def _print_batch_report(summary: dict, heading: str) -> None:
    """End-of-run terminal report: counts, then every skipped/failed/not-run
    video with its reason, so nothing has to be dug out of the JSON."""
    print(f"\n{heading}: {summary['videos_processed']} processed, "
          f"{summary['videos_skipped']} skipped, {summary['videos_failed']} failed"
          + (f", {summary['videos_cancelled']} not run" if summary['videos_cancelled'] else ''),
          flush=True)
    for title, key in (('Skipped', 'skipped'), ('Failed', 'failed'),
                       ('Not run', 'cancelled_videos')):
        records = summary.get(key) or []
        if records:
            print(f'{title} ({len(records)}):', flush=True)
            for record in records:
                print(f"  {record['relative_path']}: {record.get('reason') or 'no reason recorded'}",
                      flush=True)
    print(f"Summary: {summary['files']['summary']}", flush=True)
    if summary['files'].get('metadata'):
        print(f"Metadata: {summary['files']['metadata']}", flush=True)


# ============================================================================
# EQUIVALENT COMMAND LINES (printed after prompts; copied from the GUI)
# ============================================================================
def _script_display_path() -> str:
    """Path to this script as the user would type it from the current folder."""
    script = Path(__file__).resolve()
    try:
        return os.path.relpath(script)
    except ValueError:  # different drive on Windows
        return str(script)


def build_command_line(mode: str, **opts) -> str:
    """Build a copy-pasteable command for 'run', 'regression' or 'optimize'.

    Only options that differ from their defaults are included.
    """
    parts = ['python', _script_display_path()]

    def add(flag, value=None, condition=True):
        if condition:
            parts.append(flag)
            if value is not None:
                parts.append(str(value))

    if mode == 'run':
        add('--run')
        add('--videos', opts.get('videos'), bool(opts.get('videos')))
        for video in _as_file_list(opts.get('video')):
            add('--video', video)
        add('--recursive', condition=bool(opts.get('recursive')))
        add('--strict', condition=bool(opts.get('strict')))
    elif mode == 'regression':
        add('--regression', opts.get('regression'))
        add('--videos', opts.get('videos'))
        add('--quadrant', opts.get('quadrant'), bool(opts.get('quadrant')))
        add('--show-plot', condition=bool(opts.get('show_plot')))
        add('--skip-missing-circles', condition=bool(opts.get('skip_missing_circles')))
    elif mode == 'optimize':
        if opts.get('optimize'):
            add('--optimize', opts['optimize'])
        else:
            add('--optimize')
            add('--ground-truth', opts.get('ground_truth'))
            add('--videos', opts.get('videos'))
        add('--trials', opts.get('trials'), opts.get('trials', 30) != 30)
        add('--match-distance', opts.get('match_distance'), opts.get('match_distance', 25.0) != 25.0)
        add('--seed', opts.get('seed'), opts.get('seed', 42) != 42)
    else:
        raise ValueError(f'Unknown mode {mode!r}')
    add('--parameters', opts.get('parameters'), bool(opts.get('parameters')))
    add('--circle-file', opts.get('circle_file'), bool(opts.get('circle_file')))
    add('--output', opts.get('output'), bool(opts.get('output')))
    add('--render-video', condition=bool(opts.get('render_video')))
    add('--code-version', opts.get('code_version'), bool(opts.get('code_version')))
    return ' '.join(shlex.quote(p) for p in parts)


# ============================================================================
# INTERACTIVE PROMPTS for missing required inputs (terminal only)
# ============================================================================
class PromptCancelled(Exception):
    """User left a prompt empty, pressed Ctrl-C/Ctrl-D, or ran out of tries."""


def _ask_code_version(given: str | None, can_prompt: bool, input_fn=input) -> str:
    """The GitHub version to record in the run metadata.

    --code-version wins. Otherwise ask in a terminal, offering the version
    detected from git (Enter accepts it). Without a terminal, use the detected
    version, or 'not provided'."""
    if given:
        return given
    detected = detect_code_version()
    if not can_prompt:
        if not detected:
            print('Note: no --code-version given and none detected from git; '
                  'metadata will say "not provided".', flush=True)
        return detected or 'not provided'
    hint = f' [Enter = {detected}]' if detected else ''
    try:
        answer = input_fn(f'GitHub version of the code for this run (commit, tag or release){hint}: ')
    except (EOFError, KeyboardInterrupt):
        print('')
        raise PromptCancelled('cancelled at prompt') from None
    return answer.strip() or detected or 'not provided'


def _can_prompt(args) -> bool:
    """Prompt only in an interactive terminal and when --no-prompt is absent."""
    if getattr(args, 'no_prompt', False):
        return False
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except (AttributeError, ValueError):
        return False


def _clean_typed_path(raw: str) -> str:
    """Undo terminal quoting: surrounding quotes, backslash-escaped spaces, ~."""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in '"\'':
        text = text[1:-1]
    elif os.name != 'nt':
        # Dragging a file into macOS Terminal produces e.g. /Users/me/ant\ videos
        text = re.sub(r'\\(.)', r'\1', text)
    return os.path.expanduser(text)


def _check_prompted_path(path: Path, kind: str, recursive: bool = False) -> tuple[bool, str]:
    """Validate a typed path for a prompt kind; return (ok, message)."""
    if not path.exists():
        return False, f'Not found: {path}'
    if kind in ('video_dir', 'video_or_dir') and path.is_dir():
        try:
            count = len(_videos_from_inputs(str(path), None, recursive))
        except ValueError as exc:
            return False, str(exc)
        where = 'folder (including subfolders)' if recursive else 'folder'
        return True, f'{where.capitalize()}, {count} video(s) found'
    if kind == 'video_or_dir':
        if path.suffix.lower() in OPTIMIZE_VIDEO_EXTENSIONS:
            return True, 'Video file'
        return False, f'Not a supported video ({", ".join(sorted(OPTIMIZE_VIDEO_EXTENSIONS))}): {path.name}'
    if kind == 'video_dir':
        return False, f'Not a folder: {path}'
    if kind == 'csv_or_dir':
        if path.is_file():
            return (True, 'CSV file') if path.suffix.lower() == '.csv' else \
                   (False, f'Not a CSV file: {path.name}')
        count = sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() == '.csv')
        return (True, f'Folder, {count} CSV file(s) found') if count else \
               (False, f'No CSV files in {path}')
    if kind == 'dir':
        return (True, 'Folder') if path.is_dir() else (False, f'Not a folder: {path}')
    raise ValueError(f'Unknown prompt kind {kind!r}')


def _prompt_path(label: str, kind: str, retries: int = 3, recursive: bool = False,
                 input_fn=input, output_fn=print) -> Path:
    """Ask for a path until it validates; raise PromptCancelled on empty/Ctrl-C."""
    for attempt in range(retries):
        try:
            raw = input_fn(f'{label}: ')
        except (EOFError, KeyboardInterrupt):
            output_fn('')
            raise PromptCancelled('cancelled at prompt') from None
        if not raw.strip():
            raise PromptCancelled('empty answer')
        path = Path(_clean_typed_path(raw)).resolve()
        ok, message = _check_prompted_path(path, kind, recursive)
        if ok:
            output_fn(f'  ✓ {message}')
            return path
        left = retries - attempt - 1
        output_fn(f'  ✗ {message}' + (f' - try again ({left} left)' if left else ''))
    raise PromptCancelled('too many invalid answers')


# ============================================================================
# ARGUMENT PARSER AND ENTRY POINT
# ============================================================================
def build_argument_parser():
    """Define once: reusable inputs shared by GUI/batch/regression/Optuna."""
    parser = argparse.ArgumentParser(
        description='Ant counting: GUI, headless video batch, count regression, or Optuna detection tuning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    modes = parser.add_argument_group('modes (choose one; GUI is the default)')
    modes.add_argument('--run', action='store_true', help='Process video(s) headlessly using the production tracker')
    modes.add_argument('--gui', action='store_true',
                       help='Open the GUI with --run options (--videos/--video, --recursive, --strict, '
                            '--output, --render-video, --parameters) pre-filled; with --regression, '
                            'open the Regression tab pre-filled instead')
    modes.add_argument('--regression', nargs='?', const='', metavar='COUNTS_CSV_OR_DIR',
                       help='Compare IN/OUT totals with count CSVs; path or use --ground-truth')
    modes.add_argument('--optimize', nargs='?', const='', metavar='DATASET_DIR',
                       help='Run Optuna on labeled ant positions; path or --ground-truth + --videos')
    shared = parser.add_argument_group('shared inputs and outputs')
    shared.add_argument('--videos', metavar='DIRECTORY', help='Folder of videos for any headless mode')
    shared.add_argument('--video', metavar='FILE', action='append',
                        help='A video for --run/--gui; repeat for several videos from any folders '
                             '(not for Optuna/regression)')
    shared.add_argument('--parameters', '--settings', dest='parameters', metavar='JSON',
                        help='Parameter preset for GUI/batch/regression and base preset for Optuna')
    shared.add_argument('--ground-truth', metavar='CSV_OR_DIRECTORY',
                        help='Ground-truth count CSVs (regression) or positions (Optuna)')
    shared.add_argument('--circle-file', metavar='JSON',
                        help='Shared circle for every video; else per-video <stem>_circle.json')
    shared.add_argument('--output', '--regression-output', dest='output', metavar='DIRECTORY',
                        help='Output root directory')
    shared.add_argument('--render-video', '--render-regression-video', dest='render_video',
                        action='store_true', help='Write annotated MP4 files for batch/regression')
    shared.add_argument('--no-render-video', dest='render_video', action='store_false',
                        help='Skip annotated MP4s in headless modes')
    shared.add_argument('--code-version', metavar='TEXT',
                        help='GitHub version of the code (commit, tag or release) recorded in the run '
                             'metadata. If omitted you are asked in a terminal (Enter accepts the '
                             'version detected from git)')
    shared.add_argument('--no-prompt', action='store_true',
                        help='Never prompt for missing inputs; error instead (automatic when not in a terminal)')
    parser.set_defaults(render_video=False)
    batch = parser.add_argument_group('batch options (--run / --gui)')
    batch.add_argument('--recursive', action='store_true',
                       help='Include videos in all subfolders of --videos; outputs mirror the folder tree')
    batch.add_argument('--strict', action='store_true',
                       help='Stop before processing if any circle is missing/invalid, and on the first failed video')
    specific = parser.add_argument_group('mode-specific options')
    specific.add_argument('--show-plot', action='store_true',
                          help='Display regression plot interactively after saving PNG (requires display)')
    specific.add_argument('--quadrant', choices=_QUADRANTS,
                          help='Only evaluate this quadrant for regression')
    specific.add_argument('--skip-missing-circles', action='store_true',
                          help='Regression: skip (and list) matched videos without a valid circle '
                               'instead of stopping')
    specific.add_argument('--trials', type=int, default=30, help='Additional Optuna trials')
    specific.add_argument('--match-distance', type=float, default=25.0,
                          help='Optuna detection matching radius in original pixels')
    specific.add_argument('--seed', type=int, default=42, help='Optuna random seed')
    return parser


def _check_conflicts(parser, args, run_mode: bool, plain_gui: bool) -> None:
    """Contradictory flags are always errors; they are never prompted for."""
    regression_mode = args.regression is not None
    optimize_mode = args.optimize is not None
    batch_mode = run_mode or (args.gui and not regression_mode)
    if args.video and args.videos:
        parser.error('Use either --video FILE or --videos DIRECTORY, not both')
    if args.video and (regression_mode or optimize_mode):
        parser.error('--video is for --run/--gui; use --videos for folder modes')
    if args.recursive and args.video:
        parser.error('--recursive applies to a folder (--videos), not to --video files')
    if args.recursive and not batch_mode:
        parser.error('--recursive is only available with --run or --gui (not with --regression)')
    if args.strict and not batch_mode:
        parser.error('--strict is only available with --run or --gui (not with --regression)')
    if args.skip_missing_circles and not regression_mode:
        parser.error('--skip-missing-circles is only available with --regression')
    if args.quadrant and not regression_mode:
        parser.error('--quadrant is only available with --regression')
    if args.show_plot and not regression_mode:
        parser.error('--show-plot is only available with --regression')
    if args.ground_truth and not (regression_mode or optimize_mode):
        parser.error('--ground-truth requires --regression or --optimize')
    if args.regression and args.ground_truth:
        parser.error('Use either --regression CSV_OR_DIR or --ground-truth CSV_OR_DIR, not both')
    if args.optimize and (args.videos or args.ground_truth):
        parser.error('Use --optimize DATASET_DIR OR --optimize --ground-truth DIR --videos DIR')
    if args.circle_file and args.gui and not regression_mode:
        parser.error('--circle-file is not used by the GUI: there each video\'s circle is either '
                     'found as <name>_circle.json or clicked by hand. Use --circle-file with --run.')
    if args.circle_file and optimize_mode:
        parser.error('--circle-file does not affect positional Optuna optimization')
    if args.render_video and optimize_mode:
        parser.error('--render-video is only supported in --run, --gui and --regression')
    if plain_gui and (args.output or args.circle_file or args.render_video):
        parser.error('--output, --circle-file and --render-video need --run or --regression '
                     '(--gui can pre-fill --output and --render-video)')


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    regression_mode = args.regression is not None
    optimize_mode = args.optimize is not None
    if sum((args.run, regression_mode, optimize_mode)) > 1 or (
            args.gui and (args.run or optimize_mode)):
        parser.error('Choose only one of --run, --gui, --regression or --optimize '
                     '(--gui may be combined with --regression)')
    # Explicit video inputs without a mode mean regular batch processing.
    run_mode = args.run or (not (args.gui or regression_mode or optimize_mode)
                            and bool(args.video or args.videos))
    plain_gui = not (run_mode or args.gui or regression_mode or optimize_mode)
    _check_conflicts(parser, args, run_mode, plain_gui)

    can_prompt = _can_prompt(args)
    prompted = []

    def need(label, kind, error_message, recursive=False):
        """Prompt for a missing required path, or error when prompting is off."""
        if not can_prompt:
            parser.error(error_message + ' (run in a terminal to be prompted, '
                         'or see --help)')
        if not prompted:
            print('Missing required input. Press Enter on an empty line or '
                  'Ctrl-C to cancel.', flush=True)
        prompted.append(label)
        return str(_prompt_path(label, kind, recursive=recursive))

    try:
        if regression_mode and args.gui:
            # Regression GUI: open the Regression tab pre-filled; no prompts.
            _require_gui_dependencies()
            _apply_cli_parameters(args.parameters)
            App(prefill={'regression': {
                'ground_truth': args.regression or args.ground_truth, 'videos': args.videos,
                'parameters': args.parameters, 'circle_file': args.circle_file,
                'output': args.output, 'quadrant': args.quadrant,
                'render_video': args.render_video,
                'skip_missing_circles': args.skip_missing_circles},
                'code_version': args.code_version}).run()
        elif regression_mode:
            if not (args.regression or args.ground_truth):
                args.regression = need('Ground-truth counts CSV or folder (--regression)', 'csv_or_dir',
                                       '--regression requires a ground-truth CSV or folder')
            if not args.videos:
                args.videos = need('Video folder (--videos)', 'video_dir',
                                   '--regression requires --videos DIRECTORY')
            gt = args.regression or args.ground_truth
            args.code_version = _ask_code_version(args.code_version, can_prompt)
            if prompted:
                print('\nEquivalent command:\n' + build_command_line(
                    'regression', regression=gt, videos=args.videos, quadrant=args.quadrant,
                    show_plot=args.show_plot, skip_missing_circles=args.skip_missing_circles,
                    parameters=args.parameters,
                    circle_file=args.circle_file, output=args.output,
                    render_video=args.render_video, code_version=args.code_version) + '\n',
                    flush=True)
            _apply_cli_parameters(args.parameters)
            run_regression_mode(gt, args.videos, args.parameters, args.quadrant,
                                args.circle_file, args.output, args.render_video, args.show_plot,
                                skip_missing_circles=args.skip_missing_circles,
                                code_version=args.code_version)
        elif optimize_mode:
            if not args.optimize and not (args.videos and args.ground_truth):
                if not args.videos and not args.ground_truth and can_prompt:
                    dataset = need('Dataset folder (--optimize; holds ground-truth/ and videos/)',
                                   'dir', '')
                    if (Path(dataset) / 'ground-truth').is_dir() and (Path(dataset) / 'videos').is_dir():
                        args.optimize = dataset
                    else:
                        print('  That folder has no ground-truth/ and videos/ subfolders; '
                              'enter them separately.', flush=True)
                if not args.optimize:
                    if not args.ground_truth:
                        args.ground_truth = need('Ground-truth position CSV folder (--ground-truth)',
                                                 'csv_or_dir',
                                                 '--optimize needs DATASET_DIR or both --videos and --ground-truth')
                    if not args.videos:
                        args.videos = need('Video folder (--videos)', 'video_dir',
                                           '--optimize needs DATASET_DIR or both --videos and --ground-truth')
            if prompted:
                print('\nEquivalent command:\n' + build_command_line(
                    'optimize', optimize=args.optimize, ground_truth=args.ground_truth,
                    videos=args.videos, trials=args.trials, match_distance=args.match_distance,
                    seed=args.seed, parameters=args.parameters, output=args.output) + '\n', flush=True)
            _apply_cli_parameters(args.parameters)
            folder = args.optimize or Path(args.videos).expanduser().resolve().parent
            output = args.output
            if not args.optimize and not output:
                output = str(Path(args.videos).expanduser().resolve() / 'optuna_results')
            run_optimization_mode(folder, args.trials, args.match_distance, args.seed,
                                  args.ground_truth, args.videos, output,
                                  code_version=args.code_version or detect_code_version() or None,
                                  parameters_file=args.parameters)
        elif run_mode:
            if not (args.videos or args.video):
                if args.recursive:
                    args.videos = need('Video folder (--videos)', 'video_dir',
                                       '--run requires --videos DIRECTORY or --video FILE',
                                       recursive=True)
                else:
                    chosen = Path(need('Video folder or file (--videos / --video)', 'video_or_dir',
                                       '--run requires --videos DIRECTORY or --video FILE'))
                    if chosen.is_dir():
                        args.videos = str(chosen)
                    else:
                        args.video = [str(chosen)]
            args.code_version = _ask_code_version(args.code_version, can_prompt)
            if prompted:
                print('\nEquivalent command:\n' + build_command_line(
                    'run', videos=args.videos, video=args.video, recursive=args.recursive,
                    strict=args.strict, parameters=args.parameters, circle_file=args.circle_file,
                    output=args.output, render_video=args.render_video,
                    code_version=args.code_version) + '\n', flush=True)
            _apply_cli_parameters(args.parameters)
            summary = run_batch_mode(args.videos, args.video, args.parameters, args.circle_file,
                                     args.output, args.render_video, args.recursive, args.strict,
                                     code_version=args.code_version)
            # Skipped videos (known reasons, e.g. no circle) still exit 0;
            # a video that FAILED to process exits 1 so scripts can notice.
            if summary['videos_failed']:
                parser.exit(1, f"Exit code 1: {summary['videos_failed']} video(s) failed "
                               '(listed above); the other videos were processed.\n')
        else:
            # GUI: plain (default) or --gui with run options pre-filled.
            _require_gui_dependencies()
            _apply_cli_parameters(args.parameters)
            prefill = {'parameters': args.parameters, 'code_version': args.code_version}
            if args.gui:
                prefill.update(videos=args.videos, video=args.video, recursive=args.recursive,
                               strict=args.strict, output=args.output,
                               render_video=args.render_video)
            App(prefill=prefill).run()
    except PromptCancelled as exc:
        parser.exit(1, f'Cancelled ({exc}): nothing was run.\n')
    except BatchAbort as exc:
        if exc.summary:
            _print_batch_report(exc.summary, 'Batch stopped')
        parser.exit(2, f'\n{exc}\n')
    except KeyboardInterrupt:
        parser.exit(130, '\nInterrupted.\n')
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        # Runtime problems (bad files, missing circles): message only, no usage.
        parser.exit(2, f'{parser.prog}: error: {exc}\n')


if __name__ == '__main__':
    main()
