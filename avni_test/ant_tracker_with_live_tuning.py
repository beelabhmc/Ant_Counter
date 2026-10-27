# ============================================================================
# ANT HOLE COUNTER — EDUCATIONAL, FULLY COMMENTED EDITION
# ============================================================================
# This file preserves the original executable statements and their behavior.
# Extra comments explain the pipeline, math, array shapes, and key limitations.
#
# End-to-end data flow:
#   Video -> resized grayscale frames -> absolute inter-frame difference
#   -> threshold -> binary morphology -> connected components -> blobs
#   -> Kalman predictions -> Hungarian assignment -> persistent ant IDs
#   -> radius/buffer/hysteresis crossing decisions -> CSV and annotated MP4.
#
# IMPORTANT DESIGN OBSERVATIONS (no fixes applied in this annotated version):
# * This program uses FRAME DIFFERENCING, not adaptive background subtraction.
# * The dark-pixel "shadow_mask" currently modifies diff AFTER creating mask,
#   so it does not filter the actual connected-components detection input.
# * Once a track becomes missing>0, it is excluded from subsequent matching.
# * _save_circle_json() refuses a new filename due to isfile() precondition.
# * A summary CSV path is generated but no corresponding file is written.
# * Frames skipped for camera vibration still advance tracker missing counters.
# * Comments in the original program sometimes describe older experimental code.

# python ant_tracker_with_live_tuning.py --tune-mode=True --video ants.mp4 --circle-file ants_circle.json
# python ant_tracker_with_live_tuning.py --settings ant_tuning_parameters.json
# ============================================================================

#!/usr/bin/env python3
"""
Ant Hole Counter
================
UI for counting ants entering and exiting an anthill.

Detection method: frame-to-frame difference (NOT background subtraction).
  Comparing adjacent frames detects only things that MOVED, so static shadows
  — even if they look very different from the median background — register as
  zero motion and are ignored.  Moving ants show up clearly.

Usage:
    python ant_counter_circle_shadow.py

Workflow:
    1. Click Browse… to load a video.
    2. Select an output directory
    3. Scrub the frame slider to find a good view of the hole entrance.
    4. Click to define the center of the entrance and the direction of north
    5. Click "Process Video".  Results are written to the output directory.

Outputs (per video):
    <name>_counts.csv       — timestamp, event (enter/exit), running count
    <name>_counted.mp4      — annotated video with live count overlay
    <name>_circle.json     — circle parameters (center, radius, north_angle)
"""

from __future__ import annotations
import argparse
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import os
import json
import csv
import threading
import math
from datetime import timedelta
from scipy.optimize import linear_sum_assignment as _hungarian_solve

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit(
        "Pillow is required.  Install it with:\n"
        "    pip install Pillow\n"
        "then re-run this script."
    )


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
            clusters.append({"cx": cx, "cy": cy, "area": area})
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

# Writes the circle calibration to JSON. Known issue: the existing-file
# check prevents saving to a newly selected filename. Left unchanged here.
def _save_circle_json(circle_params, path: str):
    if not path or not os.path.isfile(path):
        return None
    with open(path, "w") as f:
        json.dump({"circle_params": circle_params}, f, indent=2)

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

    # Updates preview radius when user changes radius control.
    def _on_radius_change(self):
        self.circle_radius = float(self._radius_var.get())
        if self.current_bgr is not None:
            self._render()

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
                 stem: str | None = None):
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
    
    # ------------------------------------------------------------------
    # Reads video metadata, scales circle coordinates, writes calibration JSON,
    # processes video once, then aggregates entry/exit counts by quadrant.
    def run(self):
        """Main pipeline.  
        Returns (events, enter_totals, exit_totals, paths_dict) 
        or (None, None, None, None)."""
        cap = cv2.VideoCapture(self.video_path)
        fps      = cap.get(cv2.CAP_PROP_FPS)
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
        vid_path = os.path.join(self.out_dir, f"{self.stem}_counted.mp4")
        summary_path = os.path.join(self.out_dir, f"{self.stem}_summary.csv")
        self.progress_cb(0, "Processing frames…")
        events = self._process_frames(
            circle_proc, proc_w, proc_h, fps, total, 
            csv_path, vid_path, summary_path)
        
        # Calculate summary counts per quadrant
        enter_totals = {q: 0 for q in _QUADRANTS}
        exit_totals = {q: 0 for q in _QUADRANTS}
        for e in events:
            quad = e["quadrant"]
            if e["event"].startswith("enter_"):
                enter_totals[quad] += 1
            elif e["event"].startswith("exit_"):
                exit_totals[quad] += 1

        if self._cancel:
            return None, None, None, None

        return events, enter_totals, exit_totals, {
            "csv": csv_path, 
            "video": vid_path,
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
                        fps, total, csv_path, vid_path, summary_path):
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
        writer = cv2.VideoWriter(vid_path, fourcc, fps, (proc_w, proc_h), True)

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

        for fid in range(total):
            if self._cancel:
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
                                      "area": int(area)})
                # ant_blobs, shadow_blobs = separate_shadow_blobs(blobs, shadow_mask)
                # EXPLANATION: Merge close components to reduce split detections; might accidentally merge distinct nearby ants.
                blobs = cluster_blobs(blobs, max_dist=CLUSTER_DISTANCE)

            # ── tracking & crossing detection ─────────────────────────
            # EXPLANATION: Tracker maintains IDs across frames and emits only confirmed boundary transitions.
            events = tracker.update(blobs, circle_proc)
            ts  = fid / fps
            ts_str = str(timedelta(seconds=int(ts)))

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
        writer.release()

        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerows(csv_rows)

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

# Tkinter user interface for choosing video/output, editing the circle,
# starting the worker thread, showing progress and reporting saved outputs.
class App(CirclePreviewMixin):
    # Creates window, initializes video/circle state and builds interface.
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ant Hole Counter")
        self.root.geometry("1300x760")
        self.root.minsize(900, 600)

        # ── state ──
        self.video_path: str | None = None
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.orig_w = self.orig_h = 0
        self.current_bgr = None

        # circle setup in VIDEO pixel coords
        self.circle_center: tuple = None  # (x, y)
        self.circle_radius: float = DEFAULT_RADIUS
        self.north_point: tuple = None    # (x, y) for north direction
        self.circle_complete: bool = False

        # canvas display info
        self.disp_off_x = self.disp_off_y = 0
        self.disp_scale = 1.0
        self._photo = None

        self.processor: VideoProcessor | None = None
        self._slider_dragging = False
        self.out_dir: str | None = None


        self._build_ui()

    # ══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════════

    # Creates every Tkinter widget, binds event handlers, and lays out sidebar,
    # preview canvas, controls, and progress/status displays.
    def _build_ui(self):
        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                              sashwidth=6, sashrelief=tk.RAISED, bg="#333")
        pane.pack(fill=tk.BOTH, expand=True)

        # ── sidebar ──────────────────────────────────────────────────
        sidebar = tk.Frame(pane, width=270, bg="#f4f4f4", padx=8, pady=8)
        pane.add(sidebar, minsize=240)

        tk.Label(sidebar, text="🐜  Ant Hole Counter",
                 font=("Arial", 15, "bold"), bg="#f4f4f4").pack(pady=(4, 10))

        # Video
        vf = ttk.LabelFrame(sidebar, text="  Video  ")
        vf.pack(fill=tk.X, pady=4)

        tk.Button(vf, text="Browse…", width=20, relief=tk.GROOVE,
                  command=self.browse_video).pack(pady=(1, 4))

        self._info_lbl = tk.Label(sidebar, text="", font=("Courier", 8),
                                  justify=tk.LEFT, bg="#f4f4f4", fg="#444")
        self._info_lbl.pack(anchor=tk.W, pady=2)

        # Output
        outf = ttk.LabelFrame(sidebar, text="  Output location  ")
        outf.pack(fill=tk.X, pady=4)
        self._out_dir_lbl = tk.Label(
            outf,
            text="Output folder: default in ./outputs",
            font=("Arial", 8), fg="#444", justify=tk.LEFT, anchor=tk.W,
            wraplength=250)
        self._out_dir_lbl.pack(fill=tk.X, padx=4, pady=(4, 2))
        tk.Button(
            outf, text="Select output folder…",
            command=self._select_out_dir).pack(fill=tk.X, padx=4, pady=(0, 2))
        tk.Button(
            outf, text="Clear output folder",
            command=self._clear_out_dir).pack(fill=tk.X, padx=4, pady=(0, 4))

        # Frame scrub
        ff = ttk.LabelFrame(sidebar, text="  Frame  ")
        ff.pack(fill=tk.X, pady=4)
        self._frame_var = tk.IntVar()
        self._slider = ttk.Scale(ff, from_=0, to=100,
                                 variable=self._frame_var, orient=tk.HORIZONTAL,
                                 command=self._on_slider)
        self._slider.pack(fill=tk.X, padx=4, pady=2)
        self._frame_lbl = tk.Label(ff, text="—", font=("Arial", 8))
        self._frame_lbl.pack()

        # Circle Setup
        cf = ttk.LabelFrame(sidebar, text="  Circle Setup  ")
        cf.pack(fill=tk.X, pady=4)
        tk.Label(cf,
                 text=("1st click: Set center\n"
                       "2nd click: Set north direction\n"
                       "Right-click: Clear setup"),
                 font=("Arial", 8), justify=tk.LEFT).pack(anchor=tk.W, padx=4)
        self._circle_lbl = tk.Label(cf, text="No circle defined",
                                    fg="#b00", font=("Arial", 9, "bold"))
        self._circle_lbl.pack(pady=2)
        
        # Radius adjustment
        radius_frame = tk.Frame(cf)
        radius_frame.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(radius_frame, text="Radius:", font=("Arial", 8)).pack(side=tk.LEFT)
        self._radius_var = tk.IntVar(value=DEFAULT_RADIUS)
        radius_spinbox = tk.Spinbox(radius_frame, from_=50, to=500, 
                                   textvariable=self._radius_var, width=8,
                                   command=self._on_radius_change)
        radius_spinbox.pack(side=tk.LEFT, padx=(5,0))
        
        tk.Button(cf, text="Clear Circle",
                  command=self.clear_circle).pack(pady=(0, 4))

        # Process
        prf = ttk.LabelFrame(sidebar, text="  Process  ")
        prf.pack(fill=tk.X, pady=4)
        self._proc_btn = tk.Button(
            prf, text="▶  Process Video",
            font=("Arial", 11, "bold"),
            bg="#2e7d32", fg="white", state=tk.DISABLED,
            activebackground="#1b5e20", activeforeground="white",
            command=self.start_processing)
        self._proc_btn.pack(fill=tk.X, padx=4, pady=4)
        self._cancel_btn = tk.Button(
            prf, text="Cancel", state=tk.DISABLED,
            command=self.cancel_processing)
        self._cancel_btn.pack(fill=tk.X, padx=4, pady=(0, 4))

        self._prog_var = tk.DoubleVar()
        self._prog_bar = ttk.Progressbar(prf, variable=self._prog_var, maximum=100)
        self._prog_bar.pack(fill=tk.X, padx=4, pady=2)
        self._prog_lbl = tk.Label(prf, text="", font=("Arial", 8))
        self._prog_lbl.pack()

        # Status
        self._status_lbl = tk.Label(
            sidebar,
            text="Load a video to get started.",
            wraplength=250, justify=tk.LEFT,
            font=("Arial", 9), fg="#333", bg="#f4f4f4")
        self._status_lbl.pack(pady=6, anchor=tk.W)

        # ── canvas ───────────────────────────────────────────────────
        cf = tk.Frame(pane, bg="#111")
        pane.add(cf, minsize=500)

        self.canvas = tk.Canvas(cf, bg="#111", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>",        self._click)
        self.canvas.bind("<Double-Button-1>", self._dblclick)
        self.canvas.bind("<Button-3>",        self._rclick)
        self.canvas.bind("<Configure>",       self._resize)

    # Allows user to select the output directory.
    def _select_out_dir(self):
        out_dir = filedialog.askdirectory(title="Select output folder")
        if not out_dir:
            return
        self.out_dir = out_dir
        self._update_out_dir_label()

    # Resets the selected output directory.
    def _clear_out_dir(self):
        self.out_dir = None
        self._update_out_dir_label()

    # Updates UI text displaying the selected output directory.
    def _update_out_dir_label(self):
        if self.out_dir is None:
            self._out_dir_lbl.config(
                text="Output folder: default in ./outputs")
            return
        else:
            self._out_dir_lbl.config(
                text=f"Output folder: {self.out_dir}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Video loading
    # ══════════════════════════════════════════════════════════════════════════

    # Routes video-loading action through the UI.
    def load_video(self, path: str):
        if not os.path.exists(path):
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return
        self._open_video(path)

    # Opens the system file selector to choose the input video.
    def browse_video(self):
        p = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video", "*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI"),
                       ("All", "*.*")])
        if p:
            self._open_video(p)

    # Opens a video, reads its metadata, resets circle, and renders first frame.
    def _open_video(self, path: str):
        if self.cap:
            self.cap.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open:\n{path}")
            return
        self.cap        = cap
        self.video_path = path
        self.fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orig_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._slider.config(to=max(0, self.total_frames - 1))
        self._frame_var.set(0)
        self.clear_circle()
        self._show_frame(0)

        dur = self.total_frames / self.fps
        self._info_lbl.config(
            text=f"{os.path.basename(path)}\n"
                 f"{self.orig_w}×{self.orig_h}  {self.fps:.0f} fps\n"
                 f"{self.total_frames:,} frames  ({dur:.0f} s)")
        self._status("Video loaded.\n\n"
                     "  ANTS FOR LIFE,\n"
                     ".\n"
                     ".")
        self._refresh_proc_btn()

    # Enables Process button only if both video and circle are available.
    def _refresh_proc_btn(self):
        ok = bool(self.cap and self.circle_complete)
        self._proc_btn.config(state=tk.NORMAL if ok else tk.DISABLED)

    # ══════════════════════════════════════════════════════════════════════════
    # Processing
    # ══════════════════════════════════════════════════════════════════════════

    # Bundles circle calibration, starts VideoProcessor in a daemon thread,
    # and disables conflicting UI buttons.
    def start_processing(self):
        if not (self.cap and self.circle_complete):
            return
        self._proc_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        self._prog_var.set(0)
        self._prog_lbl.config(text="Starting…")

        # Create circle parameters dict
        north_angle = angle_deg_from_center_to_point(
            self.circle_center[0], self.circle_center[1],
            self.north_point[0], self.north_point[1])
        circle_params = {
            'center': self.circle_center,
            'radius': self.circle_radius,
            'north_angle': north_angle
        }

        self.processor = VideoProcessor(
            self.video_path, circle_params,
            self._on_progress, self._status,
            self.out_dir, None
        )
        threading.Thread(target=self._thread_body, daemon=True).start()

    # Runs processing outside the UI thread and schedules UI completion/error
    # updates using Tkinter root.after().
    def _thread_body(self):
        try:
            events, enter_totals, exit_totals, paths = self.processor.run()
            if events is None:
                self._status("Cancelled.")
                self.root.after(0, self._reset_btns)
                return

            
            msg_lines = ["Done!\n"]
            # for quad, count in enter_totals.items():
            #     msg_lines.append(f"{quad}: {count:+d}")
            msg_lines.extend([f"\nTotal events: {len(events)}", 
                             f"Saved to:\n{paths['output_dir']}"])
            
            msg = "\n".join(msg_lines)
            self._status(msg)
            self.root.after(0, lambda: self._prog_var.set(100))
            self.root.after(0, lambda: self._prog_lbl.config(text="Complete!"))
        except Exception as exc:
            import traceback
            self._status(f"⚠ Error: {type(exc)}\n{str(exc)}\n\n{traceback.format_exc()[:400]}")
            raise
        finally:
            self.root.after(0, self._reset_btns)

    # Forwards cancellation request to active VideoProcessor.
    def cancel_processing(self):
        if self.processor:
            self.processor.cancel()

    # Restores appropriate button states after processing.
    def _reset_btns(self):
        ok = bool(self.cap and self.circle_complete)
        self._proc_btn.config(state=tk.NORMAL if ok else tk.DISABLED)
        self._cancel_btn.config(state=tk.DISABLED)

    # Schedules UI-safe progress label and percent updates.
    def _on_progress(self, val: float, msg: str = ""):
        self.root.after(0, lambda: self._prog_var.set(val))
        if msg:
            self.root.after(0, lambda: self._prog_lbl.config(text=msg))

    # Schedules UI-safe status-message updates.
    def _status(self, msg: str):
        self.root.after(0, lambda: self._status_lbl.config(text=msg))

    # ══════════════════════════════════════════════════════════════════════════

    # Starts Tkinter event loop and releases the preview capture on exit.
    def run(self):
        self.root.mainloop()
        if self.cap:
            self.cap.release()


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

TUNE_WINDOW = 'Ant tracking - live tuning'
TUNE_CONTROLS = 'Controls - motion / Kalman / temporal'
TUNE_PREVIEW_SCALE = 1.0  # source panel is PROCESS_SCALE-sized


def _odd_kernel(value):
    """OpenCV morphological kernels should have odd positive side lengths."""
    value = max(1, int(value))
    return value if value % 2 else value + 1


def tune_values():
    """Return live values from the trackbars without modifying the source file."""
    result = {}
    for key, (label, low, high, factor) in TUNE_SPECS.items():
        raw = cv2.getTrackbarPos(label, TUNE_CONTROLS)
        raw = max(low, min(high, raw))
        value = raw / factor
        result[key] = int(round(value)) if factor == 1 else float(value)
    for key in ('MORPH_OPEN_SIZE', 'MORPH_CLOSE_SIZE'):
        result[key] = _odd_kernel(result[key])
    return result


def apply_tune_values(settings, tracker=None):
    """Apply controls to the globals read by the existing detection/tracking code.

    Existing Kalman filters hold Q/R matrices created in their constructors, so
    merely modifying the global KF_* constants is insufficient. Update each
    live filter's covariance matrices as well (without resetting its state).
    """
    for name, value in settings.items():
        if name not in TUNE_SPECS:
            continue
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
    """Only accept known tuning keys from JSON; no arbitrary code execution."""
    with open(filename, encoding='utf-8') as fp:
        saved = json.load(fp)
    result = {}
    for key, value in saved.items():
        if key in TUNE_SPECS and isinstance(value, (int, float)):
            label, low, high, factor = TUNE_SPECS[key]
            result[key] = max(low / factor, min(high / factor, value))
    for key in ('MORPH_OPEN_SIZE', 'MORPH_CLOSE_SIZE'):
        if key in result:
            result[key] = _odd_kernel(result[key])
    return result


def run_tune_mode(video_path, circle_file=None, preset_file=None):
    """Interactive live detector+Kalman preview with sliders and saved presets.

    Controls:
        SPACE: pause/resume  | N: step one frame | R: restart from frame 0
        S: save JSON        | Q / Escape: close tuning mode
        LEFT-CLICK on the LEFT preview pane: set the circle center
        RIGHT-CLICK: set the north direction (relative to center)

    Frame differencing, not MOG2, is used here because that is the current
    detector in the user's source. Parameter changes apply on subsequent frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pw, ph = max(1, int(orig_w * PROCESS_SCALE)), max(1, int(orig_h * PROCESS_SCALE))
    cx, cy, rad, north = pw / 2, ph / 2, min(pw, ph) / 3, 270.0

    # Circle calibration is read in ORIGINAL video coordinates and converted to
    # the resized image. Click to adjust it during tuning if no file is supplied.
    if circle_file:
        saved_circle = _load_circle_json(circle_file)
        if saved_circle is None:
            raise ValueError(f'Invalid circle JSON: {circle_file}')
        cx, cy = [v * PROCESS_SCALE for v in saved_circle['center']]
        rad = saved_circle['radius'] * PROCESS_SCALE
        north = saved_circle['north_angle']
    else:
        guesses = [
            os.path.join(os.path.dirname(video_path),
                         os.path.splitext(os.path.basename(video_path))[0] + '_circle.json'),
            os.path.join(os.path.dirname(video_path), 'outputs',
                         os.path.splitext(os.path.basename(video_path))[0] + '_circle.json'),
        ]
        for guess in guesses:
            saved_circle = _load_circle_json(guess)
            if saved_circle:
                cx, cy = [v * PROCESS_SCALE for v in saved_circle['center']]
                rad = saved_circle['radius'] * PROCESS_SCALE
                north = saved_circle['north_angle']
                print('Loaded circle:', guess)
                break

    # 'circle' is mutated by click callback, so drawing always uses latest values.
    circle = {'center': (cx, cy), 'radius': rad, 'north_angle': north}
    cv2.namedWindow(TUNE_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TUNE_CONTROLS, 560, 760)
    cv2.namedWindow(TUNE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TUNE_WINDOW, min(2 * pw, 1600), min(ph + 80, 900))

    if preset_file:
        apply_tune_values(load_tune_values(preset_file))

    def no_op(_):
        pass

    for key, (label, low, high, factor) in TUNE_SPECS.items():
        initial = max(low, min(high, int(round(globals()[key] * factor))))
        cv2.createTrackbar(label, TUNE_CONTROLS, initial, high, no_op)

    # Drawn video has two panels: the original image and its binary detection mask.
    # Mouse clicks on the original left panel configure the circle.
    def on_mouse(event, mx, my, flags, userdata):
        if not (0 <= mx < pw and 0 <= my < ph):
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            circle['center'] = (float(mx), float(my))
            print('Circle center:', circle['center'])
        elif event == cv2.EVENT_RBUTTONDOWN:
            circle['north_angle'] = angle_deg_from_center_to_point(
                *circle['center'], mx, my)
            print('Circle north angle:', circle['north_angle'])

    cv2.setMouseCallback(TUNE_WINDOW, on_mouse)

    def restart():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return Tracker(max_dist=MAX_TRACK_DIST,
                       max_missing_frames=MAX_COAST_FRAMES,
                       hysteresis_frames=CROSS_HYSTERESIS_FRAMES), [], None, 0, 0

    tracker, frame_history, displayed, frame_no, count = restart()
    prev_stride = int(FRAME_STRIDE)
    paused = False
    step_once = False
    print('Tuning: SPACE=pause  N=step  R=restart  S=save  Q=quit')
    print('LEFT click preview=set circle center; RIGHT click=set north direction')
    print('NOTE: Changes to Kalman noise apply to existing tracks; restart for a clean A/B comparison.')
    print('Changes to frame stride discard the old difference buffer.')
    try:
        while True:
            settings = tune_values()
            apply_tune_values(settings, tracker)
            if FRAME_STRIDE != prev_stride:
                frame_history.clear()
                prev_stride = FRAME_STRIDE
            # A paused frame remains visible while sliders can be changed. On
            # resume, every processing stage uses the current slider settings.
            if not paused or step_once or displayed is None:
                step_once = False
                ret, raw = cap.read()
                if not ret:
                    paused = True  # stop at end; press R to replay from start
                else:
                    small = cv2.resize(raw, (pw, ph))
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    frame_history.append(gray.copy())
                    if len(frame_history) > FRAME_STRIDE + 1:
                        frame_history.pop(0)
                    if len(frame_history) > FRAME_STRIDE:
                        diff = cv2.absdiff(gray, frame_history[0])
                        diff = cv2.GaussianBlur(diff, (3, 3), 0)
                    else:
                        diff = np.zeros((ph, pw), dtype=np.uint8)

                    _, binary = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
                    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE))
                    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (MORPH_OPEN_SIZE, MORPH_OPEN_SIZE))
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k)
                    blobs = []
                    # Keep vibration guard identical to batch mode.
                    if np.count_nonzero(binary) <= pw * ph * VIBRATION_PCT / 100:
                        n, _, stats, centers = cv2.connectedComponentsWithStats(binary)
                        for i in range(1, n):
                            area = stats[i, cv2.CC_STAT_AREA]
                            if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
                                blobs.append({'cx': float(centers[i][0]),
                                              'cy': float(centers[i][1]),
                                              'area': int(area)})
                        blobs = cluster_blobs(blobs, max_dist=CLUSTER_DISTANCE)
                    events = tracker.update(blobs, circle)
                    count += len(events)
                    frame_no += 1
                    annotated = small.copy()
                    cc = tuple(int(v) for v in circle['center'])
                    rr = int(round(circle['radius']))
                    draw_quadrant_arcs(annotated, cc, rr, circle['north_angle'])
                    for r in (rr - BUFFER_ZONE, rr + BUFFER_ZONE):
                        if r > 0:
                            cv2.circle(annotated, cc, int(r), (130, 130, 130), 1)
                    for tid, ant in tracker.ants().items():
                        draw_kalman_diagnostics(annotated, tid, ant)
                    # Status text and side-by-side segmentation make misdetections
                    # vs bad Kalman matching readily distinguishable.
                    mask_view = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                    cv2.putText(annotated, f'Frame {frame_no}  IDs {len(tracker.ants())}  blobs {len(blobs)}',
                                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 255), 1)
                    cv2.putText(mask_view, f'FG mask    confirmed events {count}',
                                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 255), 1)
                    displayed = np.hstack([annotated, mask_view])
            if displayed is not None:
                cv2.imshow(TUNE_WINDOW, displayed)
            key = cv2.waitKey(max(1, int(1000 / fps)) if not paused else 30) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                paused = not paused
            elif key == ord('n'):
                paused = True
                step_once = True
            if key == ord('r'):
                tracker, frame_history, displayed, frame_no, count = restart()
                paused = False
            if key == ord('s'):
                save_tune_values(settings, os.path.join(
                    os.path.dirname(os.path.abspath(video_path)), 'ant_tuning_parameters.json'))
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Ant tracker with optional live tuning controls')
    parser.add_argument('--tune-mode', '--control-tune-mode', dest='tune_mode',
                        nargs='?', const='True', default='False',
                        help='Show interactive OpenCV trackbars (True/False)')
    parser.add_argument('--video', help='Ant video to use in tuning mode')
    parser.add_argument('--circle-file', help='Existing _circle.json to draw correct entrance')
    parser.add_argument('--settings', help='JSON tuning preset to load for tuning or batch processing')
    args = parser.parse_args()
    enabled = str(args.tune_mode).strip().lower() in ('1', 'true', 'yes', 'on')
    if enabled:
        video = args.video
        if not video:
            # File selection only: avoid starting the full Tkinter app in tuning mode.
            root = tk.Tk()
            root.withdraw()
            video = filedialog.askopenfilename(title='Select ant video',
                                               filetypes=[('Video files', '*.mp4 *.avi *.mov *.mkv'),
                                                          ('All files', '*.*')])
            root.destroy()
        if video:
            run_tune_mode(video, args.circle_file, args.settings)
    else:
        if args.settings:
            apply_tune_values(load_tune_values(args.settings))
        App().run()


if __name__ == '__main__':
    main()

