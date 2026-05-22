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
    python ant_counter.py

Workflow:
    1. Click a video button (or Browse…) to load a video.
    2. Scrub the frame slider to find a good view of the hole entrance.
    3. Left-click to place polygon points around the hole.
       Double-click to close the polygon.  Right-click to undo the last point.
    4. Click "Process Video".  Results land in <video_folder>/outputs/.

Outputs (per video):
    <name>_counts.csv       — timestamp, event (enter/exit), running count
    <name>_counted.mp4      — annotated video with live count overlay
    <name>_polygon.json     — bounding polygon coordinates
"""

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

def build_shadow_mask(frame_bgr, kernel_size=31):

    SHADOW_VAL_DROP = 40

    SHADOW_HUE_STABLE = 15


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
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_clean = cv2.morphologyEx(shadow_raw, cv2.MORPH_OPEN, k_open)

    return shadow_clean


# ── Processing parameters (tune these if detection is noisy) ─────────────────
PROCESS_SCALE    = 0.25   # resize factor: 4K→960×540 for processing
DIFF_THRESH      = 15     # motion threshold — higher = fewer blobs (was 12, see notes)
FRAME_STRIDE     = 2      # compare frame t to frame t-STRIDE
MIN_BLOB_AREA    = 15     # min blob area in pixels at PROCESS_SCALE
MAX_BLOB_AREA    = 1200   # max blob area in pixels at PROCESS_SCALE
MAX_TRACK_DIST   = 20     # max pixel distance to link blobs between frames

# Persistence: how long to keep a track alive when its blob temporarily vanishes
# (ant stops, enters the hole, passes through shadow).  2–3 s works well.
MAX_MISSING_SEC  = 2.5 # 

# Crossing hysteresis: an ant must be on the same side of the circle boundary
# for this many consecutive frames before the crossing is committed.
# Prevents spurious counts when an ant hovers on the boundary or tracking hiccups.
# At 60 fps, 6 frames = 0.1 s.  Raise if you see double-counts at the edge.
CROSS_HYSTERESIS_FRAMES = 6

# Camera-vibration guard: if >VIBRATION_PCT % of pixels change simultaneously,
# it's a whole-frame shake → skip detection for that frame (don't count blobs)
VIBRATION_PCT    = 8.0    # percent of frame pixels; raise if too many skips

# ── Circular quadrant detection parameters ─────────────────────────────────
DEFAULT_RADIUS   = 200    # Default circle radius in pixels
QUAD_COLORS = {
    "NE": (0, 255, 255),   # yellow
    "SE": (0, 255, 0),     # green
    "SW": (255, 0, 0),     # blue
    "NW": (255, 0, 255),   # magenta
}
THICKNESS = 4
CENTER_DOT_RADIUS = 5

# ── Hardcoded video shortcuts ────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS = {
    "GX010314": os.path.join(_SCRIPT_DIR, "GX010314", "GX010314.MP4"),
    "GX010319": os.path.join(_SCRIPT_DIR, "GX010319", "GX010319.MP4"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Circular Quadrant Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

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


def is_inside_circle(cx, cy, px, py, radius):
    """Check if point is inside the circle."""
    dist = math.hypot(px - cx, py - cy)
    return dist <= radius


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

    def norm(a):
        return a % 360

    ang_N = norm(ang_N)
    ang_E = norm(ang_E)
    ang_S = norm(ang_S)
    ang_W = norm(ang_W)

    # Draw an arc from start to start+90 (clockwise)
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
# Tracker (Kalman filter + Hungarian assignment)
# ─────────────────────────────────────────────────────────────────────────────



# Kalman tuning (constant-velocity model in image pixels at PROCESS_SCALE)
KF_PROC_POS = 2.0      # position process noise
KF_PROC_VEL = 8.0      # velocity process noise
KF_MEAS_POS = 6.0      # measurement noise (centroid observation)
MIN_COAST_AGE = 8      # frames before a lost track may persist (re-acquire flicker)
MIN_CROSS_AGE = 8      # frames before crossing events are counted
# Lost tracks stay in memory up to max_missing for re-acquire, but position
# is frozen and they are not drawn / counted once missing > 0.


class KalmanFilter2D:
    """Constant-velocity Kalman filter for blob centroids (state: x, y, vx, vy)."""

    def __init__(self, x: float, y: float):
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([500.0, 500.0, 1000.0, 1000.0]).astype(np.float64)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        self.Q = np.diag([KF_PROC_POS, KF_PROC_POS, KF_PROC_VEL, KF_PROC_VEL])
        self.R = np.diag([KF_MEAS_POS, KF_MEAS_POS])
        self._I = np.eye(4, dtype=np.float64)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, zx, zy):
        z = np.array([zx, zy], dtype=np.float64)
        innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (self._I - K @ self.H) @ self.P


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

    def __init__(self, max_dist: float, max_missing_frames: int,
                 hysteresis_frames: int = 6):
        self.max_dist = max_dist
        self.max_missing = max_missing_frames
        self.hyst = hysteresis_frames
        self._ants: dict = {}
        # id → {cx, cy, missing, committed_inside, committed_quadrant, pending_inside, pending_quadrant, pending_count}
        self._next_id: int = 0

    def _sync_track_state(self, ant: dict) -> None:
        kf = ant["kf"]
        ant["cx"], ant["cy"] = float(kf.x[0]), float(kf.x[1])
        ant["vx"], ant["vy"] = float(kf.x[2]), float(kf.x[3])

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
        }

    def _freeze_lost_track(self, ant: dict) -> None:
        """Hold position at last detection; stop Kalman drift when blob is gone."""
        ant["kf"].x[0] = ant["last_cx"]
        ant["kf"].x[1] = ant["last_cy"]
        ant["kf"].x[2] = 0.0
        ant["kf"].x[3] = 0.0
        ant["cx"] = ant["last_cx"]
        ant["cy"] = ant["last_cy"]
        ant["vx"] = 0.0
        ant["vy"] = 0.0
        ant["matched_this_frame"] = False
        # Do not let coasted frames (or re-acquire) commit a stale crossing.
        ant["pending_count"] = 0
        ant["pending_inside"] = ant["committed_inside"]
        ant["pending_quadrant"] = ant["committed_quadrant"]

    def _mark_all_unmatched(self) -> None:
        for ant in self._ants.values():
            ant["matched_this_frame"] = False

    # ------------------------------------------------------------------
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
        track_ids = list(self._ants.keys())
        self._mark_all_unmatched()

        # ── 1. Kalman predict (only for tracks seen last frame) ───────
        predictions = {}
        for ant_id in track_ids:
            ant = self._ants[ant_id]
            if ant["missing"] == 0:
                px, py = ant["kf"].predict()
            else:
                px, py = ant["last_cx"], ant["last_cy"]
            predictions[ant_id] = (px, py)
            if ant["missing"] == 0:
                self._sync_track_state(ant)

        # ── 2. Cost matrix + Hungarian match ──────────────────────────
        n_tracks = len(track_ids)
        n_blobs = len(blobs)
        matched_track_rows: set[int] = set()
        matched_blob_cols: set[int] = set()

        if n_tracks and n_blobs:
            cost = np.full((n_tracks, n_blobs), self.max_dist * 100.0, dtype=np.float64)
            for ri, ant_id in enumerate(track_ids):
                ant = self._ants[ant_id]
                px, py = predictions[ant_id]
                for bi, blob in enumerate(blobs):
                    if ant["missing"] > 2 and blob["area"] < MIN_BLOB_AREA * 3:
                        continue
                    d = math.hypot(blob["cx"] - px, blob["cy"] - py)
                    if d <= self.max_dist:
                        cost[ri, bi] = d

            for ri, bi in _hungarian_match(cost, self.max_dist):
                matched_track_rows.add(ri)
                matched_blob_cols.add(bi)
                ant_id = track_ids[ri]
                ant = self._ants[ant_id]
                blob = blobs[bi]
                ant["kf"].update(blob["cx"], blob["cy"])
                ant["missing"] = 0
                ant["age"] += 1
                self._sync_track_state(ant)
                ant["last_cx"] = ant["cx"]
                ant["last_cy"] = ant["cy"]
                ant["matched_this_frame"] = True

        matched_ant_ids = {track_ids[ri] for ri in matched_track_rows}

        # ── 3. Unmatched tracks: coast or delete ──────────────────────
        for ant_id in track_ids:
            if ant_id in matched_ant_ids:
                continue
            ant = self._ants[ant_id]
            ant["missing"] += 1
            if ant["missing"] > self.max_missing:
                del self._ants[ant_id]
            elif ant["age"] < MIN_COAST_AGE:
                del self._ants[ant_id]
            else:
                self._freeze_lost_track(ant)

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

        # ── hysteresis crossing detection (detected blobs only, never coast) ──

        for ant_id, ant in self._ants.items():
            if (ant["age"] < MIN_CROSS_AGE
                    or ant["missing"] > 0
                    or not ant.get("matched_this_frame", False)):
                continue

            current_inside = is_inside_circle(cx, cy, ant["cx"], ant["cy"], radius)
            current_quadrant = (
                get_quadrant(cx, cy, ant["cx"], ant["cy"], north_angle)
                if not current_inside else "CENTER"
            )

            current_state = (current_inside, current_quadrant)
            pending_state = (ant["pending_inside"], ant["pending_quadrant"])
            committed_state = (ant["committed_inside"], ant["committed_quadrant"])

            if current_state == pending_state:
                ant["pending_count"] += 1
            else:
                ant["pending_inside"] = current_inside
                ant["pending_quadrant"] = current_quadrant
                ant["pending_count"] = 1

            if (ant["pending_count"] >= self.hyst
                    and pending_state != committed_state):
                old_inside, old_quadrant = committed_state
                new_inside = ant["pending_inside"]
                new_quadrant = ant["pending_quadrant"]

                ant["committed_inside"] = new_inside
                ant["committed_quadrant"] = new_quadrant

                if old_inside and not new_inside:
                    events.append({
                        "ant_id": ant_id,
                        "event": f"enter_{new_quadrant}",
                        "cx": ant["cx"],
                        "cy": ant["cy"],
                        "quadrant": new_quadrant,
                    })
                elif not old_inside and new_inside:
                    events.append({
                        "ant_id": ant_id,
                        "event": f"exit_{old_quadrant}",
                        "cx": ant["cx"],
                        "cy": ant["cy"],
                        "quadrant": old_quadrant,
                    })

        return events

    def ants(self) -> dict:
        return self._ants

    def committed_inside(self, ant_id) -> bool:
        """Return the committed (hysteresis-filtered) inside state."""
        ant = self._ants.get(ant_id)
        return ant["committed_inside"] if ant else False

    def get_committed_quadrant(self, ant_id) -> str:
        """Return the committed quadrant."""
        ant = self._ants.get(ant_id)
        return ant["committed_quadrant"] if ant else "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Video Processor (runs in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

class VideoProcessor:
    def __init__(self, video_path, circle_params,
                 progress_cb, status_cb):
        self.video_path = video_path
        self.circle_params = circle_params  # dict with center, radius, north_angle
        self.progress_cb = progress_cb
        self.status_cb = status_cb
        self._cancel = False

    def cancel(self):
        self._cancel = True

    # ------------------------------------------------------------------
    def run(self):
        """Main pipeline.  Returns (events, paths_dict) or (None, None)."""
        cap = cv2.VideoCapture(self.video_path)
        fps      = cap.get(cv2.CAP_PROP_FPS)
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

        # Output directory
        out_dir = os.path.join(os.path.dirname(self.video_path), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.video_path))[0]

        # Save circle parameters
        circle_path = os.path.join(out_dir, f"{stem}_circle.json")
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
        csv_path = os.path.join(out_dir, f"{stem}_counts.csv")
        vid_path = os.path.join(out_dir, f"{stem}_counted.mp4")
        self.progress_cb(0, "Processing frames…")
        events = self._process_frames(
            circle_proc, proc_w, proc_h, fps, total, csv_path, vid_path)

        if self._cancel:
            return None, None

        return events, {
            "csv": csv_path, "video": vid_path,
            "circle": circle_path, "output_dir": out_dir,
        }

    # ------------------------------------------------------------------
    def _process_frames(self, circle_proc, proc_w, proc_h,
                        fps, total, csv_path, vid_path):
        tracker = Tracker(
            max_dist=MAX_TRACK_DIST,
            max_missing_frames=int(MAX_MISSING_SEC * fps),
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
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        frame_pixels = proc_w * proc_h
        vibration_limit = int(frame_pixels * VIBRATION_PCT / 100)

        all_events = []
        # Track counts per quadrant direction
        quad_counts = {"NE": 0, "SE": 0, "SW": 0, "NW": 0}
        n_vibration_skips = 0

        # Ring buffer: diff frame t against frame t-FRAME_STRIDE.
        frame_buf: list = []

        cap = cv2.VideoCapture(self.video_path)
        csv_rows = [["timestamp_s", "time", "frame", "event", "quadrant",
                     "quad_count", "ant_id", "x_orig", "y_orig"]]

        for fid in range(total):
            if self._cancel:
                break
            ret, frame = cap.read()
            if not ret:
                break

            # ── resize + grayscale ────────────────────────────────────
            small = cv2.resize(frame, (proc_w, proc_h))
            # gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            gray = hsv[:,:,2]

            # ── frame-to-frame diff ───────────────────────────────────
            frame_buf.append(gray.copy())
            if len(frame_buf) > FRAME_STRIDE + 1:
                frame_buf.pop(0)

            if len(frame_buf) > FRAME_STRIDE:
                # Pre-blur reduces single-pixel speckle before thresholding
                raw_diff = cv2.absdiff(gray, frame_buf[0])
                diff = cv2.GaussianBlur(raw_diff, (3, 3), 0)
            else:
                diff = np.zeros((proc_h, proc_w), dtype=np.uint8)
                raw_diff = diff

            _, mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
            mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
            mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)

            shadow_mask = build_shadow_mask(small)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(shadow_mask))

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
                n_lbl, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
                blobs = []
                for i in range(1, n_lbl):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
                        blobs.append({"cx": centroids[i][0],
                                      "cy": centroids[i][1],
                                      "area": int(area)})
                # Cluster blobs to reduce shadow double-counting (changed from 15 to 40)
                blobs = cluster_blobs(blobs, max_dist=40)

            # ── tracking & crossing detection ─────────────────────────
            events = tracker.update(blobs, circle_proc)
            ts  = fid / fps
            ts_str = str(timedelta(seconds=int(ts)))

            for ev in events:
                quadrant = ev["quadrant"]
                if ev["event"].startswith("enter_"):
                    quad_counts[quadrant] += 1
                elif ev["event"].startswith("exit_"):
                    quad_counts[quadrant] -= 1
                    
                rec = {
                    "timestamp_s": ts, "time": ts_str, "frame": fid,
                    "event": ev["event"], "quadrant": quadrant,
                    "quad_count": quad_counts[quadrant],
                    "ant_id": ev["ant_id"],
                    "x_orig": ev["cx"] / PROCESS_SCALE,
                    "y_orig": ev["cy"] / PROCESS_SCALE,
                }
                all_events.append(rec)
                csv_rows.append([
                    f"{ts:.3f}", ts_str, fid, ev["event"], quadrant,
                    quad_counts[quadrant], ev["ant_id"],
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

            # circle overlay with quadrants
            draw_quadrant_arcs(out, (int(cx), int(cy)), int(radius), north_angle)

            # ant blobs — only draw actively detected tracks (not coasting ghosts)
            for ant_id, ant in tracker.ants().items():
                if ant["missing"] > 0:
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
                    
                r = 6
                cv2.circle(out, (ax, ay), r, color, 2)
                cv2.circle(out, (ax, ay), 2, color, -1)

            # quadrant count box (top-left)
            cv2.rectangle(out, (8, 8), (280, 120), (0, 0, 0), -1)
            y_offset = 25
            for i, (quad, count) in enumerate(quad_counts.items()):
                color = QUAD_COLORS[quad]
                sign = "+" if count >= 0 else ""
                label = f"{quad}: {sign}{count}"
                cv2.putText(out, label, (12, y_offset + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            # recent event flash (bottom-left, fades after 1 s)
            if all_events and fid - all_events[-1]["frame"] < fps:
                last = all_events[-1]
                event_name = last["event"]
                quadrant = last["quadrant"]
                if event_name.startswith("enter_"):
                    ev_txt = f"▲ ENTER {quadrant}"
                    ev_col = (0, 255, 90)
                elif event_name.startswith("exit_"):
                    ev_txt = f"▼ EXIT {quadrant}"
                    ev_col = (50, 50, 255)
                else:
                    ev_txt = event_name
                    ev_col = (255, 255, 255)
                cv2.rectangle(out, (6, proc_h - 45), (250, proc_h - 8), (0, 0, 0), -1)
                cv2.putText(out, ev_txt, (10, proc_h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, ev_col, 2, cv2.LINE_AA)

            # timestamp (below count box)
            cv2.putText(out, ts_str, (12, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1, cv2.LINE_AA)

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


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────

class App:
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

        self._build_ui()

    # ══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════════

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
        for name, path in VIDEOS.items():
            tk.Button(vf, text=name, width=20, relief=tk.GROOVE,
                      command=lambda p=path: self.load_video(p)).pack(pady=1)
        tk.Button(vf, text="Browse…", width=20, relief=tk.GROOVE,
                  command=self.browse_video).pack(pady=(1, 4))

        self._info_lbl = tk.Label(sidebar, text="", font=("Courier", 8),
                                  justify=tk.LEFT, bg="#f4f4f4", fg="#444")
        self._info_lbl.pack(anchor=tk.W, pady=2)

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

    # ══════════════════════════════════════════════════════════════════════════
    # Video loading
    # ══════════════════════════════════════════════════════════════════════════

    def load_video(self, path: str):
        if not os.path.exists(path):
            messagebox.showerror("File not found", f"Cannot find:\n{path}")
            return
        self._open_video(path)

    def browse_video(self):
        p = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video", "*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI"),
                       ("All", "*.*")])
        if p:
            self._open_video(p)

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

    # ══════════════════════════════════════════════════════════════════════════
    # Frame display
    # ══════════════════════════════════════════════════════════════════════════

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

    def _render(self):
        if self.current_bgr is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 4 or ch < 4:
            return

        scale = min(cw / self.orig_w, ch / self.orig_h)
        self.disp_scale = scale
        dw = int(self.orig_w  * scale)
        dh = int(self.orig_h  * scale)
        self.disp_off_x = (cw - dw) // 2
        self.disp_off_y = (ch - dh) // 2

        small = cv2.resize(self.current_bgr, (dw, dh))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # ── overlay circle and quadrants ────────────────────────────
        if self.circle_center:
            # Convert center to display coordinates
            cx_disp, cy_disp = self._v2c([self.circle_center])[0]
            cx_img = int(cx_disp - self.disp_off_x)
            cy_img = int(cy_disp - self.disp_off_y)
            radius_img = int(self.circle_radius * self.disp_scale)
            
            # If we have north direction, draw full quadrants
            if self.north_point and self.circle_complete:
                north_angle = angle_deg_from_center_to_point(
                    self.circle_center[0], self.circle_center[1],
                    self.north_point[0], self.north_point[1])
                draw_quadrant_arcs(rgb, (cx_img, cy_img), radius_img, north_angle)
            else:
                # Just draw the circle outline
                cv2.circle(rgb, (cx_img, cy_img), radius_img, (50, 255, 50), 2)
                cv2.circle(rgb, (cx_img, cy_img), 5, (50, 255, 50), -1)
                
                # If we have north point, draw the direction line
                if self.north_point:
                    nx_disp, ny_disp = self._v2c([self.north_point])[0]
                    nx_img = int(nx_disp - self.disp_off_x)
                    ny_img = int(ny_disp - self.disp_off_y)
                    cv2.line(rgb, (cx_img, cy_img), (nx_img, ny_img), (255, 255, 0), 2)
                    cv2.circle(rgb, (nx_img, ny_img), 5, (255, 255, 0), -1)

        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(self.disp_off_x, self.disp_off_y,
                                 anchor=tk.NW, image=self._photo)

    # ══════════════════════════════════════════════════════════════════════════
    # Polygon interaction
    # ══════════════════════════════════════════════════════════════════════════

    def _v2c(self, pts):
        """Video coords → canvas coords."""
        return [(x * self.disp_scale + self.disp_off_x,
                 y * self.disp_scale + self.disp_off_y)
                for x, y in pts]

    def _c2v(self, cx, cy):
        """Canvas coords → video coords."""
        return ((cx - self.disp_off_x) / self.disp_scale,
                (cy - self.disp_off_y) / self.disp_scale)

    def _click(self, event):
        if not self.cap:
            return
        vx, vy = self._c2v(event.x, event.y)
        
        if self.circle_center is None:
            # First click: set center
            self.circle_center = (vx, vy)
            self.circle_radius = self._radius_var.get()
            self._render()
            self._circle_lbl.config(
                text="Center set — click for north direction",
                fg="#664400")
        elif self.north_point is None:
            # Second click: set north direction
            self.north_point = (vx, vy)
            self.circle_complete = True
            self._render()
            self._circle_lbl.config(
                text="✓ Circle ready with quadrants",
                fg="#006600")
            self._refresh_proc_btn()

    def _dblclick(self, event):
        """Double-click does nothing in circle mode."""
        pass

    def _rclick(self, event):
        """Right-click clears the circle setup."""
        self.clear_circle()

    def clear_circle(self):
        """Clear circle setup."""
        self.circle_center = None
        self.north_point = None
        self.circle_complete = False
        self._circle_lbl.config(text="No circle defined", fg="#b00")
        if self.current_bgr is not None:
            self._render()
        self._refresh_proc_btn()
    
    def _on_radius_change(self):
        """Called when radius spinbox value changes."""
        self.circle_radius = self._radius_var.get()
        if self.current_bgr is not None:
            self._render()

    def _on_slider(self, val):
        self._show_frame(int(float(val)))

    def _resize(self, _event):
        if self.current_bgr is not None:
            self._render()

    def _refresh_proc_btn(self):
        ok = bool(self.cap and self.circle_complete)
        self._proc_btn.config(state=tk.NORMAL if ok else tk.DISABLED)

    # ══════════════════════════════════════════════════════════════════════════
    # Processing
    # ══════════════════════════════════════════════════════════════════════════

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
            progress_cb=self._on_progress,
            status_cb=self._status,
        )
        threading.Thread(target=self._thread_body, daemon=True).start()

    def _thread_body(self):
        try:
            events, paths = self.processor.run()
            if events is None:
                self._status("Cancelled.")
                self.root.after(0, self._reset_btns)
                return
                
            # Count events by quadrant
            quad_totals = {"NE": 0, "SE": 0, "SW": 0, "NW": 0}
            for e in events:
                quad = e["quadrant"]
                if e["event"].startswith("enter_"):
                    quad_totals[quad] += 1
                elif e["event"].startswith("exit_"):
                    quad_totals[quad] -= 1
            
            msg_lines = ["Done!\n"]
            for quad, count in quad_totals.items():
                msg_lines.append(f"{quad}: {count:+d}")
            msg_lines.extend([f"\nTotal events: {len(events)}", 
                             f"Saved to:\n{paths['output_dir']}"])
            
            msg = "\n".join(msg_lines)
            self._status(msg)
            self.root.after(0, lambda: self._prog_var.set(100))
            self.root.after(0, lambda: self._prog_lbl.config(text="Complete!"))
        except Exception as exc:
            import traceback
            self._status(f"⚠ Error:\n{exc}\n\n{traceback.format_exc()[:400]}")
        finally:
            self.root.after(0, self._reset_btns)

    def cancel_processing(self):
        if self.processor:
            self.processor.cancel()

    def _reset_btns(self):
        ok = bool(self.cap and self.circle_complete)
        self._proc_btn.config(state=tk.NORMAL if ok else tk.DISABLED)
        self._cancel_btn.config(state=tk.DISABLED)

    def _on_progress(self, val: float, msg: str = ""):
        self.root.after(0, lambda: self._prog_var.set(val))
        if msg:
            self.root.after(0, lambda: self._prog_lbl.config(text=msg))

    def _status(self, msg: str):
        self.root.after(0, lambda: self._status_lbl.config(text=msg))

    # ══════════════════════════════════════════════════════════════════════════

    def run(self):
        self.root.mainloop()
        if self.cap:
            self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    App().run()
