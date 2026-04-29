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

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit(
        "Pillow is required.  Install it with:\n"
        "    pip install Pillow\n"
        "then re-run this script."
    )

# ── Processing parameters (tune these if detection is noisy) ─────────────────
PROCESS_SCALE    = 0.25   # resize factor: 4K→960×540 for processing
DIFF_THRESH      = 15     # motion threshold — higher = fewer blobs (was 12, see notes)
FRAME_STRIDE     = 2      # compare frame t to frame t-STRIDE
MIN_BLOB_AREA    = 10     # min blob area in pixels at PROCESS_SCALE
MAX_BLOB_AREA    = 1200   # max blob area in pixels at PROCESS_SCALE
MAX_TRACK_DIST   = 50     # max pixel distance to link blobs between frames

# Persistence: how long to keep a track alive when its blob temporarily vanishes
# (ant stops, enters the hole, passes through shadow).  2–3 s works well.
MAX_MISSING_SEC  = 2.5 # 

# Crossing hysteresis: an ant must be on the same side of the polygon boundary
# for this many consecutive frames before the crossing is committed.
# Prevents spurious counts when an ant hovers on the boundary or tracking hiccups.
# At 60 fps, 6 frames = 0.1 s.  Raise if you see double-counts at the edge.
CROSS_HYSTERESIS_FRAMES = 6

# Camera-vibration guard: if >VIBRATION_PCT % of pixels change simultaneously,
# it's a whole-frame shake → skip detection for that frame (don't count blobs)
VIBRATION_PCT    = 8.0    # percent of frame pixels; raise if too many skips

# ── Hardcoded video shortcuts ────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS = {
    "GX010314": os.path.join(_SCRIPT_DIR, "GX010314", "GX010314.MP4"),
    "GX010319": os.path.join(_SCRIPT_DIR, "GX010319", "GX010319.MP4"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class Tracker:
    """
    Nearest-neighbour centroid tracker with hysteresis crossing detection.

    Each track carries a pending-crossing counter.  When an ant's centroid
    flips sides, we start counting frames on the new side.  Only after
    CROSS_HYSTERESIS_FRAMES consecutive frames on the new side do we commit
    the crossing as an event.  This eliminates spurious counts when an ant
    lingers on the boundary, or tracking briefly hiccups.

    With MAX_MISSING_SEC = 3 s, an ant that stops moving inside the hole
    (and temporarily loses its blob) keeps its track alive and its
    "inside" status until a blob reappears or the timeout expires.
    """

    def __init__(self, max_dist: float, max_missing_frames: int,
                 hysteresis_frames: int = 6):
        self.max_dist = max_dist
        self.max_missing = max_missing_frames
        self.hyst = hysteresis_frames
        self._ants: dict = {}
        # id → {cx, cy, missing, committed_inside, pending_side, pending_count}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    def update(self, blobs: list, poly_pts) -> list:
        """
        Update with blobs from the current frame.

        Returns list of committed crossing events:
            {"ant_id", "event": "enter"|"exit", "cx", "cy"}
        """
        events = []
        matched_ant_ids: set = set()
        matched_blob_idx: set = set()

        # ── match existing tracks to nearest unmatched blob ──────────
        for ant_id, ant in list(self._ants.items()):
            best_d, best_i = self.max_dist, -1
            for i, blob in enumerate(blobs):
                if i in matched_blob_idx:
                    continue
                d = math.hypot(blob["cx"] - ant["cx"], blob["cy"] - ant["cy"])
                if d < best_d:
                    best_d, best_i = d, i
            if best_i >= 0:
                matched_ant_ids.add(ant_id)
                matched_blob_idx.add(best_i)
                ant["cx"] = blobs[best_i]["cx"]
                ant["cy"] = blobs[best_i]["cy"]
                ant["missing"] = 0

        # ── age out long-lost tracks ──────────────────────────────────
        for ant_id in list(self._ants.keys()):
            if ant_id not in matched_ant_ids:
                self._ants[ant_id]["missing"] += 1
                if self._ants[ant_id]["missing"] > self.max_missing:
                    del self._ants[ant_id]

        # ── create new tracks for unmatched blobs ────────────────────
        for i, blob in enumerate(blobs):
            if i in matched_blob_idx:
                continue
            inside = self._in_poly(blob["cx"], blob["cy"], poly_pts)
            self._ants[self._next_id] = {
                "cx": blob["cx"], "cy": blob["cy"],
                "missing": 0,
                "committed_inside": inside,  # last committed side
                "pending_side": inside,       # side we're accumulating toward
                "pending_count": 0,
            }
            self._next_id += 1

        # ── hysteresis crossing detection ────────────────────────────
        if poly_pts and len(poly_pts) >= 3:
            for ant_id, ant in self._ants.items():
                now = self._in_poly(ant["cx"], ant["cy"], poly_pts)
                committed = ant["committed_inside"]

                if now == ant["pending_side"]:
                    # Still accumulating toward the same candidate side
                    ant["pending_count"] += 1
                else:
                    # Flipped — start accumulating toward the new side
                    ant["pending_side"]  = now
                    ant["pending_count"] = 1

                # Commit once we've been on the new side long enough
                if (ant["pending_count"] >= self.hyst
                        and ant["pending_side"] != committed):
                    ant["committed_inside"] = ant["pending_side"]
                    evt = "enter" if ant["pending_side"] else "exit"
                    events.append({"ant_id": ant_id, "event": evt,
                                   "cx": ant["cx"], "cy": ant["cy"]})

        return events

    def ants(self) -> dict:
        return self._ants

    def committed_inside(self, ant_id) -> bool:
        """Return the committed (hysteresis-filtered) inside state."""
        ant = self._ants.get(ant_id)
        return ant["committed_inside"] if ant else False

    @staticmethod
    def _in_poly(x, y, pts) -> bool:
        if not pts:
            return False
        np_pts = np.array(pts, dtype=np.float32)
        return cv2.pointPolygonTest(np_pts, (float(x), float(y)), False) >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Video Processor (runs in a background thread)
# ─────────────────────────────────────────────────────────────────────────────

class VideoProcessor:
    def __init__(self, video_path, polygon_video_coords,
                 progress_cb, status_cb):
        self.video_path = video_path
        self.poly_video = list(polygon_video_coords)
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

        # Scale polygon to process resolution
        poly_proc = [(x * PROCESS_SCALE, y * PROCESS_SCALE)
                     for x, y in self.poly_video]

        # Output directory
        out_dir = os.path.join(os.path.dirname(self.video_path), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(self.video_path))[0]

        # Save polygon
        poly_path = os.path.join(out_dir, f"{stem}_polygon.json")
        with open(poly_path, "w") as f:
            json.dump({
                "video": self.video_path,
                "polygon_video_coords": self.poly_video,
                "orig_width": orig_w, "orig_height": orig_h,
                "process_scale": PROCESS_SCALE,
                "detection": "frame_diff",
                "frame_stride": FRAME_STRIDE,
                "diff_thresh": DIFF_THRESH,
            }, f, indent=2)

        # Single-pass: frame-to-frame diff (no background estimation needed)
        csv_path = os.path.join(out_dir, f"{stem}_counts.csv")
        vid_path = os.path.join(out_dir, f"{stem}_counted.mp4")
        self.progress_cb(0, "Processing frames…")
        events = self._process_frames(
            poly_proc, proc_w, proc_h, fps, total, csv_path, vid_path)

        if self._cancel:
            return None, None

        return events, {
            "csv": csv_path, "video": vid_path,
            "polygon": poly_path, "output_dir": out_dir,
        }

    # ------------------------------------------------------------------
    def _process_frames(self, poly_proc, proc_w, proc_h,
                        fps, total, csv_path, vid_path):
        tracker = Tracker(
            max_dist=MAX_TRACK_DIST,
            max_missing_frames=int(MAX_MISSING_SEC * fps),
            hysteresis_frames=CROSS_HYSTERESIS_FRAMES,
        )
        #total = min(total, int(fps * 60))   # first 60 seconds only
        poly_int = np.array(poly_proc, dtype=np.int32)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vid_path, fourcc, fps, (proc_w, proc_h), True)

        # 7×7 close merges nearby motion pixels into one blob;
        # 5×5 open removes isolated speckles smaller than an ant.
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        frame_pixels = proc_w * proc_h
        vibration_limit = int(frame_pixels * VIBRATION_PCT / 100)

        all_events = []
        running = 0
        n_vibration_skips = 0

        # Ring buffer: diff frame t against frame t-FRAME_STRIDE.
        frame_buf: list = []

        cap = cv2.VideoCapture(self.video_path)
        csv_rows = [["timestamp_s", "time", "frame", "event",
                     "running_count", "ant_id", "x_orig", "y_orig"]]

        for fid in range(total):
            if self._cancel:
                break
            ret, frame = cap.read()
            if not ret:
                break

            # ── resize + grayscale ────────────────────────────────────
            small = cv2.resize(frame, (proc_w, proc_h))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

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

            # ── tracking & crossing detection ─────────────────────────
            events = tracker.update(blobs, poly_proc)
            ts  = fid / fps
            ts_str = str(timedelta(seconds=int(ts)))

            for ev in events:
                running += 1 if ev["event"] == "enter" else -1
                rec = {
                    "timestamp_s": ts, "time": ts_str, "frame": fid,
                    "event": ev["event"], "running_count": running,
                    "ant_id": ev["ant_id"],
                    "x_orig": ev["cx"] / PROCESS_SCALE,
                    "y_orig": ev["cy"] / PROCESS_SCALE,
                }
                all_events.append(rec)
                csv_rows.append([
                    f"{ts:.3f}", ts_str, fid, ev["event"], running,
                    ev["ant_id"],
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

            # polygon overlay
            cv2.polylines(out, [poly_int], True, (0, 220, 0), 2)

            # ant blobs
            # cyan  = committed inside (counted as entered)
            # blue  = committed outside
            # yellow = pending / accumulating toward a crossing (in hysteresis)
            for ant_id, ant in tracker.ants().items():
                cx, cy = int(ant["cx"]), int(ant["cy"])
                committed = ant["committed_inside"]
                pending   = ant["pending_side"]
                in_hyst   = (pending != committed)   # mid-crossing accumulation
                if in_hyst:
                    color = (0, 220, 255)   # yellow-ish: pending crossing
                elif committed:
                    color = (0, 255, 160)   # cyan-green: confirmed inside
                else:
                    color = (0, 120, 255)   # blue: confirmed outside
                r = 6
                cv2.circle(out, (cx, cy), r, color, 2)
                cv2.circle(out, (cx, cy), 2, color, -1)

            # count box (top-left)
            sign  = "+" if running >= 0 else ""
            label = f"Net: {sign}{running}"
            cv2.rectangle(out, (8, 8), (210, 55), (0, 0, 0), -1)
            cv2.putText(out, label, (12, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 90), 2, cv2.LINE_AA)

            # recent event flash (bottom-left, fades after 1 s)
            if all_events and fid - all_events[-1]["frame"] < fps:
                last = all_events[-1]
                ev_txt = "▲ ENTER" if last["event"] == "enter" else "▼ EXIT"
                ev_col = (0, 255, 90) if last["event"] == "enter" else (50, 50, 255)
                cv2.rectangle(out, (6, proc_h - 45), (200, proc_h - 8), (0, 0, 0), -1)
                cv2.putText(out, ev_txt, (10, proc_h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, ev_col, 2, cv2.LINE_AA)

            # timestamp (below count box)
            cv2.putText(out, ts_str, (12, 80),
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

        # polygon in VIDEO pixel coords
        self.poly_video: list = []
        self.poly_closed: bool = False

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

        # Polygon
        pf = ttk.LabelFrame(sidebar, text="  Bounding Region  ")
        pf.pack(fill=tk.X, pady=4)
        tk.Label(pf,
                 text=("Left-click : add point\n"
                       "Double-click : close polygon\n"
                       "Right-click : remove last point"),
                 font=("Arial", 8), justify=tk.LEFT).pack(anchor=tk.W, padx=4)
        self._poly_lbl = tk.Label(pf, text="No polygon drawn",
                                  fg="#b00", font=("Arial", 9, "bold"))
        self._poly_lbl.pack(pady=2)
        tk.Button(pf, text="Clear Polygon",
                  command=self.clear_polygon).pack(pady=(0, 4))

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
        self.clear_polygon()
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

        # ── overlay polygon ───────────────────────────────────────────
        if self.poly_video:
            pts_c = self._v2c(self.poly_video)   # canvas coords
            pts_i = [(int(x - self.disp_off_x), int(y - self.disp_off_y))
                     for x, y in pts_c]           # image-relative

            # draw edges
            for i in range(len(pts_i) - 1):
                cv2.line(rgb, pts_i[i], pts_i[i + 1], (50, 255, 50), 2)
            if self.poly_closed and len(pts_i) >= 3:
                cv2.line(rgb, pts_i[-1], pts_i[0], (50, 255, 50), 2)
                filled = rgb.copy()
                cv2.fillPoly(filled, [np.array(pts_i, np.int32)], (50, 255, 50))
                cv2.addWeighted(filled, 0.18, rgb, 0.82, 0, rgb)

            # draw vertices
            for j, pt in enumerate(pts_i):
                col = (255, 230, 0) if j == 0 else (255, 255, 255)
                cv2.circle(rgb, pt, 6, (0, 0, 0), -1)
                cv2.circle(rgb, pt, 5, col, -1)

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
        if not self.cap or self.poly_closed:
            return
        vx, vy = self._c2v(event.x, event.y)
        self.poly_video.append((vx, vy))
        self._render()
        n = len(self.poly_video)
        self._poly_lbl.config(
            text=f"{n} point{'s' if n != 1 else ''}  —  double-click to close",
            fg="#664400")

    def _dblclick(self, event):
        """Close the polygon.  The second click of the dbl-click already added
        a duplicate point via _click, so we pop it before closing."""
        if not self.cap:
            return
        if self.poly_video:
            self.poly_video.pop()   # remove duplicate from 2nd click
        if len(self.poly_video) < 3:
            messagebox.showinfo("Too few points",
                                "Need at least 3 points to form a polygon.")
            return
        self.poly_closed = True
        self._poly_lbl.config(
            text=f"✓  Polygon ready  ({len(self.poly_video)} pts)",
            fg="#006600")
        self._render()
        self._refresh_proc_btn()

    def _rclick(self, event):
        if self.poly_video:
            self.poly_video.pop()
            self.poly_closed = False
            self._render()
            n = len(self.poly_video)
            self._poly_lbl.config(
                text=(f"{n} point{'s' if n != 1 else ''}."
                      if n else "No polygon drawn"),
                fg="#664400" if n else "#b00")
            self._refresh_proc_btn()

    def clear_polygon(self):
        self.poly_video = []
        self.poly_closed = False
        self._poly_lbl.config(text="No polygon drawn", fg="#b00")
        if self.current_bgr is not None:
            self._render()
        self._refresh_proc_btn()

    def _on_slider(self, val):
        self._show_frame(int(float(val)))

    def _resize(self, _event):
        if self.current_bgr is not None:
            self._render()

    def _refresh_proc_btn(self):
        ok = bool(self.cap and self.poly_closed and len(self.poly_video) >= 3)
        self._proc_btn.config(state=tk.NORMAL if ok else tk.DISABLED)

    # ══════════════════════════════════════════════════════════════════════════
    # Processing
    # ══════════════════════════════════════════════════════════════════════════

    def start_processing(self):
        if not (self.cap and self.poly_closed):
            return
        self._proc_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        self._prog_var.set(0)
        self._prog_lbl.config(text="Starting…")

        self.processor = VideoProcessor(
            self.video_path, list(self.poly_video),
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
            n_in  = sum(1 for e in events if e["event"] == "enter")
            n_out = sum(1 for e in events if e["event"] == "exit")
            msg = (f"Done!\n\n"
                   f"Entries : {n_in}\n"
                   f"Exits   : {n_out}\n"
                   f"Net     : {n_in - n_out:+d}\n\n"
                   f"Saved to:\n{paths['output_dir']}")
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
        ok = bool(self.cap and self.poly_closed)
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
