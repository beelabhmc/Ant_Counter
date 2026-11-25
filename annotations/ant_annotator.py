#!/usr/bin/env python3
"""Ant Annotator: mark ant centers on 30-second clips, one frame every second.

HOW TO USE
    python -m pip install -r requirements.txt
    python ant_annotator.py
    python ant_annotator.py --tracker ../ant_tracker_modular.py            # with guessed points
    python ant_annotator.py --tracker ../ant_tracker_modular.py --parameters ant_tuning_parameters.json

1. Click Output folder and choose your project directory.
2. Click Add videos, select a source video, choose a start time and click
   "Go — annotate". The 30 s clip is saved to videos_30s/.
3. Each second of the clip (frames at 0, 1, 2, ... 29 s) is one annotation frame.
   - Click empty space to add an ant center; drag a point to move it;
     right-click a point (or select it in the list and Delete) to remove it.
   - Ctrl+Z undoes; the mouse wheel zooms. Coordinates are always in the
     clip's original pixels.
4. Press Enter (or "Confirm frame") when the points on a frame are right,
   including a frame with no ants. ONLY CONFIRMED FRAMES are written to the CSV.
   Moving on from a frame with unconfirmed points asks whether to confirm it.
5. With --tracker, the tracker runs on each new clip in the background and
   pre-fills every frame with guessed points (yellow). Fix them, then confirm.
   Guesses are only a starting point: ants that stand still are not detected,
   so look for missed ants (untick "Show guesses" to see the frame clean).
6. Work is saved after every change: progress/<clip>.json (resume with
   "Open existing annotation") and csv/<clip>.csv with columns
   Timestamp, frame, ant_x, ant_y (zero-based clip frames). There are no ant IDs.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

CLIP_SECONDS = 30
STEP_SECONDS = 1            # one annotation frame every second of the clip
GRAB_RADIUS_PX = 12         # screen pixels: click this close to a point to drag it
DELETE_RADIUS_PX = 20       # screen pixels: right-click this close to delete
PROGRESS_VERSION = 2

COLOR_CONFIRMED = "#00ff88"  # points on a confirmed frame
COLOR_USER = "#4fc3f7"       # added/moved by you, frame not yet confirmed
COLOR_GUESS = "#ffd600"      # guessed by the tracker, not yet confirmed

def safe_stem(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem).strip("._") or "video"


def timestamp(frame, fps):
    seconds = frame / fps
    h = int(seconds // 3600)
    m = int(seconds // 60) % 60
    s = int(seconds) % 60
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        return timestamp(frame + 0.000001, fps)  # practically unreachable
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def probe(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not math.isfinite(fps) or fps <= 0 or frames < 1 or width < 1 or height < 1:
        raise ValueError("Video does not report valid FPS, frame count or dimensions")
    return fps, frames, width, height


def read_frame(path, frame):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Unable to read frame {frame}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def export_clip(src, dst, start_frame, fps, width, height, progress):
    """Frame-exact sequential extraction; output frame 0 == source start_frame.

    Re-encoding ensures output frame positions are zero-based. There is no audio.
    """
    nframes = round(CLIP_SECONDS * fps)
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    dst.parent.mkdir(parents=True, exist_ok=True)
    # mp4v is widely available across OpenCV builds; no audio retained.
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot write MP4; check OpenCV codec support")
    try:
        for i in range(nframes):
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"Source ended after {i} of {nframes} requested frames")
            writer.write(img)
            if i % max(1, int(fps)) == 0:
                progress(i, nframes)
    except Exception:
        writer.release()
        cap.release()
        dst.unlink(missing_ok=True)
        raise
    writer.release()
    cap.release()
    # Inspect actual output to prevent mismatched annotation metadata.
    actual_fps, actual_n, _, _ = probe(dst)
    if actual_n != nframes or abs(actual_fps - fps) > 0.05:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"Export validation failed (expected {nframes} at {fps:.5f} FPS; got {actual_n} at {actual_fps:.5f} FPS)")
    progress(nframes, nframes)



# ─────────────────────────────────────────────────────────────────────────────
# Annotation data (no ant IDs: a frame is just a list of points)
# A point is [x, y, src]: src "g" = guessed by the tracker, "u" = placed or
# moved by the user. Only frames the user CONFIRMS are exported.
# ─────────────────────────────────────────────────────────────────────────────
def sample_frames_for(total_frames, fps):
    """One annotation frame every STEP_SECONDS of the clip (0, 1, 2, ... s)."""
    return sorted(set(min(total_frames - 1, round(sec * fps))
                      for sec in range(0, CLIP_SECONDS, STEP_SECONDS)))


def migrate_progress(data):
    """Read a progress JSON (any version) into (points, confirmed, suggestions).

    Version 1 files stored {frame: {ant_id: [x, y]}}; the IDs are dropped and,
    since those points were placed by hand, frames that have points count as
    confirmed.
    """
    version = int(data.get("version", 1))
    points = {}
    for key, value in data.get("annotations", {}).items():
        frame = int(key)
        if isinstance(value, dict):                     # v1: {ant_id: [x, y]}
            pts = [[int(xy[0]), int(xy[1]), "u"] for xy in value.values()]
        else:                                           # v2: [[x, y, src], ...]
            pts = [[int(p[0]), int(p[1]), p[2] if len(p) > 2 and p[2] in ("g", "u") else "u"]
                   for p in value]
        if pts:
            points[frame] = pts
    if version < 2:
        confirmed = set(points)
    else:
        confirmed = {int(f) for f in data.get("confirmed_frames", [])}
    suggestions = {int(k): [[int(x), int(y)] for x, y in v]
                   for k, v in data.get("suggestions", {}).items()}
    return points, confirmed, suggestions


def csv_rows(points, confirmed, fps):
    """CSV rows for confirmed frames only: Timestamp, frame, ant_x, ant_y."""
    rows = []
    for frame in sorted(confirmed):
        for x, y, _src in sorted(points.get(frame, []), key=lambda p: (p[1], p[0])):
            rows.append([timestamp(frame, fps), frame, x, y])
    return rows


def merge_guesses(points, confirmed, guesses, min_gap=8):
    """Put fresh guesses on every UNCONFIRMED frame: keep the user's own points,
    replace earlier guesses, and skip guesses within min_gap px of a user point.
    Confirmed frames are never touched."""
    for frame, guessed in guesses.items():
        if frame in confirmed:
            continue
        own = [p for p in points.get(frame, []) if p[2] == "u"]
        new = [[x, y, "g"] for x, y in guessed
               if all((x - p[0]) ** 2 + (y - p[1]) ** 2 > min_gap ** 2 for p in own)]
        if own or new:
            points[frame] = own + new
        else:
            points.pop(frame, None)


# ─────────────────────────────────────────────────────────────────────────────
# Guessed points from ant_tracker_modular.py (--tracker)
# ─────────────────────────────────────────────────────────────────────────────
class TrackerGuesser:
    """Loads ant_tracker_modular.py from a path and runs its detection +
    tracking over a clip (the same pipeline Optuna scores). Returns the
    detected ant centers on the requested frames, in original clip pixels."""

    REQUIRED = ("TrackingTrialRunner", "OptimizationVideoPair",
                "load_tune_values", "apply_tune_values", "PROCESS_SCALE")

    def __init__(self, tracker_path, parameters_path=None):
        path = Path(tracker_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Tracker not found: {path}")
        spec = importlib.util.spec_from_file_location("ant_tracker_for_annotator", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module          # needed by its dataclasses
        try:
            spec.loader.exec_module(module)      # runs imports only; main() is not called
        except SystemExit as exc:
            raise RuntimeError(f"Could not load the tracker: {exc}") from None
        missing = [name for name in self.REQUIRED if not hasattr(module, name)]
        if missing:
            raise RuntimeError(f"{path.name} is not a compatible ant tracker "
                               f"(missing: {', '.join(missing)})")
        self.module = module
        self.path = path
        self.parameters_path = None
        digest = hashlib.sha256(path.read_bytes())
        if parameters_path:
            params = Path(parameters_path).expanduser().resolve()
            values = module.load_tune_values(str(params))
            if not values:
                raise ValueError(f"No recognized parameters in {params}")
            module.apply_tune_values(values)
            self.parameters_path = params
            digest.update(params.read_bytes())
        self.settings_id = digest.hexdigest()[:16]

    def describe(self):
        return {"tracker": str(self.path),
                "parameters": str(self.parameters_path) if self.parameters_path else None,
                "settings_id": self.settings_id}

    def label(self):
        params = self.parameters_path.name if self.parameters_path else "default settings"
        return f"{self.path.name} ({params})"

    def guess(self, clip, frames, progress=None):
        module = self.module
        pair = module.OptimizationVideoPair(Path(clip).stem, Path(clip), Path(clip), 0)
        params = {"PROCESS_SCALE": float(module.PROCESS_SCALE)}
        result = module.TrackingTrialRunner().run_video(
            pair, params, tuple(int(f) for f in frames), progress_callback=progress)
        guesses = {int(f): [] for f in frames}
        for obs in result.observations:
            if obs.is_observed:   # real detections only, not Kalman predictions
                guesses[int(obs.frame_index)].append([int(round(obs.x)), int(round(obs.y))])
        return guesses


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────
class AntAnnotator:
    def __init__(self, root, guesser=None, guesser_error=None):
        self.root = root
        root.title("Ant Annotator — 30-second clips, one frame per second")
        root.geometry("1280x880")
        self.output = Path.cwd() / "ant_annotation_output"
        self.sources = []
        self.source = None
        self.clip = None
        self.stem = None
        self.fps = 0
        self.total = 0
        self.width = self.height = 0
        self.sample_frames = []
        self.sample_index = 0
        self.points = {}          # frame -> [[x, y, src], ...]
        self.confirmed = set()    # frames whose points are checked
        self.suggestions = {}     # frame -> [[x, y], ...] raw tracker guesses
        self.guesses_from = None  # describe() of the tracker that made them
        self.history = []
        self.drag = None          # (frame, index) while dragging a point
        self.guesser = guesser
        self.guessing = False
        self.zoom = tk.DoubleVar(value=1.0)
        self.show_guesses = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Add videos and select an output folder.")
        self.clip_time = tk.StringVar(value="Start: —")
        self.mode = tk.StringVar(value="Ready")
        self.preview_photo = None
        self.display_photo = None
        self.current_rgb = None
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.busy = False

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="Add videos", command=self.add_videos).pack(side="left", padx=3)
        ttk.Button(top, text="Output folder", command=self.choose_output).pack(side="left", padx=3)
        ttk.Button(top, text="Extract selected 30s clip", command=self.extract).pack(side="left", padx=3)
        ttk.Button(top, text="Open existing annotation", command=self.open_existing).pack(side="left", padx=3)
        ttk.Label(top, textvariable=self.mode).pack(side="right")

        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8)
        sidebar = ttk.Frame(body, width=320)
        body.add(sidebar, weight=0)
        main = ttk.Frame(body)
        body.add(main, weight=1)

        ttk.Label(sidebar, text="Source videos", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.video_list = tk.Listbox(sidebar, height=6, exportselection=False)
        self.video_list.pack(fill="x", pady=5)
        self.video_list.bind("<<ListboxSelect>>", self.video_selected)
        ttk.Label(sidebar, text="Clip starting position (seconds)").pack(anchor="w", pady=(8, 0))
        self.start = tk.DoubleVar(value=0)
        self.start_slider = ttk.Scale(sidebar, from_=0, to=0, variable=self.start, command=self.preview_start)
        self.start_slider.pack(fill="x")
        start_line = ttk.Frame(sidebar)
        start_line.pack(fill="x", pady=4)
        ttk.Label(start_line, text="Start sec:").pack(side="left")
        self.start_entry = ttk.Entry(start_line, width=10)
        self.start_entry.pack(side="left", padx=4)
        self.start_entry.insert(0, "0")
        ttk.Button(start_line, text="Go — annotate", command=self.start_from_entry).pack(side="left")
        ttk.Label(sidebar, textvariable=self.clip_time, wraplength=290).pack(anchor="w")
        self.preview = tk.Label(sidebar, text="Select a video", relief="groove", width=35, height=8)
        self.preview.pack(fill="x", pady=6)
        ttk.Separator(sidebar).pack(fill="x", pady=6)

        ttk.Label(sidebar, text="Annotation", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        ttk.Label(sidebar, wraplength=290, justify="left",
                  text="Click: add a point · Drag: move it · Right-click: delete\n"
                       "Enter: confirm frame and go to the next").pack(anchor="w", pady=(2, 4))
        guess_box = ttk.Frame(sidebar)
        guess_box.pack(fill="x")
        self.guess_label = ttk.Label(guess_box, wraplength=290, justify="left", text="")
        self.guess_label.pack(anchor="w")
        row = ttk.Frame(guess_box)
        row.pack(fill="x", pady=2)
        ttk.Checkbutton(row, text="Show guesses", variable=self.show_guesses,
                        command=self.render).pack(side="left")
        self.rerun_btn = ttk.Button(row, text="Re-run guesses", command=self.run_guesses)
        self.rerun_btn.pack(side="right")
        self.confirm_btn = ttk.Button(sidebar, text="✓ Confirm frame (Enter)", command=self.toggle_confirm)
        self.confirm_btn.pack(fill="x", pady=(6, 2))
        nav = ttk.Frame(sidebar)
        nav.pack(fill="x", pady=2)
        ttk.Button(nav, text="◀ Previous", command=self.prev_frame).pack(side="left", expand=True, fill="x")
        ttk.Button(nav, text="Next ▶", command=self.next_frame).pack(side="left", expand=True, fill="x")
        row = ttk.Frame(sidebar)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="Undo (Ctrl+Z)", command=self.undo).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Delete selected", command=self.delete_selected).pack(side="left", expand=True, fill="x")
        row = ttk.Frame(sidebar)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="Clear frame", command=self.clear_frame).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Save now", command=self.save).pack(side="left", expand=True, fill="x")
        ttk.Label(sidebar, text="Zoom (mouse wheel also zooms)").pack(anchor="w", pady=(6, 0))
        ttk.Scale(sidebar, from_=0.5, to=4, variable=self.zoom, command=lambda _: self.render()).pack(fill="x")
        ttk.Button(sidebar, text="Reset zoom", command=lambda: (self.zoom.set(1.0), self.render())).pack(fill="x")
        self.frame_label = ttk.Label(sidebar, text="Frame: —", justify="left")
        self.frame_label.pack(anchor="w", pady=4)
        self.progress_label = ttk.Label(sidebar, text="", font=("TkDefaultFont", 10, "bold"))
        self.progress_label.pack(anchor="w")
        ttk.Label(sidebar, text="Points on this frame:").pack(anchor="w")
        self.ant_list = tk.Listbox(sidebar, height=7)
        self.ant_list.pack(fill="both", expand=True, pady=4)

        viewer = ttk.Frame(main)
        viewer.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(viewer, bg="#151515", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.bump_zoom(1.15))
        self.canvas.bind("<Button-5>", lambda e: self.bump_zoom(1 / 1.15))
        self.canvas.bind("<Button-3>", self.delete_nearest)
        self.canvas.bind("<Button-2>", self.delete_nearest)   # macOS right-click
        ttk.Label(root, textvariable=self.status, relief="sunken", padding=5, anchor="w").pack(fill="x")
        root.bind("<Left>", lambda e: self.prev_frame())
        root.bind("<Right>", lambda e: self.next_frame())
        root.bind("<Return>", lambda e: self.confirm_and_next())
        root.bind("<Control-z>", lambda e: self.undo())
        root.bind("<Command-z>", lambda e: self.undo())
        root.bind("<Delete>", lambda e: self.delete_selected())
        root.bind("<BackSpace>", lambda e: self.delete_selected())
        root.protocol("WM_DELETE_WINDOW", self.close)

        if guesser is not None:
            self.guess_label.configure(text=f"Guesses: {guesser.label()}")
        elif guesser_error:
            self.guess_label.configure(text=f"Guesses off: {guesser_error}")
        else:
            self.guess_label.configure(text="Guesses off (start with --tracker PATH "
                                            "to pre-fill points)")
        self._update_rerun_btn()

    # ── videos and clips ────────────────────────────────────────────────
    def choose_output(self):
        folder = filedialog.askdirectory(title="Choose output folder", initialdir=str(self.output.parent))
        if folder:
            self.output = Path(folder)
            self.status.set(f"Output: {self.output}")

    def add_videos(self):
        paths = filedialog.askopenfilenames(title="Select videos", filetypes=[("Videos", "*.mp4 *.mov *.avi *.mkv *.m4v *.webm"), ("All files", "*.*")])
        for p in paths:
            p = Path(p)
            if p not in self.sources:
                self.sources.append(p)
                self.video_list.insert("end", p.name)
        if paths and self.source is None:
            self.video_list.selection_set(0)
            self.video_selected()

    def video_selected(self, _=None):
        if self.busy or not self.video_list.curselection():
            return
        p = self.sources[self.video_list.curselection()[0]]
        try:
            fps, frames, width, height = probe(p)
        except Exception as ex:
            messagebox.showerror("Video error", str(ex))
            return
        if frames < round(CLIP_SECONDS * fps):
            messagebox.showerror("Video too short", f"{p.name} contains less than {CLIP_SECONDS} seconds of frames.")
            return
        self.source = p
        self.clip = None
        self.current_rgb = None  # never display an old clip while switching videos
        self.fps, self.total, self.width, self.height = fps, frames, width, height
        self.start_slider.configure(to=max(0, (frames - round(CLIP_SECONDS * fps)) / fps))
        self.start.set(0)
        self.start_entry.delete(0, "end")
        self.start_entry.insert(0, "0")
        self.preview_start()
        self.status.set(f"Selected {p.name}: {fps:.3f} FPS, {frames} frames ({frames / fps:.2f}s). Enter start seconds, then click Go — annotate.")
        self.canvas.delete("all")
        self.frame_label.configure(text="Frame: — (click Go — annotate to open a frame)")

    def start_from_entry(self):
        try:
            self.start.set(float(self.start_entry.get()))
        except ValueError:
            messagebox.showerror("Invalid start", "Enter a start time in seconds.")
            return
        self.preview_start()
        self.extract()

    def preview_start(self, _=None):
        if self.source is None or self.clip is not None:
            return
        max_start = max(0, (self.total - round(CLIP_SECONDS * self.fps)) / self.fps)
        v = max(0, min(max_start, self.start.get()))
        frame = max(0, min(self.total - round(CLIP_SECONDS * self.fps), round(v * self.fps)))
        v = frame / self.fps
        self.start_entry.delete(0, "end")
        self.start_entry.insert(0, f"{v:.6f}")
        self.clip_time.set(f"Start: {v:.3f}s (source frame {frame}); end: ~{v+CLIP_SECONDS:.3f}s")
        try:
            img = Image.fromarray(read_frame(self.source, frame))
            img.thumbnail((290, 165))
            self.preview_photo = ImageTk.PhotoImage(img)
            self.preview.configure(image=self.preview_photo, text="", width=290, height=165)
        except Exception as ex:
            self.status.set(f"Preview error: {ex}")

    def extract(self):
        if self.busy or self.source is None:
            return
        max_start_frame = self.total - round(CLIP_SECONDS * self.fps)
        try:
            requested = float(self.start_entry.get())
        except ValueError:
            requested = self.start.get()
        if not 0 <= requested <= max_start_frame / self.fps + 1e-8:
            messagebox.showerror("Start time", f"Choose between 0 and {max_start_frame/self.fps:.3f} seconds")
            return
        start_frame = min(max_start_frame, round(requested * self.fps))
        stem = safe_stem(self.source.name)
        dst = self.output / "videos_30s" / f"{stem}_30s.mp4"
        metadata = self.output / "progress" / f"{stem}.json"
        if (dst.exists() or metadata.exists()) and not messagebox.askyesno("Existing annotations", f"Files for {stem} exist. Overwrite clip and annotations?\nChoose No to preserve previous work."):
            return
        self.busy = True
        self.mode.set("Extracting...")
        self.status.set(f"Extracting {self.source.name} at original frame {start_frame}...")
        self.root.update_idletasks()
        source, fps, width, height = self.source, self.fps, self.width, self.height

        def worker():
            try:
                export_clip(source, dst, start_frame, fps, width, height, lambda i, n: self.root.after(0, lambda x=i, y=n: self.status.set(f"Extracting {x}/{y} frames")))
                self.root.after(0, lambda: self.ready_clip(dst, stem, source, start_frame))
            except Exception as ex:
                self.root.after(0, lambda msg=str(ex): self.extract_failed(msg))
        threading.Thread(target=worker, daemon=True).start()

    def extract_failed(self, msg):
        self.busy = False
        self.mode.set("Ready")
        messagebox.showerror("Extraction failed", msg)
        self.status.set("Extraction failed: " + msg)

    def ready_clip(self, dst, stem, source, start_frame):
        self.busy = False
        self.mode.set("Annotating")
        self.clip, self.stem = dst, stem
        self.fps, self.total, self.width, self.height = probe(dst)
        self.sample_frames = sample_frames_for(self.total, self.fps)
        self.sample_index = 0
        self.points, self.confirmed, self.suggestions = {}, set(), {}
        self.guesses_from = None
        self.history = []
        self.source_origin = str(source)
        self.source_start_frame = start_frame
        self.zoom.set(1.0)
        self.save()
        self.load_frame()
        self.status.set(f"Clip created: {dst.name} ({len(self.sample_frames)} frames to annotate, one per second).")
        if self.guesser is not None:
            self.run_guesses()

    def open_existing(self):
        filename = filedialog.askopenfilename(title="Open saved annotation progress", filetypes=[("Annotation progress", "*.json")])
        if not filename:
            return
        try:
            data = json.loads(Path(filename).read_text(encoding="utf-8"))
            clip = Path(data["clip"])
            if not clip.exists():
                # allow output directory to be moved as a unit
                clip = Path(filename).parent.parent / "videos_30s" / Path(data["clip"]).name
            if not clip.exists():
                raise FileNotFoundError(f"Clipped video not found: {data['clip']}")
            self.output = Path(filename).parent.parent
            self.clip = clip
            self.stem = data["stem"]
            self.fps, self.total, self.width, self.height = probe(clip)
            # Keep any frames annotated before and add the one-per-second frames.
            saved = {int(x) for x in data.get("sample_frames", [])}
            self.sample_frames = sorted(saved | set(sample_frames_for(self.total, self.fps)))
            current = int(data.get("sample_index", 0))
            old_frames = [int(x) for x in data.get("sample_frames", [])]
            current_frame = old_frames[current] if 0 <= current < len(old_frames) else 0
            self.sample_index = self.sample_frames.index(current_frame) if current_frame in self.sample_frames else 0
            self.points, self.confirmed, self.suggestions = migrate_progress(data)
            self.guesses_from = data.get("guesses_from")
            self.source_origin = data.get("source", "")
            self.source_start_frame = data.get("source_start_frame", 0)
            self.history = []
            self.mode.set("Annotating")
            self.zoom.set(1.0)
            self.load_frame()
            message = f"Resumed {self.stem}; output folder: {self.output}"
            if len(self.sample_frames) > len(saved):
                message += f" ({len(self.sample_frames)} frames now: one per second)"
            self.status.set(message)
            if self.guesser is not None and not self.suggestions:
                self.run_guesses()
        except Exception as ex:
            messagebox.showerror("Load error", str(ex))

    # ── guesses ─────────────────────────────────────────────────────────
    def _update_rerun_btn(self):
        enabled = self.guesser is not None and not self.guessing
        self.rerun_btn.state(["!disabled"] if enabled else ["disabled"])

    def run_guesses(self):
        """Run the tracker on the current clip in the background."""
        if self.guesser is None or self.clip is None or self.guessing:
            return
        clip, frames = self.clip, list(self.sample_frames)
        self.guessing = True
        self._update_rerun_btn()
        self.guess_label.configure(text=f"Guessing with {self.guesser.label()}…")

        def progress(current, total):
            self.root.after(0, lambda c=current, t=total: self.guess_label.configure(
                text=f"Guessing with {self.guesser.label()}: {100 * c // max(1, t)}%"))

        def worker():
            try:
                guesses = self.guesser.guess(clip, frames, progress)
                self.root.after(0, lambda g=guesses: self._guesses_ready(clip, g, None))
            except Exception as ex:
                self.root.after(0, lambda m=f"{type(ex).__name__}: {ex}": self._guesses_ready(clip, None, m))
        threading.Thread(target=worker, daemon=True).start()

    def _guesses_ready(self, clip, guesses, error):
        self.guessing = False
        self._update_rerun_btn()
        if error:
            self.guess_label.configure(text=f"Guessing failed: {error}")
            self.status.set("Guessing failed; you can still annotate by hand.")
            return
        if clip != self.clip:
            self.guess_label.configure(text="Guesses finished for a clip that is no longer open; "
                                            "click Re-run guesses.")
            return
        self.suggestions = guesses
        self.guesses_from = self.guesser.describe()
        merge_guesses(self.points, self.confirmed, guesses)
        found = sum(len(v) for v in guesses.values())
        self.guess_label.configure(text=f"Guesses: {self.guesser.label()} — {found} points on "
                                        f"{sum(1 for v in guesses.values() if v)} frames")
        self.status.set("Guesses added to unconfirmed frames (yellow). Fix them, then press Enter "
                        "to confirm each frame. Stationary ants are not detected: look for them too.")
        self.persist_changes()

    # ── frames ──────────────────────────────────────────────────────────
    def current_frame(self):
        return self.sample_frames[self.sample_index]

    def load_frame(self):
        if self.clip is None:
            return
        frame = self.current_frame()
        try:
            self.current_rgb = read_frame(self.clip, frame)
        except Exception as ex:
            messagebox.showerror("Frame read failed", str(ex))
            return
        self.render()
        self.refresh_ants()
        self.save()

    def _leave_frame_ok(self):
        """Moving away from a frame with unconfirmed points: offer to confirm."""
        f = self.current_frame()
        if f in self.confirmed or not self.points.get(f):
            return True
        answer = messagebox.askyesnocancel(
            "Confirm frame?", "This frame has unconfirmed points.\n\n"
            "Yes: confirm them and continue\nNo: continue without confirming\nCancel: stay")
        if answer is None:
            return False
        if answer:
            self.confirmed.add(f)
            self.save()
        return True

    def prev_frame(self):
        if self.clip is not None and self.sample_index > 0 and self._leave_frame_ok():
            self.sample_index -= 1
            self.load_frame()

    def next_frame(self):
        if self.clip is not None and self.sample_index < len(self.sample_frames)-1 and self._leave_frame_ok():
            self.sample_index += 1
            self.load_frame()

    def toggle_confirm(self):
        if self.clip is None:
            return
        f = self.current_frame()
        self.push_history(f)
        if f in self.confirmed:
            self.confirmed.discard(f)
        else:
            self.confirmed.add(f)
        self.persist_changes()

    def confirm_and_next(self):
        if self.clip is None:
            return
        f = self.current_frame()
        if f not in self.confirmed:
            self.push_history(f)
            self.confirmed.add(f)
            self.persist_changes()
        if self.sample_index < len(self.sample_frames) - 1:
            self.sample_index += 1
            self.load_frame()
        else:
            self.status.set("Last frame confirmed." if len(self.confirmed) >= len(self.sample_frames)
                            else "Last frame. Some earlier frames are not confirmed yet.")

    # ── drawing ─────────────────────────────────────────────────────────
    def _visible(self, frame, point):
        return frame in self.confirmed or point[2] != "g" or self.show_guesses.get()

    def render(self):
        if self.current_rgb is None or self.clip is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        fit = min(cw/self.width, ch/self.height)
        self.display_scale = max(0.01, fit * self.zoom.get())
        dw, dh = max(1, round(self.width*self.display_scale)), max(1, round(self.height*self.display_scale))
        img = Image.fromarray(self.current_rgb).resize((dw, dh), Image.Resampling.BILINEAR)
        self.display_photo = ImageTk.PhotoImage(img)
        ox = (cw - dw)//2
        oy = (ch - dh)//2
        self.display_offset = (ox, oy)
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.display_photo)
        f = self.current_frame()
        confirmed = f in self.confirmed
        for number, point in enumerate(self.points.get(f, []), 1):
            if not self._visible(f, point):
                continue
            x, y, src = point
            color = COLOR_CONFIRMED if confirmed else (COLOR_GUESS if src == "g" else COLOR_USER)
            px = ox + x*self.display_scale
            py = oy + y*self.display_scale
            radius = 7
            self.canvas.create_oval(px-radius, py-radius, px+radius, py+radius, outline=color, width=2)
            self.canvas.create_line(px-10, py, px+10, py, fill=color)
            self.canvas.create_line(px, py-10, px, py+10, fill=color)
            self.canvas.create_text(px+11, py-12, text=str(number), anchor="w", fill=color,
                                    font=("Arial", 11, "bold"))
        n_done = len(self.confirmed & set(self.sample_frames))
        state = "✓ CONFIRMED" if confirmed else "not confirmed"
        self.frame_label.configure(
            text=f"Frame {self.sample_index+1}/{len(self.sample_frames)} | clip frame {f} | "
                 f"{timestamp(f, self.fps)}\nThis frame: {state}")
        self.progress_label.configure(text=f"Confirmed {n_done}/{len(self.sample_frames)} frames")
        self.confirm_btn.configure(text="Unconfirm frame" if confirmed
                                   else "✓ Confirm frame (Enter)")

    def refresh_ants(self):
        self.ant_list.delete(0, "end")
        if self.clip is None:
            return
        f = self.current_frame()
        for number, (x, y, src) in enumerate(self.points.get(f, []), 1):
            kind = "guess" if src == "g" and f not in self.confirmed else ""
            self.ant_list.insert("end", f"#{number}  ({x}, {y})  {kind}".rstrip())

    # ── editing ─────────────────────────────────────────────────────────
    def _to_image(self, event):
        ox, oy = self.display_offset
        return (event.x - ox) / self.display_scale, (event.y - oy) / self.display_scale

    def _nearest(self, f, x, y, radius_px):
        best, best_d = None, (radius_px / self.display_scale) ** 2
        for index, point in enumerate(self.points.get(f, [])):
            if not self._visible(f, point):
                continue
            d = (point[0] - x) ** 2 + (point[1] - y) ** 2
            if d <= best_d:
                best, best_d = index, d
        return best

    def on_press(self, event):
        if self.clip is None or self.busy or self.current_rgb is None:
            return
        f = self.current_frame()
        x, y = self._to_image(event)
        index = self._nearest(f, x, y, GRAB_RADIUS_PX)
        if index is None:
            if not (0 <= x < self.width and 0 <= y < self.height):
                return
            self.push_history(f)
            self.points.setdefault(f, []).append([int(round(x)), int(round(y)), "u"])
            index = len(self.points[f]) - 1
            self.render()
        else:
            self.push_history(f)
        self.drag = (f, index)

    def on_drag(self, event):
        if self.drag is None:
            return
        f, index = self.drag
        x, y = self._to_image(event)
        point = self.points[f][index]
        point[0] = int(round(min(self.width - 1, max(0, x))))
        point[1] = int(round(min(self.height - 1, max(0, y))))
        point[2] = "u"
        self.render()

    def on_release(self, _event):
        if self.drag is None:
            return
        self.drag = None
        self.persist_changes()

    def delete_nearest(self, event):
        if self.clip is None:
            return
        f = self.current_frame()
        x, y = self._to_image(event)
        index = self._nearest(f, x, y, DELETE_RADIUS_PX)
        if index is None:
            return
        self.push_history(f)
        self.points[f].pop(index)
        self.persist_changes()

    def delete_selected(self):
        if self.clip is None or not self.ant_list.curselection():
            return
        f = self.current_frame()
        index = self.ant_list.curselection()[0]
        if index < len(self.points.get(f, [])):
            self.push_history(f)
            self.points[f].pop(index)
            self.persist_changes()

    def clear_frame(self):
        if self.clip is None:
            return
        f = self.current_frame()
        if self.points.get(f):
            self.push_history(f)
            self.points.pop(f, None)
            self.persist_changes()

    def push_history(self, frame):
        self.history.append((frame, [p[:] for p in self.points.get(frame, [])],
                             frame in self.confirmed))
        if len(self.history) > 200:
            self.history.pop(0)

    def undo(self):
        if not self.history:
            self.status.set("Nothing to undo (history is in-memory for this session).")
            return
        f, pts, was_confirmed = self.history.pop()
        if pts:
            self.points[f] = pts
        else:
            self.points.pop(f, None)
        if was_confirmed:
            self.confirmed.add(f)
        else:
            self.confirmed.discard(f)
        if f in self.sample_frames and f != self.current_frame():
            self.sample_index = self.sample_frames.index(f)
            self.load_frame()
        self.persist_changes()

    def persist_changes(self):
        self.refresh_ants()
        self.render()
        self.save()

    def on_wheel(self, event):
        self.bump_zoom(1.15 if event.delta > 0 else 1/1.15)

    def bump_zoom(self, factor):
        self.zoom.set(min(4.0, max(0.5, self.zoom.get()*factor)))
        self.render()

    # ── saving ──────────────────────────────────────────────────────────
    def save(self):
        if self.clip is None or self.stem is None:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "csv").mkdir(exist_ok=True)
        (self.output / "progress").mkdir(exist_ok=True)
        data = {
            "version": PROGRESS_VERSION,
            "clip": str(self.clip.resolve()),
            "stem": self.stem,
            "source": getattr(self, "source_origin", ""),
            "source_start_frame": getattr(self, "source_start_frame", 0),
            "fps": self.fps,
            "sample_frames": self.sample_frames,
            "sample_index": self.sample_index,
            "annotations": {str(f): pts for f, pts in sorted(self.points.items())},
            "confirmed_frames": sorted(self.confirmed),
            "suggestions": {str(f): pts for f, pts in sorted(self.suggestions.items())},
            "guesses_from": self.guesses_from,
        }
        progress_file = self.output / "progress" / f"{self.stem}.json"
        tmp = progress_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, progress_file)
        csv_file = self.output / "csv" / f"{self.stem}.csv"
        tmpcsv = csv_file.with_suffix(".csv.tmp")
        with tmpcsv.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["Timestamp", "frame", "ant_x", "ant_y"])
            writer.writerows(csv_rows(self.points, self.confirmed, self.fps))
        os.replace(tmpcsv, csv_file)

    def close(self):
        if not self.busy:
            self.save()
            self.root.destroy()
        else:
            messagebox.showinfo("Extraction running", "Finish the current extraction before closing the window.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Annotate ant centers on 30-second clips (one frame per second).")
    parser.add_argument("--tracker", metavar="PATH",
                        help="Path to ant_tracker_modular.py: run it on each clip and pre-fill "
                             "guessed points for you to adjust and confirm")
    parser.add_argument("--parameters", metavar="JSON",
                        help="Tracker settings for the guesses (e.g. ant_tuning_parameters.json); "
                             "needs --tracker")
    args = parser.parse_args(argv)
    if args.parameters and not args.tracker:
        parser.error("--parameters needs --tracker")
    guesser, error = None, None
    if args.tracker:
        try:
            guesser = TrackerGuesser(args.tracker, args.parameters)
            print(f"Guesses enabled: {guesser.label()}", flush=True)
        except Exception as ex:  # keep annotating by hand if the tracker can't load
            error = f"{type(ex).__name__}: {ex}"
            print(f"Guesses disabled: {error}", file=sys.stderr, flush=True)
    root = tk.Tk()
    AntAnnotator(root, guesser, error)
    if error:
        root.after(200, lambda: messagebox.showwarning(
            "Guesses disabled", f"Could not load the tracker, so there are no guessed points:\n\n{error}"))
    root.mainloop()


if __name__ == "__main__":
    main()
