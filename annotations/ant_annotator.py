#!/usr/bin/env python3
"""HOW TO USE:
python -m pip install -r requirements.txt
python ant_annotator.py

Usage
1. Click Output folder and choose your project directory.
2. Click Add videos and select multiple files. Select a source video from the list.
3. Use the start slider or type a start time (seconds) and click Go. The preview shows the frame at the proposed start.
4. Click Extract selected 30s clip. The app exports the selected interval into videos_30s/ and opens annotation mode.
5. Click the center of each ant. Each click writes the selected ant ID and coordinate at the current sampled frame. Click New ant ID to assign a new ID; enter a previously used ant ID to track the same ant in a later frame.
6. Use Next/Previous or the left/right arrow keys to move between frames sampled at 0, 2, 4, ..., 28 seconds. The zoom slider / mouse wheel zooms; coordinates remain relative to the original clip's pixel resolution.
7. Click an item in Ants on this frame to select its ID. Delete selected / active ID removes the annotation from this frame; right-click near a marker to delete it. Undo (Ctrl+Z) reverses recent in-session annotations/deletions.
8. Annotations automatically save after each edit and frame change. If you close and reopen the app, use Open existing annotation and select progress/<video_stem>.json to resume.
9. Switch to the next source video and repeat.  


"""
import csv
import json
import math
import os
import re
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

CLIP_SECONDS = 30
STEP_SECONDS = 2


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


class AntAnnotator:
    def __init__(self, root):
        self.root = root
        root.title("Ant Annotator — 30-second clips")
        root.geometry("1260x850")
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
        self.annotations = {}
        self.history = []
        self.next_id = 1
        self.active_id = tk.StringVar(value="1")
        self.zoom = tk.DoubleVar(value=1.0)
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
        sidebar = ttk.Frame(body, width=310)
        body.add(sidebar, weight=0)
        main = ttk.Frame(body)
        body.add(main, weight=1)

        ttk.Label(sidebar, text="Source videos", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.video_list = tk.Listbox(sidebar, height=8, exportselection=False)
        self.video_list.pack(fill="x", pady=5)
        self.video_list.bind("<<ListboxSelect>>", self.video_selected)
        ttk.Label(sidebar, text="Clip starting position (seconds)").pack(anchor="w", pady=(12, 0))
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
        ttk.Label(sidebar, textvariable=self.clip_time, wraplength=280).pack(anchor="w")
        self.preview = tk.Label(sidebar, text="Select a video", relief="groove", width=35, height=10)
        self.preview.pack(fill="x", pady=8)
        ttk.Separator(sidebar).pack(fill="x", pady=8)

        ttk.Label(sidebar, text="Annotation", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        ttk.Label(sidebar, text="Active ant ID (click an ant to label):").pack(anchor="w", pady=(6, 0))
        id_row = ttk.Frame(sidebar)
        id_row.pack(fill="x", pady=4)
        ttk.Entry(id_row, textvariable=self.active_id, width=12).pack(side="left")
        ttk.Button(id_row, text="New ant ID", command=self.new_ant_id).pack(side="left", padx=5)
        ttk.Label(sidebar, text="Keep the same ID for the same ant across frames.\nSelect an ID, then click the ant center.", wraplength=280).pack(anchor="w", pady=4)
        nav = ttk.Frame(sidebar)
        nav.pack(fill="x", pady=6)
        ttk.Button(nav, text="◀ Previous", command=self.prev_frame).pack(side="left", padx=2)
        ttk.Button(nav, text="Next ▶", command=self.next_frame).pack(side="left", padx=2)
        ttk.Button(sidebar, text="Undo last action", command=self.undo).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Delete selected / active ID on frame", command=self.delete_active).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Save now", command=self.save).pack(fill="x", pady=2)
        ttk.Label(sidebar, text="Zoom (mousewheel also zooms)").pack(anchor="w", pady=(10, 0))
        ttk.Scale(sidebar, from_=0.5, to=4, variable=self.zoom, command=lambda _: self.render()).pack(fill="x")
        ttk.Button(sidebar, text="Reset zoom", command=lambda: self.zoom.set(1.0)).pack(fill="x")
        ttk.Label(sidebar, text="Coordinates are measured in original clip pixels,\nindependent of zoom.", wraplength=280).pack(anchor="w", pady=7)
        self.frame_label = ttk.Label(sidebar, text="Frame: —")
        self.frame_label.pack(anchor="w", pady=5)
        ttk.Label(sidebar, text="Ants on this frame:").pack(anchor="w")
        self.ant_list = tk.Listbox(sidebar, height=9)
        self.ant_list.pack(fill="both", expand=True, pady=4)
        self.ant_list.bind("<<ListboxSelect>>", self.ant_selected)

        viewer = ttk.Frame(main)
        viewer.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(viewer, bg="#151515", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<Button-1>", self.click_ant)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.bump_zoom(1.15))
        self.canvas.bind("<Button-5>", lambda e: self.bump_zoom(1 / 1.15))
        self.canvas.bind("<Button-3>", self.delete_nearest)
        ttk.Label(root, textvariable=self.status, relief="sunken", padding=5, anchor="w").pack(fill="x")
        root.bind("<Left>", lambda e: self.prev_frame())
        root.bind("<Right>", lambda e: self.next_frame())
        root.bind("<Control-z>", lambda e: self.undo())
        root.protocol("WM_DELETE_WINDOW", self.close)

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
        if frames < round(30 * fps):
            messagebox.showerror("Video too short", f"{p.name} contains less than 30 seconds of frames.")
            return
        self.source = p
        self.clip = None
        self.current_rgb = None  # never display an old clip while switching videos
        self.fps, self.total, self.width, self.height = fps, frames, width, height
        self.start_slider.configure(to=max(0, (frames - round(30 * fps)) / fps))
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
        # Go is the primary workflow: extract and immediately display the
        # first annotation frame. The small preview alone is not annotation.
        self.extract()

    def preview_start(self, _=None):
        if self.source is None or self.clip is not None:
            return
        max_start = max(0, (self.total - round(30 * self.fps)) / self.fps)
        v = max(0, min(max_start, self.start.get()))
        # align preview to an exact source frame
        frame = max(0, min(self.total - round(30 * self.fps), round(v * self.fps)))
        v = frame / self.fps
        self.start_entry.delete(0, "end")
        self.start_entry.insert(0, f"{v:.6f}")
        self.clip_time.set(f"Start: {v:.3f}s (source frame {frame}); end: ~{v+30:.3f}s")
        try:
            img = Image.fromarray(read_frame(self.source, frame))
            img.thumbnail((290, 185))
            self.preview_photo = ImageTk.PhotoImage(img)
            self.preview.configure(image=self.preview_photo, text="", width=290, height=185)
        except Exception as ex:
            self.status.set(f"Preview error: {ex}")

    def extract(self):
        if self.busy or self.source is None:
            return
        max_start_frame = self.total - round(30 * self.fps)
        try:
            requested = float(self.start_entry.get())
        except ValueError:
            requested = self.start.get()
        if not 0 <= requested <= max_start_frame / self.fps + 1e-8:
            messagebox.showerror("Start time", f"Choose between 0 and {max_start_frame/self.fps:.3f} seconds")
            return
        start_frame = min(max_start_frame, round(requested * self.fps))
        # Avoid silently overwriting different original files that share a basename.
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
        self.sample_frames = sorted(set(min(self.total - 1, round(sec * self.fps)) for sec in range(0, 30, 2)))
        self.sample_index = 0
        self.annotations = {}
        self.history = []
        self.next_id = 1
        self.active_id.set("1")
        self.source_origin = str(source)
        self.source_start_frame = start_frame
        self.zoom.set(1.0)
        self.save()
        self.load_frame()
        self.status.set(f"Clip created: {dst.name}. Click ant centers in the LARGE frame on the right. CSV frames start at 0.")

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
            self.sample_frames = [int(x) for x in data["sample_frames"]]
            self.sample_index = min(len(self.sample_frames)-1, max(0, int(data.get("sample_index", 0))))
            self.annotations = {int(k): {str(a): [int(z) for z in xy] for a, xy in ants.items()} for k, ants in data["annotations"].items()}
            self.next_id = int(data.get("next_id", 1))
            self.active_id.set(data.get("active_id", "1"))
            self.source_origin = data.get("source", "")
            self.source_start_frame = data.get("source_start_frame", 0)
            self.history = []
            self.mode.set("Annotating")
            self.zoom.set(1.0)
            self.load_frame()
            self.status.set(f"Resumed {self.stem}; output folder: {self.output}")
        except Exception as ex:
            messagebox.showerror("Load error", str(ex))

    def load_frame(self):
        if self.clip is None:
            return
        frame = self.sample_frames[self.sample_index]
        try:
            self.current_rgb = read_frame(self.clip, frame)
        except Exception as ex:
            messagebox.showerror("Frame read failed", str(ex))
            return
        self.frame_label.configure(text=f"Sample {self.sample_index+1}/{len(self.sample_frames)} | Clip frame {frame}\nTimestamp {timestamp(frame, self.fps)} | FPS {self.fps:.4f}")
        self.render()
        self.refresh_ants()
        self.save()

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
        f = self.sample_frames[self.sample_index]
        for ant_id, (x, y) in sorted(self.annotations.get(f, {}).items(), key=lambda z: z[0]):
            px = ox + x*self.display_scale
            py = oy + y*self.display_scale
            radius = 7
            self.canvas.create_oval(px-radius, py-radius, px+radius, py+radius, outline="#00ff88", width=2)
            self.canvas.create_line(px-10,py,px+10,py, fill="#00ff88")
            self.canvas.create_line(px,py-10,px,py+10, fill="#00ff88")
            self.canvas.create_text(px+11,py-12,text=f"ID {ant_id}",anchor="w",fill="#00ff88",font=("Arial", 11,"bold"))

    def click_ant(self, event):
        if self.clip is None or self.busy:
            return
        ant_id = self.active_id.get().strip()
        if not ant_id:
            messagebox.showerror("Ant ID", "Enter an ant ID first")
            return
        ox, oy = self.display_offset
        x = round((event.x - ox)/self.display_scale)
        y = round((event.y - oy)/self.display_scale)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        f = self.sample_frames[self.sample_index]
        self.push_history(f)
        self.annotations.setdefault(f, {})[ant_id] = [x, y]
        if ant_id.isdigit():
            self.next_id = max(self.next_id, int(ant_id)+1)
        self.persist_changes()

    def new_ant_id(self):
        self.active_id.set(str(self.next_id))
        self.next_id += 1
        if self.clip:
            self.save()

    def ant_selected(self, _=None):
        if not self.ant_list.curselection() or self.clip is None:
            return
        line = self.ant_list.get(self.ant_list.curselection()[0])
        self.active_id.set(line.split(":",1)[0])

    def push_history(self, frame):
        self.history.append((frame, dict((k, v[:]) for k,v in self.annotations.get(frame, {}).items())))
        if len(self.history) > 100:
            self.history.pop(0)

    def undo(self):
        if not self.history:
            self.status.set("Nothing to undo (history is in-memory for this session).")
            return
        f, data = self.history.pop()
        if data:
            self.annotations[f] = data
        else:
            self.annotations.pop(f, None)
        self.persist_changes()

    def delete_active(self):
        if self.clip is None:
            return
        f = self.sample_frames[self.sample_index]
        ant_id = self.active_id.get().strip()
        if ant_id not in self.annotations.get(f, {}):
            return
        self.push_history(f)
        self.annotations[f].pop(ant_id)
        self.persist_changes()

    def delete_nearest(self, event):
        if self.clip is None:
            return
        f = self.sample_frames[self.sample_index]
        ants = self.annotations.get(f, {})
        if not ants:
            return
        ox, oy = self.display_offset
        x = (event.x-ox)/self.display_scale
        y = (event.y-oy)/self.display_scale
        ant_id, dist = min(((ant_id, (xy[0]-x)**2 + (xy[1]-y)**2) for ant_id, xy in ants.items()), key=lambda z:z[1])
        if dist > (20/self.display_scale)**2:
            return
        self.push_history(f)
        ants.pop(ant_id)
        self.persist_changes()

    def persist_changes(self):
        self.refresh_ants()
        self.render()
        self.save()

    def refresh_ants(self):
        self.ant_list.delete(0, "end")
        if self.clip is None:
            return
        f = self.sample_frames[self.sample_index]
        for ant_id, xy in sorted(self.annotations.get(f, {}).items(), key=lambda z: (not z[0].isdigit(), int(z[0]) if z[0].isdigit() else z[0])):
            self.ant_list.insert("end", f"{ant_id}: ({xy[0]}, {xy[1]})")

    def prev_frame(self):
        if self.clip is not None and self.sample_index > 0:
            self.sample_index -= 1
            self.load_frame()

    def next_frame(self):
        if self.clip is not None and self.sample_index < len(self.sample_frames)-1:
            self.sample_index += 1
            self.load_frame()

    def on_wheel(self, event):
        self.bump_zoom(1.15 if event.delta > 0 else 1/1.15)

    def bump_zoom(self, factor):
        self.zoom.set(min(4.0, max(0.5, self.zoom.get()*factor)))
        self.render()

    def save(self):
        if self.clip is None or self.stem is None:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "csv").mkdir(exist_ok=True)
        (self.output / "progress").mkdir(exist_ok=True)
        data = {
            "version": 1,
            "clip": str(self.clip.resolve()),
            "stem": self.stem,
            "source": getattr(self, "source_origin", ""),
            "source_start_frame": getattr(self, "source_start_frame", 0),
            "fps": self.fps,
            "sample_frames": self.sample_frames,
            "sample_index": self.sample_index,
            "next_id": self.next_id,
            "active_id": self.active_id.get(),
            "annotations": self.annotations,
        }
        progress_file = self.output / "progress" / f"{self.stem}.json"
        tmp = progress_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, progress_file)
        csv_file = self.output / "csv" / f"{self.stem}.csv"
        tmpcsv = csv_file.with_suffix(".csv.tmp")
        with tmpcsv.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["Timestamp", "frame", "ant_id", "ant_x", "ant_y"])
            for frame in sorted(self.annotations):
                for ant_id, (x, y) in sorted(self.annotations[frame].items()):
                    writer.writerow([timestamp(frame, self.fps), frame, ant_id, x, y])
        os.replace(tmpcsv, csv_file)

    def close(self):
        if not self.busy:
            self.save()
            self.root.destroy()
        else:
            messagebox.showinfo("Extraction running", "Finish the current extraction before closing the window.")


if __name__ == "__main__":
    root = tk.Tk()
    AntAnnotator(root)
    root.mainloop()
