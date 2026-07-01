#!/usr/bin/env python3
"""
Ant Hole Counter — Batch Mode
=============================
Process multiple videos with one circle/quadrant setup.

Usage:
    python ant_counter_circle_shadow_batch.py

Workflow:
    1. Add videos (files, folder, or shortcuts).
    2. Select an output directory
    2. Preview any video in the queue; scrub to a representative frame.
    3. Set circle center + north direction once (same camera setup assumed)
    4. Select other videos in the queue to override shared circles if needed
    5. Click "Process All".  Each video writes several files to the output directory.

Outputs (per video):
    <name>_counts.csv       — events with separate enter_count / exit_count columns
    <name>_summary.csv      — total enters and exits per quadrant
    <name>_counted.mp4      — annotated video (in/out shown separately per quadrant)
    <name>_circle.json      — circle parameters

Note: Circle params are saved/loaded as JSON to reuse across sessions.
If a video already has <stem>_circle.json in the output directory, it is loaded when you
select that video in the queue.
"""

from __future__ import annotations
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

from ant_counter_circle_shadow import (
    DEFAULT_RADIUS,
    VideoProcessor,
    CirclePreviewMixin,
    _stem_circle_path,
    _load_circle_json,
    _save_circle_json,
    _circle_params_dict,
)

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
_QUADRANTS = ("NE", "SE", "SW", "NW")




# NOTE: circle JSON helpers and preview/circle UI helpers are
# imported from ant_counter_circle_shadow.py to avoid duplication.


class BatchApp(CirclePreviewMixin):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ant Hole Counter — Batch")
        self.root.geometry("1400x780")
        self.root.minsize(1000, 620)

        self.video_queue: list[str] = []
        self._per_video_circles: dict[str, dict] = {}

        self.cap = None
        self.preview_path: str | None = None
        self.fps = 30.0
        self.total_frames = 0
        self.orig_w = self.orig_h = 0
        self.current_bgr = None

        self.circle_center = None
        self.circle_radius = float(DEFAULT_RADIUS)
        self.north_point = None
        self.circle_complete = False

        self.disp_off_x = self.disp_off_y = 0
        self.disp_scale = 1.0
        self._photo = None

        self.processor: VideoProcessor | None = None
        self._batch_cancel = False
        self._batch_running = False
        self.output_root: str | None = None
        self.output_root_base: str | None = None

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                              sashwidth=6, sashrelief=tk.RAISED, bg="#333")
        pane.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(pane, width=300, bg="#f4f4f4", padx=8, pady=8)
        pane.add(sidebar, minsize=280)

        tk.Label(sidebar, text="Ant Hole Counter",
                 font=("Arial", 15, "bold"), bg="#f4f4f4").pack(pady=(4, 4))
        tk.Label(sidebar, text="Batch mode",
                 font=("Arial", 10), fg="#555", bg="#f4f4f4").pack(pady=(0, 8))

        # Queue
        qf = ttk.LabelFrame(sidebar, text="  Video queue  ")
        qf.pack(fill=tk.BOTH, expand=True, pady=4)

        list_frame = tk.Frame(qf)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._queue_list = tk.Listbox(
            list_frame, height=8, selectmode=tk.SINGLE,
            yscrollcommand=scroll.set, font=("Courier", 9))
        scroll.config(command=self._queue_list.yview)
        self._queue_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._queue_list.bind("<<ListboxSelect>>", self._on_queue_select)

        btn_row = tk.Frame(qf)
        btn_row.pack(fill=tk.X, padx=4, pady=2)
        tk.Button(btn_row, text="Add files…", width=10,
                  command=self._add_files).pack(side=tk.LEFT, padx=1)
        tk.Button(btn_row, text="Add folder…", width=10,
                  command=self._add_folder).pack(side=tk.LEFT, padx=1)
        tk.Button(btn_row, text="Remove", width=8,
                  command=self._remove_selected).pack(side=tk.LEFT, padx=1)
        tk.Button(btn_row, text="Clear", width=6,
                  command=self._clear_queue).pack(side=tk.LEFT, padx=1)

        sf = tk.Frame(qf)
        sf.pack(fill=tk.X, padx=4, pady=(0, 4))

        self._queue_count_lbl = tk.Label(
            qf, text="0 videos", font=("Arial", 8), fg="#444")
        self._queue_count_lbl.pack(anchor=tk.W, padx=4, pady=(0, 4))

        outf = ttk.LabelFrame(sidebar, text="  Output location  ")
        outf.pack(fill=tk.X, pady=4)
        self._output_root_lbl = tk.Label(
            outf,
            text="Output root: default in ./outputs",
            font=("Arial", 8), fg="#444", justify=tk.LEFT, anchor=tk.W,
            wraplength=250)
        self._output_root_lbl.pack(fill=tk.X, padx=4, pady=(4, 2))
        tk.Button(
            outf, text="Select output folder…",
            command=self._select_output_root).pack(fill=tk.X, padx=4, pady=(0, 2))
        tk.Button(
            outf, text="Clear output folder",
            command=self._clear_output_root).pack(fill=tk.X, padx=4, pady=(0, 4))

        # Preview frame scrub
        ff = ttk.LabelFrame(sidebar, text="  Preview frame  ")
        ff.pack(fill=tk.X, pady=4)
        self._frame_var = tk.IntVar()
        self._slider = ttk.Scale(
            ff, from_=0, to=100, variable=self._frame_var,
            orient=tk.HORIZONTAL, command=self._on_slider)
        self._slider.pack(fill=tk.X, padx=4, pady=2)
        self._frame_lbl = tk.Label(ff, text="—", font=("Arial", 8))
        self._frame_lbl.pack()

        # Circle setup
        cf = ttk.LabelFrame(sidebar, text="  Circle setup  ")
        cf.pack(fill=tk.X, pady=4)
        tk.Label(cf,
                 text=("1st click: center\n"
                       "2nd click: north\n"
                       "Right-click: clear"),
                 font=("Arial", 8), justify=tk.LEFT).pack(anchor=tk.W, padx=4)
        self._circle_lbl = tk.Label(
            cf, text="No circle defined", fg="#b00", font=("Arial", 9, "bold"))
        self._circle_lbl.pack(pady=2)

        radius_frame = tk.Frame(cf)
        radius_frame.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(radius_frame, text="Radius:", font=("Arial", 8)).pack(side=tk.LEFT)
        self._radius_var = tk.IntVar(value=DEFAULT_RADIUS)
        tk.Spinbox(
            radius_frame, from_=50, to=500, textvariable=self._radius_var,
            width=8, command=self._on_radius_change,
        ).pack(side=tk.LEFT, padx=(5, 0))

        assign_row = tk.Frame(cf)
        assign_row.pack(fill=tk.X, padx=4, pady=2)
        self._assign_btn = tk.Button(
            assign_row, text="Assign to this video",
            command=self._assign_circle_to_video, state=tk.DISABLED)
        self._assign_btn.pack(side=tk.LEFT, padx=1)
        self._clear_assign_btn = tk.Button(
            assign_row, text="Clear assignment",
            command=self._clear_video_assignment, state=tk.DISABLED)
        self._clear_assign_btn.pack(side=tk.LEFT, padx=1)

        cfg_row = tk.Frame(cf)
        cfg_row.pack(fill=tk.X, padx=4, pady=2)
        tk.Button(cfg_row, text="Save circle…", command=self._save_circle).pack(
            side=tk.LEFT, padx=1)
        tk.Button(cfg_row, text="Load circle…", command=self._load_circle).pack(
            side=tk.LEFT, padx=1)
        tk.Button(cf, text="Clear circle", command=self.clear_circle).pack(pady=(0, 4))

        # Batch process
        prf = ttk.LabelFrame(sidebar, text="  Batch process  ")
        prf.pack(fill=tk.X, pady=4)
        self._proc_btn = tk.Button(
            prf, text="Process All",
            font=("Arial", 11, "bold"),
            bg="#2e7d32", fg="white", state=tk.DISABLED,
            activebackground="#1b5e20", activeforeground="white",
            command=self.start_batch)
        self._proc_btn.pack(fill=tk.X, padx=4, pady=4)
        self._cancel_btn = tk.Button(
            prf, text="Cancel", state=tk.DISABLED, command=self.cancel_batch)
        self._cancel_btn.pack(fill=tk.X, padx=4, pady=(0, 4))

        self._batch_prog_var = tk.DoubleVar()
        ttk.Progressbar(
            prf, variable=self._batch_prog_var, maximum=100,
        ).pack(fill=tk.X, padx=4, pady=2)
        self._batch_prog_lbl = tk.Label(prf, text="", font=("Arial", 8))
        self._batch_prog_lbl.pack()

        self._vid_prog_var = tk.DoubleVar()
        ttk.Progressbar(
            prf, variable=self._vid_prog_var, maximum=100,
        ).pack(fill=tk.X, padx=4, pady=2)
        self._vid_prog_lbl = tk.Label(prf, text="", font=("Arial", 8))
        self._vid_prog_lbl.pack()

        self._status_lbl = tk.Label(
            sidebar, text="Add videos and define a circle to begin.",
            wraplength=270, justify=tk.LEFT, font=("Arial", 9),
            fg="#333", bg="#f4f4f4")
        self._status_lbl.pack(pady=6, anchor=tk.W)

        # Canvas
        canvas_frame = tk.Frame(pane, bg="#111")
        pane.add(canvas_frame, minsize=500)
        self.canvas = tk.Canvas(
            canvas_frame, bg="#111", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Button-3>", self._rclick)
        self.canvas.bind("<Configure>", self._resize)

    # ── Queue management ─────────────────────────────────────────────────────

    def _enqueue(self, paths: list[str]):
        added = 0
        for path in paths:
            path = os.path.abspath(path)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext not in _VIDEO_EXTS:
                continue
            out_dir, stem = self._find_outdir(path)
            if stem.startswith("."):
                continue
            if path in self.video_queue:
                continue
            self.video_queue.append(path)

            added += 1
        if self.output_root:
            self.output_root_base = self._compute_output_root_base()
            self._update_output_root_label()
        if added and self.preview_path is None and self.video_queue:
            self._queue_list.selection_set(0)
            self._preview_video(self.video_queue[0])
        self._load_saved_circles()

    def _load_saved_circles(self):
        for path in self.video_queue:
            out_dir, stem = self._find_outdir(path)
            saved = _load_circle_json(_stem_circle_path(out_dir, stem))
            if saved:
                self._per_video_circles[path] = saved
        self._refresh_queue_ui()
        self._refresh_proc_btn()

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select videos",
            filetypes=[
                ("Video", "*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI *.mkv *.MKV"),
                ("All", "*.*"),
            ],
        )
        if paths:
            self._enqueue(list(paths))

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Select folder with videos")
        if not folder:
            return
        paths = []
        for name in sorted(os.listdir(folder)):
            full = os.path.join(folder, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].lower() in _VIDEO_EXTS:
                paths.append(full)
        if not paths:
            messagebox.showinfo("No videos", f"No video files found in:\n{folder}")
            return
        self._enqueue(paths)

    def _remove_selected(self):
        sel = self._queue_list.curselection()
        if not sel:
            return
        idx = sel[0]
        path = self.video_queue.pop(idx)
        self._per_video_circles.pop(path, None)
        self._refresh_queue_ui()
        if path == self.preview_path:
            self._close_preview()
            if self.video_queue:
                self._queue_list.selection_set(min(idx, len(self.video_queue) - 1))
                self._preview_video(self.video_queue[min(idx, len(self.video_queue) - 1)])
        self._refresh_proc_btn()

    def _clear_queue(self):
        if self._batch_running:
            return
        self.video_queue.clear()
        self._per_video_circles.clear()
        self._close_preview()
        if self.output_root:
            self.output_root_base = None
            self._update_output_root_label()
        self._refresh_queue_ui()
        self._refresh_proc_btn()

    def _refresh_queue_ui(self):
        self._queue_list.delete(0, tk.END)
        for path in self.video_queue:
            name = os.path.basename(path)
            tag = " [circle assigned]" if path in self._per_video_circles else " [uses shared]" if self.circle_complete else " [NO CIRCLE]"
            self._queue_list.insert(tk.END, name + tag)
        n = len(self.video_queue)
        self._queue_count_lbl.config(text=f"{n} video{'s' if n != 1 else ''}")

    def _on_queue_select(self, _event=None):
        sel = self._queue_list.curselection()
        if not sel:
            return
        path = self.video_queue[sel[0]]
        self._preview_video(path)
        # Load this video's assigned circle into the display, if it has one
        if path in self._per_video_circles:
            self._apply_circle_params(self._per_video_circles[path])
            self._circle_lbl.config(
                text=f"Circle: assigned to {os.path.basename(path)}", fg="#006600")
        self._refresh_assign_btns()

    def _select_output_root(self):
        root = filedialog.askdirectory(title="Select output root directory")
        if not root:
            return
        self.output_root = root
        self.output_root_base = self._compute_output_root_base()
        self._load_saved_circles()
        self._refresh_queue_ui()
        self._update_output_root_label()

    def _clear_output_root(self):
        self.output_root = None
        self.output_root_base = None
        self._update_output_root_label()

    def _update_output_root_label(self):
        if self.output_root is None:
            self._output_root_lbl.config(
                text="Output root: default in ./outputs")
            return
        if self.output_root_base:
            self._output_root_lbl.config(
                text=(f"Output root: {self.output_root}\n"
                      f"Mirror base: {self.output_root_base}"))
        else:
            self._output_root_lbl.config(
                text=f"Output root: {self.output_root}")

    def _compute_output_root_base(self) -> str | None:
        if not self.video_queue:
            return None
        dirs = [os.path.dirname(p) for p in self.video_queue]
        try:
            base = os.path.commonpath(dirs)
        except ValueError:
            return None
        return base
    
    def _find_outdir(self, path):
        # Based on input video path, output directory and output root
        # determines what is the output directory for this video
        # and creates it if it doesn't already exist
        # Returns the full path of the output directory and the 
        # video name without its extension to construct output files
        if self.output_root:
            if self.output_root_base:
                rel_path = os.path.relpath(
                    os.path.dirname(path), self.output_root_base)
                out_dir = os.path.join(self.output_root, rel_path)
            else:
                out_dir = self.output_root
        else:
            out_dir = os.path.join(os.path.abspath(os.path.curdir),"outputs")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(path))[0]
        return out_dir, stem

    # ── Preview ──────────────────────────────────────────────────────────────

    def _close_preview(self):
        if self.cap:
            self.cap.release()
        self.cap = None
        self.preview_path = None
        self.current_bgr = None
        # Canvas may already be destroyed when closing the app; ignore errors.
        try:
            self.canvas.delete("all")
        except tk.TclError:
            pass

    def _preview_video(self, path: str):
        if self._batch_running:
            return
        if self.cap:
            self.cap.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open:\n{path}")
            return
        self.cap = cap
        self.preview_path = path
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._slider.config(to=max(0, self.total_frames - 1))
        self._frame_var.set(0)
        self._show_frame(0)
        self._status(
            f"Preview: {os.path.basename(path)}\n"
            f"{self.orig_w}x{self.orig_h}  {self.fps:.0f} fps  "
            f"{self.total_frames:,} frames")

    def _click(self, event):
        if not self.cap or self._batch_running:
            return
        super()._click(event)
        if self.circle_complete:
            self._refresh_assign_btns()

    def _rclick(self, _event):
        if not self._batch_running:
            super()._rclick(_event)

    def clear_circle(self):
        super().clear_circle()
        self._refresh_assign_btns()

    def _apply_circle_params(self, params: dict):
        super()._apply_circle_params(params)
        if self.circle_complete:
            self._refresh_assign_btns()

    def _shared_circle_params(self) -> dict | None:
        if not self.circle_complete:
            return None
        return _circle_params_dict(
            self.circle_center, self.circle_radius, self.north_point)

    def _circle_for_video(self, video_path: str) -> dict | None:
        if video_path in self._per_video_circles:
            return self._per_video_circles[video_path]
        return self._shared_circle_params()

    def _assign_circle_to_video(self):
        if not self.circle_complete:
            messagebox.showwarning("Circle", "Define center and north first.")
            return
        if not self.preview_path:
            messagebox.showerror("Please select a video in the video queue")
            return
        params = _circle_params_dict(
            self.circle_center, self.circle_radius, self.north_point)
        self._per_video_circles[self.preview_path] = params
        # Also save to disk in the designated output directory
        out_dir, stem = self._find_outdir(self.preview_path)
        circle_path = _stem_circle_path(out_dir, stem)
        _save_circle_json(params, circle_path)
        self._circle_lbl.config(
            text=f"Circle: assigned to {os.path.basename(self.preview_path)}",
            fg="#006600")
        self._refresh_queue_ui()
        self._refresh_assign_btns()
        self._refresh_proc_btn()

    def _clear_video_assignment(self):
        if not self.preview_path:
            return
        self._per_video_circles.pop(self.preview_path, None)
        self._circle_lbl.config(text="Circle cleared for this video", fg="#664400")
        self._refresh_queue_ui()
        self._refresh_assign_btns()
        self._refresh_proc_btn()

    def _refresh_assign_btns(self):
        has_preview = self.preview_path is not None and not self._batch_running
        has_assignment = self.preview_path in self._per_video_circles if self.preview_path else False
        self._assign_btn.config(
            state=tk.NORMAL if (has_preview and self.circle_complete) else tk.DISABLED)
        self._clear_assign_btn.config(
            state=tk.NORMAL if (has_preview and has_assignment) else tk.DISABLED)

    def _refresh_proc_btn(self):
        all_have_circle = bool(self.video_queue) and all(
            self._circle_for_video(p) for p in self.video_queue)
        ok = all_have_circle and not self._batch_running
        self._proc_btn.config(state=tk.NORMAL if ok else tk.DISABLED)

    # ── Batch processing ─────────────────────────────────────────────────────

    def start_batch(self):
        if not self.video_queue:
            return
        missing = [p for p in self.video_queue if not self._circle_for_video(p)]
        if missing:
            names = "\n".join(os.path.basename(p) for p in missing[:5])
            if len(missing) > 5:
                names += f"\n…and {len(missing) - 5} more"
            messagebox.showerror(
                "Circle missing",
                f"These videos have no circle assigned and no shared circle is set:\n\n"
                f"{names}\n\n"
                "Select each video in the queue, set a circle, and click\n"
                "\"Assign to this video\" — or define a shared circle for all.")
            return
        self._batch_cancel = False
        self._batch_running = True
        self._proc_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        self._batch_prog_var.set(0)
        self._vid_prog_var.set(0)
        threading.Thread(target=self._batch_thread, daemon=True).start()

    def cancel_batch(self):
        self._batch_cancel = True
        if self.processor:
            self.processor.cancel()

    def _batch_thread(self):
        n = len(self.video_queue)
        summaries = []
        try:
            for i, video_path in enumerate(self.video_queue):
                if self._batch_cancel:
                    self._status("Batch cancelled.")
                    break

                name = os.path.basename(video_path)
                out_dir, stem = self._find_outdir(video_path)
                circle_params = self._circle_for_video(video_path)
                batch_pct = 100 * i / n
                self._set_batch_progress(
                    batch_pct, f"Video {i + 1}/{n}: {name}")

                self.processor = VideoProcessor(
                    video_path, circle_params,
                    lambda v, m, _i=i, _n=n, _name=name: self._on_vid_progress(
                        v, m, _i, _n, _name),
                    self._status,
                    out_dir, stem
                )
                events, enter_totals, exit_totals, paths = self.processor.run()

                if self._batch_cancel or events is None:
                    if not self._batch_cancel:
                        summaries.append((name, "cancelled", None))
                    continue

                quadrant_values = self.processor.calculate_quadrant_arc_pixel_values()

                # Write summary CSV for this video
                with open(paths["summary"], "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["quadrant", "enters", "exits", "mean_grayscale"])
                    for q in _QUADRANTS:
                        w.writerow([q, enter_totals[q], exit_totals[q], quadrant_values[q]])
                    w.writerow(["TOTAL", sum(enter_totals.values()), sum(exit_totals.values()), None])

                summaries.append((
                    name, "ok",
                    enter_totals, exit_totals,
                    len(events), paths["output_dir"],
                ))

            if not self._batch_cancel:
                self._set_batch_progress(100, f"Done — {n} video(s)")
                lines = ["Batch complete:\n"]
                for row in summaries:
                    if row[1] == "ok":
                        _, _, enters, exits, n_ev, out = row
                        qstr = "  ".join(
                            f"{q}:+{enters[q]}/-{exits[q]}" for q in _QUADRANTS)
                        lines.append(f"  {row[0]}: {n_ev} events ({qstr})")
                        lines.append(f"    -> {out}")
                    else:
                        lines.append(f"  {row[0]}: {row[1]}")
                self._status("\n".join(lines))
        except Exception as exc:
            import traceback
            self._status(f"⚠ Error: {type(exc)}\n{str(exc)}\n\n{traceback.format_exc()[:400]}")
            raise
        finally:
            self.processor = None
            self._batch_running = False
            self.root.after(0, self._refresh_proc_btn)
            self.root.after(0, lambda: self._cancel_btn.config(state=tk.DISABLED))

    def _on_vid_progress(self, val, msg, batch_idx, batch_total, name):
        overall = 100 * (batch_idx + val / 100) / batch_total
        self._set_batch_progress(overall, f"Video {batch_idx + 1}/{batch_total}: {name}")
        self._set_vid_progress(val, msg)

    def _set_batch_progress(self, val, msg):
        self.root.after(0, lambda: self._batch_prog_var.set(val))
        self.root.after(0, lambda: self._batch_prog_lbl.config(text=msg))

    def _set_vid_progress(self, val, msg):
        self.root.after(0, lambda: self._vid_prog_var.set(val))
        if msg:
            self.root.after(0, lambda: self._vid_prog_lbl.config(text=msg))

    def _status(self, msg: str):
        self.root.after(0, lambda: self._status_lbl.config(text=msg))

    def run(self):
        self.root.mainloop()
        self._close_preview()


if __name__ == "__main__":
    BatchApp().run()
