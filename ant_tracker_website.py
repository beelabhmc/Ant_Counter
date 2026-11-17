#!/usr/bin/env python3
# =============================================================================
# ANT TRACKER WEBSITE — LOCAL BROWSER CONTROL PANEL
# =============================================================================
# START:
#   python ant_tracker_website.py --tracker /path/to/ant_tracker_modular.py
#   python ant_tracker_website.py --tracker ./ant_tracker_modular.py --port 8765
#   python ant_tracker_website.py /path/to/ant_tracker_modular.py
#   python ant_tracker_website.py --tracker ./ant_tracker_modular.py --no-browser
#
# Website: http://127.0.0.1:8765 (defaults to a browser auto-open)
# --tracker PATH  REQUIRED: existing ant_tracker_modular.py; kept unchanged.
# --port N        Local port (default 8765).
# --host ADDRESS  Bind address (default 127.0.0.1; localhost only recommended).
# --no-browser    Do not automatically open the website.
#
# WEBSITE FEATURES:
# * Normal batch processing: --run, --video / --videos, --parameters, --output,
#   --circle-file, --render-video.
# * Regression: --regression (directory or consolidated CSV), --quadrant,
#   --videos, --parameters, --circle-file, --output, --render-video, --show-plot
#   (plots are available as saved PNG; interactive --show-plot intentionally
#   disabled in the website to avoid blocking a server worker).
# * Optuna: --optimize with --ground-truth and --videos, --parameters, --trials,
#   --match-distance, --seed, --output; or legacy --optimize DATASET.
# * Browser-side video frame scrubber and 2-click calibration:
#     click 1 = center, click 2 = north; radius in ORIGINAL video pixels.
#   Save per-video *_circle.json alongside the source video (the tracker's
#   supported path), or a shared JSON at a user-selected path.
# * Select or upload a parameter JSON preset; no parameter editor.
# * Upload one or several videos directly from the browser (no path required).
# * Optional ground-truth CSV upload and gold/black theme with help tooltips.
# * Folder browser, job launch / live console logs / cancel, results explorer,
#   CSV/JSON preview, image/video playback and download.
#
# SECURITY: This is a LOCAL-DEVELOPMENT app for trusted local users. By default
# it binds only 127.0.0.1. It intentionally reads/writes arbitrary local paths
# chosen through the UI, runs the selected local Python script, and should NOT
# be exposed to the Internet, to an untrusted LAN or through a public tunnel.
# No shell=True is used. Jobs are invoked with sys.executable and argv lists.
# Python requirements: OpenCV (already required by ant_tracker_modular.py).
# Standard-library-only web server; no Flask installation necessary.
# =============================================================================
from __future__ import annotations
import argparse
import ast
import io
import uuid
import json
import math
import mimetypes
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}
PARAM_EXTRA = {
    'PROCESS_SCALE': ('Processing resolution multiplier', .1, 1.0, .25),
    'BUFFER_ZONE': ('Circle crossing buffer (scaled px)', 0, 100, 10),
    'CROSS_HYSTERESIS_FRAMES': ('Crossing hysteresis frames', 1, 30, 4),
    'PATH_CONFIRM_FRAMES': ('Observed frames to confirm crossing', 1, 30, 4),
}


def path_of(raw):
    return Path(os.path.expandvars(str(raw or '').strip())).expanduser().resolve()


def find_specs(tracker: Path):
    """Read the real TUNE_SPECS from the tracker without importing its GUI."""
    source = tracker.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(tracker))
    specs = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'TUNE_SPECS':
                    specs = ast.literal_eval(node.value)
    # Read the tracker's top-level numeric defaults using literal AST values.
    defaults = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and (target.id in specs or target.id in PARAM_EXTRA):
                try:
                    v = ast.literal_eval(node.value)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        defaults[target.id] = v
                except (ValueError, TypeError):
                    pass
    fields = []
    for key, (label, lo, hi, factor) in specs.items():
        fields.append(dict(name=key, label=label, minimum=lo/factor,
                           maximum=hi/factor, default=defaults.get(key, lo/factor),
                           integer=key not in {'VIBRATION_PCT', 'KF_PROC_POS', 'KF_PROC_VEL',
                                               'KF_MEAS_POS', 'REACQUIRE_DISTANCE_SCALE'}))
    for key, (label, lo, hi, fallback) in PARAM_EXTRA.items():
        fields.append(dict(name=key, label=label, minimum=lo, maximum=hi,
                           default=defaults.get(key, fallback), integer=key != 'PROCESS_SCALE'))
    return fields


class WebsiteState:
    def __init__(self, tracker: Path):
        self.tracker = tracker
        self.fields = find_specs(tracker)
        self.param_map = {f['name']: f for f in self.fields}
        self.lock = threading.Lock()
        self.jobs = {}
        self.counter = 0

    def job_list(self):
        with self.lock:
            return [self._job_view(j) for j in reversed(list(self.jobs.values()))]

    def _job_view(self, job):
        p = job['process']
        return {'id': job['id'], 'mode': job['mode'], 'started': job['started'],
                'finished': job['finished'], 'status': job['status'], 'code': job['code'],
                'command': job['command'], 'output': job['output'], 'log': list(job['log']),
                'pid': p.pid if p is not None else None}

    def job(self, job_id):
        with self.lock:
            j = self.jobs.get(str(job_id))
            if not j:
                raise ValueError('Job not found')
            return self._job_view(j)

    def start(self, payload):
        mode = str(payload.get('mode', 'run'))
        if mode not in ('run', 'regression', 'optimize'):
            raise ValueError('Select run, regression or optimize')
        if not self.tracker.is_file():
            raise ValueError('Tracker script not found')
        args = [sys.executable, '-u', str(self.tracker)]
        videos = str(payload.get('videos') or '').strip()
        video = str(payload.get('video') or '').strip()
        output = str(payload.get('output') or '').strip()
        parameters = str(payload.get('parameters') or '').strip()
        circle = str(payload.get('circle_file') or '').strip()
        truth = str(payload.get('ground_truth') or '').strip()
        dataset = str(payload.get('dataset') or '').strip()
        if videos and video:
            raise ValueError('Use either a single video or a video folder')
        if parameters:
            self.require_file(parameters, '.json', 'Parameter JSON')
            args += ['--parameters', str(path_of(parameters))]
        if output:
            path_of(output).mkdir(parents=True, exist_ok=True)
            args += ['--output', str(path_of(output))]
        if mode == 'run':
            args.append('--run')
            if video:
                self.require_file(video, None, 'Video')
                args += ['--video', str(path_of(video))]
            elif videos:
                self.require_dir(videos, 'Video directory')
                args += ['--videos', str(path_of(videos))]
            else:
                raise ValueError('Provide a single video or video folder')
            if circle:
                self.require_file(circle, '.json', 'Circle JSON')
                args += ['--circle-file', str(path_of(circle))]
            if payload.get('render_video'):
                args.append('--render-video')
        elif mode == 'regression':
            if not videos or not truth:
                raise ValueError('Regression requires a video directory and count ground truth (CSV or folder)')
            self.require_dir(videos, 'Video directory')
            if not path_of(truth).exists():
                raise ValueError('Ground truth path does not exist')
            args += ['--regression', str(path_of(truth)), '--videos', str(path_of(videos))]
            quadrant = str(payload.get('quadrant') or 'all').upper()
            if quadrant != 'ALL':
                if quadrant not in ('NE', 'SE', 'SW', 'NW'):
                    raise ValueError('Unknown quadrant')
                args += ['--quadrant', quadrant]
            if circle:
                self.require_file(circle, '.json', 'Circle JSON')
                args += ['--circle-file', str(path_of(circle))]
            if payload.get('render_video'):
                args.append('--render-video')
        else:
            if dataset:
                if videos or truth:
                    raise ValueError('Use the legacy dataset folder OR separate videos and ground-truth folders')
                self.require_dir(dataset, 'Optuna dataset directory')
                args += ['--optimize', str(path_of(dataset))]
            else:
                if not videos or not truth:
                    raise ValueError('Optuna requires videos and positional ground-truth folders')
                self.require_dir(videos, 'Video directory')
                self.require_dir(truth, 'Positional ground-truth directory')
                args += ['--optimize', '--videos', str(path_of(videos)), '--ground-truth', str(path_of(truth))]
            trials = int(payload.get('trials', 30))
            match_distance = float(payload.get('match_distance', 25))
            seed = int(payload.get('seed', 42))
            if trials < 1 or trials > 100000 or not 0 < match_distance < 100000:
                raise ValueError('Invalid trials or match-distance')
            args += ['--trials', str(trials), '--match-distance', str(match_distance), '--seed', str(seed)]
            if circle:
                raise ValueError('Circle calibration is not used for positional Optuna optimization')
        with self.lock:
            self.counter += 1
            key = str(self.counter)
            job = {'id': key, 'mode': mode, 'started': datetime.now().isoformat(timespec='seconds'),
                   'finished': None, 'status': 'starting', 'code': None,
                   'command': args, 'output': str(path_of(output)) if output else '',
                   'log': deque(maxlen=2500), 'process': None}
            self.jobs[key] = job
        def worker():
            try:
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                env.setdefault('MPLBACKEND', 'Agg')
                proc = subprocess.Popen(args, cwd=str(self.tracker.parent),
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, text=True,
                                        errors='replace', bufsize=1, env=env)
                with self.lock:
                    job['process'] = proc
                    job['status'] = 'running'
                for line in proc.stdout:
                    with self.lock:
                        job['log'].append(line.rstrip('\r\n'))
                code = proc.wait()
                with self.lock:
                    job['code'] = code
                    job['status'] = 'completed' if code == 0 else ('cancelled' if job['status'] == 'cancelling' else 'failed')
            except Exception as exc:
                with self.lock:
                    job['log'].append(f'[website error] {exc}')
                    job['status'] = 'failed'
            finally:
                with self.lock:
                    job['finished'] = datetime.now().isoformat(timespec='seconds')
        threading.Thread(target=worker, daemon=True, name=f'tracker-job-{key}').start()
        return self.job(key)

    @staticmethod
    def require_file(raw, suffix, label):
        p = path_of(raw)
        if not p.is_file() or (suffix and p.suffix.lower() != suffix):
            raise ValueError(f'{label} not found or incorrect file type: {p}')

    @staticmethod
    def require_dir(raw, label):
        p = path_of(raw)
        if not p.is_dir():
            raise ValueError(f'{label} not found: {p}')

    def stop(self, id_):
        with self.lock:
            j = self.jobs.get(str(id_))
            if not j:
                raise ValueError('Job not found')
            proc = j['process']
            if proc is None or proc.poll() is not None:
                return self._job_view(j)
            j['status'] = 'cancelling'
            proc.terminate()
            return self._job_view(j)

    def read_parameters(self, raw):
        p = path_of(raw)
        self.require_file(p, '.json', 'Parameters')
        content = json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(content, dict):
            raise ValueError('Parameter file must be a JSON object')
        return {k: v for k, v in content.items() if k in self.param_map and isinstance(v, (float, int)) and not isinstance(v, bool)}

    def write_parameters(self, raw, values):
        p = path_of(raw)
        if p.suffix.lower() != '.json':
            raise ValueError('Parameter file must end with .json')
        if not isinstance(values, dict) or not values:
            raise ValueError('Provide at least one parameter')
        validated = {}
        for name, value in values.items():
            spec = self.param_map.get(name)
            if spec is None:
                raise ValueError(f'Unrecognized parameter: {name}')
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f'{name}: numeric value required')
            if not spec['minimum'] <= value <= spec['maximum']:
                raise ValueError(f'{name}: use {spec["minimum"]} to {spec["maximum"]}')
            if spec['integer'] and int(value) != value:
                raise ValueError(f'{name}: integer value required')
            if name in ('MORPH_OPEN_SIZE', 'MORPH_CLOSE_SIZE') and int(value) % 2 == 0:
                raise ValueError(f'{name}: use an odd kernel size')
            validated[name] = value
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(validated, indent=2) + '\n', encoding='utf-8')
        return {'path': str(p), 'count': len(validated)}


def video_metadata(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f'Cannot open video: {path}')
    data = {'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': float(cap.get(cv2.CAP_PROP_FPS))}
    cap.release()
    return data


def frame_jpeg(path, frame_num=0, max_width=1200):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Could not open video')
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_num)))
        ok, frame = cap.read()
        if not ok:
            raise ValueError('Could not read requested video frame')
        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / width
            frame = cv2.resize(frame, (max_width, max(1, round(height * scale))))
        ok, result = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError('Failed to encode preview')
        return result.tobytes()
    finally:
        cap.release()


def normalize_circle(raw):
    if not isinstance(raw, dict):
        raise ValueError('Circle JSON must be an object')
    raw = raw.get('circle_params', raw)
    center = raw.get('center')
    radius = float(raw.get('radius', 0))
    if not isinstance(center, (list, tuple)) or len(center) != 2 or radius <= 0:
        raise ValueError('Circle must contain center [x,y] and positive radius')
    x, y = float(center[0]), float(center[1])
    if 'north_angle' in raw:
        angle = float(raw['north_angle'])
    elif 'north_point' in raw:
        nx, ny = map(float, raw['north_point'])
        angle = math.degrees(math.atan2(ny-y, nx-x)) % 360
    else:
        raise ValueError('Circle must contain north_angle or north_point')
    if not all(map(math.isfinite, (x, y, radius, angle))):
        raise ValueError('Circle has invalid coordinates')
    return {'center': [x, y], 'radius': radius, 'north_angle': angle % 360}


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ant Tracker Studio</title>
<style>
:root{color-scheme:dark;--bg:#0c111b;--panel:#151d2b;--line:#2b3546;--text:#eaf0f8;--muted:#9baabe;--accent:#65d7bc;--red:#f48787}*{box-sizing:border-box}body{margin:0;background:var(--bg);font:14px system-ui,-apple-system,Segoe UI,sans-serif;color:var(--text)}header{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:18px 26px;background:#111a28;border-bottom:1px solid var(--line)}h1{font-size:21px;margin:0}h2{font-size:17px;margin:0 0 13px}h3{font-size:14px;margin:15px 0 8px}.muted,.hint{color:var(--muted)}.hint{font-size:12px}main{max-width:1550px;margin:auto;padding:20px;display:grid;grid-template-columns:minmax(315px,400px) minmax(430px,1fr);gap:17px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:18px;margin-bottom:16px;min-width:0}.wide{grid-column:1/-1}.tabs{display:flex;gap:6px;flex-wrap:wrap}.tab,button{background:#24334a;color:var(--text);border:1px solid #3c4a60;border-radius:7px;padding:9px 12px;cursor:pointer}.tab.on,.primary{background:#137b6d;border-color:#199782;color:white}button:hover{filter:brightness(1.15)}button:disabled{opacity:.45;cursor:not-allowed}label{display:block;font-size:12px;margin:11px 0 5px;color:#c6d3e0}input,select,textarea{width:100%;background:#0d1521;color:var(--text);padding:10px;border:1px solid #394658;border-radius:7px;font:inherit}input[type=checkbox]{width:auto}.field{display:flex;gap:6px}.field input{min-width:0}.inline{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.inline label{margin:0}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:10px}.hidden{display:none!important}.preview{position:relative;width:100%;min-height:120px;background:#060b11;border:1px solid #3b4758;display:flex;justify-content:center;align-items:center;overflow:hidden}.preview img{width:100%;height:auto;display:block;cursor:crosshair}.preview svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.toolbar{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.toolbar button{flex:1}.mono,pre{font-family:ui-monospace,Consolas,monospace}pre{background:#07101b;border:1px solid var(--line);padding:13px;max-height:340px;overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:12px}.job{padding:12px;border:1px solid var(--line);border-radius:9px;margin-bottom:10px}.badge{color:var(--accent)}.error{color:var(--red)}.file{display:flex;gap:8px;align-items:center;border-bottom:1px solid var(--line);padding:7px 0}.file>span{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.file button{padding:5px 8px}.paramgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:8px 12px}.param label{margin:0 0 4px}.param input{padding:8px}.resultpreview{max-width:100%;max-height:570px}.small{font-size:12px}a{color:#86e6d6}@media(max-width:1000px){main{display:block}} 

/* Gold and black visual identity */
:root{color-scheme:dark;--bg:#080808;--panel:#151411;--line:#4a3d20;--text:#f8f3e5;--muted:#b4a98c;--accent:#f5bd35;--red:#ff7d74}
body{background:radial-gradient(ellipse at 12% 0%,#30250d 0%,#0a0908 42%,#080808 100%)}
header{background:#11100c;border-bottom:1px solid #74551d}h1{color:#ffd064}
.panel{background:linear-gradient(140deg,#181611,#100f0d);border-color:#46381a;box-shadow:0 8px 30px #0004}
.tab,button{background:#282216;color:#f6df9d;border:1px solid #695123}
.tab.on,.primary{background:#eab53d;border-color:#ffd672;color:#1b1405;font-weight:700}
button:hover{filter:brightness(1.13)}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid #ffd15b;outline-offset:2px}
label{color:#e6d3a9}input,select,textarea{background:#0c0b09;color:#faf0d4;border-color:#5a4724}
.preview{background:#000;border-color:#715620}pre{background:#0c0b09}a{color:#ffce5d}.badge{color:#ffd064}
.upload-zone{position:relative;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:120px;gap:7px;background:#19150e;border:2px dashed #a77d2d;border-radius:11px;margin:8px 0;padding:17px;cursor:pointer;text-align:center}
.upload-zone:hover,.upload-zone.dragging{border-color:#ffd76f;background:#29200e}.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer}.upload-zone strong{font-size:15px;color:#ffcf64}
.upload-progress{height:7px;background:#302713;border-radius:5px;overflow:hidden;margin-top:8px}.upload-progress>div{height:100%;background:#edb632;width:0%;transition:width .15s}
.info{display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:bold;width:18px;height:18px;border:1px solid #b08b3c;color:#ffd35b;border-radius:50%;margin-left:5px;cursor:help;vertical-align:middle;position:relative}
.info:hover::after,.info:focus::after{content:attr(data-help);position:absolute;z-index:30;top:24px;left:0;min-width:230px;max-width:310px;padding:10px;white-space:normal;line-height:1.5;background:#32250d;color:#fff3cb;border:1px solid #daa943;border-radius:8px;box-shadow:0 6px 22px #000b;text-align:left;font-size:12px;font-weight:400}
.pill{display:inline-block;background:#463312;color:#ffd46b;padding:3px 8px;border-radius:20px;font-size:12px}
</style></head><body><header><div><h1>🐜 Ant Tracker Studio</h1><div class="muted small" id="engineLabel">Local processing dashboard</div></div><div class="muted">Localhost · browser interface</div></header>
<main><div><section class="panel"><h2>1. Select a mode</h2><div class="tabs"><button class="tab on" data-mode="run">Process videos</button><button class="tab" data-mode="regression">Regression</button><button class="tab" data-mode="optimize">Optuna</button></div><p class="hint" id="modeHelp">Run the production ant detector and counter against one video or a folder.</p>
<label>Upload videos <span class="info" tabindex="0" data-help="Choose one or multiple video files from your computer. Uploads are saved on the localhost server and the same collection can be processed, used in regression, or optimized with Optuna.">i</span></label>
<div class="upload-zone" id="videoDrop"><strong>⬆ Click to upload or drop videos here</strong><span class="hint">One or multiple .mp4, .avi, .mov, .mkv, .m4v videos</span><input id="videoUpload" type="file" accept=".mp4,.avi,.mov,.mkv,.m4v,video/*" multiple></div>
<div class="upload-progress"><div id="videoProgress"></div></div><p class="hint" id="uploadStatus">No videos uploaded yet.</p>
<div id="singleVideoRow" class="hidden"><input id="video" placeholder="Single video path"></div>
<label>Uploaded videos folder <span class="info" tabindex="0" data-help="This server-local path is filled automatically after upload. You may also choose an existing folder on this same computer.">i</span></label><div class="field"><input id="videos" placeholder="Upload videos above or choose a local directory"><button data-pick="videos" data-kind="directory">Browse</button></div>
<div class="toolbar"><button id="refreshVideoList">List videos</button><button id="selectFirst">Calibrate first video</button></div><div id="videoList" class="hint">Upload videos to list them here.</div>
<div id="datasetRow" class="hidden"><label>Legacy Optuna dataset root (alternative to video + truth folders)</label><div class="field"><input id="dataset" placeholder="/path/dataset"><button data-pick="dataset" data-kind="directory">Browse</button></div></div>
<div id="groundTruthRow"><label><span id="groundTruthLabel">Ground truth (CSV or folder)</span> <span class="info" tabindex="0" data-help="Regression uses a CSV of correct quadrant IN/OUT counts. Optuna instead needs a directory of frame-by-frame ant position labels.">i</span></label><div class="field"><input id="ground_truth" placeholder="/path/counts.csv"><button data-pick="ground_truth" data-kind="any">Browse</button></div><div class="toolbar"><input type="file" id="groundTruthUpload" accept=".csv,.json" multiple aria-label="Upload ground truth CSV files"></div><p class="hint" id="groundTruthStatus">You can upload a CSV or choose an existing local folder.</p><p class="hint" id="groundTruthHint">For regression: consolidated count CSV or per-video count CSV folder.</p></div>
<label>Parameters JSON <span class="info" tabindex="0" data-help="Upload a .json preset to control the detector, tracker, and counting thresholds. The website does not modify parameter values. Leave blank to use the script defaults.">i</span></label>
<div class="field"><input id="parameters" readonly placeholder="Upload your JSON preset (optional)"><button data-pick="parameters" data-kind="json">Local file</button></div>
<div class="toolbar"><input type="file" id="parametersUpload" accept=".json,application/json" aria-label="Choose parameter JSON" style="max-width:100%"></div><p class="hint" id="parameterStatus">No parameter file selected — script defaults will be used.</p>
<label>Output directory (optional) <span class="info" tabindex="0" data-help="All reports, annotated videos, CSVs, and regression plots will be saved in this folder on your computer.">i</span></label><div class="field"><input id="output" placeholder="/path/results"><button data-pick="output" data-kind="directory" data-new="1">Browse</button></div>
<label>Shared circle JSON (optional) <span class="info" tabindex="0" data-help="A single calibrated circle used for all videos. Individual *_circle.json files saved next to each uploaded video can be used instead.">i</span></label><div class="field"><input id="circle_file" placeholder="/path/circle.json"><button data-pick="circle_file" data-kind="json">Browse</button></div>
<div id="regressionOptions" class="hidden"><label>Regression quadrant <span class="info" tabindex="0" data-help="Compare one direction sector or all four; the engine fits ground-truth counts against predicted IN/OUT counts.">i</span></label><select id="quadrant"><option value="all">All quadrants</option><option>NE</option><option>SE</option><option>SW</option><option>NW</option></select><p class="hint">Linear fit, R², MAE, RMSE, count-error table and two-panel PNG are saved.</p></div>
<div id="optunaOptions" class="hidden"><div class="twocol"><div><label>Optuna trials <span class="info" tabindex="0" data-help="Number of additional optimization trials. More trials can improve parameter search but take longer.">i</span></label><input id="trials" type="number" min="1" value="30"></div><div><label>Match distance (px) <span class="info" tabindex="0" data-help="Maximum accepted distance in original video pixels between a detected ant and its labeled location during optimization.">i</span></label><input id="match_distance" type="number" min="0.1" step="0.5" value="25"></div></div><label>Random seed <span class="info" tabindex="0" data-help="Controls reproducibility of the optimization sampler. A persisted study may still be resumed.">i</span></label><input id="seed" type="number" value="42"><p class="hint">Optuna ground truth contains ant locations, not entry/exit counts.</p></div>
<div id="renderRow"><label class="inline"><input type="checkbox" id="render_video"> Render annotated MP4 files <span class="info" tabindex="0" data-help="Generate an MP4 with ant tracks, quadrant boundaries and count overlays; this can slow processing and use substantial disk space.">i</span></label></div>
<div class="toolbar"><button class="primary" id="startJob">▶ Start processing</button></div><div class="hint" id="runNotice"></div></section></div>
<div><section class="panel"><h2>2. Circle calibration <span class="info" tabindex="0" data-help="Click the video entrance center, click a point indicating north, then save the calibration JSON. Circle coordinates and radius use original video pixels.">i</span></h2><p class="hint">Select a video, choose a frame, click center and then click north. Radius and coordinates use original video pixels. Saved calibrations are used by normal processing and regression.</p>
<label>Video for preview</label><div class="field"><input id="previewVideo" placeholder="/path/ants.mp4"><button data-pick="previewVideo" data-kind="video">Browse</button><button id="openPreview">Open</button></div><div class="hint" id="videoMeta">No video loaded</div>
<label>Video frame <span id="frameNumber">0</span></label><input type="range" id="frameSlider" min="0" max="0" value="0"><div class="preview" id="preview"><span class="hint">Video frame preview appears here.</span></div>
<div class="twocol"><div><label>Circle radius (original px)</label><input type="number" id="radius" value="410" min="1" step="1"></div><div><label>North angle (degrees)</label><input type="number" id="northAngle" value="270" min="0" max="359.99" step="0.1"></div></div>
<div class="toolbar"><button id="resetCircle">Clear selection</button><button id="loadCircle">Load circle JSON</button><button id="saveCircle" class="primary">Save per-video circle</button></div><label>Custom circle save/load path (optional)</label><div class="field"><input id="circleSavePath" placeholder="Auto: next to video, *_circle.json"><button data-pick="circleSavePath" data-kind="json" data-new="1">Browse</button></div>
<div class="hint" id="circleInfo">Click video to choose center, then north.</div></section>
<section class="panel"><h2>3. Processing jobs and logs</h2><div class="toolbar"><button id="refreshJobs">Refresh</button><button id="stopJob">Stop selected job</button></div><div id="jobList" class="hint">No jobs yet.</div><h3>Live log</h3><pre id="jobLog">Start a job to view its progress here.</pre></section>
</div>
</main>
<div id="picker" class="hidden" style="position:fixed;inset:0;background:#000b;z-index:10;display:flex;align-items:center;justify-content:center;padding:18px"><div class="panel" style="width:min(730px,98vw);max-height:90vh;overflow:auto"><h2>Local file browser</h2><div class="field"><input id="pickerDir" placeholder="/path"><button id="pickerGo">Go</button><button id="pickerParent">Up</button></div><div class="hint" id="pickerHelp"></div><div id="pickerList" style="max-height:50vh;overflow:auto"></div><div class="toolbar"><button id="pickerSelectFolder">Select current folder</button><button id="pickerManual">Use typed path</button><button id="pickerCancel">Cancel</button></div></div></div>
<script>
const $=s=>document.getElementById(s);let mode='run',frameInfo=null,circle={center:null,north:null},jobId=null,jobTimer=null,pickerTarget=null,pickerKind=null;let uploadSession=null;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(url,data){let opt=data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)};let r=await fetch(url,opt),body=await r.json();if(!r.ok||!body.ok)throw Error(body.error||'Request failed');return body.data}
function message(s,bad=false){$('runNotice').textContent=s;$('runNotice').className=bad?'error':'hint'}
function get(id){return $(id).value.trim()}
function setMode(m){mode=m;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on',x.dataset.mode===m));$('singleVideoRow').classList.add('hidden');$('regressionOptions').classList.toggle('hidden',m!=='regression');$('optunaOptions').classList.toggle('hidden',m!=='optimize');$('datasetRow').classList.toggle('hidden',m!=='optimize');$('renderRow').classList.toggle('hidden',m==='optimize');$('groundTruthRow').classList.toggle('hidden',m==='run');$('circle_file').disabled=m==='optimize';$('ground_truth').placeholder=m==='regression'?'/path/representative_counts.csv or counts_dir':'/path/position_labels';$('groundTruthLabel').textContent=m==='regression'?'Count ground truth (CSV or folder)': 'Ground truth ant positions (Optuna only)';$('groundTruthUpload').accept=m==='optimize'?'.csv':'.csv,.json';$('groundTruthHint').textContent=m==='regression'?'One CSV with Video Title + eight count columns, or a directory of per-video CSVs.':'For Optuna: positional annotation CSV files (frame, ant_x, ant_y, etc.)';$('modeHelp').textContent={run:'Process a video or folder without the desktop GUI.',regression:'Run videos and compare observed quadrant IN/OUT counts to annotated count totals, with fitted regression and plots.',optimize:'Optimize detection F1 using Optuna, positional labels, and a video folder.'}[m]}
document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
async function listVideos(){try{let d=await api('/api/videos?directory='+encodeURIComponent(get('videos')));$('videoList').innerHTML=d.videos.length?d.videos.map(f=>`<div class="file"><span>${esc(f.name)}</span><button data-preview="${esc(f.path)}">Calibrate</button></div>`).join(''):'No supported videos in folder';document.querySelectorAll('[data-preview]').forEach(b=>b.onclick=()=>{$('previewVideo').value=b.dataset.preview;openPreview()})}catch(e){$('videoList').textContent=e.message}}
$('dataset').addEventListener('input',()=>{if(get('dataset')){$('videos').value='';$('ground_truth').value=''}});$('videos').addEventListener('input',()=>{if(get('videos'))$('dataset').value=''});$('ground_truth').addEventListener('input',()=>{if(get('ground_truth'))$('dataset').value=''});$('refreshVideoList').onclick=listVideos;$('selectFirst').onclick=async()=>{try{let d=await api('/api/videos?directory='+encodeURIComponent(get('videos')));if(!d.videos.length)throw Error('No videos found');$('previewVideo').value=d.videos[0].path;openPreview()}catch(e){alert(e.message)}};
function payload(){return {mode,video:get('video'),videos:get('videos'),output:get('output'),parameters:get('parameters'),circle_file:mode==='optimize'?'':get('circle_file'),ground_truth:get('ground_truth'),dataset:mode==='optimize'?get('dataset'):'',quadrant:get('quadrant'),trials:Number(get('trials')),match_distance:Number(get('match_distance')),seed:Number(get('seed')),render_video:$('render_video').checked}}
$('startJob').onclick=async()=>{try{let j=await api('/api/jobs',payload());jobId=j.id;message('Job '+j.id+' started.');await refreshJobs();if(jobTimer)clearInterval(jobTimer);jobTimer=setInterval(refreshJobs,1800)}catch(e){message(e.message,true)}};
async function refreshJobs(){try{let data=await api('/api/jobs');$('jobList').innerHTML=data.length?data.map(j=>`<div class="job" style="cursor:pointer;border-color:${j.id===jobId?'#f5bd35':''}" data-job="${j.id}"><b>#${j.id} ${esc(j.mode)}</b> <span class="badge">${esc(j.status)}</span> ${esc(j.started)} ${j.code!==null?'(exit '+j.code+')':''}<div class="hint mono">${esc(j.command.join(' '))}</div></div>`).join(''):'No jobs yet.';document.querySelectorAll('[data-job]').forEach(x=>x.onclick=()=>{jobId=x.dataset.job;refreshJobs()});if(jobId){let j=data.find(x=>x.id===jobId);if(j){let el=$('jobLog'),nearBottom=el.scrollHeight-el.scrollTop-el.clientHeight<100;el.textContent=j.log.join('\n')||'Waiting for output...';if(nearBottom)el.scrollTop=el.scrollHeight;}}}catch(e){$('jobLog').textContent=e.message}}
$('refreshJobs').onclick=refreshJobs;$('stopJob').onclick=async()=>{if(!jobId)return alert('Select a job first');if(!confirm('Stop job '+jobId+'?'))return;try{await api('/api/stop',{id:jobId});refreshJobs()}catch(e){alert(e.message)}};
function readCircleInputs(){if(circle.center&&circle.north){let dx=circle.north[0]-circle.center[0],dy=circle.north[1]-circle.center[1];$('northAngle').value=(((Math.atan2(dy,dx)*180/Math.PI)%360)+360)%360}drawCircle()}
async function openPreview(){try{let video=get('previewVideo');frameInfo=await api('/api/video-info?path='+encodeURIComponent(video));$('videoMeta').textContent=`${frameInfo.width} × ${frameInfo.height} · ${frameInfo.frames} frames · ${frameInfo.fps.toFixed(2)} fps`;let slider=$('frameSlider');slider.max=Math.max(0,frameInfo.frames-1);slider.value=0;circle={center:null,north:null};showFrame()}catch(e){alert(e.message)}}
function showFrame(){if(!frameInfo)return;let frame=Number($('frameSlider').value);$('frameNumber').textContent=frame;let p=$('preview');p.innerHTML=`<img id="frameImg" src="/api/frame?path=${encodeURIComponent(get('previewVideo'))}&frame=${frame}&t=${Date.now()}" alt="Video frame"><svg id="circleOverlay" viewBox="0 0 ${frameInfo.width} ${frameInfo.height}" preserveAspectRatio="none"></svg>`;$('frameImg').onclick=ev=>{let rect=ev.target.getBoundingClientRect();let x=(ev.clientX-rect.left)*frameInfo.width/rect.width,y=(ev.clientY-rect.top)*frameInfo.height/rect.height;if(!circle.center){circle.center=[x,y];circle.north=null}else if(!circle.north){circle.north=[x,y]}else{circle.center=[x,y];circle.north=null}readCircleInputs()};$('frameImg').onload=drawCircle}
let frameTimeout=null;$('frameSlider').oninput=()=>{if(frameTimeout)clearTimeout(frameTimeout);frameTimeout=setTimeout(showFrame,100)};$('openPreview').onclick=openPreview;
function drawCircle(){let svg=$('circleOverlay');if(!svg||!frameInfo)return;let r=Math.max(1,Number(get('radius'))||1);if(!circle.center){svg.innerHTML='';$('circleInfo').textContent='Click frame: center, then north direction';return}let [x,y]=circle.center;let ang=(Number(get('northAngle'))||0)*Math.PI/180;let nx=x+Math.cos(ang)*r,ny=y+Math.sin(ang)*r;svg.innerHTML=`<circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="#f5bd35" stroke-width="4"/><line x1="${x}" y1="${y}" x2="${nx}" y2="${ny}" stroke="#ffe082" stroke-width="4"/><circle cx="${x}" cy="${y}" r="6" fill="#fff"/>${circle.north?`<circle cx="${circle.north[0]}" cy="${circle.north[1]}" r="6" fill="#ffe082"/>`:''}`;$('circleInfo').textContent=`Center (${x.toFixed(1)}, ${y.toFixed(1)}), radius ${r.toFixed(1)}, north angle ${get('northAngle')}°${circle.north?' — ready to save':' — click to set north'}`}
$('radius').oninput=drawCircle;$('northAngle').oninput=drawCircle;$('resetCircle').onclick=()=>{circle={center:null,north:null};drawCircle()};
$('saveCircle').onclick=async()=>{try{if(!circle.center||!circle.north)throw Error('Click the center and the north direction first');let d=await api('/api/circle-save',{video:get('previewVideo'),path:get('circleSavePath'),center:circle.center,radius:Number(get('radius')),north_angle:Number(get('northAngle'))});$('circleSavePath').value=d.path;$('circleInfo').textContent='Saved: '+d.path;message('Circle saved. It will be automatically discovered for this video if saved next to it.')}catch(e){alert(e.message)}};
$('loadCircle').onclick=async()=>{try{let path=get('circleSavePath')||get('circle_file');if(!path&&get('previewVideo')){let p=get('previewVideo');path=p.replace(/\.[^/.]+$/, '')+'_circle.json'}if(!path)throw Error('Provide a circle JSON path');let d=await api('/api/circle-load?path='+encodeURIComponent(path));circle.center=d.center;let a=d.north_angle*Math.PI/180;circle.north=[d.center[0]+Math.cos(a)*d.radius,d.center[1]+Math.sin(a)*d.radius];$('radius').value=d.radius;$('northAngle').value=d.north_angle;drawCircle();message('Loaded circle from '+path)}catch(e){alert(e.message)}};
function pickerClose(){$('picker').classList.add('hidden')}async function pickerGo(path){try{let d=await api('/api/browse?path='+encodeURIComponent(path));$('pickerDir').value=d.path;$('pickerList').innerHTML=d.entries.map(f=>`<div class="file"><span>${f.directory?'📁':'📄'} ${esc(f.name)}</span><button data-pickpath="${esc(f.path)}" data-pickdir="${f.directory?'1':'0'}">${f.directory?'Open':'Choose'}</button>${f.directory?`<button data-seldir="${esc(f.path)}">Select</button>`:''}</div>`).join('');document.querySelectorAll('[data-pickpath]').forEach(b=>b.onclick=()=>{if(b.dataset.pickdir==='1')pickerGo(b.dataset.pickpath);else{$(pickerTarget).value=b.dataset.pickpath;pickerClose()}});document.querySelectorAll('[data-seldir]').forEach(b=>b.onclick=()=>{$(pickerTarget).value=b.dataset.seldir;pickerClose()})}catch(e){$('pickerList').textContent=e.message}}
document.querySelectorAll('[data-pick]').forEach(b=>b.onclick=()=>{pickerTarget=b.dataset.pick;pickerKind=b.dataset.kind;$('pickerHelp').textContent='Browse folders or paste an absolute path; selects server-local files (same computer as Python).';$('picker').classList.remove('hidden');let v=get(pickerTarget)||get('videos')||'.';pickerGo(v)});$('pickerGo').onclick=()=>pickerGo(get('pickerDir'));$('pickerParent').onclick=()=>{let path=get('pickerDir').replace(/[/\\]+$/,'');pickerGo(path.substring(0,Math.max(1,path.lastIndexOf('/')))||'/')};$('pickerSelectFolder').onclick=()=>{$(pickerTarget).value=get('pickerDir');pickerClose()};$('pickerManual').onclick=()=>{$(pickerTarget).value=get('pickerDir');pickerClose()};$('pickerCancel').onclick=pickerClose;
// Upload individual browser-selected files. Each video stays in the same session folder.
function sessionId(){if(!uploadSession){uploadSession=(crypto.randomUUID?crypto.randomUUID().replaceAll('-',''):[...crypto.getRandomValues(new Uint8Array(16))].map(x=>x.toString(16).padStart(2,'0')).join(''))}return uploadSession}
async function uploadFile(file,kind,onProgress){
  const url='/api/upload?kind='+encodeURIComponent(kind)+'&filename='+encodeURIComponent(file.name)+'&session='+sessionId();
  // XMLHttpRequest reports byte-level progress during large video uploads.
  return await new Promise((resolve,reject)=>{const xhr=new XMLHttpRequest();xhr.open('POST',url);xhr.setRequestHeader('Content-Type','application/octet-stream');xhr.upload.onprogress=e=>{if(e.lengthComputable&&onProgress)onProgress(e.loaded/e.total)};xhr.onload=()=>{try{const r=JSON.parse(xhr.responseText);if(xhr.status>=200&&xhr.status<300&&r.ok)resolve(r.data);else reject(Error(r.error||'Upload failed'))}catch(err){reject(err)}};xhr.onerror=()=>reject(Error('Upload connection failed'));xhr.send(file)})
}
$('videoUpload').onchange=async ev=>{const files=Array.from(ev.target.files||[]);if(!files.length)return;const allowed=['.mp4','.avi','.mov','.mkv','.m4v'];let uploaded=[];try{for(let i=0;i<files.length;i++){let f=files[i];if(!allowed.some(ext=>f.name.toLowerCase().endsWith(ext)))throw Error('Unsupported video: '+f.name);$('uploadStatus').textContent=`Uploading ${i+1}/${files.length}: ${f.name}`;const d=await uploadFile(f,'video',percent=>$('videoProgress').style.width=(((i+percent)/files.length)*100).toFixed(1)+'%');uploaded.push(d);$('videos').value=d.folder;$('video').value='';$('dataset').value=''}$('uploadStatus').textContent=`Uploaded ${uploaded.length} video(s) · ${get('videos')}`;message('Video upload complete');await listVideos();if(uploaded.length===1){$('previewVideo').value=uploaded[0].path;openPreview()}}catch(e){message(e.message,true);$('uploadStatus').textContent='Upload error: '+e.message}ev.target.value=''};
const drop=$('videoDrop');drop.ondragover=e=>{e.preventDefault();drop.classList.add('dragging')};drop.ondragleave=()=>drop.classList.remove('dragging');drop.ondrop=e=>{e.preventDefault();drop.classList.remove('dragging');const inp=$('videoUpload');inp.files=e.dataTransfer.files;inp.dispatchEvent(new Event('change'))};
$('parametersUpload').onchange=async e=>{let f=e.target.files[0];if(!f)return;try{const result=await uploadFile(f,'parameters');$('parameters').value=result.path;$('parameterStatus').textContent='Selected: '+result.filename+' ('+(result.size/1024).toFixed(1)+' KB)';message('Parameter preset uploaded')}catch(err){message(err.message,true)}e.target.value=''};
$('groundTruthUpload').onchange=async e=>{const files=Array.from(e.target.files||[]);if(!files.length)return;try{let last=null;for(let i=0;i<files.length;i++){last=await uploadFile(files[i],'ground_truth');$('groundTruthStatus').textContent=`Uploaded ${i+1}/${files.length}: ${files[i].name}`}if(mode==='optimize'||files.length>1){$('ground_truth').value=last.folder}else{$('ground_truth').value=last.path}$('dataset').value='';message(`${files.length} ground truth file(s) uploaded`)}catch(err){message(err.message,true)}e.target.value=''};
async function init(){try{let d=await api('/api/config');$('engineLabel').textContent='Engine: '+d.tracker;$('videos').value='';$('output').value=d.folder+'/website_results';}catch(e){message(e.message,true)}setMode('run');refreshJobs()}init();
</script></body></html>'''


def make_handler(state: WebsiteState):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'AntTrackerWebsite/1.0'

        def respond(self, payload, status=200, content_type='application/json; charset=utf-8'):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(payload)

        def get_params(self):
            return urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)

        def query(self, name, default=''):
            return self.get_params().get(name, [default])[0]

        def json_body(self):
            size = int(self.headers.get('Content-Length', 0))
            if size > 1_000_000:
                raise ValueError('Request body too large')
            return json.loads(self.rfile.read(size).decode('utf-8')) if size else {}

        def do_GET(self):
            try:
                route = urllib.parse.urlsplit(self.path).path
                if route == '/':
                    return self.respond(HTML.encode('utf-8'), content_type='text/html; charset=utf-8')
                if route == '/favicon.ico':
                    return self.respond(b'', 204, 'image/x-icon')
                if route == '/api/config':
                    data = {'tracker': str(state.tracker), 'folder': str(state.tracker.parent), 'supported_uploads': ['video','parameters','ground_truth','circle']}
                elif route == '/api/jobs':
                    data = state.job_list()
                elif route == '/api/job':
                    data = state.job(self.query('id'))
                elif route == '/api/videos':
                    folder = path_of(self.query('directory'))
                    if not folder.is_dir():
                        raise ValueError(f'Video folder does not exist: {folder}')
                    data = {'videos': [{'name': p.name, 'path': str(p)} for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXT]}
                elif route == '/api/browse':
                    raw = self.query('path') or str(state.tracker.parent)
                    p = path_of(raw)
                    if p.is_file():
                        p = p.parent
                    if not p.is_dir():
                        # For output directory creation, open nearest existing parent.
                        v = p
                        while not v.exists() and v != v.parent:
                            v = v.parent
                        p = v
                    if not p.is_dir():
                        raise ValueError('Not a directory')
                    entries = [{'path': str(f), 'name': f.name, 'directory': f.is_dir()} for f in p.iterdir() if not f.name.startswith('.')]
                    entries.sort(key=lambda d: (not d['directory'], d['name'].lower()))
                    data = {'path': str(p), 'entries': entries[:1000]}
                elif route == '/api/video-info':
                    p = path_of(self.query('path'))
                    state.require_file(p, None, 'Video')
                    if p.suffix.lower() not in VIDEO_EXT:
                        raise ValueError('Unsupported video format')
                    data = video_metadata(p)
                elif route == '/api/frame':
                    p = path_of(self.query('path'))
                    state.require_file(p, None, 'Video')
                    if p.suffix.lower() not in VIDEO_EXT:
                        raise ValueError('Unsupported video format')
                    return self.respond(frame_jpeg(p, int(self.query('frame', '0'))), content_type='image/jpeg')
                elif route == '/api/circle-load':
                    p = path_of(self.query('path'))
                    state.require_file(p, '.json', 'Circle calibration')
                    data = normalize_circle(json.loads(p.read_text(encoding='utf-8')))
                elif route == '/api/text':
                    p = path_of(self.query('path'))
                    if p.suffix.lower() not in ('.csv', '.json', '.txt', '.log'):
                        raise ValueError('Unsupported text extension')
                    state.require_file(p, None, 'Text file')
                    data = {'text': p.read_bytes()[:250_000].decode('utf-8', 'replace')}
                elif route == '/api/download':
                    p = path_of(self.query('path'))
                    state.require_file(p, None, 'Result file')
                    # Browser video players need Range requests for seeking.
                    size = p.stat().st_size
                    range_header = self.headers.get('Range', '')
                    start, end, status = 0, size - 1, 200
                    if range_header.startswith('bytes='):
                        spec = range_header[6:].split(',')[0]
                        a, _, b = spec.partition('-')
                        if a:
                            start = int(a)
                            end = min(end, int(b)) if b else end
                        elif b:
                            start = max(0, size - int(b))
                        if start > end or start >= size:
                            self.send_response(416)
                            self.send_header('Content-Range', f'bytes */{size}')
                            self.end_headers()
                            return
                        status = 206
                    self.send_response(status)
                    self.send_header('Content-Type', mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
                    self.send_header('Accept-Ranges', 'bytes')
                    self.send_header('Content-Length', str(max(0, end-start+1)))
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    if status == 206:
                        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                    self.end_headers()
                    with p.open('rb') as stream:
                        stream.seek(start)
                        remaining = end-start+1
                        while remaining > 0:
                            chunk = stream.read(min(65536, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                    return
                else:
                    return self.respond({'ok': False, 'error': 'Not found'}, 404)
                self.respond({'ok': True, 'data': data})
            except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self.respond({'ok': False, 'error': str(exc)}, 400)
            except Exception:
                traceback.print_exc()
                self.respond({'ok': False, 'error': 'Internal server error'}, 500)

        def do_POST(self):
            try:
                route = urllib.parse.urlsplit(self.path).path
                if route == '/api/upload':
                    # Stream binary uploads in chunks: large videos never enter RAM.
                    kind = self.query('kind')
                    filename = Path(self.query('filename')).name
                    if not filename or filename in ('.', '..') or filename.startswith('.'):
                        raise ValueError('A valid upload filename is required')
                    extensions = {'video': VIDEO_EXT, 'parameters': {'.json'},
                                  'ground_truth': {'.csv', '.json'}, 'circle': {'.json'}}
                    if kind not in extensions or Path(filename).suffix.lower() not in extensions[kind]:
                        raise ValueError('Unsupported upload type or extension')
                    length = int(self.headers.get('Content-Length', '0'))
                    if length <= 0 or length > 10 * 1024**3:
                        raise ValueError('Upload must be between 1 byte and 10 GB')
                    session = self.query('session')
                    if not session or len(session) != 32 or not all(c in '0123456789abcdef' for c in session):
                        raise ValueError('Invalid upload session')
                    base = state.tracker.parent / 'ant_tracker_uploads' / session
                    folder = base / ('videos' if kind == 'video' else kind)
                    folder.mkdir(parents=True, exist_ok=True)
                    target = folder / filename
                    # Ensure duplicate names do not silently replace uploaded videos.
                    if target.exists():
                        stem, ext = target.stem, target.suffix
                        for idx in range(2, 10001):
                            possible = folder / f'{stem}_{idx}{ext}'
                            if not possible.exists():
                                target = possible
                                break
                    temp = folder / ('.upload-' + uuid.uuid4().hex)
                    try:
                        left = length
                        with temp.open('wb') as stream:
                            while left:
                                chunk = self.rfile.read(min(left, 1024 * 1024))
                                if not chunk:
                                    raise ValueError('Upload interrupted')
                                stream.write(chunk)
                                left -= len(chunk)
                        if kind in ('parameters', 'circle'):
                            content = json.loads(temp.read_text(encoding='utf-8'))
                            if not isinstance(content, dict):
                                raise ValueError('JSON file must contain an object')
                        temp.replace(target)
                    finally:
                        temp.unlink(missing_ok=True)
                    return self.respond({'ok': True, 'data': {'path': str(target),
                                         'folder': str(folder), 'filename': target.name,
                                         'size': length}})
                body = self.json_body()
                if route == '/api/jobs':
                    data = state.start(body)
                elif route == '/api/stop':
                    data = state.stop(body.get('id'))
                elif route == '/api/circle-save':
                    video = path_of(body.get('video'))
                    state.require_file(video, None, 'Video for circle')
                    target = body.get('path') or str(video.with_name(video.stem + '_circle.json'))
                    p = path_of(target)
                    if p.suffix.lower() != '.json':
                        raise ValueError('Circle save path must end in .json')
                    circle = normalize_circle(body)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps({'video': str(video), 'circle_params': circle}, indent=2)+'\n', encoding='utf-8')
                    data = {'path': str(p), 'circle': circle}
                else:
                    return self.respond({'ok': False, 'error': 'Not found'}, 404)
                self.respond({'ok': True, 'data': data})
            except (ValueError, OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self.respond({'ok': False, 'error': str(exc)}, 400)
            except Exception:
                traceback.print_exc()
                self.respond({'ok': False, 'error': 'Internal server error'}, 500)

        def log_message(self, format, *args):
            if not self.path.startswith('/api/frame'):
                print('[web]', format % args, flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description='Local website for ant_tracker_modular.py')
    parser.add_argument('tracker_path', nargs='?', help='Path to ant_tracker_modular.py')
    parser.add_argument('--tracker', help='Path to ant_tracker_modular.py')
    parser.add_argument('--host', default='127.0.0.1', help='Interface to bind (default: localhost only)')
    parser.add_argument('--port', type=int, default=8765, help='Web UI port')
    parser.add_argument('--no-browser', action='store_true', help='Do not auto-open a browser tab')
    args = parser.parse_args()
    p = path_of(args.tracker or args.tracker_path or 'ant_tracker_modular.py')
    if not p.is_file():
        parser.error(f'Tracker not found: {p}; use --tracker /path/to/ant_tracker_modular.py')
    if not 0 < args.port < 65536:
        parser.error('--port must be 1–65535')
    state = WebsiteState(p)
    app = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    url = f'http://127.0.0.1:{args.port}/'
    print(f'Ant Tracker Studio: {url}\nTracker: {p}\nPress Ctrl+C to stop the web server.', flush=True)
    if not args.no_browser:
        threading.Timer(.6, lambda: webbrowser.open(url)).start()
    try:
        app.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        print('\nShutting down website...', flush=True)
    finally:
        app.server_close()


if __name__ == '__main__':
    main()
