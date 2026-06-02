

#!/usr/bin/env python3
"""
Threshold Calibration
=====================
Collect raw (unscaled) LD1 diff values from a video to determine the best
scalar factor and DIFF_THRESH for compute_ld1_diff().




Usage:
   python calibrate_threshold.py




Steps:
   1. Set VIDEO_PATH and CIRCLE_JSON_PATH (or fill in CIRCLE_PARAMS manually).
   2. Set QUIET_FRAMES to a range with no ants (background noise only).
   3. Set ACTIVE_FRAMES to a range with ants moving.
   4. Run the script — it prints recommended values and shows a histogram.
"""



from __future__ import annotations
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt




from ant_counter_circle_shadow import LD1_VEC, PROCESS_SCALE




# ── Configuration ─────────────────────────────────────────────────────────────




VIDEO_PATH       = "GX010103 copy - 9_30 colony 17.MP4"
CIRCLE_JSON_PATH = "outputs/GX010103 copy - 9_30 colony 17_circle.json"   # leave blank to use CIRCLE_PARAMS below




# Used only if CIRCLE_JSON_PATH is blank or missing
CIRCLE_PARAMS = {
   "center": (1234, 567),   # (x, y) in original video pixels
   "radius": 200,
   "north_angle": 45.0,
}




QUIET_FRAMES  = (3180, 3380)   # (start, end) frame range — no ants
ACTIVE_FRAMES = (0, 300)   # (start, end) frame range — ants present




# ── Helpers ───────────────────────────────────────────────────────────────────




def load_circle_params(video_path: str) -> dict | None:
   """Try to load circle params from outputs/<stem>_circle.json."""
   stem    = os.path.splitext(os.path.basename(video_path))[0]
   out_dir = os.path.join(os.path.dirname(video_path), "outputs")
   path    = os.path.join(out_dir, f"{stem}_circle.json")
   if not os.path.isfile(path):
       return None
   with open(path) as f:
       data = json.load(f)
   return data.get("circle_params")








def collect_raw_ld1_values(video_path: str, circle_params: dict,
                           start_frame: int, end_frame: int) -> np.ndarray:
   """
   Return a flat array of raw (unscaled) LD1 diff values for all pixels
   inside the circle, over the given frame range.
   """
   cap    = cv2.VideoCapture(video_path)
   orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   proc_w = int(orig_w * PROCESS_SCALE)
   proc_h = int(orig_h * PROCESS_SCALE)




   cx = circle_params["center"][0] * PROCESS_SCALE
   cy = circle_params["center"][1] * PROCESS_SCALE
   r  = circle_params["radius"]    * PROCESS_SCALE
   circle_mask = np.zeros((proc_h, proc_w), dtype=np.uint8)
   cv2.circle(circle_mask, (int(cx), int(cy)), int(r), 255, -1)




   raw_values: list[float] = []
   cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
   prev = None




   for _ in range(end_frame - start_frame):
       ret, frame = cap.read()
       if not ret:
           break
       small = cv2.resize(frame, (proc_w, proc_h))
       if prev is not None:
           hsv1 = cv2.cvtColor(prev,  cv2.COLOR_BGR2HSV).astype(np.float32)
           hsv2 = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
           raw  = np.abs(((hsv2 - hsv1) * LD1_VEC).sum(axis=2))
           raw_values.extend(raw[circle_mask == 255].tolist())
       prev = small




   cap.release()
   return np.array(raw_values)




# ── Main ──────────────────────────────────────────────────────────────────────




def main():
   # Resolve circle params
   circle_params = None
   if CIRCLE_JSON_PATH and os.path.isfile(CIRCLE_JSON_PATH):
       with open(CIRCLE_JSON_PATH) as f:
           circle_params = json.load(f).get("circle_params")
       print(f"Loaded circle from: {CIRCLE_JSON_PATH}")
   if circle_params is None:
       circle_params = load_circle_params(VIDEO_PATH)
       if circle_params:
           stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
           print(f"Loaded circle from outputs/{stem}_circle.json")
   if circle_params is None:
       print("whats up")
       circle_params = CIRCLE_PARAMS
       print("Using hard-coded CIRCLE_PARAMS")




   print(f"\nCollecting background (quiet) frames {QUIET_FRAMES}…")
   quiet = collect_raw_ld1_values(VIDEO_PATH, circle_params, *QUIET_FRAMES)




   print(f"Collecting active frames {ACTIVE_FRAMES}…")
   active = collect_raw_ld1_values(VIDEO_PATH, circle_params, *ACTIVE_FRAMES)




   # ── Statistics ────────────────────────────────────────────────────────────
   bg_99   = np.percentile(quiet,  99)
   bg_999  = np.percentile(quiet,  99.5)
   ant_95  = np.percentile(active, 95)
   ant_999 = np.percentile(active, 100)




   # Scalar: map the brightest real ant signal (99.9th pct) to 255
   recommended_scalar = 255 / ant_999 if ant_999 > 0 else 255 / 30




   # Threshold in scaled space: sit just above background 99th pct
   recommended_thresh = bg_99 * recommended_scalar




   print("\n── Raw LD1 value statistics ──────────────────────────────────")
   print(f"  Background  99th pct : {bg_99:.3f}")
   print(f"  Background  99.9th pct: {bg_999:.3f}")
   print(f"  Active      95th pct : {ant_95:.3f}")
   print(f"  Active      99.9th pct: {ant_999:.3f}")
   print()
   print("── Recommended values ────────────────────────────────────────")
   print(f"  Scalar  (255 / X)  : 255 / {ant_999:.2f} = {recommended_scalar:.2f}")
   print(f"  DIFF_THRESH        : {recommended_thresh:.1f}  (background 99th pct × scalar)")
   print()
   print("  In ant_counter_circle_shadow.py set:")
   print(f"    return (diff * {recommended_scalar:.2f}).clip(0,255).astype(np.uint8)")
   print(f"    DIFF_THRESH = {recommended_thresh:.0f}")


if __name__ == "__main__":
    main()
