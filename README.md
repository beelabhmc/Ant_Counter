# Ant Counter

Counts ants entering and leaving a hole through each quadrant (NE, SE, SW, NW). Ants are detected by frame differencing (only things that move show up) and tracked between frames with a Kalman filter and Hungarian assignment.

## Contents

- [Set up](#set-up)
- [Ant Annotator](#ant-annotator)
- [Ant Tracking](#ant-tracking)
  - [Inputs](#tracking-inputs) · [Options](#tracking-options) · [GUI](#tracking-in-the-gui) · [CLI](#tracking-from-the-cli) · [Outputs](#tracking-outputs)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Regression](#regression)
  - [Inputs](#regression-inputs) · [Options](#regression-options) · [CLI and GUI](#running-regression) · [Outputs](#regression-outputs)

## Set up

1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. Create the environment (Optuna comes from conda-forge):
   ```bash
   conda create --name ant_counter -c conda-forge --file requirements.txt python=3.9
   conda activate ant_counter
   ```
3. Optional, for dragging and dropping videos onto the GUI: `pip install tkinterdnd2`

Run commands from the `Ant_Counter` folder. `python ant_tracker_modular.py --help` lists every option.

## Ant Annotator

Marks ant centers on 30-second clips to make the labeled positions that [Hyperparameter Optimization](#hyperparameter-optimization) uses. One frame per second is annotated (30 frames per clip).

```bash
cd annotations
python ant_annotator.py                                          # click every ant yourself
python ant_annotator.py --tracker ../ant_tracker_modular.py      # start from guessed points
python ant_annotator.py --tracker ../ant_tracker_modular.py --parameters ant_tuning_parameters.json
```

| Option | Mandatory? | What it does |
|---|---|---|
| `--tracker PATH` | optional | Path to `ant_tracker_modular.py`. The tracker runs on each new clip and pre-fills every frame with guessed points. |
| `--parameters JSON` | optional (needs `--tracker`) | Tracker settings for the guesses; tuned settings guess better than the defaults |

1. Choose the **Output folder**, **Add videos**, pick a start time and click **Go — annotate**.
2. On each frame:
   - **Click** to add a point.
   - **Drag** a point to move it.
   - **Right-click** a point to delete it.
   - **Ctrl+Z** undoes.

   Guesses show in yellow, your own points in blue, and points on a confirmed frame in green.
3. Press **Enter** (Confirm frame) when the frame is right, including a frame with no ants; this also moves to the next frame.
   - **Only confirmed frames are saved to the CSV.**
   - Leaving a frame with unconfirmed points asks whether to confirm it.
4. The tracker can't detect ants that stand still, so look for missed ants. Untick **Show guesses** to see the frame clean.

Output, saved after every change:
- `csv/<clip>.csv` with columns `Timestamp, frame, ant_x, ant_y` (no ant IDs)
- `progress/<clip>.json`, which **Open existing annotation** resumes; older files with IDs open fine

Copy the CSVs into your ground-truth folder for `--optimize`. Guessing needs the tracker's packages (numpy, scipy), which the conda environment already has.

## Ant Tracking

Counts ants in one video or many, from the GUI or the command line (CLI). Both give identical results.

### Tracking inputs

Mandatory:

| Input | CLI | GUI (Count ants tab) |
|---|---|---|
| **Videos** (`.mp4 .mov .avi .mkv .m4v`) | `--videos FOLDER`, **or** `--video FILE` (repeat for videos from different folders) | **Videos**: add files from any folders, or drag & drop. **Folder**: choose one folder. |
| **A circle per video** (hole center, radius, north) | `<name>_circle.json` next to the video or in the output folder, **or** one `--circle-file` for all | **Find `<name>_circle.json`** in the input/output folders, **or** **Click it manually** for each video |

**Making circle files:** in the GUI, open the video, click the hole center, then toward north, set the radius, and click **Save circle…**. Keep the suggested name.

In a terminal, a missing video path is asked for instead of causing an error. Before each run (CLI or GUI) you're asked for the **GitHub version** of the code, which is saved in the run's metadata. Press Enter to accept the version detected from git, or pass `--code-version`.

### Tracking options

| CLI | GUI | Default | What it does |
|---|---|---|---|
| `--parameters FILE` | Parameters → **Load…** | built-in | Detection settings, e.g. `annotations/ant_tuning_parameters.json` |
| `--output FOLDER` | Output → **Select output folder…** | `./outputs` (files), `<folder>/batch_results` (folder) | Where results go |
| `--recursive` | **Include subfolders** | off | Also process subfolders; results mirror the folder tree |
| `--strict` | **Strict** | off | Stop instead of skipping on a missing circle or a failed video |
| `--render-video` | **Render annotated MP4** | off (GUI: on for one video) | Write an annotated video (slow) |
| `--circle-file FILE` | — | per-video files | One circle for every video |
| `--code-version TEXT` | asked before each run | asked | GitHub version (commit, tag or release) saved in the metadata |
| `--no-prompt` | — | off | Error instead of asking for missing inputs |
| `--gui` | — | — | Open the GUI with these options pre-filled |

### Tracking in the GUI

```bash
python ant_tracker_modular.py
python ant_tracker_modular.py --gui --videos /path/to/videos --recursive    # pre-filled
```

1. **Choose videos:** click **Add videos…** / **Add a folder's videos…**, or drag them in; or switch to **Folder**.
2. **Check the table:** each video shows *ready*, *missing* or *invalid* (with the reason) for its circle.
3. **Fix missing circles:** double-click a row, then click the center and north.
   - In *Find* mode, also click **Save circle…** so the file is found.
   - In *Click it manually* mode, the circle is used straight away.
4. Optionally load parameters, pick an output folder, and tick **Strict** or **Render**.
5. Click **Process**. Videos without a circle are skipped, unless Strict is on. **Cancel** stops the run.

**Copy CLI command** copies the equivalent command, to run later without the GUI.

### Tracking from the CLI

```bash
# one video (GX010287_circle.json next to it)
python ant_tracker_modular.py --run --video /path/to/GX010287.MP4 --render-video

# a folder and all its subfolders, with tuned settings
python ant_tracker_modular.py --run --videos /path/to/videos --recursive \
    --parameters annotations/ant_tuning_parameters.json --output ./outputs/batch

# videos from different folders
python ant_tracker_modular.py --run --video /data/day1/A.MP4 --video /data/day2/B.MP4
```

Before processing, every circle is checked (`Found 42 video(s) | 40 with circles | 2 missing | 0 invalid`). Videos without a circle are skipped. With `--strict`, the run stops instead.

### Tracking outputs

| File | Contents |
|---|---|
| `<name>_counts.csv` | Each enter/exit event with time and quadrant |
| `<name>_summary.csv` | Enters and exits per quadrant |
| `<name>_ant_boxes.csv` | Each tracked ant's box per frame |
| `<name>_circle.json` | The circle used |
| `<name>_counted.mp4` | Annotated video (if rendering is on) |
| `batch_summary.json` | One per run: processed/skipped/failed videos with reasons, totals |
| `run_metadata.json` | One per run: date, code version, parameter file and exact values used, inputs |

The terminal ends with a list of skipped and failed videos.

Exit codes:
- `0`: done (skipped videos are fine)
- `1`: a video failed (the rest were processed)
- `2`: stopped early

## Hyperparameter Optimization

Tunes detection settings with Optuna against hand-labeled ant positions. CLI only.

| | Option | Notes |
|---|---|---|
| **Mandatory** | `--ground-truth FOLDER` + `--videos FOLDER` | One label CSV per video, same name, columns `timestamp,frame,ant_x,ant_y` (zero-based frames, original pixels; an old `ant_id` column is ignored) |
| | *or* `--optimize DATASET` | Folder containing `ground-truth/` and `videos/` |
| Optional | `--trials N` | Default 30 |
| | `--parameters FILE` | Starting settings |
| | `--match-distance PX` | Default 25: how close a detection must be to a label |
| | `--seed N` | Default 42 |
| | `--output FOLDER` | Default: dataset folder or `<videos>/optuna_results` |
| | `--code-version TEXT` | Saved in the metadata (default: detected from git) |

```bash
python ant_tracker_modular.py --optimize --ground-truth annotations/ground-truth --videos /path/to/videos \
    --trials 50 --output annotations/optuna_run2
```

Each run's results go in a new `optuna_<date-time>/` folder inside the output folder:
- `ant_tuning_parameters.json`: the best settings, ready for `--parameters`
- `optuna_results.json` and `optuna_trials.csv`: scores and every trial
- `run_metadata.json`: date, code version and base settings

The study database stays in the output folder itself, so running again continues the study; use a new output folder to start fresh. Optuna scores detection, not counts, so check the result with a regression run.

## Regression

Compares the tracker's enter/exit counts with hand counts and fits a straight line (slope, intercept, R², MAE, RMSE).

### Regression inputs

Mandatory:

| Input | CLI | GUI (Regression tab) |
|---|---|---|
| **Hand counts:** one CSV (`Video_Title, Colony_ID, Video_Date, NW_Enter, NW_Exit, ...`), or a folder of per-video CSVs | `--regression FILE_OR_FOLDER` | **Choose CSV…** / **Choose folder…**, or drop it |
| **Video folder:** matched to `Video_Title` by file name | `--videos FOLDER` | **Choose video folder…**, or drop it |
| **A circle per matched video:** `<name>_circle.json` next to the video, in the hand-counts folder, or in `<videos>/circles/` | (found automatically) | (found automatically) |

Missing circles stop the run. All of them are listed first.

### Regression options

| CLI | GUI | What it does |
|---|---|---|
| `--skip-missing-circles` | **Skip videos without a circle** | Skip those videos instead of stopping |
| `--parameters FILE` | Parameters → **Load…** | Settings to test |
| `--quadrant NE` | **Quadrant** | Score one quadrant only |
| `--circle-file FILE` | **Circle file for all…** | One circle for every video |
| `--output FOLDER` | **Select output folder…** | Default: `<videos>/regression_results`. Each run gets its own `regression_<date-time>/` folder inside it |
| `--code-version TEXT` | asked before each run | GitHub version saved in the metadata |
| `--render-video` | **Render annotated MP4** | Also write annotated videos |
| `--show-plot` | (always shown) | Open the plot window |

### Running regression

```bash
# CLI
python ant_tracker_modular.py --regression "regression_test/insample-outputs/in-sample-ground truth.csv" \
    --videos /path/to/videos --parameters annotations/ant_tuning_parameters.json --skip-missing-circles

# GUI, pre-filled (or open the GUI and go to the Regression tab)
python ant_tracker_modular.py --gui --regression "regression_test/insample-outputs/in-sample-ground truth.csv" \
    --videos /path/to/videos
```

In the GUI, check the table of matched videos and their circles, then click **Run regression**. The statistics, the plot and each video's errors appear in the tab.

### Regression outputs

Each run writes to a new `regression_<date-time>/` folder, so earlier results are never overwritten.

| File | Contents |
|---|---|
| `regression_summary.json` | Fit statistics, totals, skipped/unmatched videos, settings |
| `regression_comparison.csv` | Predicted vs hand counts per video and quadrant |
| `regression_results.csv` | One row per plotted point |
| `regression_plot.png` | Predicted vs hand counts, and squared error |
| `regression_run_metadata.json` | Date, code version, git details, parameter file and exact values used, videos and circles |

put photo of regression graph on here
