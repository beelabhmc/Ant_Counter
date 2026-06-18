# Ant Counter

These scripts are used to count ant activity through a quadrant. The script works by using the frame differencing computer vision techinique to detect act movement and uses kalman filter + hungarian assignment to track ants between frames

## Set up

If you are running the pipeline for the first time, please install conda [this page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) 

After you install conda, install the packages in the requirements.txt using the command conda create --name <env> --file requirements.txt python=3.9

## Running the scripts

### Single Video

```
python ant_counter_circle_shadow.py
```
1. Click a video shortcut or Browse… to load a video.
2. Scrub the frame slider to a frame with a clear view of the entrance.
3. Left-click to place the circle center, left-click again to set the north direction.
4. Adjust the radius spinner if needed.
5. Click Process Video. Outputs appear in <video_folder>/outputs/

### Batch Mode

```
python ant_counter_circle_shadow_batch.py
```

1. Add videos with Add files… or Add folder….
2. Select a video in the queue, draw its circle, and click Assign to this video. Repeat for each video that needs a different circle. Alternatively, draw one shared circle — it applies to any video without an individual assignment.
3. Click Process All.

The queue list shows [circle assigned], [uses shared], or [NO CIRCLE] next to each video.

## Outputs

| File | Contents |
|------|---------|
| `<stem>_counts.csv` | One row per crossing event with timestamp, quadrant, and running enter/exit counts |
| `<stem>_summary.csv` | Total enters and exits per quadrant plus an overall total |
| `<stem>_counted.mp4` | Annotated video with circle overlay, blob markers, and live counts |
| `<stem>_circle.json` | Saved circle parameters (center, radius, north angle) — auto-loaded in future sessions |
