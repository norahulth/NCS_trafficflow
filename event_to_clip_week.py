from metavision_core.event_io import EventsIterator
import h5py
import numpy as np
import pandas as pd
import os
from pathlib import Path

week = 37

WEEK_INPUT_DIR = Path(f"//172.16.222.2/EBVrecordings/2024-TrafficMonitoring/data_collection/week_{week}")
WEEK_OUTPUT_DIR = Path(f"//172.16.222.2/tmp/Nora/traffic_flow_data/week_{week}")

# Set to None to process all boxes, or e.g. ["box_1"] / ["box_1", "box_2"]
BOXES_TO_PROCESS = ["box3"]

CLIP_LEN_FRAMES = 5400      # 1 minute at nominal 90 fps
CLIP_STEP_FRAMES = CLIP_LEN_FRAMES

# Event dtype
event_dtype = np.dtype([
    ("x", np.int16),
    ("y", np.int16),
    ("p", np.int8),
    ("t", np.int64),
])

def find_recordings_for_week(week_input_dir, boxes_to_process=None):
    """
    Finds all recording folders in a week folder that contain both:
      - events.raw
      - timestamps.csv
    """
    week_input_dir = Path(week_input_dir)

    if boxes_to_process is None:
        search_roots = sorted([p for p in week_input_dir.iterdir() if p.is_dir() and p.name.startswith("box")])
    else:
        search_roots = [week_input_dir / box for box in boxes_to_process]

    recording_dirs = []
    for root in search_roots:
        if not root.exists():
            print(f"Skipping missing box folder: {root}")
            continue

        for events_path in sorted(root.rglob("events.raw")):
            recording_dir = events_path.parent
            timestamps_path = recording_dir / "timestamps.csv"

            if timestamps_path.exists():
                recording_dirs.append(recording_dir)
            else:
                print(f"Skipping {recording_dir}: missing timestamps.csv")

    return recording_dirs


def build_clip_ranges(timestamp_path, clip_len_frames=5400, clip_step_frames=None):
    if clip_step_frames is None:
        clip_step_frames = clip_len_frames

    df_timestamps = pd.read_csv(timestamp_path)

    shift = df_timestamps.index[df_timestamps["event_time"] != 0][0]

    time_windows1 = df_timestamps.iloc[shift:, 0].to_numpy().astype(float)
    time_windows = time_windows1 - time_windows1[0]

    clip_ranges = []
    clip_idx = 0

    while True:
        start_frame = clip_idx * clip_step_frames
        end_frame = start_frame + clip_len_frames

        if end_frame >= len(time_windows):
            break

        clip_ranges.append({
            "label": clip_idx,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_ts": time_windows[start_frame],
            "end_ts": time_windows[end_frame],
        })

        clip_idx += 1

    return clip_ranges


def write_clip(out_path, events_buffer, current_range, event_dtype):
    with h5py.File(out_path, "w") as f:
        f.create_dataset("events", data=np.array(events_buffer, dtype=event_dtype))
        f.create_dataset("start_frame", data=current_range["start_frame"])
        f.create_dataset("end_frame", data=current_range["end_frame"])
        f.create_dataset("start_ts", data=current_range["start_ts"])
        f.create_dataset("end_ts", data=current_range["end_ts"])


def process_recording(recording_dir, week_input_dir, week_output_dir):
    recording_dir = Path(recording_dir)
    week_input_dir = Path(week_input_dir)
    week_output_dir = Path(week_output_dir)

    input_raw = recording_dir / "events.raw"
    timestamp_path = recording_dir / "timestamps.csv"

    # Mirror the input folder structure under WEEK_OUTPUT_DIR
    relative_recording_dir = recording_dir.relative_to(week_input_dir)
    output_dir = week_output_dir / relative_recording_dir / "event_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / "event_clip"

    print(f"\nProcessing: {recording_dir}")
    print(f"Output:     {output_dir}")

    clip_ranges = build_clip_ranges(
        timestamp_path,
        clip_len_frames=CLIP_LEN_FRAMES,
        clip_step_frames=CLIP_STEP_FRAMES,
    )

    print(f"Prepared {len(clip_ranges)} clip windows.")

    if len(clip_ranges) == 0:
        print(f"Skipping {recording_dir}: no valid clip ranges.")
        return 0

    events_iterator = EventsIterator(str(input_raw))

    current_clip_idx = 0
    events_buffer = []
    current_range = clip_ranges[current_clip_idx]
    clips_written = 0

    for evs in events_iterator:
        for ev in evs:
            t = int(ev["t"])

            # Skip events before the current clip starts
            if t < current_range["start_ts"]:
                continue

            # Passed the end of the current clip, write it and move on
            while t >= current_range["end_ts"]:
                out_path = f"{output_prefix}_{current_range['label']:03d}.h5"
                write_clip(out_path, events_buffer, current_range, event_dtype)
                clips_written += 1
                print("Wrote:", out_path)

                events_buffer = []
                current_clip_idx += 1

                if current_clip_idx >= len(clip_ranges):
                    break

                current_range = clip_ranges[current_clip_idx]

                # If this event belongs before the next clip end, stop advancing
                if t < current_range["end_ts"]:
                    break

            if current_clip_idx >= len(clip_ranges):
                break

            # Keep event only if it falls inside current clip window
            if current_range["start_ts"] <= t < current_range["end_ts"]:
                events_buffer.append((
                    int(ev["x"]),
                    int(ev["y"]),
                    int(ev["p"]),
                    t
                ))

        if current_clip_idx >= len(clip_ranges):
            break

    # Flush final clip if it has collected events
    if current_clip_idx < len(clip_ranges) and len(events_buffer) > 0:
        current_range = clip_ranges[current_clip_idx]
        out_path = f"{output_prefix}_{current_range['label']:03d}.h5"
        write_clip(out_path, events_buffer, current_range, event_dtype)
        clips_written += 1
        print("Wrote:", out_path)

    print(f"Finished {recording_dir.name}: wrote {clips_written} clips.")
    return clips_written


recording_dirs = find_recordings_for_week(WEEK_INPUT_DIR, BOXES_TO_PROCESS)
print(f"Found {len(recording_dirs)} recordings to process.")

total_clips = 0
failed_recordings = []

for recording_dir in recording_dirs:
    try:
        total_clips += process_recording(recording_dir, WEEK_INPUT_DIR, WEEK_OUTPUT_DIR)
    except Exception as e:
        print(f"FAILED {recording_dir}: {e}")
        failed_recordings.append((recording_dir, str(e)))

print("\nDone.")
print(f"Total clips written: {total_clips}")
print(f"Failed recordings: {len(failed_recordings)}")

if failed_recordings:
    for recording_dir, error in failed_recordings:
        print(f"- {recording_dir}: {error}")