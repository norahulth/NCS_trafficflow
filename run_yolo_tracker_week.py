from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
import pandas as pd
import time
import subprocess
import psutil


def get_gpu_temp():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
        )
        return int(out.decode().strip())
    except Exception:
        return None


def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None

        for name in temps:
            if name in ("coretemp", "k10temp", "cpu_thermal"):
                entries = temps[name]
                vals = [entry.current for entry in entries if entry.current is not None]
                if vals:
                    return max(vals)

        for entries in temps.values():
            vals = [entry.current for entry in entries if entry.current is not None]
            if vals:
                return max(vals)

        return None
    except Exception:
        return None


DATA_ROOT = Path("/media/EBVrecordings/2024-TrafficMonitoring/data_collection")
WEEK_NAME = "week_33"
LOCAL_OUTPUT_ROOT = Path("./labels_output")

FRAMES_PER_CLIP = 5400
CONF_THRESHOLD = 0.1
MODEL_PATH = "yolo26x.pt"
TRACKER_CFG = "bytetrack.yaml"
DEVICE = 0

# Skip clips here
CLIPS_TO_SKIP = 0


def find_recording_dirs(week_dir: Path):
    """
    Find all folders under the week directory that contain both:
      - video.avi
      - timestamps.csv
    """
    recording_dirs = []

    for video_path in week_dir.rglob("video.avi"):
        recording_dir = video_path.parent
        timestamp_path = recording_dir / "timestamps.csv"
        if timestamp_path.exists():
            recording_dirs.append(recording_dir)

    return sorted(recording_dirs)


def process_recording(recording_dir: Path, week_dir: Path, output_root: Path, model: YOLO):
    video_path = recording_dir / "video.avi"
    timestamp_path = recording_dir / "timestamps.csv"

    # Mirror folder structure locally
    relative_recording_path = recording_dir.relative_to(DATA_ROOT)
    output_dir = output_root / relative_recording_path / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Processing: {recording_dir} ===")
    print(f"Output labels: {output_dir}")

    df_timestamps = pd.read_csv(timestamp_path)

    non_zero_indices = df_timestamps.index[df_timestamps["event_time"] != 0]
    if len(non_zero_indices) == 0:
        print(f"Skipping {recording_dir}: no non-zero event_time found.")
        return

    shift = non_zero_indices[0]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print(f"GPU temp: {get_gpu_temp()}")
    print(f"CPU temp: {get_cpu_temp()}")

    global_frame_idx = 0
    t = 0

    # Skip frames until SHIFT
    while t < shift:
        ret, _ = cap.read()
        if not ret:
            print(f"Video ended before SHIFT in {recording_dir}")
            cap.release()
            return
        t += 1

    # Skip full clips if requested
    frames_to_skip = FRAMES_PER_CLIP * CLIPS_TO_SKIP
    while global_frame_idx < frames_to_skip:
        ret, _ = cap.read()
        if not ret:
            print(f"Video ended before clip skipping completed in {recording_dir}")
            cap.release()
            return
        global_frame_idx += 1

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gpu_temp = get_gpu_temp()
        if gpu_temp is not None and gpu_temp > 85:
            print(f"GPU too hot ({gpu_temp}°C). Stopping {recording_dir}.")
            break

        cpu_temp = get_cpu_temp()
        if cpu_temp is not None and cpu_temp > 90:
            print(f"CPU too hot ({cpu_temp}°C). Stopping {recording_dir}.")
            break

        clip_idx = global_frame_idx // FRAMES_PER_CLIP
        frame_in_clip = global_frame_idx % FRAMES_PER_CLIP

        clip_dir = output_dir / f"clip_{clip_idx:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)

        if global_frame_idx % FRAMES_PER_CLIP == 0:
            print(f"Processing clip_{clip_idx:03d}")
            print(f"Elapsed: {time.time() - start_time:.2f}s")
            print(f"GPU temp: {get_gpu_temp()}")
            print(f"CPU temp: {get_cpu_temp()}")

        label_path = clip_dir / f"{frame_in_clip}.txt"

        results = model.track(
            frame,
            conf=CONF_THRESHOLD,
            tracker=TRACKER_CFG,
            persist=True,
            verbose=False,
            device=DEVICE
        )[0]

        with open(label_path, "w") as f:
            for box in results.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                x_center, y_center, width, height = box.xywhn[0].tolist()
                track_id = int(box.id.item()) if box.id is not None else -1

                f.write(
                    f"{cls} "
                    f"{x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f} "
                    f"{conf:.6f} "
                    f"{track_id}\n"
                )

        global_frame_idx += 1

    cap.release()
    print(f"Done with {recording_dir}")


def main():
    week_dir = DATA_ROOT / WEEK_NAME

    if not week_dir.exists():
        raise FileNotFoundError(f"Week folder not found: {week_dir}")

    recording_dirs = find_recording_dirs(week_dir)

    if not recording_dirs:
        print(f"No recordings found under {week_dir}")
        return

    print(f"Found {len(recording_dirs)} recording(s) in {week_dir}")

    model = YOLO(MODEL_PATH)

    for recording_dir in recording_dirs:
        try:
            process_recording(recording_dir, week_dir, LOCAL_OUTPUT_ROOT, model)
        except Exception as e:
            print(f"Error while processing {recording_dir}: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main()