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
    except:
        return None

import psutil

def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None

        for name in temps:
            if name in ("coretemp", "k10temp", "cpu_thermal"):
                entries = temps[name]
                if entries:
                    return max([entry.current for entry in entries if entry.current])

        for entries in temps.values():
            if entries:
                return max([entry.current for entry in entries if entry.current])

        return None
    except:
        return None

VIDEO_PATH = "/media/EBVrecordings/2024-TrafficMonitoring/data_collection/week_34/box_1/2024_08_28_16_57_02_recordings/video.avi"
OUTPUT_DIR = Path("week_34/box_1/2024_08_28_16_57_02_recordings/labels")
TIMESTAMP_PATH = "/media/EBVrecordings/2024-TrafficMonitoring/data_collection/week_34/box_1/2024_08_28_16_57_02_recordings/timestamps.csv"
FRAMES_PER_CLIP = 5400
CONF_THRESHOLD = 0.1

df_timestamps = pd.read_csv(TIMESTAMP_PATH)
SHIFT = df_timestamps.index[df_timestamps["event_time"] != 0][0]

model = YOLO("yolo26x.pt")

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print(f"GPU temp: {get_gpu_temp()}")
print(f"CPU temp: {get_cpu_temp()}")

cap = cv2.VideoCapture(VIDEO_PATH)
global_frame_idx = 0
clips_to_skip = 0
t = 0

while t < SHIFT:
    ret, frame = cap.read()
    t += 1

start_time = time.time()

while global_frame_idx < (FRAMES_PER_CLIP * clips_to_skip):
    ret, frame = cap.read()
    global_frame_idx += 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gpu_temp = get_gpu_temp()
    if gpu_temp and gpu_temp > 85:
        print(f"GPU too hot ({gpu_temp}°C). Stopping.")
        break

    cpu_temp = get_cpu_temp()
    if cpu_temp > 90:
        print(f"CPU too hot ({cpu_temp}°C). Stopping.")
        break

    clip_idx = global_frame_idx // FRAMES_PER_CLIP
    frame_in_clip = global_frame_idx % FRAMES_PER_CLIP

    clip_dir = OUTPUT_DIR / f"clip_{clip_idx:03d}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    if global_frame_idx % FRAMES_PER_CLIP == 0:
        print(f"Processing clip_{clip_idx:03d}")
        print(f"Time:{time.time()-start_time}")
        print(f"GPU temp: {get_gpu_temp()}")
        print(f"CPU temp: {get_cpu_temp()}")

    label_path = clip_dir / f"{frame_in_clip}.txt"
    results = model.track(
        frame,
        conf=CONF_THRESHOLD,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
        device=0
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
print("Done.")
