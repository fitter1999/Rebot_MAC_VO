#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.AdHoc.DECXIN3261V.Device import DECXIN_DEVICE_BY_ID, find_decxin_device


def write_sample_crops(output: Path) -> None:
    sample = PROJECT_ROOT / "DECXIN-3261V-message/时间戳IMU解码(2)/时间戳IMU解码/TimeStamp_Data_Decode_DemoCode_V010002/TST_3D_Cam_Image_Decode/4000x1200_0_0_10.bmp"
    image = cv2.imread(str(sample), cv2.IMREAD_COLOR)
    if image is None:
        print(f"sample image not found: {sample}")
        return
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "sample_preview_1000x300.png"), cv2.resize(image, (1000, 300), interpolation=cv2.INTER_AREA))
    crops = {
        "sample_default_left_160_2080.png": image[:, 160:2080],
        "sample_default_right_2080_4000.png": image[:, 2080:4000],
        "sample_left_0_1920.png": image[:, 0:1920],
        "sample_right_1920_3840.png": image[:, 1920:3840],
        "sample_right_2080_4000.png": image[:, 2080:4000],
        "sample_middle_1900_2100.png": image[:, 1900:2100],
        "sample_right_edge_3840_4000.png": image[:, 3840:4000],
    }
    for name, crop in crops.items():
        cv2.imwrite(str(output / name), crop)
    print(f"wrote sample crops to {output}")


def run_command(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout)
    except FileNotFoundError:
        print(f"missing command: {cmd[0]}")


def try_opencv(device: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"opencv opened={cap.isOpened()} device={device}")
    ok, frame = cap.read()
    print(f"opencv read={ok} shape={None if frame is None else frame.shape}")
    if ok and frame is not None:
        cv2.imwrite(str(output / "opencv_raw_4000x1200.png"), frame)
        cv2.imwrite(str(output / "opencv_preview_1000x300.png"), cv2.resize(frame, (1000, 300), interpolation=cv2.INTER_AREA))
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe DECXIN-3261V camera and sample stereo split.")
    parser.add_argument("--device", default=DECXIN_DEVICE_BY_ID)
    parser.add_argument("--output", default="DECXIN-3261V-message/probe")
    parser.add_argument("--try-opencv", action="store_true")
    args = parser.parse_args()

    output = PROJECT_ROOT / args.output
    args.device = find_decxin_device(args.device)
    write_sample_crops(output)
    run_command(["lsusb"])
    run_command(["bash", "-lc", "ls -l /dev/video* /dev/v4l/by-id /dev/v4l/by-path 2>/dev/null"])
    run_command(["gst-device-monitor-1.0", "Video/Source"])
    if args.try_opencv:
        try_opencv(args.device, output)


if __name__ == "__main__":
    main()
