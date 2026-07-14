#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.AdHoc.Capture_OV2710_Calibration import detect_checkerboard, draw_debug, is_new_pose, pair_feature
from Scripts.AdHoc.DECXIN3261V.Device import DECXIN_DEVICE_BY_ID, find_decxin_device
from Scripts.AdHoc.DECXIN3261V.Run_Realtime import SplitStereoCameraReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-capture DECXIN-3261V split stereo checkerboard pairs.")
    parser.add_argument("--device", default=DECXIN_DEVICE_BY_ID)
    parser.add_argument("--raw-width", type=int, default=4000)
    parser.add_argument("--raw-height", type=int, default=1200)
    parser.add_argument("--eye-width", type=int, default=1920)
    parser.add_argument("--eye-height", type=int, default=1200)
    parser.add_argument("--left-x", type=int, default=2080)
    parser.add_argument("--right-x", type=int, default=160)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--inner-cols", type=int, default=9)
    parser.add_argument("--inner-rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=28.0)
    parser.add_argument("--output", default="Calibration/decxin3261v_screen_640x480")
    parser.add_argument("--max-pairs", type=int, default=35)
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--min-save-interval", type=float, default=0.8)
    parser.add_argument("--min-pose-delta", type=float, default=0.055)
    parser.add_argument("--save-debug", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = find_decxin_device(args.device)
    pattern_size = (args.inner_cols, args.inner_rows)
    out_root = Path(args.output)
    left_dir = out_root / "left"
    right_dir = out_root / "right"
    debug_dir = out_root / "debug"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    if args.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    (out_root / "capture_info.txt").write_text(
        "\n".join(
            [
                f"device={args.device}",
                f"raw_width={args.raw_width}",
                f"raw_height={args.raw_height}",
                f"eye_width={args.eye_width}",
                f"eye_height={args.eye_height}",
                f"left_x={args.left_x}",
                f"right_x={args.right_x}",
                f"width={args.width}",
                f"height={args.height}",
                f"fourcc={args.fourcc}",
                f"inner_cols={args.inner_cols}",
                f"inner_rows={args.inner_rows}",
                f"square_mm={args.square_mm}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    reader = SplitStereoCameraReader(
        args.device,
        args.raw_width,
        args.raw_height,
        args.camera_fps,
        args.fourcc,
        args.left_x,
        args.right_x,
        args.eye_width,
        args.eye_height,
        args.width,
        args.height,
    )
    reader.start()

    saved_features: list[np.ndarray] = []
    saved_count = 0
    frame_count = 0
    last_pair_id = 0
    last_save_t = 0.0
    start_t = time.monotonic()
    last_status_t = 0.0

    print("\nDECXIN 标定采集已开始。把屏幕棋盘格全屏，移动/倾斜相机，脚本会自动保存新姿态。")
    print("终止：Ctrl-C；完成后会自动退出。\n")

    try:
        while saved_count < args.max_pairs and (time.monotonic() - start_t) < args.timeout_sec:
            pair = reader.wait_for_newer(last_pair_id, timeout=2.0)
            if pair is None:
                continue
            last_pair_id = pair.pair_id
            frame_count += 1

            left_gray = cv2.cvtColor(pair.left_rgb, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(pair.right_rgb, cv2.COLOR_RGB2GRAY)
            left_det = detect_checkerboard(left_gray, pattern_size)
            right_det = detect_checkerboard(right_gray, pattern_size)
            now = time.monotonic()

            if now - last_status_t > 1.0:
                status = "FOUND" if (left_det.found and right_det.found) else f"L={left_det.found} R={right_det.found}"
                print(f"\r保存 {saved_count:02d}/{args.max_pairs} | 检测 {status} | 帧 {frame_count}", end="", flush=True)
                last_status_t = now

            if not (left_det.found and right_det.found):
                continue

            feature = pair_feature(left_det, right_det)
            if (now - last_save_t) < args.min_save_interval:
                continue
            if not is_new_pose(feature, saved_features, args.min_pose_delta):
                continue

            stem = f"{saved_count:04d}"
            cv2.imwrite(str(left_dir / f"{stem}.png"), cv2.cvtColor(pair.left_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(right_dir / f"{stem}.png"), cv2.cvtColor(pair.right_rgb, cv2.COLOR_RGB2BGR))
            if args.save_debug:
                left_dbg = draw_debug(pair.left_rgb, pattern_size, left_det)
                right_dbg = draw_debug(pair.right_rgb, pattern_size, right_det)
                merged = np.concatenate([left_dbg, right_dbg], axis=1)
                for y in range(0, merged.shape[0], 40):
                    cv2.line(merged, (0, y), (merged.shape[1], y), (0, 255, 255), 1)
                cv2.imwrite(str(debug_dir / f"{stem}.png"), merged)

            saved_features.append(feature)
            saved_count += 1
            last_save_t = now
            print(f"\n\a已保存第 {saved_count:02d} 对：{stem}.png")
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C，停止采集。")
    finally:
        reader.stop()

    print(f"\n完成：保存 {saved_count} 对到 {out_root}")
    if saved_count < 15:
        print("提示：少于 15 对通常不够稳定，建议至少 25-35 对。")


if __name__ == "__main__":
    main()
