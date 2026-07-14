#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.AdHoc.Run_OV2710_Realtime import StereoCameraReader


@dataclass
class Detection:
    found: bool
    corners: np.ndarray | None
    feature: np.ndarray | None


def detect_checkerboard(gray: np.ndarray, pattern_size: tuple[int, int]) -> Detection:
    flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags_sb)

    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
            corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)

    if not found or corners is None:
        return Detection(False, None, None)

    pts = corners.reshape(-1, 2)
    h, w = gray.shape[:2]
    center = pts.mean(axis=0) / np.array([w, h], dtype=np.float32)
    span = (pts.max(axis=0) - pts.min(axis=0)) / np.array([w, h], dtype=np.float32)
    first_row = pts[: pattern_size[0]]
    row_vec = first_row[-1] - first_row[0]
    angle = np.arctan2(row_vec[1], row_vec[0]) / np.pi
    feature = np.array([center[0], center[1], span[0], span[1], angle], dtype=np.float32)
    return Detection(True, corners, feature)


def pair_feature(left: Detection, right: Detection) -> np.ndarray:
    assert left.feature is not None and right.feature is not None
    return np.concatenate([left.feature, right.feature], axis=0)


def is_new_pose(feature: np.ndarray, saved_features: list[np.ndarray], min_delta: float) -> bool:
    if not saved_features:
        return True
    return min(float(np.linalg.norm(feature - prev)) for prev in saved_features) >= min_delta


def draw_debug(rgb: np.ndarray, pattern_size: tuple[int, int], detection: Detection) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if detection.found and detection.corners is not None:
        cv2.drawChessboardCorners(bgr, pattern_size, detection.corners, True)
    return bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-capture OV2710 stereo checkerboard pairs.")
    parser.add_argument("--left", default="/dev/video4")
    parser.add_argument("--right", default="/dev/video6")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--inner-cols", type=int, default=9)
    parser.add_argument("--inner-rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=28.0)
    parser.add_argument("--output", default="Calibration/ov2710_screen")
    parser.add_argument("--max-pairs", type=int, default=35)
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--min-save-interval", type=float, default=0.8)
    parser.add_argument("--min-pose-delta", type=float, default=0.055)
    parser.add_argument("--max-skew-ms", type=float, default=8.0)
    parser.add_argument("--save-debug", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern_size = (args.inner_cols, args.inner_rows)
    out_root = Path(args.output)
    left_dir = out_root / "left"
    right_dir = out_root / "right"
    debug_dir = out_root / "debug"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    if args.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    meta = out_root / "capture_info.txt"
    meta.write_text(
        "\n".join(
            [
                f"left={args.left}",
                f"right={args.right}",
                f"width={args.width}",
                f"height={args.height}",
                f"fourcc={args.fourcc}",
                f"inner_cols={args.inner_cols}",
                f"inner_rows={args.inner_rows}",
                f"square_mm={args.square_mm}",
                f"max_skew_ms={args.max_skew_ms}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    reader = StereoCameraReader(args.left, args.right, args.width, args.height, args.camera_fps, args.fourcc)
    reader.start()

    saved_features: list[np.ndarray] = []
    saved_count = 0
    frame_count = 0
    last_pair_id = 0
    last_save_t = 0.0
    start_t = time.monotonic()
    last_status_t = 0.0

    print("\n采集已开始。现在切到浏览器棋盘格页面并全屏。")
    print("移动/倾斜双目相机，让棋盘格出现在画面不同位置；脚本会自动保存新姿态。")
    print("终止：Ctrl-C；完成后会自动退出。\n")

    try:
        while saved_count < args.max_pairs and (time.monotonic() - start_t) < args.timeout_sec:
            pair = reader.wait_for_newer(last_pair_id, timeout=2.0)
            if pair is None:
                continue
            last_pair_id = pair.pair_id
            frame_count += 1

            if pair.software_skew_ms > args.max_skew_ms:
                now = time.monotonic()
                if now - last_status_t > 1.0:
                    print(
                        f"\r保存 {saved_count:02d}/{args.max_pairs} | 跳过 skew={pair.software_skew_ms:.1f}ms | 帧 {frame_count}",
                        end="",
                        flush=True,
                    )
                    last_status_t = now
                continue

            left_rgb = pair.left_rgb
            right_rgb = pair.right_rgb
            left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)

            left_det = detect_checkerboard(left_gray, pattern_size)
            right_det = detect_checkerboard(right_gray, pattern_size)
            now = time.monotonic()

            if now - last_status_t > 1.0:
                status = "FOUND" if (left_det.found and right_det.found) else f"L={left_det.found} R={right_det.found}"
                print(
                    f"\r保存 {saved_count:02d}/{args.max_pairs} | 检测 {status} | skew={pair.software_skew_ms:.1f}ms | 帧 {frame_count}",
                    end="",
                    flush=True,
                )
                last_status_t = now

            if not (left_det.found and right_det.found):
                continue

            feature = pair_feature(left_det, right_det)
            if (now - last_save_t) < args.min_save_interval:
                continue
            if not is_new_pose(feature, saved_features, args.min_pose_delta):
                continue

            stem = f"{saved_count:04d}"
            cv2.imwrite(str(left_dir / f"{stem}.png"), cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(right_dir / f"{stem}.png"), cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))
            if args.save_debug:
                left_dbg = draw_debug(left_rgb, pattern_size, left_det)
                right_dbg = draw_debug(right_rgb, pattern_size, right_det)
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
