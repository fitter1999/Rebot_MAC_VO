#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2


def convert_image(src: Path, dst: Path, crop_width: int, target_width: int, target_height: int) -> None:
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(src)
    h, w = image.shape[:2]
    if crop_width > w:
        raise ValueError(f"crop width {crop_width} exceeds image width {w}: {src}")
    x0 = (w - crop_width) // 2
    crop = image[:, x0 : x0 + crop_width]
    resized = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), resized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 640x480 center-cropped DECXIN calibration images from 640x400 captures.")
    parser.add_argument("--input", default="Calibration/decxin3261v_screen")
    parser.add_argument("--output", default="Calibration/decxin3261v_screen_640x480")
    parser.add_argument("--crop-width", type=int, default=533)
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--target-height", type=int, default=480)
    args = parser.parse_args()

    src_root = Path(args.input)
    dst_root = Path(args.output)
    for side in ("left", "right"):
        for src in sorted((src_root / side).glob("*.png")):
            convert_image(src, dst_root / side / src.name, args.crop_width, args.target_width, args.target_height)

    info_src = src_root / "capture_info.txt"
    if info_src.exists():
        shutil.copy2(info_src, dst_root / "capture_info_source.txt")
    (dst_root / "capture_info.txt").write_text(
        "\n".join(
            [
                f"source={src_root}",
                f"crop_width={args.crop_width}",
                f"target_width={args.target_width}",
                f"target_height={args.target_height}",
                "square_mm=28.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote 640x480 calibration set: {dst_root}")


if __name__ == "__main__":
    main()
