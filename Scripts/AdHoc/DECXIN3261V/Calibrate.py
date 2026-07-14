#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.AdHoc.Calibrate_OV2710_Stereo import main


def add_default_arg(name: str, value: str) -> None:
    if not any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:]):
        sys.argv.extend([name, value])


def get_arg_value(name: str, default: str) -> str:
    for idx, arg in enumerate(sys.argv[1:], start=1):
        if arg == name and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return default


def read_capture_value(info_file: Path, key: str) -> int | None:
    if not info_file.exists():
        return None
    for line in info_file.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            try:
                return int(line.split("=", 1)[1])
            except ValueError:
                return None
    return None


def add_swap_for_legacy_capture() -> None:
    if "--swap" in sys.argv[1:]:
        return
    input_root = Path(get_arg_value("--input", "Calibration/decxin3261v_screen_640x480"))
    left_x = None
    right_x = None
    for info_file in (input_root / "capture_info.txt", input_root / "capture_info_source.txt"):
        left_x = read_capture_value(info_file, "left_x")
        right_x = read_capture_value(info_file, "right_x")
        if left_x is not None and right_x is not None:
            break
    if left_x is not None and right_x is not None and left_x < right_x:
        sys.argv.append("--swap")


if __name__ == "__main__":
    add_default_arg("--input", "Calibration/decxin3261v_screen_640x480")
    add_default_arg("--known-baseline-m", "0.06")
    add_default_arg("--square-mm", "28")
    add_swap_for_legacy_capture()
    main()
