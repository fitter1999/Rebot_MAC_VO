#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline dense MAC-VO mapping from a recorded DECXIN stereo_sequence.")
    parser.add_argument("--result", required=True, help="Online result folder containing stereo_sequence/left and stereo_sequence/right.")
    parser.add_argument("--odom", default="Config/Experiment/MACVO/MACVO_DECXIN3261V_Mapping.yaml")
    parser.add_argument("--resultRoot", default="./Results_decxin3261v_offline_mapping")
    parser.add_argument("--seq-from", type=int, default=0)
    parser.add_argument("--seq-to", type=int, default=None)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--timing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = Path(args.result).resolve()
    seq = result / "stereo_sequence"
    if not (seq / "left").exists() or not (seq / "right").exists():
        raise FileNotFoundError(f"No stereo_sequence/left,right under {result}")

    data_cfg = result / "decxin_offline_sequence.yaml"
    data = {
        "type": "GeneralStereo",
        "name": f"DECXIN3261V-offline-{result.name}",
        "args": {
            "root": str(seq),
            "camera": {
                "fx": 456.8243713378906,
                "fy": 456.8243713378906,
                "cx": 339.35308837890625,
                "cy": 248.93231201171875,
            },
            "bl": 0.06,
            "format": "png",
        },
    }
    with data_cfg.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    cmd = [
        str(PROJECT_ROOT / "run_macvo_wjy.sh"),
        "python",
        "MACVO.py",
        "--odom",
        args.odom,
        "--data",
        str(data_cfg),
        "--resultRoot",
        args.resultRoot,
        "--noeval",
    ]
    if args.seq_from:
        cmd.extend(["--seq_from", str(args.seq_from)])
    if args.seq_to is not None:
        cmd.extend(["--seq_to", str(args.seq_to)])
    if args.preload:
        cmd.append("--preload")
    if args.timing:
        cmd.append("--timing")

    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=PROJECT_ROOT))


if __name__ == "__main__":
    main()
