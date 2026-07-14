#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def find_latest_result(root: Path) -> Path:
    candidates = sorted(root.glob("MACVO-OV2710-Realtime@OV2710-live/*"), key=lambda p: p.stat().st_mtime)
    candidates = [p for p in candidates if (p / "poses.npy").exists()]
    if not candidates:
        raise FileNotFoundError(f"No result with poses.npy under {root}")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a saved OV2710 MAC-VO trajectory.")
    parser.add_argument("--result", default=None, help="Result folder containing poses.npy. Defaults to latest under --root.")
    parser.add_argument("--root", default="Results_ov2710_live")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result) if args.result else find_latest_result(Path(args.root))
    poses = np.load(result_dir / "poses.npy")
    xyz = poses[:, 1:4]

    output = Path(args.output) if args.output else result_dir / "trajectory_xy.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(xyz[:, 0], xyz[:, 1], "-o", markersize=2, linewidth=1.4)
    ax.scatter([xyz[0, 0]], [xyz[0, 1]], c="green", s=60, label="start")
    ax.scatter([xyz[-1, 0]], [xyz[-1, 1]], c="red", s=60, label="end")
    ax.set_title(f"{result_dir.name} ({len(xyz)} poses)")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.axis("equal")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(f"result: {result_dir}")
    print(f"poses: {poses.shape}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
