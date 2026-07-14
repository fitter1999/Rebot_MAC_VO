#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose recorded DECXIN stereo sequence geometry without MAC-VO.")
    parser.add_argument("--result", required=True, help="Folder containing stereo_sequence/, or stereo_sequence itself.")
    parser.add_argument("--every", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all sampled frames.")
    parser.add_argument("--fx", type=float, default=456.8243713378906)
    parser.add_argument("--baseline", type=float, default=0.06)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def sequence_dir(path: Path) -> Path:
    if (path / "left").exists() and (path / "right").exists():
        return path
    seq = path / "stereo_sequence"
    if (seq / "left").exists() and (seq / "right").exists():
        return seq
    raise FileNotFoundError(f"No stereo_sequence/left,right under {path}")


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def feature_stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    orb = cv2.ORB_create(2500)
    kp_l, des_l = orb.detectAndCompute(left, None)
    kp_r, des_r = orb.detectAndCompute(right, None)
    if des_l is None or des_r is None:
        return {
            "matches": 0.0,
            "disp_median": float("nan"),
            "disp_pos_ratio": float("nan"),
            "dy_median": float("nan"),
            "dy_mad": float("nan"),
        }
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des_l, des_r), key=lambda m: m.distance)[:600]
    if not matches:
        return {
            "matches": 0.0,
            "disp_median": float("nan"),
            "disp_pos_ratio": float("nan"),
            "dy_median": float("nan"),
            "dy_mad": float("nan"),
        }
    disp = []
    dy = []
    for match in matches:
        pl = kp_l[match.queryIdx].pt
        pr = kp_r[match.trainIdx].pt
        disp.append(pl[0] - pr[0])
        dy.append(pl[1] - pr[1])
    disp_a = np.asarray(disp, dtype=np.float32)
    dy_a = np.asarray(dy, dtype=np.float32)
    dy_med = float(np.median(dy_a))
    dy_mad = float(np.median(np.abs(dy_a - dy_med)))
    epi_mask = np.abs(dy_a - dy_med) < max(2.0, 2.5 * dy_mad)
    good_disp = disp_a[epi_mask]
    return {
        "matches": float(len(matches)),
        "epi_matches": float(good_disp.size),
        "disp_median": float(np.median(good_disp)) if good_disp.size else float("nan"),
        "disp_pos_ratio": float(np.mean(good_disp > 0)) if good_disp.size else float("nan"),
        "dy_median": dy_med,
        "dy_mad": dy_mad,
    }


def sgbm_stats(left: np.ndarray, right: np.ndarray, fx: float, baseline: float) -> dict[str, float]:
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 3 * 5 * 5,
        P2=32 * 3 * 5 * 5,
        disp12MaxDiff=2,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = matcher.compute(left, right).astype(np.float32) / 16.0
    valid = np.isfinite(disp) & (disp > 0.5)
    if not np.any(valid):
        return {
            "sgbm_valid_ratio": 0.0,
            "sgbm_disp_median": float("nan"),
            "sgbm_depth_median": float("nan"),
            "sgbm_depth_p90": float("nan"),
        }
    d = disp[valid]
    depth = (fx * baseline) / d
    depth = depth[np.isfinite(depth)]
    return {
        "sgbm_valid_ratio": float(valid.mean()),
        "sgbm_disp_median": float(np.median(d)),
        "sgbm_depth_median": float(np.median(depth)) if depth.size else float("nan"),
        "sgbm_depth_p90": float(np.percentile(depth, 90)) if depth.size else float("nan"),
    }


def main() -> None:
    args = parse_args()
    seq = sequence_dir(Path(args.result).expanduser())
    left_files = sorted((seq / "left").glob("*.png"))
    right_files = sorted((seq / "right").glob("*.png"))
    if not left_files or len(left_files) != len(right_files):
        raise FileNotFoundError(f"Invalid stereo sequence: {seq}")

    indices = list(range(0, len(left_files), max(args.every, 1)))
    if args.max_frames > 0:
        indices = indices[: args.max_frames]

    rows: list[dict[str, float | int]] = []
    for idx in indices:
        left = read_gray(left_files[idx])
        right = read_gray(right_files[idx])
        row: dict[str, float | int] = {"frame": idx}
        row.update(feature_stats(left, right))
        row.update(sgbm_stats(left, right, args.fx, args.baseline))
        rows.append(row)

    keys = list(rows[0].keys()) if rows else []
    out = Path(args.out) if args.out else seq / "diagnostic_report.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"sequence: {seq}")
    print(f"frames: {len(left_files)}, sampled: {len(rows)}, report: {out}")
    for key in (
        "disp_median",
        "disp_pos_ratio",
        "dy_mad",
        "sgbm_valid_ratio",
        "sgbm_depth_median",
        "sgbm_depth_p90",
    ):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            print(f"{key}: median={np.median(values):.4f}, p10={np.percentile(values, 10):.4f}, p90={np.percentile(values, 90):.4f}")


if __name__ == "__main__":
    main()
