#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a saved MAC-VO result as sparse map reconstruction files.")
    parser.add_argument("--result", required=True, help="Result folder containing tensor_map.npz and poses.npy.")
    parser.add_argument("--output-dir", default=None, help="Defaults to RESULT/reconstruction.")
    parser.add_argument("--max-distance", type=float, default=8.0, help="Drop map points farther than this many meters from origin.")
    parser.add_argument("--cov-det-percentile", type=float, default=99.0, help="Drop points above this covariance determinant percentile.")
    parser.add_argument("--max-cov-det", type=float, default=None, help="Absolute covariance determinant cutoff; overrides percentile.")
    parser.add_argument("--keep-macvo-coordinates", action="store_true", help="Keep MAC-VO NED/FRD coordinates in PLY files instead of exporting z-up viewer coordinates.")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def write_points_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb is None:
        rgb = np.full((xyz.shape[0], 3), 220, dtype=np.uint8)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb, strict=True):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_trajectory_ply(path: Path, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = xyz.shape[0]
    edges = max(0, n - 1)
    colors = np.full((n, 3), [40, 100, 240], dtype=np.uint8)
    if n > 0:
        colors[0] = [40, 220, 80]
        colors[-1] = [240, 60, 50]

    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {edges}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, colors, strict=True):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for idx in range(edges):
            f.write(f"{idx} {idx + 1} 40 100 240\n")


def to_viewer_xyz(xyz: np.ndarray, keep_macvo_coordinates: bool) -> np.ndarray:
    if keep_macvo_coordinates:
        return xyz
    out = xyz.copy()
    out[:, 1] *= -1.0
    out[:, 2] *= -1.0
    return out


def plot_topdown(path: Path, points: np.ndarray, poses: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], s=0.5, c="0.25", alpha=0.45, label="sparse points")
    if poses.size:
        ax.plot(poses[:, 0], poses[:, 1], color="#1f5eff", linewidth=1.5, label="trajectory")
        ax.scatter([poses[0, 0]], [poses[0, 1]], c="green", s=50, label="start")
        ax.scatter([poses[-1, 0]], [poses[-1, 1]], c="red", s=50, label="end")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def main() -> None:
    args = parse_args()
    result = Path(args.result)
    output_dir = Path(args.output_dir) if args.output_dir else result / "reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)

    map_file = result / "tensor_map.npz"
    pose_file = result / "poses.npy"
    if not map_file.exists():
        raise FileNotFoundError(map_file)
    if not pose_file.exists():
        raise FileNotFoundError(pose_file)

    tensor_map = np.load(map_file)
    points = tensor_map["points//pos_Tw"].astype(np.float32)
    colors = tensor_map["points//color"].astype(np.uint8)
    cov = tensor_map["points//cov_Tw"].astype(np.float64)
    poses = np.load(pose_file)[:, 1:4].astype(np.float32)

    def filter_points(xyz: np.ndarray, rgb: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        finite_mask = np.isfinite(xyz).all(axis=1)
        dist = np.linalg.norm(xyz, axis=1)
        distance_mask = dist <= args.max_distance

        cov_det = np.linalg.det(covariance)
        cov_finite = np.isfinite(cov_det)
        if args.max_cov_det is not None:
            cov_cutoff = float(args.max_cov_det)
        else:
            valid_det = cov_det[cov_finite]
            cov_cutoff = float(np.percentile(valid_det, args.cov_det_percentile)) if valid_det.size else float("inf")
        cov_mask = cov_finite & (cov_det <= cov_cutoff)

        filtered_mask = finite_mask & distance_mask & cov_mask
        return finite_mask, filtered_mask, rgb[filtered_mask], cov_cutoff

    finite_mask, filtered_mask, filtered_colors, cov_cutoff = filter_points(points, colors, cov)
    filtered_points = points[filtered_mask]

    write_points_ply(output_dir / "vo_points_all.ply", to_viewer_xyz(points[finite_mask], args.keep_macvo_coordinates), colors[finite_mask])
    write_points_ply(output_dir / "vo_points_filtered.ply", to_viewer_xyz(filtered_points, args.keep_macvo_coordinates), filtered_colors)

    mapping_summary = None
    map_points_filtered = np.empty((0, 3), dtype=np.float32)
    if "map_points//pos_Tw" in tensor_map:
        map_points = tensor_map["map_points//pos_Tw"].astype(np.float32)
        map_colors = tensor_map["map_points//color"].astype(np.uint8)
        map_cov = tensor_map["map_points//cov_Tw"].astype(np.float64)
        map_finite, map_filtered, map_filtered_colors, map_cov_cutoff = filter_points(map_points, map_colors, map_cov)
        map_points_filtered = map_points[map_filtered]
        write_points_ply(output_dir / "mapping_points_all.ply", to_viewer_xyz(map_points[map_finite], args.keep_macvo_coordinates), map_colors[map_finite])
        write_points_ply(output_dir / "mapping_points_filtered.ply", to_viewer_xyz(map_points_filtered, args.keep_macvo_coordinates), map_filtered_colors)
        mapping_summary = {
            "points_total": int(map_points.shape[0]),
            "points_finite": int(map_finite.sum()),
            "points_filtered": int(map_filtered.sum()),
            "cov_det_cutoff": map_cov_cutoff,
            "all_points_ply": str(output_dir / "mapping_points_all.ply"),
            "filtered_points_ply": str(output_dir / "mapping_points_filtered.ply"),
        }

    write_trajectory_ply(output_dir / "trajectory.ply", to_viewer_xyz(poses, args.keep_macvo_coordinates))
    np.savetxt(output_dir / "trajectory_xyz.txt", poses, fmt="%.9f")

    if not args.no_plot:
        plot_points = map_points_filtered if map_points_filtered.size else filtered_points
        plot_topdown(output_dir / "topdown_preview.png", plot_points, poses)

    summary = {
        "result": str(result),
        "coordinate": "viewer z-up right-handed PLY by default: x=MACVO_x, y=-MACVO_y, z=-MACVO_z. trajectory_xyz.txt keeps raw MAC-VO coordinates.",
        "poses": int(poses.shape[0]),
        "vo_points_total": int(points.shape[0]),
        "vo_points_finite": int(finite_mask.sum()),
        "vo_points_filtered": int(filtered_mask.sum()),
        "mapping_points": mapping_summary,
        "max_distance_m": args.max_distance,
        "vo_cov_det_cutoff": cov_cutoff,
        "xyz_min_filtered": filtered_points.min(axis=0).tolist() if filtered_points.size else None,
        "xyz_max_filtered": filtered_points.max(axis=0).tolist() if filtered_points.size else None,
        "trajectory_start_xyz": poses[0].tolist() if poses.size else None,
        "trajectory_end_xyz": poses[-1].tolist() if poses.size else None,
        "trajectory_end_distance_m": float(np.linalg.norm(poses[-1])) if poses.size else None,
        "outputs": {
            "all_points_ply": str(output_dir / "vo_points_all.ply"),
            "filtered_points_ply": str(output_dir / "vo_points_filtered.ply"),
            "trajectory_ply": str(output_dir / "trajectory.ply"),
            "trajectory_xyz": str(output_dir / "trajectory_xyz.txt"),
            "topdown_preview": str(output_dir / "topdown_preview.png"),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
