#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pypose as pp
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ROOTS = (
    Path("Results_decxin3261v_mapping/MACVO-DECXIN3261V-Mapping@DECXIN3261V-live"),
    Path("Results_decxin3261v_offline_mapping"),
    Path("Results_decxin3261v_growth"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a saved DECXIN MAC-VO map in Rerun.")
    parser.add_argument("--result", default=None, help="Result folder containing tensor_map.npz, or a parent directory to search. Defaults to latest DECXIN mapping result.")
    parser.add_argument("--growth", action="store_true", help="Replay map points frame by frame instead of showing one static map.")
    parser.add_argument("--every", type=int, default=1, help="For --growth, log one frame chunk every N saved frames.")
    parser.add_argument("--max-points", type=int, default=300000, help="Maximum points shown for the static map; 0 means no limit.")
    parser.add_argument("--max-distance", type=float, default=6.0, help="Drop points farther than this many meters from origin; 0 disables.")
    parser.add_argument("--cov-det-percentile", type=float, default=95.0, help="Drop points above this covariance determinant percentile; 100 disables.")
    parser.add_argument("--vo-points", action="store_true", help="Also show sparse VO tracking points.")
    parser.add_argument("--camera-every", type=int, default=10, help="Show one camera frustum every N poses in static mode.")
    parser.add_argument("--image-every", type=int, default=1, help="For --growth, log one image every N frames.")
    parser.add_argument("--no-images", action="store_true", help="Do not log saved stereo images or preview images.")
    return parser.parse_args()


def latest_result() -> Path:
    candidates: list[Path] = []
    for root in DEFAULT_ROOTS:
        if root.exists():
            candidates.extend(p.parent for p in root.rglob("tensor_map.npz"))
    if not candidates:
        searched = ", ".join(str(p) for p in DEFAULT_ROOTS)
        raise FileNotFoundError(f"No tensor_map.npz found under: {searched}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_result(path: str) -> Path:
    result = Path(path).expanduser()
    if (result / "tensor_map.npz").exists():
        return result

    candidates = [p.parent for p in result.rglob("tensor_map.npz")]
    if not candidates:
        raise FileNotFoundError(f"No tensor_map.npz found under {result}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def is_stereo_sequence(path: Path) -> bool:
    return (path / "left").exists() and (path / "right").exists()


def normalize_sequence_path(path: Path, base: Path) -> Path | None:
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([base / path, PROJECT_ROOT / path])

    for candidate in candidates:
        candidate = candidate.expanduser()
        if is_stereo_sequence(candidate):
            return candidate
        nested = candidate / "stereo_sequence"
        if is_stereo_sequence(nested):
            return nested
    return None


def iter_yaml_roots(node) -> list[str]:
    roots: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "root" and isinstance(value, str):
                roots.append(value)
            else:
                roots.extend(iter_yaml_roots(value))
    elif isinstance(node, list):
        for value in node:
            roots.extend(iter_yaml_roots(value))
    return roots


def sequence_from_yaml(path: Path) -> Path | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for raw_root in iter_yaml_roots(data):
        seq = normalize_sequence_path(Path(raw_root), path.parent)
        if seq is not None:
            return seq
    return None


def find_sequence_dir(result: Path) -> Path | None:
    direct = normalize_sequence_path(result / "stereo_sequence", result)
    if direct is not None:
        return direct

    for cfg in (result / "config.yaml", result / "decxin_offline_sequence.yaml"):
        seq = sequence_from_yaml(cfg)
        if seq is not None:
            return seq
    return None


def point_filter(
    xyz: np.ndarray,
    cov: np.ndarray | None,
    max_distance: float,
    cov_det_percentile: float,
) -> np.ndarray:
    mask = np.isfinite(xyz).all(axis=1)
    if max_distance > 0:
        mask &= np.linalg.norm(xyz, axis=1) <= max_distance
    if cov is not None and cov_det_percentile < 100.0:
        det = np.linalg.det(cov)
        finite = np.isfinite(det)
        valid = det[finite]
        if valid.size > 0:
            cutoff = np.percentile(valid, cov_det_percentile)
            mask &= finite & (det <= cutoff)
    return mask


def limit_points(xyz: np.ndarray, rgb: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz, rgb
    idx = np.linspace(0, xyz.shape[0] - 1, max_points, dtype=np.int64)
    return xyz[idx], rgb[idx]


def log_trajectory(rr, poses_xyz: np.ndarray) -> None:
    if poses_xyz.shape[0] == 0:
        return
    if poses_xyz.shape[0] > 1:
        segments = np.stack([poses_xyz[:-1], poses_xyz[1:]], axis=1)
        rr.log("/world/trajectory", rr.LineStrips3D(segments, colors=[255, 218, 0], radii=0.01), static=True)


def log_camera(rr_plt, path: str, pose_xyzw: np.ndarray, K: np.ndarray) -> None:
    rr_plt.log_camera(
        path,
        pp.SE3(torch.from_numpy(pose_xyzw.astype(np.float32))),
        torch.from_numpy(K.astype(np.float32)),
    )


def log_static_cameras(rr_plt, poses_xyzw: np.ndarray, Ks: np.ndarray, every: int) -> None:
    if poses_xyzw.shape[0] == 0:
        return
    step = max(every, 1)
    indices = list(range(0, poses_xyzw.shape[0], step))
    if indices[-1] != poses_xyzw.shape[0] - 1:
        indices.append(poses_xyzw.shape[0] - 1)
    for idx in indices:
        log_camera(rr_plt, f"/world/cameras/frame_{idx:06d}/cam_left", poses_xyzw[idx], Ks[idx])


def read_rgb(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sequence_image_pair(sequence_dir: Path | None, frame_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    if sequence_dir is None:
        return None, None
    name = f"{frame_idx:06d}.png"
    return read_rgb(sequence_dir / "left" / name), read_rgb(sequence_dir / "right" / name)


def preview_image_pair(result: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    preview = result / "preview"
    return read_rgb(preview / "first_pair_left.png"), read_rgb(preview / "first_pair_right.png")


def log_image_pair(rr, left: np.ndarray | None, right: np.ndarray | None) -> bool:
    logged = False
    if left is not None:
        rr.log("/images/left", rr.Image(left).compress())
        logged = True
    if right is not None:
        rr.log("/images/right", rr.Image(right).compress())
        logged = True
    return logged


def frame_map_indices(ranges: np.ndarray, frame_idx: int) -> np.ndarray:
    if frame_idx >= ranges.shape[0]:
        return np.empty((0,), dtype=np.int64)
    idx_parts: list[np.ndarray] = []
    for start, length in ranges[frame_idx]:
        if start < 0 or length <= 0:
            continue
        idx_parts.append(np.arange(start, start + length, dtype=np.int64))
    if not idx_parts:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(idx_parts)


def main() -> None:
    args = parse_args()
    result = resolve_result(args.result) if args.result else latest_result()
    map_file = result / "tensor_map.npz"
    pose_file = result / "poses.npy"
    if not map_file.exists():
        raise FileNotFoundError(map_file)
    if not pose_file.exists():
        raise FileNotFoundError(pose_file)

    import rerun as rr
    from Utility.Visualize import rr_plt

    tensor_map = np.load(map_file)
    image_sequence = find_sequence_dir(result)
    poses = np.load(pose_file)[:, 1:4].astype(np.float32)
    frame_poses = tensor_map["frames//pose"].astype(np.float32) if "frames//pose" in tensor_map else None
    frame_Ks = tensor_map["frames//K"].astype(np.float32) if "frames//K" in tensor_map else None

    if "map_points//pos_Tw" in tensor_map:
        map_xyz = tensor_map["map_points//pos_Tw"].astype(np.float32)
        map_rgb = tensor_map["map_points//color"].astype(np.uint8)
        map_cov = tensor_map["map_points//cov_Tw"].astype(np.float64)
    else:
        map_xyz = np.empty((0, 3), dtype=np.float32)
        map_rgb = np.empty((0, 3), dtype=np.uint8)
        map_cov = None

    rr.init("MACVO-DECXIN3261V-SavedMap", spawn=False)
    rr_plt.default_mode = "rerun"
    rr.spawn(connect=True)
    rr.log("/", rr.ViewCoordinates(xyz=rr.ViewCoordinates.FRD), static=True)
    rr.log(
        "/world/origin",
        rr.Points3D([[0.0, 0.0, 0.0]], colors=[255, 255, 255], radii=0.05, labels=["origin"]),
        static=True,
    )
    log_trajectory(rr, poses)
    if frame_poses is not None and frame_Ks is not None:
        if args.growth:
            log_camera(rr_plt, "/world/current_camera/cam_left", frame_poses[0], frame_Ks[0])
        else:
            log_static_cameras(rr_plt, frame_poses, frame_Ks, args.camera_every)

    logged_preview = False
    if not args.no_images and not args.growth:
        left, right = sequence_image_pair(image_sequence, 0)
        if left is None and right is None:
            left, right = preview_image_pair(result)
        logged_preview = log_image_pair(rr, left, right)

    if args.vo_points and "points//pos_Tw" in tensor_map:
        vo_xyz = tensor_map["points//pos_Tw"].astype(np.float32)
        vo_rgb = tensor_map["points//color"].astype(np.uint8)
        vo_cov = tensor_map["points//cov_Tw"].astype(np.float64)
        vo_mask = point_filter(vo_xyz, vo_cov, args.max_distance, args.cov_det_percentile)
        vo_xyz, vo_rgb = limit_points(vo_xyz[vo_mask], vo_rgb[vo_mask], args.max_points)
        if vo_xyz.shape[0] > 0:
            rr.log("/world/vo_points", rr.Points3D(vo_xyz, colors=vo_rgb, radii=0.006))

    if map_xyz.shape[0] == 0:
        print(f"No mapping points found in {map_file}")
        return

    map_mask = point_filter(map_xyz, map_cov, args.max_distance, args.cov_det_percentile)
    if args.growth and "edge/frame2map/ranges" in tensor_map:
        ranges = tensor_map["edge/frame2map/ranges"].astype(np.int64)
        total_logged = 0
        for frame_idx in range(0, min(ranges.shape[0], poses.shape[0]), max(args.every, 1)):
            rr.set_time_sequence("frame_idx", frame_idx)
            if frame_poses is not None and frame_Ks is not None and frame_idx < frame_poses.shape[0]:
                log_camera(rr_plt, "/world/current_camera/cam_left", frame_poses[frame_idx], frame_Ks[frame_idx])
            if not args.no_images and frame_idx % max(args.image_every, 1) == 0:
                left, right = sequence_image_pair(image_sequence, frame_idx)
                log_image_pair(rr, left, right)
            idx = frame_map_indices(ranges, frame_idx)
            if idx.size == 0:
                continue
            idx = idx[map_mask[idx]]
            if idx.size == 0:
                continue
            rr.log(
                f"/world/map_chunks/frame_{frame_idx:06d}",
                rr.Points3D(map_xyz[idx], colors=map_rgb[idx], radii=0.004),
            )
            total_logged += int(idx.size)
        if not args.no_images and image_sequence is None:
            left, right = preview_image_pair(result)
            if log_image_pair(rr, left, right):
                print("No stereo_sequence found; logged preview/first_pair images only.")
        if image_sequence is not None:
            print(f"Using image sequence: {image_sequence}")
        print(f"Opened Rerun growth replay: {result} ({total_logged} mapping points logged)")
    else:
        xyz, rgb = limit_points(map_xyz[map_mask], map_rgb[map_mask], args.max_points)
        rr.log("/world/mapping_points", rr.Points3D(xyz, colors=rgb, radii=0.004))
        if not args.no_images and not logged_preview:
            print("No saved stereo_sequence or preview images found for this result.")
        print(f"Opened Rerun static map: {result} ({xyz.shape[0]} / {map_xyz.shape[0]} mapping points shown)")


if __name__ == "__main__":
    main()
