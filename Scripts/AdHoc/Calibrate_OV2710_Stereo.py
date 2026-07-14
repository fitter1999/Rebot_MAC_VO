#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.AdHoc.Capture_OV2710_Calibration import detect_checkerboard, draw_debug


def object_points(pattern_size: tuple[int, int], square_m: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_m
    return objp


def homography_rmse(src: np.ndarray, dst: np.ndarray) -> float:
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        return float("inf")
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((projected - dst) ** 2, axis=1))))


def align_right_corners_to_left(
    left_corners: np.ndarray,
    right_corners: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[np.ndarray, str, float]:
    cols, rows = pattern_size
    left_pts = left_corners.reshape(-1, 2).astype(np.float32)
    right_grid = right_corners.reshape(rows, cols, 1, 2)

    variants = {
        "same": right_grid,
        "reverse": right_grid[::-1, ::-1],
        "flip_rows": right_grid[::-1, :],
        "flip_cols": right_grid[:, ::-1],
    }
    scored = {
        name: homography_rmse(left_pts, corners.reshape(-1, 2).astype(np.float32))
        for name, corners in variants.items()
    }
    best_name = min(scored, key=scored.get)
    best = variants[best_name].reshape(-1, 1, 2).astype(np.float32)
    return best, best_name, scored[best_name]


def relative_pose_from_board(
    object_pts: np.ndarray,
    left_corners: np.ndarray,
    right_corners: np.ndarray,
    K_l: np.ndarray,
    D_l: np.ndarray,
    K_r: np.ndarray,
    D_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    ok_l, rvec_l, tvec_l = cv2.solvePnP(object_pts, left_corners, K_l, D_l, flags=cv2.SOLVEPNP_ITERATIVE)
    ok_r, rvec_r, tvec_r = cv2.solvePnP(object_pts, right_corners, K_r, D_r, flags=cv2.SOLVEPNP_ITERATIVE)
    if not (ok_l and ok_r):
        return None
    R_l, _ = cv2.Rodrigues(rvec_l)
    R_r, _ = cv2.Rodrigues(rvec_r)
    R_lr = R_r @ R_l.T
    T_lr = tvec_r - R_lr @ tvec_l
    return R_lr, T_lr.reshape(3)


def load_pairs(root: Path) -> list[tuple[Path, Path]]:
    left = sorted((root / "left").glob("*.png"))
    right = sorted((root / "right").glob("*.png"))
    right_map = {p.name: p for p in right}
    return [(lp, right_map[lp.name]) for lp in left if lp.name in right_map]


def to_list(array: np.ndarray) -> list:
    return np.asarray(array).tolist()


def reprojection_errors(
    object_points_list: list[np.ndarray],
    image_points_list: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    K: np.ndarray,
    D: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for obj, img, rvec, tvec in zip(object_points_list, image_points_list, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        err = cv2.norm(img, proj, cv2.NORM_L2) / len(proj)
        errors.append(float(err))
    return errors


def save_rectified_preview(
    pair: tuple[Path, Path],
    out_file: Path,
    maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    left_img = cv2.imread(str(pair[0]), cv2.IMREAD_COLOR)
    right_img = cv2.imread(str(pair[1]), cv2.IMREAD_COLOR)
    map_lx, map_ly, map_rx, map_ry = maps
    left_rect = cv2.remap(left_img, map_lx, map_ly, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, map_rx, map_ry, cv2.INTER_LINEAR)
    merged = np.concatenate([left_rect, right_rect], axis=1)
    for y in range(0, merged.shape[0], 40):
        cv2.line(merged, (0, y), (merged.shape[1], y), (0, 255, 255), 1)
    cv2.imwrite(str(out_file), merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate stereo cameras from captured checkerboard pairs.")
    parser.add_argument("--input", default="Calibration/ov2710_screen")
    parser.add_argument("--inner-cols", type=int, default=9)
    parser.add_argument("--inner-rows", type=int, default=6)
    parser.add_argument("--square-mm", type=float, default=28.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-valid-pairs", type=int, default=15)
    parser.add_argument("--known-baseline-m", type=float, default=0.10)
    parser.add_argument("--baseline-filter-ratio", type=float, default=0.30)
    parser.add_argument("--swap", action="store_true", help="Swap left/right image folders before calibration.")
    parser.add_argument("--debug", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input)
    out_dir = Path(args.output) if args.output else root / "calibration_result"
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern_size = (args.inner_cols, args.inner_rows)
    square_m = args.square_mm / 1000.0
    objp = object_points(pattern_size, square_m)

    pairs = load_pairs(root)
    if args.swap:
        pairs = [(right, left) for left, right in pairs]
    if not pairs:
        raise FileNotFoundError(f"No matching left/right PNG pairs under {root}")

    object_points_list: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    valid_pairs: list[tuple[Path, Path]] = []
    align_counts: dict[str, int] = {}
    align_errors: list[float] = []
    image_size: tuple[int, int] | None = None

    debug_dir = out_dir / "corner_debug"
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for lp, rp in pairs:
        left = cv2.imread(str(lp), cv2.IMREAD_COLOR)
        right = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        if left is None or right is None:
            continue
        if left.shape[:2] != right.shape[:2]:
            print(f"skip {lp.name}: left/right shape mismatch")
            continue
        image_size = (left.shape[1], left.shape[0])

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        left_det = detect_checkerboard(left_gray, pattern_size)
        right_det = detect_checkerboard(right_gray, pattern_size)

        if not (left_det.found and right_det.found and left_det.corners is not None and right_det.corners is not None):
            print(f"skip {lp.name}: detect failed L={left_det.found} R={right_det.found}")
            continue

        aligned_right, align_name, align_err = align_right_corners_to_left(left_det.corners, right_det.corners, pattern_size)
        align_counts[align_name] = align_counts.get(align_name, 0) + 1
        align_errors.append(align_err)

        object_points_list.append(objp.copy())
        left_points.append(left_det.corners.astype(np.float32))
        right_points.append(aligned_right)
        valid_pairs.append((lp, rp))

        if args.debug:
            left_dbg = draw_debug(cv2.cvtColor(left, cv2.COLOR_BGR2RGB), pattern_size, left_det)
            aligned_det = type(right_det)(True, aligned_right, right_det.feature)
            right_dbg = draw_debug(cv2.cvtColor(right, cv2.COLOR_BGR2RGB), pattern_size, aligned_det)
            cv2.imwrite(str(debug_dir / lp.name), np.concatenate([left_dbg, right_dbg], axis=1))

    if image_size is None:
        raise RuntimeError("No readable image pairs")
    if len(valid_pairs) < args.min_valid_pairs:
        raise RuntimeError(f"Only {len(valid_pairs)} valid pairs; need at least {args.min_valid_pairs}")

    flags_mono = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        object_points_list, left_points, image_size, None, None, flags=flags_mono, criteria=criteria
    )
    rms_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        object_points_list, right_points, image_size, None, None, flags=flags_mono, criteria=criteria
    )

    baseline_filter_low = args.known_baseline_m * (1.0 - args.baseline_filter_ratio)
    baseline_filter_high = args.known_baseline_m * (1.0 + args.baseline_filter_ratio)
    filtered_indices: list[int] = []
    rejected_by_baseline: list[tuple[str, float]] = []
    if args.known_baseline_m > 0:
        for idx, ((lp, _), obj, lpts, rpts) in enumerate(zip(valid_pairs, object_points_list, left_points, right_points)):
            rel = relative_pose_from_board(obj, lpts, rpts, K_l, D_l, K_r, D_r)
            if rel is None:
                rejected_by_baseline.append((lp.name, float("nan")))
                continue
            _, t_lr = rel
            norm = float(np.linalg.norm(t_lr))
            if baseline_filter_low <= norm <= baseline_filter_high:
                filtered_indices.append(idx)
            else:
                rejected_by_baseline.append((lp.name, norm))

        if len(filtered_indices) >= args.min_valid_pairs:
            object_points_list = [object_points_list[i] for i in filtered_indices]
            left_points = [left_points[i] for i in filtered_indices]
            right_points = [right_points[i] for i in filtered_indices]
            valid_pairs = [valid_pairs[i] for i in filtered_indices]
        else:
            print(
                f"warning: baseline filter kept only {len(filtered_indices)} pairs; "
                "using all valid pairs instead."
            )

    rms_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        object_points_list, left_points, image_size, K_l, D_l, flags=cv2.CALIB_USE_INTRINSIC_GUESS, criteria=criteria
    )
    rms_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        object_points_list, right_points, image_size, K_r, D_r, flags=cv2.CALIB_USE_INTRINSIC_GUESS, criteria=criteria
    )

    flags_stereo = cv2.CALIB_FIX_INTRINSIC
    rms_stereo, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        object_points_list,
        left_points,
        right_points,
        K_l,
        D_l,
        K_r,
        D_r,
        image_size,
        criteria=criteria,
        flags=flags_stereo,
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    baseline_m = float(abs(P2[0, 3] / P1[0, 0]))
    map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, image_size, cv2.CV_32FC1)
    map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, image_size, cv2.CV_32FC1)

    left_errs = reprojection_errors(object_points_list, left_points, rvecs_l, tvecs_l, K_l, D_l)
    right_errs = reprojection_errors(object_points_list, right_points, rvecs_r, tvecs_r, K_r, D_r)

    result = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "pattern_size": [args.inner_cols, args.inner_rows],
        "square_size_m": square_m,
        "swap_left_right": bool(args.swap),
        "valid_pairs": len(valid_pairs),
        "baseline_filter": {
            "known_baseline_m": float(args.known_baseline_m),
            "ratio": float(args.baseline_filter_ratio),
            "kept": len(valid_pairs),
            "rejected": [[name, None if np.isnan(norm) else float(norm)] for name, norm in rejected_by_baseline],
        },
        "right_corner_alignment_counts": align_counts,
        "right_corner_alignment_mean_homography_rmse_px": float(np.mean(align_errors)),
        "rms_left": float(rms_l),
        "rms_right": float(rms_r),
        "rms_stereo": float(rms_stereo),
        "mean_reproj_left_px": float(np.mean(left_errs)),
        "mean_reproj_right_px": float(np.mean(right_errs)),
        "baseline_m_from_P": baseline_m,
        "baseline_m_from_T": float(np.linalg.norm(T)),
        "known_baseline_m": float(args.known_baseline_m),
        "left": {"K": to_list(K_l), "D": to_list(D_l.reshape(-1))},
        "right": {"K": to_list(K_r), "D": to_list(D_r.reshape(-1))},
        "stereo": {
            "R": to_list(R),
            "T": to_list(T.reshape(-1)),
            "E": to_list(E),
            "F": to_list(F),
            "R1": to_list(R1),
            "R2": to_list(R2),
            "P1": to_list(P1),
            "P2": to_list(P2),
            "Q": to_list(Q),
            "roi1": list(map(int, roi1)),
            "roi2": list(map(int, roi2)),
        },
        "macvo_rectified_camera": {
            "fx": float(P1[0, 0]),
            "fy": float(P1[1, 1]),
            "cx": float(P1[0, 2]),
            "cy": float(P1[1, 2]),
            "baseline": baseline_m,
        },
    }

    if np.linalg.norm(T) > 1e-9 and args.known_baseline_m > 0:
        T_known = T * (args.known_baseline_m / float(np.linalg.norm(T)))
        R1k, R2k, P1k, P2k, Qk, roi1k, roi2k = cv2.stereoRectify(
            K_l, D_l, K_r, D_r, image_size, R, T_known, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, D_l, R1k, P1k, image_size, cv2.CV_32FC1)
        map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, D_r, R2k, P2k, image_size, cv2.CV_32FC1)
        R1, R2, P1, P2, Q, roi1, roi2 = R1k, R2k, P1k, P2k, Qk, roi1k, roi2k
        baseline_m = float(abs(P2[0, 3] / P1[0, 0]))
        result["stereo"]["R1"] = to_list(R1)
        result["stereo"]["R2"] = to_list(R2)
        result["stereo"]["P1"] = to_list(P1)
        result["stereo"]["P2"] = to_list(P2)
        result["stereo"]["Q"] = to_list(Q)
        result["stereo"]["roi1"] = list(map(int, roi1))
        result["stereo"]["roi2"] = list(map(int, roi2))
        result["baseline_m_from_P"] = baseline_m
        result["macvo_rectified_camera"] = {
            "fx": float(P1[0, 0]),
            "fy": float(P1[1, 1]),
            "cx": float(P1[0, 2]),
            "cy": float(P1[1, 2]),
            "baseline": baseline_m,
        }
        result["known_baseline_rectified_camera"] = {
            "fx": float(P1k[0, 0]),
            "fy": float(P1k[1, 1]),
            "cx": float(P1k[0, 2]),
            "cy": float(P1k[1, 2]),
            "baseline": float(abs(P2k[0, 3] / P1k[0, 0])),
        }

    yaml_file = out_dir / "calibration.yaml"
    with yaml_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False)

    maps_file = out_dir / "rectify_maps.npz"
    np.savez_compressed(
        maps_file,
        map_lx=map_lx,
        map_ly=map_ly,
        map_rx=map_rx,
        map_ry=map_ry,
        K_l=K_l,
        D_l=D_l,
        K_r=K_r,
        D_r=D_r,
        R=R,
        T=T,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
    )

    save_rectified_preview(valid_pairs[len(valid_pairs) // 2], out_dir / "rectified_preview.png", (map_lx, map_ly, map_rx, map_ry))

    print(f"valid pairs: {len(valid_pairs)}")
    if args.known_baseline_m > 0:
        print(f"baseline filter rejected: {len(rejected_by_baseline)}")
    print(f"right corner alignment: {align_counts}, mean H rmse={np.mean(align_errors):.4f}px")
    print(f"rms left/right/stereo: {rms_l:.4f}, {rms_r:.4f}, {rms_stereo:.4f}")
    print(f"mean reproj px left/right: {np.mean(left_errs):.4f}, {np.mean(right_errs):.4f}")
    print(f"baseline from T: {np.linalg.norm(T):.4f} m")
    print(f"baseline from rectified P: {baseline_m:.4f} m")
    print(f"MAC-VO rectified camera: {result['macvo_rectified_camera']}")
    if "known_baseline_rectified_camera" in result:
        print(f"Known-baseline candidate: {result['known_baseline_rectified_camera']}")
    print(f"wrote: {yaml_file}")
    print(f"wrote: {maps_file}")
    print(f"preview: {out_dir / 'rectified_preview.png'}")


if __name__ == "__main__":
    main()
