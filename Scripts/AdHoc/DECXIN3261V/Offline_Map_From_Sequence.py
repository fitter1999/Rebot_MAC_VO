#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline dense MAC-VO mapping from a recorded DECXIN stereo_sequence.")
    parser.add_argument("--result", required=True, help="Online result folder containing stereo_sequence/left and stereo_sequence/right.")
    parser.add_argument("--odom", default="Config/Experiment/MACVO/MACVO_DECXIN3261V_Mapping.yaml")
    parser.add_argument("--resultRoot", default="./Results_decxin3261v_offline_mapping")
    parser.add_argument("--target-fps", type=float, default=None, help="Subsample the recorded sequence to this approximate FPS before running MAC-VO.")
    parser.add_argument("--stride", type=int, default=1, help="Use every N-th recorded frame before running MAC-VO.")
    parser.add_argument("--sampled-dir", default=None, help="Optional output directory for the sampled stereo_sequence.")
    parser.add_argument("--prepare-only", action="store_true", help="Only generate the sampled sequence/config and print the MAC-VO command.")
    parser.add_argument("--seq-from", type=int, default=0)
    parser.add_argument("--seq-to", type=int, default=None)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--timing", action="store_true")
    return parser.parse_args()


def format_fps_tag(fps: float) -> str:
    text = f"{fps:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def load_timestamp_ns(seq: Path) -> list[int] | None:
    csv_path = seq / "timestamps.csv"
    if not csv_path.exists():
        return None

    times: list[int] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                times.append(int(row["time_ns"]))
            except (KeyError, TypeError, ValueError):
                return None

    return times


def select_by_target_fps(times_ns: list[int], target_fps: float) -> list[int]:
    if target_fps <= 0:
        raise ValueError("--target-fps must be positive")
    interval_ns = int(round(1_000_000_000 / target_fps))
    selected = [0]
    next_time = times_ns[0] + interval_ns
    for idx, ts in enumerate(times_ns[1:], start=1):
        if ts >= next_time:
            selected.append(idx)
            while next_time <= ts:
                next_time += interval_ns
    return selected


def average_fps(times_ns: list[int], indices: list[int]) -> float | None:
    if len(indices) < 2:
        return None
    duration_s = (times_ns[indices[-1]] - times_ns[indices[0]]) / 1e9
    if duration_s <= 0:
        return None
    return (len(indices) - 1) / duration_s


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_sampled_sequence(
    seq: Path,
    result: Path,
    indices: list[int],
    tag: str,
    sampled_dir_arg: str | None,
    times_ns: list[int] | None,
) -> Path:
    left_files = sorted((seq / "left").glob("*.png"))
    right_files = sorted((seq / "right").glob("*.png"))
    if not left_files or len(left_files) != len(right_files):
        raise FileNotFoundError(f"Invalid stereo_sequence under {seq}")

    if sampled_dir_arg:
        out = Path(sampled_dir_arg).expanduser().resolve()
    else:
        stamp = time.strftime("%m_%d_%H%M%S")
        out = result / "sampled_sequences" / f"stereo_sequence{tag}_{stamp}"

    if out.exists() and sampled_dir_arg:
        raise FileExistsError(f"--sampled-dir already exists: {out}")
    if out.exists():
        shutil.rmtree(out)
    left_out = out / "left"
    right_out = out / "right"
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, int | str]] = []
    for out_idx, src_idx in enumerate(indices):
        name = f"{out_idx:06d}.png"
        link_or_copy(left_files[src_idx], left_out / name)
        link_or_copy(right_files[src_idx], right_out / name)
        record: dict[str, int | str] = {
            "frame_idx": out_idx,
            "source_frame_idx": src_idx,
            "left": f"left/{name}",
            "right": f"right/{name}",
            "source_left": str(left_files[src_idx]),
            "source_right": str(right_files[src_idx]),
        }
        if times_ns is not None and src_idx < len(times_ns):
            record["source_time_ns"] = times_ns[src_idx]
        records.append(record)

    with (out / "source_indices.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return out


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.target_fps is not None and args.stride != 1:
        raise ValueError("Use either --target-fps or --stride, not both.")

    result = Path(args.result).resolve()
    seq = result / "stereo_sequence"
    if not (seq / "left").exists() or not (seq / "right").exists():
        raise FileNotFoundError(f"No stereo_sequence/left,right under {result}")

    left_files = sorted((seq / "left").glob("*.png"))
    right_files = sorted((seq / "right").glob("*.png"))
    if not left_files or len(left_files) != len(right_files):
        raise FileNotFoundError(f"Invalid stereo_sequence under {seq}")

    image_len = len(left_files)
    source_len = image_len
    times_ns = load_timestamp_ns(seq)
    if times_ns is not None and len(times_ns) != image_len:
        source_len = min(image_len, len(times_ns))
        print(
            f"warning: image count ({image_len}) and timestamps.csv rows ({len(times_ns)}) differ; "
            f"using first {source_len} frames for timestamp-based sampling.",
            file=sys.stderr,
        )
    indices = list(range(source_len))
    sample_tag = ""
    sequence_root = seq

    if args.target_fps is not None:
        if times_ns is None:
            raise FileNotFoundError(f"--target-fps requires {seq / 'timestamps.csv'}")
        indices = select_by_target_fps(times_ns[:source_len], args.target_fps)
        sample_tag = f"-fps{format_fps_tag(args.target_fps)}"
    elif args.stride > 1:
        indices = list(range(0, source_len, args.stride))
        sample_tag = f"-stride{args.stride}"

    if len(indices) < 2:
        raise ValueError(f"Only selected {len(indices)} frame(s); choose a lower --target-fps or --stride")

    if sample_tag:
        sequence_root = build_sampled_sequence(seq, result, indices, sample_tag, args.sampled_dir, times_ns)
        fps = average_fps(times_ns, indices) if times_ns is not None else None
        fps_text = f", approx_fps={fps:.3f}" if fps is not None else ""
        print(f"Prepared sampled sequence: {source_len} -> {len(indices)} frames{fps_text}")
        print(f"Sampled stereo_sequence: {sequence_root}")

    data_cfg = result / f"decxin_offline_sequence{sample_tag}.yaml"
    data = {
        "type": "GeneralStereo",
        "name": f"DECXIN3261V-offline-{result.name}{sample_tag}",
        "args": {
            "root": str(sequence_root),
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
    if args.prepare_only:
        return
    raise SystemExit(subprocess.call(cmd, cwd=PROJECT_ROOT))


if __name__ == "__main__":
    main()
