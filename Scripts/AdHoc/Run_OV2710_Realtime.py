#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pypose as pp
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DataLoader import StereoData, StereoFrame
from Odometry.MACVO import MACVO
from Utility.Config import asNamespace, load_config
from Utility.PrettyPrint import Logger
from Utility.Sandbox import Sandbox
from Utility.Timer import Timer


@dataclass
class CameraSample:
    sample_id: int
    time_ns: int
    rgb: np.ndarray
    read_span_ns: int


@dataclass
class StereoPair:
    pair_id: int
    time_ns: int
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    left_grab_mid_ns: int
    right_grab_mid_ns: int
    grab_span_ns: int

    @property
    def software_skew_ms(self) -> float:
        return abs(self.right_grab_mid_ns - self.left_grab_mid_ns) / 1e6


@dataclass
class DisplaySnapshot:
    pair: StereoPair | None = None
    poses_xyz: np.ndarray | None = None
    vo_points_xyz: np.ndarray | None = None
    map_points_xyz: np.ndarray | None = None
    processed: int = 0
    step_ms: float = 0.0
    skipped_pairs: int = 0
    stats: dict[str, float] | None = None


def fourcc_to_str(value: float) -> str:
    value_i = int(value)
    if value_i <= 0:
        return "N/A"
    return "".join(chr((value_i >> (8 * idx)) & 0xFF) for idx in range(4))


def parse_device(device: str) -> int | str:
    video_prefix = "/dev/video"
    path = Path(device)
    if path.exists():
        try:
            resolved = path.resolve()
            resolved_s = str(resolved)
            if resolved_s.startswith(video_prefix) and resolved_s[len(video_prefix):].isdigit():
                return int(resolved_s[len(video_prefix):])
        except OSError:
            pass
    if device.isdigit():
        return int(device)
    if device.startswith(video_prefix) and device[len(video_prefix):].isdigit():
        return int(device[len(video_prefix):])
    return device


def convert_to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"Unsupported camera frame shape: {frame.shape}")


def open_camera(device: str, width: int, height: int, fps: float, fourcc: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(parse_device(device), cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {device}")

    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Opened camera {device}, but failed to read a frame")

    actual = {
        "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    Logger.write("info", f"{device}: {actual}, first frame shape={frame.shape}")
    return cap


class StereoCameraReader:
    def __init__(
        self,
        left_device: str,
        right_device: str,
        width: int,
        height: int,
        fps: float,
        fourcc: str,
        pairing_delay_ms: float,
    ) -> None:
        self.left_device = left_device
        self.right_device = right_device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.pairing_delay_ns = int(max(pairing_delay_ms, 0.0) * 1e6)

        self.left_cap: cv2.VideoCapture | None = None
        self.right_cap: cv2.VideoCapture | None = None
        self.left_thread: threading.Thread | None = None
        self.right_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.cond = threading.Condition()

        self.left_buffer: deque[CameraSample] = deque(maxlen=8)
        self.right_buffer: deque[CameraSample] = deque(maxlen=8)
        self.left_sample_id = 0
        self.right_sample_id = 0
        self.last_paired_left_id = 0
        self.last_paired_right_id = 0
        self.latest_pair: StereoPair | None = None
        self.pair_id = 0
        self.error_count = 0
        self.left_intervals_ms: deque[float] = deque(maxlen=300)
        self.right_intervals_ms: deque[float] = deque(maxlen=300)
        self.pair_intervals_ms: deque[float] = deque(maxlen=300)
        self.skews_ms: deque[float] = deque(maxlen=300)
        self.last_left_time_ns: int | None = None
        self.last_right_time_ns: int | None = None
        self.last_pair_time_ns: int | None = None

    def start(self) -> None:
        self.left_cap = open_camera(self.left_device, self.width, self.height, self.fps, self.fourcc)
        self.right_cap = open_camera(self.right_device, self.width, self.height, self.fps, self.fourcc)
        self.left_thread = threading.Thread(
            target=self._camera_loop,
            args=("left", self.left_cap),
            name="OV2710LeftCapture",
            daemon=True,
        )
        self.right_thread = threading.Thread(
            target=self._camera_loop,
            args=("right", self.right_cap),
            name="OV2710RightCapture",
            daemon=True,
        )
        self.left_thread.start()
        self.right_thread.start()

    def _camera_loop(self, side: str, cap: cv2.VideoCapture) -> None:
        while not self.stop_event.is_set():
            t0 = time.monotonic_ns()
            ok, frame_bgr = cap.read()
            t1 = time.monotonic_ns()

            if not ok or frame_bgr is None:
                with self.cond:
                    self.error_count += 1
                    self.cond.notify_all()
                time.sleep(0.005)
                continue

            if frame_bgr.shape[:2] != (self.height, self.width):
                frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)

            sample = CameraSample(
                sample_id=0,
                time_ns=(t0 + t1) // 2,
                rgb=convert_to_rgb(frame_bgr),
                read_span_ns=t1 - t0,
            )

            with self.cond:
                if side == "left":
                    self.left_sample_id += 1
                    sample.sample_id = self.left_sample_id
                    if self.last_left_time_ns is not None:
                        self.left_intervals_ms.append((sample.time_ns - self.last_left_time_ns) / 1e6)
                    self.last_left_time_ns = sample.time_ns
                    self.left_buffer.append(sample)
                else:
                    self.right_sample_id += 1
                    sample.sample_id = self.right_sample_id
                    if self.last_right_time_ns is not None:
                        self.right_intervals_ms.append((sample.time_ns - self.last_right_time_ns) / 1e6)
                    self.last_right_time_ns = sample.time_ns
                    self.right_buffer.append(sample)

                self._publish_pair_locked()
                self.cond.notify_all()

    def _publish_pair_locked(self) -> None:
        if not self.left_buffer or not self.right_buffer:
            return

        now_ns = time.monotonic_ns()
        candidates: list[tuple[int, CameraSample, CameraSample]] = []
        for left in self.left_buffer:
            if left.sample_id <= self.last_paired_left_id:
                continue
            if now_ns - left.time_ns < self.pairing_delay_ns:
                continue
            for right in self.right_buffer:
                if right.sample_id <= self.last_paired_right_id:
                    continue
                if now_ns - right.time_ns < self.pairing_delay_ns:
                    continue
                candidates.append((abs(left.time_ns - right.time_ns), left, right))
        if not candidates:
            return

        _, left, right = min(candidates, key=lambda item: item[0])
        pair = StereoPair(
            pair_id=self.pair_id + 1,
            time_ns=(left.time_ns + right.time_ns) // 2,
            left_rgb=left.rgb,
            right_rgb=right.rgb,
            left_grab_mid_ns=left.time_ns,
            right_grab_mid_ns=right.time_ns,
            grab_span_ns=max(left.read_span_ns, right.read_span_ns),
        )

        if self.last_pair_time_ns is not None:
            self.pair_intervals_ms.append((pair.time_ns - self.last_pair_time_ns) / 1e6)
        self.last_pair_time_ns = pair.time_ns
        self.skews_ms.append(pair.software_skew_ms)
        self.last_paired_left_id = left.sample_id
        self.last_paired_right_id = right.sample_id
        while self.left_buffer and self.left_buffer[0].sample_id <= self.last_paired_left_id:
            self.left_buffer.popleft()
        while self.right_buffer and self.right_buffer[0].sample_id <= self.last_paired_right_id:
            self.right_buffer.popleft()
        self.pair_id = pair.pair_id
        self.latest_pair = pair

    def wait_for_newer(self, last_pair_id: int, timeout: float) -> StereoPair | None:
        deadline = time.monotonic() + timeout
        with self.cond:
            while not self.stop_event.is_set():
                if self.latest_pair is not None and self.latest_pair.pair_id > last_pair_id:
                    return self.latest_pair
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.cond.wait(timeout=remaining)
        return None

    def stop(self) -> None:
        self.stop_event.set()
        with self.cond:
            self.cond.notify_all()
        if self.left_thread is not None:
            self.left_thread.join(timeout=2.0)
        if self.right_thread is not None:
            self.right_thread.join(timeout=2.0)
        if self.left_cap is not None:
            self.left_cap.release()
        if self.right_cap is not None:
            self.right_cap.release()

    def stats(self) -> dict[str, float | str]:
        with self.cond:
            left_intervals = list(self.left_intervals_ms)
            right_intervals = list(self.right_intervals_ms)
            pair_intervals = list(self.pair_intervals_ms)
            skews = list(self.skews_ms)
            pair_id = self.pair_id

        stats: dict[str, float | str] = {
            "pairs": float(pair_id),
            "errors": float(self.error_count),
            "pairing_mode": "nearest",
            "pairing_delay_ms": self.pairing_delay_ns / 1e6,
        }
        if left_intervals:
            left_mean_interval = statistics.fmean(left_intervals)
            stats["left_fps"] = 1000.0 / left_mean_interval if left_mean_interval > 0 else 0.0
        if right_intervals:
            right_mean_interval = statistics.fmean(right_intervals)
            stats["right_fps"] = 1000.0 / right_mean_interval if right_mean_interval > 0 else 0.0
        if pair_intervals:
            mean_interval = statistics.fmean(pair_intervals)
            stats["capture_fps"] = 1000.0 / mean_interval if mean_interval > 0 else 0.0
            stats["interval_mean_ms"] = mean_interval
            stats["interval_max_ms"] = max(pair_intervals)
        if skews:
            stats["skew_mean_ms"] = statistics.fmean(skews)
            stats["skew_max_ms"] = max(skews)
        return stats


def image_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(rgb)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return tensor / 255.0


def make_stereo_frame(
    pair: StereoPair,
    frame_idx: int,
    K: torch.Tensor,
    baseline: float,
    T_BS: pp.LieTensor,
    swap: bool,
) -> StereoFrame:
    left_rgb, right_rgb = (pair.right_rgb, pair.left_rgb) if swap else (pair.left_rgb, pair.right_rgb)

    image_l = image_to_tensor(left_rgb)
    image_r = image_to_tensor(right_rgb)

    return StereoFrame(
        idx=[frame_idx],
        time_ns=[pair.time_ns],
        stereo=StereoData(
            T_BS=T_BS,
            K=K,
            baseline=torch.tensor([baseline], dtype=torch.float32),
            width=image_l.size(-1),
            height=image_l.size(-2),
            time_ns=[pair.time_ns],
            imageL=image_l,
            imageR=image_r,
        ),
    )


def resize_pair(pair: StereoPair, width: int | None, height: int | None) -> StereoPair:
    if width is None and height is None:
        return pair
    target_w = pair.left_rgb.shape[1] if width is None else width
    target_h = pair.left_rgb.shape[0] if height is None else height
    if pair.left_rgb.shape[:2] == (target_h, target_w) and pair.right_rgb.shape[:2] == (target_h, target_w):
        return pair

    left = cv2.resize(pair.left_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    right = cv2.resize(pair.right_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return StereoPair(
        pair_id=pair.pair_id,
        time_ns=pair.time_ns,
        left_rgb=left,
        right_rgb=right,
        left_grab_mid_ns=pair.left_grab_mid_ns,
        right_grab_mid_ns=pair.right_grab_mid_ns,
        grab_span_ns=pair.grab_span_ns,
    )


def scale_intrinsic(K: torch.Tensor, raw_width: int, raw_height: int, target_width: int, target_height: int) -> torch.Tensor:
    scaled = K.clone()
    scaled[:, 0, :] *= float(target_width) / float(raw_width)
    scaled[:, 1, :] *= float(target_height) / float(raw_height)
    return scaled


def save_pair_preview(pair: StereoPair, folder: Path, prefix: str, swap: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    left_rgb, right_rgb = (pair.right_rgb, pair.left_rgb) if swap else (pair.left_rgb, pair.right_rgb)
    cv2.imwrite(str(folder / f"{prefix}_left.png"), cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(folder / f"{prefix}_right.png"), cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))


class StereoSequenceRecorder:
    def __init__(self, folder: Path, swap: bool) -> None:
        self.folder = folder
        self.left_dir = folder / "left"
        self.right_dir = folder / "right"
        self.swap = swap
        self.records: list[dict[str, int | float | str]] = []
        self.left_dir.mkdir(parents=True, exist_ok=True)
        self.right_dir.mkdir(parents=True, exist_ok=True)

    def write(self, pair: StereoPair, frame_idx: int) -> None:
        left_rgb, right_rgb = (pair.right_rgb, pair.left_rgb) if self.swap else (pair.left_rgb, pair.right_rgb)
        name = f"{frame_idx:06d}.png"
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        cv2.imwrite(str(self.left_dir / name), cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR), png_params)
        cv2.imwrite(str(self.right_dir / name), cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR), png_params)
        self.records.append(
            {
                "frame_idx": frame_idx,
                "pair_id": pair.pair_id,
                "time_ns": pair.time_ns,
                "left_time_ns": pair.left_grab_mid_ns,
                "right_time_ns": pair.right_grab_mid_ns,
                "software_skew_ms": pair.software_skew_ms,
                "left": f"left/{name}",
                "right": f"right/{name}",
            }
        )

    def close(self) -> None:
        if not self.records:
            return
        import csv

        with (self.folder / "timestamps.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.records[0].keys()))
            writer.writeheader()
            writer.writerows(self.records)


class StereoRectifier:
    def __init__(self, maps_file: str | None) -> None:
        self.enabled = False
        self.map_lx: np.ndarray | None = None
        self.map_ly: np.ndarray | None = None
        self.map_rx: np.ndarray | None = None
        self.map_ry: np.ndarray | None = None
        self.K: torch.Tensor | None = None
        self.baseline: float | None = None

        if maps_file is None:
            return

        data = np.load(maps_file)
        self.map_lx = data["map_lx"]
        self.map_ly = data["map_ly"]
        self.map_rx = data["map_rx"]
        self.map_ry = data["map_ry"]
        P1 = data["P1"]
        P2 = data["P2"]

        self.K = torch.tensor(
            [[[float(P1[0, 0]), 0.0, float(P1[0, 2])], [0.0, float(P1[1, 1]), float(P1[1, 2])], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )
        self.baseline = float(abs(P2[0, 3] / P1[0, 0]))
        self.enabled = True
        Logger.write("info", f"Loaded rectification maps from {maps_file}")
        Logger.write("info", f"Rectified camera K={self.K.tolist()}, baseline={self.baseline:.6f} m")

    def rectify_pair(self, pair: StereoPair, swap: bool) -> StereoPair:
        if not self.enabled:
            return pair
        assert self.map_lx is not None and self.map_ly is not None and self.map_rx is not None and self.map_ry is not None

        left_rgb, right_rgb = (pair.right_rgb, pair.left_rgb) if swap else (pair.left_rgb, pair.right_rgb)
        left_rect = cv2.remap(left_rgb, self.map_lx, self.map_ly, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_rgb, self.map_rx, self.map_ry, cv2.INTER_LINEAR)

        if swap:
            left_out, right_out = right_rect, left_rect
        else:
            left_out, right_out = left_rect, right_rect

        return StereoPair(
            pair_id=pair.pair_id,
            time_ns=pair.time_ns,
            left_rgb=left_out,
            right_rgb=right_out,
            left_grab_mid_ns=pair.left_grab_mid_ns,
            right_grab_mid_ns=pair.right_grab_mid_ns,
            grab_span_ns=pair.grab_span_ns,
        )


class LiveDisplay:
    def __init__(self, scale: float, window_name: str = "MAC-VO OV2710 Live") -> None:
        self.scale = scale
        self.window_name = window_name
        self.enabled = True

    @staticmethod
    def _draw_text(image: np.ndarray, lines: list[str]) -> None:
        y = 24
        for line in lines:
            cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22

    @staticmethod
    def _trajectory_panel(system: MACVO, height: int) -> np.ndarray:
        width = max(260, height // 2)
        panel = np.full((height, width, 3), 245, dtype=np.uint8)
        cv2.putText(panel, "trajectory x-y", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

        if len(system.graph.frames) == 0:
            return panel

        poses = system.graph.frames.data["pose"].tensor.detach().cpu().numpy()
        pts = poses[:, :2].astype(np.float32)
        if pts.shape[0] == 0:
            return panel

        center = pts.mean(axis=0)
        span = np.maximum(np.ptp(pts, axis=0), 0.05)
        scale = 0.82 * min((width - 24) / span[0], (height - 48) / span[1])
        xy = (pts - center) * scale
        pix = np.empty_like(xy)
        pix[:, 0] = width / 2 + xy[:, 0]
        pix[:, 1] = height / 2 - xy[:, 1]
        pix = np.round(pix).astype(np.int32)

        if pix.shape[0] > 1:
            cv2.polylines(panel, [pix.reshape(-1, 1, 2)], False, (30, 90, 220), 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(pix[0]), 4, (40, 160, 40), -1, cv2.LINE_AA)
        cv2.circle(panel, tuple(pix[-1]), 5, (30, 30, 220), -1, cv2.LINE_AA)
        return panel

    def update(
        self,
        pair: StereoPair,
        system: MACVO,
        processed: int,
        step_ms: float,
        skipped_pairs: int,
        stats: dict[str, float],
    ) -> bool:
        if not self.enabled:
            return True

        left = cv2.cvtColor(pair.left_rgb, cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(pair.right_rgb, cv2.COLOR_RGB2BGR)
        stereo = np.concatenate([left, right], axis=1)

        if self.scale != 1.0:
            stereo = cv2.resize(stereo, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        lines = [
            f"VO frame: {processed}",
            f"step: {step_ms:.1f} ms",
            f"capture fps: {stats.get('capture_fps', 0.0):.1f}",
            f"skew mean/max: {stats.get('skew_mean_ms', 0.0):.1f}/{stats.get('skew_max_ms', 0.0):.1f} ms",
            f"skipped pairs: {skipped_pairs}",
            "q/Esc: quit",
        ]
        self._draw_text(stereo, lines)

        traj = self._trajectory_panel(system, stereo.shape[0])
        canvas = np.concatenate([stereo, traj], axis=1)
        try:
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            Logger.write("warn", f"OpenCV display failed; disabling live display: {exc}")
            self.enabled = False
            return True

        return key not in (27, ord("q"))

    def close(self) -> None:
        if self.enabled:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass


class WebDisplay:
    def __init__(self, host: str, port: int, scale: float, fps: float, max_points: int) -> None:
        self.host = host
        self.port = port
        self.scale = scale
        self.fps = fps
        self.max_points = max_points
        self.cond = threading.Condition()
        self.state_lock = threading.Lock()
        self.snapshot = DisplaySnapshot()
        self.jpeg: bytes | None = None
        self.running = True
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.render_thread: threading.Thread | None = None

    @staticmethod
    def _draw_text(image: np.ndarray, lines: list[str]) -> None:
        LiveDisplay._draw_text(image, lines)

    @staticmethod
    def _project_points(points: np.ndarray, dims: tuple[int, int], axes: tuple[int, int], margin: int = 18) -> tuple[np.ndarray, np.ndarray]:
        width, height = dims
        if points.size == 0:
            return np.empty((0, 2), dtype=np.int32), np.zeros((2,), dtype=np.float32)

        pts2 = points[:, axes].astype(np.float32)
        center = pts2.mean(axis=0)
        span = np.maximum(np.ptp(pts2, axis=0), 0.05)
        scale = 0.86 * min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
        xy = (pts2 - center) * scale
        pix = np.empty_like(xy)
        pix[:, 0] = width / 2 + xy[:, 0]
        pix[:, 1] = height / 2 - xy[:, 1]
        return np.round(pix).astype(np.int32), center

    def _plot_panel(
        self,
        title: str,
        poses_xyz: np.ndarray | None,
        vo_points_xyz: np.ndarray | None,
        map_points_xyz: np.ndarray | None,
        width: int,
        height: int,
        axes: tuple[int, int],
    ) -> np.ndarray:
        panel = np.full((height, width, 3), 246, dtype=np.uint8)
        cv2.putText(panel, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 1, cv2.LINE_AA)

        clouds: list[np.ndarray] = []
        if poses_xyz is not None and poses_xyz.size:
            clouds.append(poses_xyz)
        if vo_points_xyz is not None and vo_points_xyz.size:
            clouds.append(vo_points_xyz)
        if map_points_xyz is not None and map_points_xyz.size:
            clouds.append(map_points_xyz)
        if not clouds:
            return panel

        all_pts = np.concatenate(clouds, axis=0)
        pix_all, center = self._project_points(all_pts, (width, height), axes)
        offset = 0

        def take_pix(points: np.ndarray | None) -> np.ndarray:
            nonlocal offset
            if points is None or points.size == 0:
                return np.empty((0, 2), dtype=np.int32)
            n = points.shape[0]
            out = pix_all[offset : offset + n]
            offset += n
            return out

        traj_pix = take_pix(poses_xyz)
        vo_pix = take_pix(vo_points_xyz)
        map_pix = take_pix(map_points_xyz)

        for p in map_pix:
            cv2.circle(panel, tuple(p), 1, (150, 150, 150), -1, cv2.LINE_AA)
        for p in vo_pix:
            cv2.circle(panel, tuple(p), 2, (220, 120, 30), -1, cv2.LINE_AA)
        if traj_pix.shape[0] > 1:
            cv2.polylines(panel, [traj_pix.reshape(-1, 1, 2)], False, (30, 110, 230), 2, cv2.LINE_AA)
        if traj_pix.shape[0] > 0:
            cv2.circle(panel, tuple(traj_pix[0]), 5, (40, 170, 40), -1, cv2.LINE_AA)
            cv2.circle(panel, tuple(traj_pix[-1]), 6, (40, 40, 220), -1, cv2.LINE_AA)

        cv2.putText(
            panel,
            f"poses={0 if poses_xyz is None else len(poses_xyz)} vo_pts={0 if vo_points_xyz is None else len(vo_points_xyz)} map_pts={0 if map_points_xyz is None else len(map_points_xyz)}",
            (12, height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        return panel

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                return

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    body = (
                        "<!doctype html><html><head><meta charset='utf-8'>"
                        "<title>MAC-VO Live</title>"
                        "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif;}"
                        "header{padding:10px 14px;background:#222;position:sticky;top:0;}"
                        "img{display:block;max-width:100vw;width:100%;height:auto;}</style>"
                        "</head><body><header>MAC-VO OV2710 Live - refresh if stream stalls</header>"
                        "<img src='/stream.mjpg'></body></html>"
                    ).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path != "/stream.mjpg":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return

                self.send_response(HTTPStatus.OK)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                last_sent: bytes | None = None
                while owner.running:
                    with owner.cond:
                        owner.cond.wait_for(lambda: owner.jpeg is not None and owner.jpeg is not last_sent or not owner.running, timeout=2.0)
                        if not owner.running:
                            break
                        frame = owner.jpeg
                    if frame is None:
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        last_sent = frame
                    except (BrokenPipeError, ConnectionResetError):
                        break

        self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, name="MACVOWebDisplay", daemon=True)
        self.thread.start()
        self.render_thread = threading.Thread(target=self._render_loop, name="MACVOWebRender", daemon=True)
        self.render_thread.start()
        Logger.write("info", f"Web display: http://{self.host}:{self.port}")

    def update(
        self,
        pair: StereoPair,
        poses_xyz: np.ndarray | None,
        vo_points_xyz: np.ndarray | None,
        map_points_xyz: np.ndarray | None,
        processed: int,
        step_ms: float,
        skipped_pairs: int,
        stats: dict[str, float],
    ) -> None:
        with self.state_lock:
            self.snapshot = DisplaySnapshot(
                pair=pair,
                poses_xyz=poses_xyz,
                vo_points_xyz=vo_points_xyz,
                map_points_xyz=map_points_xyz,
                processed=processed,
                step_ms=step_ms,
                skipped_pairs=skipped_pairs,
                stats=dict(stats),
            )

    def _render_loop(self) -> None:
        interval = 1.0 / max(self.fps, 0.1)
        while self.running:
            start = time.monotonic()
            with self.state_lock:
                snap = self.snapshot
            if snap.pair is not None:
                self._render_snapshot(snap)
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, interval - elapsed))

    def _render_snapshot(self, snap: DisplaySnapshot) -> None:
        assert snap.pair is not None
        stats = snap.stats or {}
        left = cv2.cvtColor(snap.pair.left_rgb, cv2.COLOR_RGB2BGR)
        right = cv2.cvtColor(snap.pair.right_rgb, cv2.COLOR_RGB2BGR)
        stereo = np.concatenate([left, right], axis=1)
        if self.scale != 1.0:
            stereo = cv2.resize(stereo, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        lines = [
            f"VO frame: {snap.processed}",
            f"step: {snap.step_ms:.1f} ms",
            f"capture fps: {stats.get('capture_fps', 0.0):.1f}",
            f"skew mean/max: {stats.get('skew_mean_ms', 0.0):.1f}/{stats.get('skew_max_ms', 0.0):.1f} ms",
            f"skipped pairs: {snap.skipped_pairs}",
            f"web fps target: {self.fps:.1f}",
        ]
        self._draw_text(stereo, lines)

        panel_w = max(360, stereo.shape[0] // 2)
        top = self._plot_panel("3D x-z projection", snap.poses_xyz, snap.vo_points_xyz, snap.map_points_xyz, panel_w, stereo.shape[0] // 2, (0, 2))
        bottom = self._plot_panel("top-down x-y", snap.poses_xyz, snap.vo_points_xyz, snap.map_points_xyz, panel_w, stereo.shape[0] - top.shape[0], (0, 1))
        side = np.concatenate([top, bottom], axis=0)
        canvas = np.concatenate([stereo, side], axis=1)
        ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return
        with self.cond:
            self.jpeg = encoded.tobytes()
            self.cond.notify_all()

    def close(self) -> None:
        self.running = False
        with self.cond:
            self.cond.notify_all()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.render_thread is not None:
            self.render_thread.join(timeout=2.0)


class RerunImageDisplay:
    def __init__(self, application_id: str, every: int) -> None:
        self.application_id = application_id
        self.every = max(every, 1)
        self.rr = None

    def start(self) -> None:
        try:
            import rerun as rr
        except ImportError as exc:
            raise RuntimeError("Rerun is not installed in this environment; install rerun-sdk or run without --useRR.") from exc
        self.rr = rr
        rr.init(self.application_id, spawn=False)
        rr.spawn(connect=True)
        Logger.write("info", "Rerun image-only UI enabled.")

    def update(self, pair: StereoPair, frame_idx: int) -> None:
        if self.rr is None or frame_idx % self.every != 0:
            return
        self.rr.set_time_sequence("frame_idx", frame_idx)
        self.rr.log("/world/macvo/cam_left", self.rr.Image(pair.left_rgb).compress())
        self.rr.log("/world/macvo/cam_right", self.rr.Image(pair.right_rgb).compress())


class RerunLiveDisplay:
    def __init__(
        self,
        application_id: str,
        every: int,
        max_points: int,
        cov_mode: str,
        log_image: bool,
        save_path: str | None,
        fixed_bounds: float,
        camera_centered: bool,
        local_radius: float,
        trail_frames: int,
        map_chunks: bool,
    ) -> None:
        self.application_id = application_id
        self.every = max(every, 1)
        self.max_points = max(max_points, 0)
        self.cov_mode = cov_mode
        self.log_image = log_image
        self.save_path = save_path
        self.fixed_bounds = max(fixed_bounds, 0.0)
        self.camera_centered = camera_centered
        self.local_radius = max(local_radius, 0.0)
        self.trail_frames = max(trail_frames, 0)
        self.map_chunks = map_chunks
        self.rr = None
        self.rr_plt = None
        self.logged_map_frames: set[int] = set()

    @staticmethod
    def _limit_bundle(bundle, max_points: int):
        if max_points <= 0 or len(bundle) <= max_points:
            return bundle
        idx = torch.linspace(0, len(bundle) - 1, max_points, dtype=torch.long)
        return bundle[idx]

    @staticmethod
    def _limit_tensors(max_points: int, *values: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        first = next((value for value in values if value is not None), None)
        if first is None or max_points <= 0 or first.size(0) <= max_points:
            return values
        idx = torch.linspace(0, first.size(0) - 1, max_points, dtype=torch.long, device=first.device)
        return tuple(value[idx.to(value.device)] if value is not None else None for value in values)

    def _log_highlight_trajectory(self, path: str, poses: torch.Tensor, bounds: float | None = None) -> None:
        if self.rr is None or poses.size(0) < 2:
            return
        xyz_t = poses[:, :3].detach().cpu()
        if bounds is not None and bounds > 0.0:
            in_bounds = (xyz_t.abs() <= bounds).all(dim=1)
            segment_mask = in_bounds[:-1] & in_bounds[1:]
            if not bool(segment_mask.any()):
                self.rr.log(path, self.rr.Clear(recursive=True))
                return
            xyz = xyz_t.numpy()
            segments = np.stack([xyz[:-1], xyz[1:]], axis=1)[segment_mask.numpy()]
        else:
            xyz = xyz_t.numpy()
            segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
        self.rr.log(
            path,
            self.rr.LineStrips3D(
                segments,
                colors=[[255, 218, 0]],
                radii=self.rr.Radius.ui_points(1.5),
            ),
        )

    def _log_clipped_points(
        self,
        path: str,
        position: torch.Tensor,
        color: torch.Tensor | None,
        cov: torch.Tensor | None,
        bounds: float,
    ) -> None:
        if self.rr is None or self.rr_plt is None:
            return

        pos = position.detach().cpu()
        col = color.detach().cpu() if color is not None else None
        covariance = cov.detach().cpu() if cov is not None else None

        if bounds > 0.0 and pos.size(0) > 0:
            mask = (pos.abs() <= bounds).all(dim=1)
            pos = pos[mask]
            col = col[mask] if col is not None else None
            covariance = covariance[mask] if covariance is not None else None

        pos, col, covariance = self._limit_tensors(self.max_points, pos, col, covariance)
        if pos is None or pos.size(0) == 0:
            self.rr.log(path, self.rr.Clear(recursive=True))
            return

        self.rr_plt.log_points(path, pos, col, covariance, self.cov_mode)

    def _is_inside_fixed_world(self, xyz: torch.Tensor) -> bool:
        if self.fixed_bounds <= 0.0:
            return True
        return bool((xyz.detach().cpu().abs() <= self.fixed_bounds).all().item())

    def start(self) -> None:
        try:
            import rerun as rr
            from Utility.Visualize import rr_plt
        except ImportError as exc:
            raise RuntimeError("Rerun is not installed in this environment; install rerun-sdk or run without --useRR.") from exc

        self.rr = rr
        self.rr_plt = rr_plt
        rr_plt.default_mode = "rerun"
        rr.init(self.application_id, spawn=False)
        if self.save_path is not None:
            rr.save(self.save_path)
            Logger.write("info", f"Rerun recording enabled: {self.save_path}")
        else:
            rr.spawn(connect=True)
            Logger.write("info", "Rerun live UI enabled.")
        self._send_blueprint()
        rr.log("/", rr.ViewCoordinates(xyz=rr.ViewCoordinates.FRD), static=True)
        rr.log(
            "/world/origin_axes",
            rr.Arrows3D(
                origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                vectors=[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
                colors=[[230, 40, 40], [40, 200, 70], [60, 120, 240]],
                radii=[0.01, 0.01, 0.01],
                labels=["x", "y", "z"],
            ),
            static=True,
        )
        rr.log(
            "/world/origin",
            rr.Points3D([[0.0, 0.0, 0.0]], radii=[0.04], colors=[[255, 255, 255]], labels=["origin"]),
            static=True,
        )
        rr.log("/world_fixed", rr.ViewCoordinates(xyz=rr.ViewCoordinates.FRD), static=True)
        rr.log(
            "/world_fixed/origin_axes",
            rr.Arrows3D(
                origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                vectors=[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
                colors=[[230, 40, 40], [40, 200, 70], [60, 120, 240]],
                radii=[0.01, 0.01, 0.01],
                labels=["x", "y", "z"],
            ),
            static=True,
        )
        rr.log(
            "/world_fixed/origin",
            rr.Points3D([[0.0, 0.0, 0.0]], radii=[0.04], colors=[[255, 255, 255]], labels=["origin"]),
            static=True,
        )
        if self.fixed_bounds > 0.0:
            size = float(self.fixed_bounds) * 2.0
            rr.log(
                "/world/fixed_view_bounds",
                rr.Boxes3D(
                    centers=[[0.0, 0.0, 0.0]],
                    sizes=[[size, size, size]],
                    colors=[[120, 120, 120, 18]],
                    radii=[0.002],
                ),
                static=True,
            )
            rr.log(
                "/world_fixed/fixed_view_bounds",
                rr.Boxes3D(
                    centers=[[0.0, 0.0, 0.0]],
                    sizes=[[size, size, size]],
                    colors=[[120, 120, 120, 24]],
                    radii=[0.002],
                ),
                static=True,
            )
            Logger.write("info", f"Rerun fixed 3D view bounds: +/-{self.fixed_bounds:.1f} m")
        if self.camera_centered:
            rr.log("/camera_centered", rr.ViewCoordinates(xyz=rr.ViewCoordinates.FRD), static=True)
            if self.local_radius > 0.0:
                size = self.local_radius * 2.0
                rr.log(
                    "/camera_centered/local_view_bounds",
                    rr.Boxes3D(
                        centers=[[0.0, 0.0, 0.0]],
                        sizes=[[size, size, size]],
                        colors=[[120, 120, 120, 18]],
                        radii=[0.002],
                    ),
                    static=True,
                )
            rr.log(
                "/camera_centered/current_camera",
                rr.Arrows3D(
                    origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    vectors=[[0.35, 0.0, 0.0], [0.0, 0.35, 0.0], [0.0, 0.0, 0.35]],
                    colors=[[230, 40, 40], [40, 200, 70], [60, 120, 240]],
                    radii=[0.012, 0.012, 0.012],
                    labels=["cam x", "cam y", "cam z"],
                ),
                static=True,
            )
            Logger.write(
                "info",
                f"Rerun camera-centered visualization enabled at /camera_centered "
                f"(local_radius={self.local_radius:.2f}m, trail_frames={self.trail_frames})",
            )

    def _send_blueprint(self) -> None:
        if self.rr is None or not self.camera_centered:
            return
        try:
            import rerun.blueprint as rrb

            self.rr.send_blueprint(
                rrb.Blueprint(
                    rrb.Horizontal(
                        rrb.Spatial3DView(origin="/camera_centered", contents="/camera_centered/**", name="Camera follow view"),
                        rrb.Vertical(
                            rrb.Spatial3DView(origin="/world_fixed", contents="/world_fixed/**", name="World view"),
                            rrb.Spatial2DView(origin="/world/macvo/cam_left", name="Left image"),
                            row_shares=[2, 1],
                        ),
                        column_shares=[3, 1],
                    ),
                    collapse_panels=True,
                )
            )
        except Exception as exc:
            Logger.write("warn", f"Unable to set Rerun layout: {exc}")

    def update(self, frame: StereoFrame, system: MACVO, processed: int) -> None:
        if processed % self.every != 0:
            return
        if self.rr is None or self.rr_plt is None:
            return
        if len(system.graph.frames) == 0:
            return

        rr = self.rr
        rr_plt = self.rr_plt
        rr.set_time_sequence("frame_idx", frame.frame_idx)

        try:
            if len(system.graph.frames) > 1:
                all_poses = system.graph.frames.data["pose"].tensor
                rr_plt.log_trajectory("/world/est", pp.SE3(all_poses))
                self._log_highlight_trajectory("/world/est_highlight", all_poses)
                self._log_highlight_trajectory("/world_fixed/est_highlight", all_poses, self.fixed_bounds)

            latest_frame = system.graph.frames[-1:]
            latest_pose = pp.SE3(system.graph.frames.data["pose"][-1])
            latest_xyz = latest_pose.translation().detach().cpu()
            rr_plt.log_camera("/world/macvo/cam_left", latest_pose, frame.stereo.frame_K)
            rr.log(
                "/world/macvo/current_axes",
                rr.Arrows3D(
                    origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    vectors=[[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]],
                    colors=[[230, 40, 40], [40, 200, 70], [60, 120, 240]],
                    radii=rr.Radius.ui_points(2.0),
                ),
            )
            if self.log_image:
                rr_plt.log_image("/world/macvo/cam_left", frame.stereo.imageL[0].permute(1, 2, 0))

            if self._is_inside_fixed_world(latest_xyz):
                rr_plt.log_camera("/world_fixed/macvo/cam_left", latest_pose, frame.stereo.frame_K)
                rr.log(
                    "/world_fixed/macvo/current_axes",
                    rr.Arrows3D(
                        origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        vectors=[[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]],
                        colors=[[230, 40, 40], [40, 200, 70], [60, 120, 240]],
                        radii=rr.Radius.ui_points(2.0),
                    ),
                )
            else:
                rr.log("/world_fixed/macvo", rr.Clear(recursive=True))

            if self.camera_centered:
                local_pose_tensor = system.graph.frames.data["pose"][-1].detach().cpu().clone()
                local_pose_tensor[:3] = 0.0
                rr_plt.log_camera("/camera_centered/macvo/cam_left", pp.SE3(local_pose_tensor), frame.stereo.frame_K)
                rr.log(
                    "/camera_centered/macvo/current_axes",
                    rr.Arrows3D(
                        origins=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        vectors=[[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]],
                        colors=[[230, 40, 40], [40, 200, 70], [60, 120, 240]],
                        radii=rr.Radius.ui_points(2.0),
                    ),
                )
                if self.log_image:
                    rr_plt.log_image("/camera_centered/macvo/cam_left", frame.stereo.imageL[0].permute(1, 2, 0))

            if self.camera_centered and len(system.graph.frames) > 1:
                poses = system.graph.frames.data["pose"].tensor.detach().cpu().clone()
                if self.trail_frames > 0 and poses.size(0) > self.trail_frames:
                    poses = poses[-self.trail_frames :]
                poses[:, :3] -= latest_xyz
                rr_plt.log_trajectory("/camera_centered/relative_trajectory", pp.SE3(poses))
                self._log_highlight_trajectory("/camera_centered/relative_trajectory_highlight", poses)

            latest_frame_idx = int(latest_frame.index[0].item())
            frame_map_points = system.graph.get_frame2map(latest_frame)
            if self.map_chunks and len(frame_map_points) > 0 and latest_frame_idx not in self.logged_map_frames:
                map_chunk = self._limit_bundle(frame_map_points, self.max_points)
                rr_plt.log_points(
                    f"/world/map_chunks/frame_{latest_frame_idx:06d}",
                    map_chunk.data["pos_Tw"],
                    map_chunk.data["color"],
                    map_chunk.data["cov_Tw"],
                    self.cov_mode,
                )
                self._log_clipped_points(
                    f"/world_fixed/map_chunks/frame_{latest_frame_idx:06d}",
                    map_chunk.data["pos_Tw"],
                    map_chunk.data["color"],
                    map_chunk.data["cov_Tw"],
                    self.fixed_bounds,
                )
                self.logged_map_frames.add(latest_frame_idx)

            if len(system.graph.map_points) > 0:
                if self.camera_centered:
                    map_all = system.graph.map_points
                    map_pos_local = map_all.data["pos_Tw"].detach().cpu() - latest_xyz
                    map_color = map_all.data["color"].detach().cpu()
                    map_cov = map_all.data["cov_Tw"].detach().cpu()
                    if self.local_radius > 0.0:
                        local_mask = torch.linalg.norm(map_pos_local, dim=1) <= self.local_radius
                        map_pos_local = map_pos_local[local_mask]
                        map_color = map_color[local_mask]
                        map_cov = map_cov[local_mask]
                    map_pos_local, map_color, map_cov = self._limit_tensors(self.max_points, map_pos_local, map_color, map_cov)
                    if map_pos_local.size(0) > 0:
                        rr_plt.log_points(
                            "/camera_centered/point_cloud",
                            map_pos_local,
                            map_color,
                            map_cov,
                            self.cov_mode,
                        )

            vo_points = system.graph.get_match2point(system.graph.get_frame2match(latest_frame))
            vo_points = self._limit_bundle(vo_points, self.max_points)
            if len(vo_points) > 0:
                rr_plt.log_points(
                    "/world/vo_tracking",
                    vo_points.data["pos_Tw"],
                    vo_points.data["color"],
                    vo_points.data["cov_Tw"],
                    self.cov_mode,
                )
                self._log_clipped_points(
                    "/world_fixed/vo_tracking",
                    vo_points.data["pos_Tw"],
                    vo_points.data["color"],
                    vo_points.data["cov_Tw"],
                    self.fixed_bounds,
                )
                if self.camera_centered:
                    vo_all = system.graph.get_match2point(system.graph.get_frame2match(latest_frame))
                    vo_pos_local = vo_all.data["pos_Tw"].detach().cpu() - latest_xyz
                    vo_color = vo_all.data["color"].detach().cpu()
                    vo_cov = vo_all.data["cov_Tw"].detach().cpu()
                    if self.local_radius > 0.0:
                        local_mask = torch.linalg.norm(vo_pos_local, dim=1) <= self.local_radius
                        vo_pos_local = vo_pos_local[local_mask]
                        vo_color = vo_color[local_mask]
                        vo_cov = vo_cov[local_mask]
                    vo_pos_local, vo_color, vo_cov = self._limit_tensors(self.max_points, vo_pos_local, vo_color, vo_cov)
                    if vo_pos_local.size(0) > 0:
                        rr_plt.log_points(
                            "/camera_centered/vo_tracking",
                            vo_pos_local,
                            vo_color,
                            vo_cov,
                            self.cov_mode,
                        )
        except Exception as exc:
            Logger.write("warn", f"Rerun visualization skipped for frame {processed}: {exc}")


def save_system_outputs(system: MACVO, saveto: Sandbox) -> None:
    global_map = system.get_map()
    if len(global_map.frames) == 0:
        Logger.write("warn", "No frame in MAC-VO map; skip saving outputs.")
        return

    sensor_poses = pp.SE3(global_map.frames.data["pose"].tensor)
    T_BS = pp.SE3(global_map.frames.data["T_BS"].tensor)
    body_poses = (T_BS @ sensor_poses @ T_BS.Inv()).tensor().cpu().numpy()
    time_ns = global_map.frames.data["time_ns"].tensor.cpu().numpy()[:, np.newaxis]

    np.save(saveto.path("poses.npy"), np.concatenate([time_ns, body_poses], axis=-1))
    np.savez_compressed(saveto.path("tensor_map.npz"), **global_map.serialize())
    Logger.write("info", f"Saved trajectory and map to {saveto.folder}")


def terminate_and_save(system: MACVO, exp_space: Sandbox) -> bool:
    num_frames = len(system.graph.frames)
    if num_frames == 0:
        Logger.write("warn", "No processed frame; skip saving outputs.")
        return True
    if num_frames < 3:
        Logger.write("warn", f"Only {num_frames} processed frame(s); skip MAC-VO postprocess and save raw partial map.")
        save_system_outputs(system, exp_space)
        return True
    system.terminate()
    save_system_outputs(system, exp_space)
    return True


def format_stats(stats: dict[str, float | str]) -> str:
    keys = [
        "pairs",
        "left_fps",
        "right_fps",
        "capture_fps",
        "interval_mean_ms",
        "interval_max_ms",
        "skew_mean_ms",
        "skew_max_ms",
        "errors",
    ]
    parts = []
    for key in keys:
        if key in stats:
            value = stats[key]
            if isinstance(value, (float, int)):
                parts.append(f"{key}={value:.3f}")
            else:
                parts.append(f"{key}={value}")
    if "pairing_mode" in stats:
        parts.append(f"pairing_mode={stats['pairing_mode']}")
    return ", ".join(parts)


def format_pose_status(system: MACVO) -> str:
    try:
        if len(system.graph.frames) == 0:
            return "pose=N/A"
        xyz = system.graph.frames.data["pose"].tensor[-1, :3].detach().cpu().numpy()
        dist = float(np.linalg.norm(xyz))
        return f"pose=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f})m, origin_dist={dist:.3f}m"
    except Exception:
        return "pose=N/A"


def sample_visual_state(system: MACVO, max_points: int) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    poses_xyz: np.ndarray | None = None
    vo_points_xyz: np.ndarray | None = None
    map_points_xyz: np.ndarray | None = None

    try:
        if len(system.graph.frames) > 0:
            poses_xyz = system.graph.frames.data["pose"].tensor.detach().cpu().numpy()[:, :3].astype(np.float32)
    except Exception:
        poses_xyz = None

    try:
        if len(system.graph.points) > 0:
            pts = system.graph.points.data["pos_Tw"].tensor.detach().cpu().numpy().astype(np.float32)
            if pts.shape[0] > max_points:
                pts = pts[-max_points:]
            vo_points_xyz = pts
    except Exception:
        vo_points_xyz = None

    try:
        if len(system.graph.map_points) > 0:
            pts = system.graph.map_points.data["pos_Tw"].tensor.detach().cpu().numpy().astype(np.float32)
            if pts.shape[0] > max_points:
                pts = pts[-max_points:]
            map_points_xyz = pts
    except Exception:
        map_points_xyz = None

    return poses_xyz, vo_points_xyz, map_points_xyz


def run_realtime_from_reader(
    args: argparse.Namespace,
    reader,
    rectifier: StereoRectifier,
    vo_width: int,
    vo_height: int,
    project_suffix: str,
) -> None:
    reader.start()

    last_pair_id = 0
    live_display = None
    web_display = None
    rr_display = None
    sequence_recorder = None

    try:
        for _ in range(max(args.warmup_pairs, 0)):
            pair = reader.wait_for_newer(last_pair_id, args.wait_timeout)
            if pair is None:
                raise RuntimeError("Timed out while warming up cameras")
            last_pair_id = pair.pair_id

        Logger.write("info", f"Camera warmup done: {format_stats(reader.stats())}")

        if args.capture_only:
            capture_result_dir: Path | None = None
            if args.record_sequence:
                capture_result_dir = Path(args.record_dir) if args.record_dir else Path(args.resultRoot) / time.strftime("%m_%d_%H%M%S")
                sequence_recorder = StereoSequenceRecorder(capture_result_dir / "stereo_sequence", args.swap)
                preview_dir = capture_result_dir / "preview"
                Logger.write("info", f"Capture-only recording to {capture_result_dir}")
            else:
                preview_dir = Path(args.preview_dir) if args.preview_dir is not None else None

            rr_display = RerunImageDisplay(f"MACVO-CaptureOnly@{project_suffix}", args.rr_every) if args.useRR else None
            if rr_display is not None:
                rr_display.start()

            max_skew = 0.0
            captured = 0
            next_capture_time = time.monotonic()
            try:
                while args.max_frames <= 0 or captured < args.max_frames:
                    pair = reader.wait_for_newer(last_pair_id, args.wait_timeout)
                    if pair is None:
                        raise RuntimeError("Timed out while waiting for stereo pair")
                    last_pair_id = pair.pair_id
                    max_skew = max(max_skew, pair.software_skew_ms)
                    now = time.monotonic()
                    if args.vo_fps > 0 and now < next_capture_time:
                        continue
                    next_capture_time = max(next_capture_time + 1.0 / args.vo_fps, now)
                    pair_for_record = rectifier.rectify_pair(pair, args.swap)
                    if sequence_recorder is not None:
                        sequence_recorder.write(pair_for_record, captured)
                    if rr_display is not None:
                        rr_display.update(pair_for_record, captured)
                    if preview_dir is not None and (captured == 0 or (args.save_preview_every > 0 and captured % args.save_preview_every == 0)):
                        save_pair_preview(pair_for_record, preview_dir, f"pair_{captured:04d}", False)
                    captured += 1
                    if captured % max(args.status_every, 1) == 0:
                        Logger.write("info", f"Capture-only frame={captured}, capture=({format_stats(reader.stats())})")
            except KeyboardInterrupt:
                Logger.write("warn", f"Interrupted by user after {captured} capture-only frames; saving sequence.")
            if sequence_recorder is not None:
                sequence_recorder.close()
            Logger.write("info", f"Capture-only finished: {format_stats(reader.stats())}")
            if capture_result_dir is not None:
                Logger.write("info", f"Saved capture-only stereo sequence to {capture_result_dir}")
            if max_skew > args.max_software_skew_ms:
                Logger.write("warn", f"Software timestamp skew exceeded threshold: max={max_skew:.3f} ms")
            return

        cfg, cfg_dict = load_config(Path(args.odom))
        odomcfg, odomcfg_dict = cfg.Odometry, cfg_dict["Odometry"]
        project_name = f"{odomcfg.name}@{project_suffix}"
        exp_space = Sandbox.create(Path(args.resultRoot), project_name)
        camera_config = {
            "left": args.left,
            "right": args.right,
            "swap": args.swap,
            "width": args.width,
            "height": args.height,
            "vo_width": vo_width,
            "vo_height": vo_height,
            "camera_fps": args.camera_fps,
            "vo_fps": args.vo_fps,
            "pairing_delay_ms": args.pairing_delay_ms,
            "baseline": args.baseline,
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
            "fourcc": args.fourcc,
            "rectify_maps": args.rectify_maps,
            "drop_high_skew_vo": args.drop_high_skew_vo,
            "vo_max_skew_ms": args.vo_max_skew_ms,
            "record_sequence": args.record_sequence,
            "record_dir": args.record_dir,
        }
        for key in ("device", "raw_width", "raw_height", "eye_width", "eye_height", "left_x", "right_x"):
            if hasattr(args, key):
                camera_config[key] = getattr(args, key)
        exp_space.config = {
            "Project": project_name,
            "Odometry": odomcfg_dict,
            "Camera": camera_config,
        }

        Timer.setup(active=args.timing)
        system = MACVO[StereoFrame].from_config(asNamespace(exp_space.config))

        K = torch.tensor(
            [[[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
        )
        K_vo = scale_intrinsic(K, args.width, args.height, vo_width, vo_height)
        T_BS = pp.identity_SE3(1, dtype=torch.float64)

        preview_dir = exp_space.path("preview")
        sequence_recorder = StereoSequenceRecorder(
            Path(args.record_dir) if args.record_dir else exp_space.path("stereo_sequence"),
            args.swap,
        ) if args.record_sequence else None
        live_display = LiveDisplay(args.display_scale) if args.display else None
        web_display = WebDisplay(args.web_host, args.web_port, args.display_scale, args.web_fps, args.web_max_points) if args.web_display else None
        if web_display is not None:
            web_display.start()
        rr_display = RerunLiveDisplay(
            project_name,
            args.rr_every,
            args.rr_max_points,
            args.rr_cov_mode,
            not args.rr_no_image,
            args.rr_save,
            args.rr_fixed_bounds,
            args.rr_camera_centered,
            args.rr_local_radius,
            args.rr_trail_frames,
            args.rr_map_chunks,
        ) if args.useRR else None
        if rr_display is not None:
            rr_display.start()

        processed = 0
        skipped_pairs = 0
        next_vo_time = time.monotonic()
        last_display_update_pair_id = -1
        poses_xyz: np.ndarray | None = None
        vo_points_xyz: np.ndarray | None = None
        map_points_xyz: np.ndarray | None = None
        last_step_ms = 0.0
        max_skew = 0.0
        rejected_skew_pairs = 0
        low_skew_timeout_count = 0
        last_low_skew_warn = 0.0
        terminated = False

        try:
            while args.max_frames <= 0 or processed < args.max_frames:
                now = time.monotonic()
                if now < next_vo_time:
                    pair = reader.wait_for_newer(last_pair_id, timeout=min(args.wait_timeout, max(0.001, next_vo_time - now)))
                    if pair is not None:
                        skipped_pairs += max(0, pair.pair_id - last_pair_id - 1)
                        last_pair_id = pair.pair_id
                        max_skew = max(max_skew, pair.software_skew_ms)
                        pair_for_display = rectifier.rectify_pair(pair, args.swap)
                        if web_display is not None and pair_for_display.pair_id != last_display_update_pair_id:
                            web_display.update(
                                pair_for_display,
                                poses_xyz,
                                vo_points_xyz,
                                map_points_xyz,
                                processed,
                                last_step_ms,
                                skipped_pairs,
                                reader.stats(),
                            )
                            last_display_update_pair_id = pair_for_display.pair_id
                    continue

                next_vo_time = max(next_vo_time + 1.0 / args.vo_fps, time.monotonic())
                vo_deadline = time.monotonic() + args.wait_timeout
                pair_for_vo_source: StereoPair | None = None
                while True:
                    pair = reader.wait_for_newer(last_pair_id, max(0.001, vo_deadline - time.monotonic()))
                    if pair is None:
                        if args.drop_high_skew_vo:
                            low_skew_timeout_count += 1
                            now_warn = time.monotonic()
                            if now_warn - last_low_skew_warn > 2.0:
                                Logger.write(
                                    "warn",
                                    f"No stereo pair passed skew gate <= {args.vo_max_skew_ms:.3f} ms "
                                    f"within {args.wait_timeout:.1f}s; continuing. "
                                    f"skew_rejected={rejected_skew_pairs}, timeouts={low_skew_timeout_count}",
                                )
                                last_low_skew_warn = now_warn
                            next_vo_time = time.monotonic()
                            break
                        raise RuntimeError("Timed out while waiting for stereo pair")

                    skipped_pairs += max(0, pair.pair_id - last_pair_id - 1)
                    last_pair_id = pair.pair_id
                    max_skew = max(max_skew, pair.software_skew_ms)
                    if args.drop_high_skew_vo and pair.software_skew_ms > args.vo_max_skew_ms:
                        rejected_skew_pairs += 1
                        pair_for_display = rectifier.rectify_pair(pair, args.swap)
                        if web_display is not None and pair_for_display.pair_id != last_display_update_pair_id:
                            web_display.update(
                                pair_for_display,
                                poses_xyz,
                                vo_points_xyz,
                                map_points_xyz,
                                processed,
                                last_step_ms,
                                skipped_pairs,
                                reader.stats(),
                            )
                            last_display_update_pair_id = pair_for_display.pair_id
                        if time.monotonic() >= vo_deadline:
                            low_skew_timeout_count += 1
                            now_warn = time.monotonic()
                            if now_warn - last_low_skew_warn > 2.0:
                                Logger.write(
                                    "warn",
                                    f"No stereo pair passed skew gate <= {args.vo_max_skew_ms:.3f} ms "
                                    f"within {args.wait_timeout:.1f}s; continuing. "
                                    f"skew_rejected={rejected_skew_pairs}, timeouts={low_skew_timeout_count}",
                                )
                                last_low_skew_warn = now_warn
                            next_vo_time = time.monotonic()
                            break
                        continue
                    pair_for_vo_source = pair
                    break

                if pair_for_vo_source is None:
                    continue

                pair_for_vo = rectifier.rectify_pair(pair_for_vo_source, args.swap)
                if processed == 0:
                    save_pair_preview(pair_for_vo, preview_dir, "first_pair", args.swap)
                elif args.save_preview_every > 0 and processed % args.save_preview_every == 0:
                    save_pair_preview(pair_for_vo, preview_dir, f"pair_{processed:04d}", args.swap)
                if sequence_recorder is not None:
                    sequence_recorder.write(pair_for_vo, processed)

                pair_for_vo_net = resize_pair(pair_for_vo, vo_width, vo_height)
                frame = make_stereo_frame(pair_for_vo_net, processed, K_vo, args.baseline, T_BS, args.swap)
                step_start = time.perf_counter()
                system.run(frame)
                step_ms = (time.perf_counter() - step_start) * 1000.0
                last_step_ms = step_ms
                processed += 1
                poses_xyz, vo_points_xyz, map_points_xyz = sample_visual_state(system, args.web_max_points)

                if processed % max(args.status_every, 1) == 0:
                    Logger.write(
                        "info",
                        f"VO frame={processed}, step={step_ms:.1f} ms, "
                        f"{format_pose_status(system)}, "
                        f"skipped_pairs={skipped_pairs}, skew_rejected={rejected_skew_pairs}, "
                        f"capture=({format_stats(reader.stats())})",
                    )
                if live_display is not None and processed % max(args.display_every, 1) == 0:
                    if not live_display.update(pair_for_vo, system, processed, step_ms, skipped_pairs, reader.stats()):
                        raise KeyboardInterrupt
                if web_display is not None and processed % max(args.display_every, 1) == 0:
                    web_display.update(pair_for_vo, poses_xyz, vo_points_xyz, map_points_xyz, processed, step_ms, skipped_pairs, reader.stats())
                if rr_display is not None:
                    rr_display.update(frame, system, processed)

            terminated = terminate_and_save(system, exp_space)
            Timer.report()
            Timer.save_elapsed(exp_space.path("elapsed_time.json"))

            if max_skew > args.max_software_skew_ms:
                Logger.write("warn", f"Software timestamp skew exceeded threshold: max={max_skew:.3f} ms")
            Logger.write("info", f"Finished {processed} VO frames, skipped {skipped_pairs} camera pairs.")
        except KeyboardInterrupt:
            Logger.write("warn", "Interrupted by user; saving partial result.")
            if processed > 0 and not terminated:
                terminated = terminate_and_save(system, exp_space)
        except Exception:
            Logger.show_exception()
            if processed > 0 and not terminated:
                terminate_and_save(system, exp_space)
            raise
    finally:
        if live_display is not None:
            live_display.close()
        if web_display is not None:
            web_display.close()
        if sequence_recorder is not None:
            sequence_recorder.close()
        reader.stop()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MAC-VO on a live OV2710 stereo pair.")
    parser.add_argument("--odom", default="Config/Experiment/MACVO/MACVO_OV2710_Realtime.yaml")
    parser.add_argument("--left", default="/dev/video4")
    parser.add_argument("--right", default="/dev/video6")
    parser.add_argument("--swap", action="store_true", help="Swap left/right images before feeding MAC-VO.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--vo-width", type=int, default=None, help="Resize rectified images to this width before MAC-VO.")
    parser.add_argument("--vo-height", type=int, default=None, help="Resize rectified images to this height before MAC-VO.")
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--vo-fps", type=float, default=10.0)
    parser.add_argument("--pairing-delay-ms", type=float, default=10.0, help="Delay pairing by this many ms to choose the nearest timestamp match.")
    parser.add_argument("--fourcc", default="YUYV")
    parser.add_argument("--baseline", type=float, default=0.10)
    parser.add_argument("--fx", type=float, default=500.0)
    parser.add_argument("--fy", type=float, default=500.0)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--rectify-maps", default=None)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until Ctrl-C.")
    parser.add_argument("--warmup-pairs", type=int, default=15)
    parser.add_argument("--wait-timeout", type=float, default=2.0)
    parser.add_argument("--max-software-skew-ms", type=float, default=8.0)
    parser.add_argument("--drop-high-skew-vo", action="store_true", help="Skip stereo pairs whose software skew is above --vo-max-skew-ms.")
    parser.add_argument("--vo-max-skew-ms", type=float, default=None, help="VO skew gate used with --drop-high-skew-vo. Defaults to --max-software-skew-ms.")
    parser.add_argument("--status-every", type=int, default=5)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-every", type=int, default=1)
    parser.add_argument("--display-scale", type=float, default=0.7)
    parser.add_argument("--web-display", action="store_true")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8765)
    parser.add_argument("--web-fps", type=float, default=10.0)
    parser.add_argument("--web-max-points", type=int, default=800)
    parser.add_argument("--useRR", action="store_true", help="Use the project's Rerun UI for live 3D visualization.")
    parser.add_argument("--rr-every", type=int, default=1, help="Send one Rerun update every N processed VO frames.")
    parser.add_argument("--rr-max-points", type=int, default=200, help="Limit VO/map points sent to Rerun per update.")
    parser.add_argument("--rr-cov-mode", choices=("none", "axis", "sphere", "color"), default="none")
    parser.add_argument("--rr-no-image", action="store_true", help="Do not stream camera images to Rerun.")
    parser.add_argument("--rr-save", default=None, help="Save a .rrd recording instead of opening the live Rerun UI.")
    parser.add_argument("--rr-fixed-bounds", type=float, default=2.5, help="Add a fixed +/-meter 3D bounds box to stop Rerun auto-fit zooming.")
    parser.add_argument("--rr-camera-centered", action=argparse.BooleanOptionalAction, default=True, help="Also log a camera-centered view where the current camera is always at the origin.")
    parser.add_argument("--rr-local-radius", type=float, default=1.5, help="Camera-centered view radius in meters; 0 disables local point filtering.")
    parser.add_argument("--rr-trail-frames", type=int, default=80, help="Number of recent poses shown in the camera-centered trajectory; 0 shows all.")
    parser.add_argument("--rr-map-chunks", action=argparse.BooleanOptionalAction, default=True, help="Log per-frame mapping chunks so the Rerun map grows over time.")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--save-preview-every", type=int, default=0)
    parser.add_argument("--record-sequence", action="store_true", help="Save every rectified stereo pair used by VO.")
    parser.add_argument("--record-dir", default=None, help="Defaults to RESULT/stereo_sequence.")
    parser.add_argument("--resultRoot", default="./Results_ov2710_live")
    parser.add_argument("--timing", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.cx = float(args.width / 2.0 if args.cx is None else args.cx)
    args.cy = float(args.height / 2.0 if args.cy is None else args.cy)

    if args.vo_fps <= 0:
        raise ValueError("--vo-fps must be positive")
    if args.vo_max_skew_ms is None:
        args.vo_max_skew_ms = args.max_software_skew_ms
    if args.vo_max_skew_ms <= 0:
        raise ValueError("--vo-max-skew-ms must be positive")
    if len(args.fourcc) not in (0, 4):
        raise ValueError("--fourcc must be empty or exactly four characters")

    Logger.write(
        "warn",
        "Using camera intrinsics from command line unless --rectify-maps is provided.",
    )
    Logger.write(
        "warn",
        "OpenCV/V4L2 software timing can only check host-side skew; it cannot prove hardware sync.",
    )

    rectifier = StereoRectifier(args.rectify_maps)
    if rectifier.enabled:
        assert rectifier.K is not None and rectifier.baseline is not None
        args.fx = float(rectifier.K[0, 0, 0])
        args.fy = float(rectifier.K[0, 1, 1])
        args.cx = float(rectifier.K[0, 0, 2])
        args.cy = float(rectifier.K[0, 1, 2])
        args.baseline = rectifier.baseline

    vo_width = args.width if args.vo_width is None else args.vo_width
    vo_height = args.height if args.vo_height is None else args.vo_height
    if vo_width <= 0 or vo_height <= 0:
        raise ValueError("--vo-width and --vo-height must be positive")
    if vo_width > args.width or vo_height > args.height:
        Logger.write(
            "warn",
            f"VO input is larger than capture ({vo_width}x{vo_height} vs {args.width}x{args.height}); this will not improve speed.",
        )
    Logger.write("info", f"Capture/display resolution: {args.width}x{args.height}; VO input resolution: {vo_width}x{vo_height}")

    reader = StereoCameraReader(
        args.left,
        args.right,
        args.width,
        args.height,
        args.camera_fps,
        args.fourcc,
        args.pairing_delay_ms,
    )
    run_realtime_from_reader(args, reader, rectifier, vo_width, vo_height, "OV2710-live")


if __name__ == "__main__":
    main()
