#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Scripts.AdHoc import Run_OV2710_Realtime as common
from Scripts.AdHoc.DECXIN3261V.Device import DECXIN_DEVICE_BY_ID, find_decxin_device


class SplitStereoCameraReader:
    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: float,
        fourcc: str,
        left_x: int,
        right_x: int,
        eye_width: int,
        eye_height: int,
        output_width: int,
        output_height: int,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.left_x = left_x
        self.right_x = right_x
        self.eye_width = eye_width
        self.eye_height = eye_height
        self.output_width = output_width
        self.output_height = output_height

        self.cap: cv2.VideoCapture | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.cond = threading.Condition()
        self.latest_pair: common.StereoPair | None = None
        self.pair_id = 0
        self.error_count = 0
        self.last_pair_time_ns: int | None = None
        self.pair_intervals_ms: deque[float] = deque(maxlen=300)
        self.read_spans_ms: deque[float] = deque(maxlen=300)

    def start(self) -> None:
        self.cap = common.open_camera(self.device, self.width, self.height, self.fps, self.fourcc)
        self.thread = threading.Thread(target=self._loop, name="DECXIN3261VCapture", daemon=True)
        self.thread.start()

    def _crop_eye(self, rgb: np.ndarray, x: int) -> np.ndarray:
        crop = rgb[: self.eye_height, x : x + self.eye_width]
        if crop.shape[0] <= 0 or crop.shape[1] <= 0:
            raise RuntimeError(
                f"Invalid DECXIN eye crop x={x}, eye={self.eye_width}x{self.eye_height}, "
                f"raw={rgb.shape[1]}x{rgb.shape[0]}"
            )

        src_h, src_w = crop.shape[:2]
        target_aspect = self.output_width / self.output_height
        src_aspect = src_w / src_h
        if abs(src_aspect - target_aspect) > 1e-6:
            if src_aspect > target_aspect:
                new_w = max(1, min(src_w, round(src_h * target_aspect)))
                x0 = (src_w - new_w) // 2
                crop = crop[:, x0 : x0 + new_w]
            else:
                new_h = max(1, min(src_h, round(src_w / target_aspect)))
                y0 = (src_h - new_h) // 2
                crop = crop[y0 : y0 + new_h, :]

        if crop.shape[:2] != (self.output_height, self.output_width):
            crop = cv2.resize(crop, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA)
        return crop

    def _loop(self) -> None:
        assert self.cap is not None
        while not self.stop_event.is_set():
            t0 = time.monotonic_ns()
            ok, frame_bgr = self.cap.read()
            t1 = time.monotonic_ns()
            if not ok or frame_bgr is None:
                with self.cond:
                    self.error_count += 1
                    self.cond.notify_all()
                time.sleep(0.005)
                continue

            if frame_bgr.shape[:2] != (self.height, self.width):
                frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)

            rgb = common.convert_to_rgb(frame_bgr)
            left = self._crop_eye(rgb, self.left_x)
            right = self._crop_eye(rgb, self.right_x)
            ts = (t0 + t1) // 2
            pair = common.StereoPair(
                pair_id=self.pair_id + 1,
                time_ns=ts,
                left_rgb=left,
                right_rgb=right,
                left_grab_mid_ns=ts,
                right_grab_mid_ns=ts,
                grab_span_ns=t1 - t0,
            )

            with self.cond:
                if self.last_pair_time_ns is not None:
                    self.pair_intervals_ms.append((pair.time_ns - self.last_pair_time_ns) / 1e6)
                self.last_pair_time_ns = pair.time_ns
                self.read_spans_ms.append((t1 - t0) / 1e6)
                self.pair_id = pair.pair_id
                self.latest_pair = pair
                self.cond.notify_all()

    def wait_for_newer(self, last_pair_id: int, timeout: float) -> common.StereoPair | None:
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
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()

    def stats(self) -> dict[str, float | str]:
        with self.cond:
            intervals = list(self.pair_intervals_ms)
            spans = list(self.read_spans_ms)
            pair_id = self.pair_id
            errors = self.error_count

        stats: dict[str, float | str] = {
            "pairs": float(pair_id),
            "errors": float(errors),
            "pairing_mode": "single_uvc_split",
            "skew_mean_ms": 0.0,
            "skew_max_ms": 0.0,
        }
        if intervals:
            mean_interval = statistics.fmean(intervals)
            stats["capture_fps"] = 1000.0 / mean_interval if mean_interval > 0 else 0.0
            stats["left_fps"] = stats["capture_fps"]
            stats["right_fps"] = stats["capture_fps"]
            stats["interval_mean_ms"] = mean_interval
            stats["interval_max_ms"] = max(intervals)
        if spans:
            stats["grab_span_mean_ms"] = statistics.fmean(spans)
            stats["grab_span_max_ms"] = max(spans)
        return stats


class GstSplitStereoCameraReader(SplitStereoCameraReader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.Gst = None
        self.pipeline = None
        self.left_sink = None
        self.right_sink = None
        self.bus = None

    def start(self) -> None:
        sys.path.append("/usr/lib/python3/dist-packages")
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as exc:
            raise RuntimeError(
                "GStreamer Python bindings are unavailable; use --capture-backend opencv."
            ) from exc

        Gst.init(None)
        self.Gst = Gst
        pipeline_desc = self._build_pipeline_desc()
        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.left_sink = self.pipeline.get_by_name("leftsink")
        self.right_sink = self.pipeline.get_by_name("rightsink")
        self.bus = self.pipeline.get_bus()

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            error = self._pop_bus_error()
            self.pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"Failed to start GStreamer camera pipeline: {error}")
        self.pipeline.get_state(5 * Gst.SECOND)
        common.Logger.write("info", f"GStreamer DECXIN capture enabled: {pipeline_desc}")
        self.thread = threading.Thread(target=self._loop, name="DECXIN3261VGstCapture", daemon=True)
        self.thread.start()

    def _build_pipeline_desc(self) -> str:
        left_crop = self._crop_borders(self.left_x)
        right_crop = self._crop_borders(self.right_x)
        fps_num = max(1, int(round(self.fps)))

        if self.fourcc.upper() in ("MJPG", "MJPEG"):
            source_caps = f"image/jpeg,width={self.width},height={self.height},framerate={fps_num}/1 ! jpegdec"
        elif self.fourcc.upper() in ("YUY2", "YUYV"):
            source_caps = f"video/x-raw,format=YUY2,width={self.width},height={self.height},framerate={fps_num}/1"
        else:
            raise ValueError(f"Unsupported GStreamer DECXIN fourcc: {self.fourcc}")

        def branch(name: str, crop: tuple[int, int, int, int]) -> str:
            left, right, top, bottom = crop
            return (
                f"t. ! queue leaky=downstream max-size-buffers=1 ! "
                f"videocrop left={left} right={right} top={top} bottom={bottom} ! "
                f"videoscale ! videoconvert ! "
                f"video/x-raw,format=RGB,width={self.output_width},height={self.output_height} ! "
                f"appsink name={name} drop=true max-buffers=1 sync=false emit-signals=false"
            )

        return (
            f"v4l2src device={self.device} do-timestamp=true ! {source_caps} ! tee name=t "
            f"{branch('leftsink', left_crop)} "
            f"{branch('rightsink', right_crop)}"
        )

    def _crop_borders(self, x: int) -> tuple[int, int, int, int]:
        src_w = self.eye_width
        src_h = self.eye_height
        target_aspect = self.output_width / self.output_height
        src_aspect = src_w / src_h
        inner_x = 0
        inner_y = 0
        crop_w = src_w
        crop_h = src_h
        if abs(src_aspect - target_aspect) > 1e-6:
            if src_aspect > target_aspect:
                crop_w = max(1, min(src_w, round(src_h * target_aspect)))
                inner_x = (src_w - crop_w) // 2
            else:
                crop_h = max(1, min(src_h, round(src_w / target_aspect)))
                inner_y = (src_h - crop_h) // 2

        left = x + inner_x
        right = self.width - (left + crop_w)
        top = inner_y
        bottom = self.height - (top + crop_h)
        if min(left, right, top, bottom) < 0:
            raise RuntimeError(
                f"Invalid DECXIN GStreamer crop borders {(left, right, top, bottom)} "
                f"for raw={self.width}x{self.height}"
            )
        return left, right, top, bottom

    def _pop_bus_error(self) -> str:
        if self.bus is None or self.Gst is None:
            return "unknown error"
        msg = self.bus.timed_pop_filtered(
            0,
            self.Gst.MessageType.ERROR | self.Gst.MessageType.WARNING | self.Gst.MessageType.EOS,
        )
        if msg is None:
            return "unknown error"
        if msg.type == self.Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            return f"{err}; {debug}"
        if msg.type == self.Gst.MessageType.WARNING:
            err, debug = msg.parse_warning()
            return f"{err}; {debug}"
        return str(msg.type)

    def _pull_rgb(self, sink):
        assert self.Gst is not None
        sample = sink.emit("try-pull-sample", int(1 * self.Gst.SECOND))
        if sample is None:
            return None
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        width = int(struct.get_value("width"))
        height = int(struct.get_value("height"))
        buf = sample.get_buffer()
        ok, map_info = buf.map(self.Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            expected = width * height * 3
            data = np.frombuffer(map_info.data, dtype=np.uint8)
            if data.size < expected:
                raise RuntimeError(f"Short GStreamer frame: got {data.size} bytes, expected {expected}")
            return data[:expected].reshape((height, width, 3)).copy()
        finally:
            buf.unmap(map_info)

    def _loop(self) -> None:
        assert self.left_sink is not None and self.right_sink is not None
        while not self.stop_event.is_set():
            t0 = time.monotonic_ns()
            left = self._pull_rgb(self.left_sink)
            right = self._pull_rgb(self.right_sink)
            t1 = time.monotonic_ns()
            if left is None or right is None:
                with self.cond:
                    self.error_count += 1
                    self.cond.notify_all()
                time.sleep(0.005)
                continue

            ts = (t0 + t1) // 2
            pair = common.StereoPair(
                pair_id=self.pair_id + 1,
                time_ns=ts,
                left_rgb=left,
                right_rgb=right,
                left_grab_mid_ns=ts,
                right_grab_mid_ns=ts,
                grab_span_ns=t1 - t0,
            )

            with self.cond:
                if self.last_pair_time_ns is not None:
                    self.pair_intervals_ms.append((pair.time_ns - self.last_pair_time_ns) / 1e6)
                self.last_pair_time_ns = pair.time_ns
                self.read_spans_ms.append((t1 - t0) / 1e6)
                self.pair_id = pair.pair_id
                self.latest_pair = pair
                self.cond.notify_all()

    def stop(self) -> None:
        self.stop_event.set()
        with self.cond:
            self.cond.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.pipeline is not None and self.Gst is not None:
            self.pipeline.set_state(self.Gst.State.NULL)


def build_argparser() -> argparse.ArgumentParser:
    parser = common.build_argparser()
    parser.description = "Run MAC-VO on a DECXIN-3261V single-USB stereo stream."
    parser.set_defaults(
        left=DECXIN_DEVICE_BY_ID,
        right="",
        width=640,
        height=480,
        vo_width=640,
        vo_height=480,
        camera_fps=30.0,
        fourcc="MJPG",
        baseline=0.06,
        fx=457.0,
        fy=457.0,
        cx=339.0,
        cy=249.0,
        max_software_skew_ms=1.0,
        resultRoot="./Results_decxin3261v_live",
    )
    parser.add_argument("--device", default=DECXIN_DEVICE_BY_ID)
    parser.add_argument("--raw-width", type=int, default=4000)
    parser.add_argument("--raw-height", type=int, default=1200)
    parser.add_argument("--eye-width", type=int, default=1920)
    parser.add_argument("--eye-height", type=int, default=1200)
    parser.add_argument("--left-x", type=int, default=2080)
    parser.add_argument("--right-x", type=int, default=160)
    parser.add_argument(
        "--capture-backend",
        choices=("opencv", "gst-v4l2"),
        default="opencv",
        help="DECXIN capture backend. gst-v4l2 decodes/crops/scales in GStreamer before Python.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    args.device = find_decxin_device(args.device)
    if not Path(args.device).exists() and not args.device.isdigit():
        common.Logger.write(
            "warn",
            f"DECXIN video node not found: {args.device}. "
            "If lsusb still shows the camera, unplug/replug it or move it to a USB3 port.",
        )
    args.left = args.device
    args.right = "single_uvc_split"
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

    common.Logger.write("warn", "DECXIN-3261V mode: single UVC frame is split horizontally into left/right images.")
    common.Logger.write("warn", "Use calibration maps once available; command-line intrinsics are only a temporary fallback.")

    rectifier = common.StereoRectifier(args.rectify_maps)
    if rectifier.enabled:
        assert rectifier.K is not None and rectifier.baseline is not None
        args.fx = float(rectifier.K[0, 0, 0])
        args.fy = float(rectifier.K[0, 1, 1])
        args.cx = float(rectifier.K[0, 0, 2])
        args.cy = float(rectifier.K[0, 1, 2])
        args.baseline = rectifier.baseline

    vo_width = args.width if args.vo_width is None else args.vo_width
    vo_height = args.height if args.vo_height is None else args.vo_height
    common.Logger.write(
        "info",
        f"Raw stream: {args.raw_width}x{args.raw_height}; eye crop: {args.eye_width}x{args.eye_height}; "
        f"capture/display: {args.width}x{args.height}; VO input: {vo_width}x{vo_height}",
    )

    reader_cls = GstSplitStereoCameraReader if args.capture_backend == "gst-v4l2" else SplitStereoCameraReader
    reader = reader_cls(
        args.device,
        args.raw_width,
        args.raw_height,
        args.camera_fps,
        args.fourcc,
        args.left_x,
        args.right_x,
        args.eye_width,
        args.eye_height,
        args.width,
        args.height,
    )
    common.run_realtime_from_reader(args, reader, rectifier, vo_width, vo_height, "DECXIN3261V-live")


if __name__ == "__main__":
    main()
