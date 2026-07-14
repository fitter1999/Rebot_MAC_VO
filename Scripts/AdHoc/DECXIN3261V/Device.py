from __future__ import annotations

import glob
import subprocess
from pathlib import Path


DECXIN_DEVICE_BY_ID = "/dev/v4l/by-id/usb-DECXIN_DECXIN_Camera_01.00.000-video-index0"
DECXIN_USB_VENDOR_ID = "1bcf"
DECXIN_USB_MODEL_ID = "2d50"


def _udev_properties(device: str) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["udevadm", "info", "--query=property", f"--name={device}"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return {}

    props: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            props[key] = value
    return props


def _is_decxin_video_node(device: str) -> bool:
    props = _udev_properties(device)
    vendor = props.get("ID_VENDOR_ID", "").lower()
    model = props.get("ID_MODEL_ID", "").lower()
    text = " ".join(
        props.get(key, "")
        for key in ("ID_MODEL", "ID_SERIAL", "ID_V4L_PRODUCT", "ID_VENDOR_FROM_DATABASE")
    ).lower()
    return (
        (vendor == DECXIN_USB_VENDOR_ID and model == DECXIN_USB_MODEL_ID)
        or "decxin" in text
    )


def find_decxin_device(requested: str | None = None) -> str:
    if requested in (None, "", "auto"):
        requested = None

    if requested and Path(requested).exists():
        return requested

    for candidate in sorted(glob.glob("/dev/v4l/by-id/*DECXIN*video-index0")):
        if Path(candidate).exists():
            return candidate

    for candidate in sorted(glob.glob("/dev/video*")):
        if _is_decxin_video_node(candidate):
            return candidate

    if requested:
        return requested
    return DECXIN_DEVICE_BY_ID
