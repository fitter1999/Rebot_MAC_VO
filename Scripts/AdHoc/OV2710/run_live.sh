#!/usr/bin/env bash
set -euo pipefail

cd /home/wjy/WJY/MAC-VO

exec ./run_macvo_wjy.sh python Scripts/AdHoc/OV2710/Run_Realtime.py \
  --odom Config/Experiment/MACVO/MACVO_OV2710_Realtime.yaml \
  --left /dev/v4l/by-path/pci-0000:80:14.0-usb-0:1.1.1:1.0-video-index0 \
  --right /dev/v4l/by-path/pci-0000:80:14.0-usb-0:1.1.4:1.0-video-index0 \
  --width 640 \
  --height 480 \
  --vo-width 320 \
  --vo-height 240 \
  --camera-fps 30 \
  --vo-fps 10 \
  --fourcc MJPG \
  --rectify-maps Calibration/ov2710_screen/calibration_result/rectify_maps.npz \
  --status-every 5 \
  --rr-fixed-bounds 2.5 \
  --rr-local-radius 1.5 \
  --rr-trail-frames 80 \
  --rr-max-points 200 \
  --rr-cov-mode none \
  --resultRoot ./Results_ov2710_live \
  "$@"
