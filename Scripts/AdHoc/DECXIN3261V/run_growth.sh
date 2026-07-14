#!/usr/bin/env bash
set -euo pipefail

cd /home/wjy/WJY/MAC-VO

exec ./run_macvo_wjy.sh python Scripts/AdHoc/DECXIN3261V/Run_Realtime.py \
  --odom Config/Experiment/MACVO/MACVO_DECXIN3261V_Realtime.yaml \
  --raw-width 4000 \
  --raw-height 1200 \
  --eye-width 1920 \
  --eye-height 1200 \
  --left-x 2080 \
  --right-x 160 \
  --capture-backend gst-v4l2 \
  --width 640 \
  --height 480 \
  --vo-width 320 \
  --vo-height 240 \
  --camera-fps 30 \
  --vo-fps 10 \
  --fourcc MJPG \
  --rectify-maps Calibration/decxin3261v_screen_640x480/calibration_result/rectify_maps.npz \
  --status-every 2 \
  --record-sequence \
  --rr-every 2 \
  --rr-fixed-bounds 6.0 \
  --rr-local-radius 1.0 \
  --rr-trail-frames 120 \
  --rr-max-points 300 \
  --rr-cov-mode none \
  --rr-no-image \
  --resultRoot ./Results_decxin3261v_growth \
  "$@"
