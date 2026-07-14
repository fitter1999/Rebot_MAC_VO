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
  --vo-width 640 \
  --vo-height 480 \
  --camera-fps 30 \
  --vo-fps 10 \
  --fourcc MJPG \
  --rectify-maps Calibration/decxin3261v_screen_640x480/calibration_result/rectify_maps.npz \
  --baseline 0.06 \
  --fx 457 \
  --fy 457 \
  --cx 339 \
  --cy 249 \
  --status-every 5 \
  --rr-fixed-bounds 2.5 \
  --rr-local-radius 1.5 \
  --rr-trail-frames 80 \
  --rr-max-points 200 \
  --rr-cov-mode none \
  --resultRoot ./Results_decxin3261v_live \
  "$@"
