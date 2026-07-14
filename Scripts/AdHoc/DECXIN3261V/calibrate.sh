#!/usr/bin/env bash
set -euo pipefail

cd /home/wjy/WJY/MAC-VO

exec ./run_macvo_wjy.sh python Scripts/AdHoc/DECXIN3261V/Calibrate.py \
  --input Calibration/decxin3261v_screen_640x480 \
  --known-baseline-m 0.06 \
  --square-mm 28 \
  "$@"
