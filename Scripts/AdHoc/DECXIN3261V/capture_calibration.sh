#!/usr/bin/env bash
set -euo pipefail

cd /home/wjy/WJY/MAC-VO

exec ./run_macvo_wjy.sh python Scripts/AdHoc/DECXIN3261V/Capture_Calibration.py "$@"
