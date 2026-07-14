#!/usr/bin/env bash
set -euo pipefail

cd /home/wjy/WJY/MAC-VO

exec Scripts/AdHoc/OV2710/run_live.sh "$@"
