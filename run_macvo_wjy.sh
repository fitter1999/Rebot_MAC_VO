#!/usr/bin/env bash
set -euo pipefail

export TMPDIR=/home/wjy/WJY/.cache/tmp
export XDG_CACHE_HOME=/home/wjy/WJY/.cache/macvo_mamba
export HOME=/home/wjy/WJY
export PIP_CACHE_DIR=/home/wjy/WJY/.cache/pip
export MAMBA_ROOT_PREFIX=/home/wjy/WJY/miniforge3
export MAMBA_PKGS_DIRS=/home/wjy/WJY/.cache/mamba/pkgs

cd /home/wjy/WJY/MAC-VO
exec /home/wjy/WJY/miniforge3/micromamba run -n macvo_wjy "$@"
