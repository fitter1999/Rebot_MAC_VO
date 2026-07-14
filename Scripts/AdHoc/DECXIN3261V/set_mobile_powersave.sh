#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  exec sudo "$0" "$@"
fi

echo "[macvo] CPU governor -> powersave"
for p in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [[ -w "$p" ]] && echo powersave > "$p" || true
done

echo "[macvo] CPU EPP -> balance_performance"
for p in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
  [[ -w "$p" ]] && echo balance_performance > "$p" || true
done

echo "[macvo] Restore USB autosuspend policy"
for dev in /sys/bus/usb/devices/4-2.1 /sys/bus/usb/devices/4-2 /sys/bus/usb/devices/usb4; do
  [[ -w "$dev/power/control" ]] && echo auto > "$dev/power/control" || true
done

nvidia-smi -pm 0 >/dev/null 2>&1 || true
echo "[macvo] Done."
