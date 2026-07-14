#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  exec sudo "$0" "$@"
fi

echo "[macvo] CPU governor -> performance"
for p in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [[ -w "$p" ]] && echo performance > "$p" || true
done

echo "[macvo] CPU EPP -> performance"
for p in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
  [[ -w "$p" ]] && echo performance > "$p" || true
done

echo "[macvo] Disable USB autosuspend for DECXIN and root hubs"
for dev in /sys/bus/usb/devices/4-2.1 /sys/bus/usb/devices/4-2 /sys/bus/usb/devices/usb4; do
  [[ -w "$dev/power/control" ]] && echo on > "$dev/power/control" || true
done

if [[ -w /sys/module/usbcore/parameters/autosuspend ]]; then
  echo -1 > /sys/module/usbcore/parameters/autosuspend || true
fi

echo "[macvo] Keep NVIDIA PCIe device awake"
for dev in /sys/bus/pci/devices/*; do
  if [[ -f "$dev/vendor" ]] && [[ "$(cat "$dev/vendor")" == "0x10de" ]]; then
    [[ -w "$dev/power/control" ]] && echo on > "$dev/power/control" || true
  fi
done

echo "[macvo] Try NVIDIA persistence / power limit"
nvidia-smi -pm 1 >/dev/null 2>&1 || true
nvidia-smi -pl 65 >/dev/null 2>&1 || true

echo "[macvo] Done. Current status:"
grep -H . /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor \
          /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference \
          /sys/bus/usb/devices/4-2.1/power/control 2>/dev/null || true
nvidia-smi --query-gpu=power.limit,pstate,clocks.sm,clocks.mem,power.draw --format=csv,noheader,nounits 2>/dev/null || true
