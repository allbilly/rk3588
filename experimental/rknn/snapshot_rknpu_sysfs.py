#!/usr/bin/env python3
"""Read non-privileged RK3588 RKNPU runtime/devfreq state."""

import argparse
from pathlib import Path


SYSFS_FILES = (
    "/sys/devices/platform/fdab0000.npu/power/runtime_status",
    "/sys/devices/platform/fdab0000.npu/power/runtime_usage",
    "/sys/devices/platform/fdab0000.npu/power/runtime_active_time",
    "/sys/devices/platform/fdab0000.npu/power/runtime_suspended_time",
    "/sys/devices/platform/fdab0000.npu/power/control",
    "/sys/class/devfreq/fdab0000.npu/cur_freq",
    "/sys/class/devfreq/fdab0000.npu/target_freq",
    "/sys/class/devfreq/fdab0000.npu/governor",
    "/sys/class/devfreq/fdab0000.npu/load",
    "/sys/class/devfreq/fdab0000.npu/trans_stat",
)


def read_text(path):
    try:
        return Path(path).read_text(errors="replace").strip().replace("\n", "\\n")
    except OSError as exc:
        return f"<unreadable:{exc.errno}>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="snapshot")
    args = parser.parse_args()

    print(f"SYSFS_SNAPSHOT {args.label}")
    for path in SYSFS_FILES:
        print(f"  {path} = {read_text(path)}")


if __name__ == "__main__":
    main()
